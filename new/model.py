import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon

        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim), requires_grad=False)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", torch.randn(num_codes, code_dim))

    def forward(self, x):  # [B, S, D]
        B, S, D = x.shape
        flat_x = x.view(-1, D)
        codebook = self.codebook.to(x.device)

        distances = torch.cdist(flat_x, codebook)  # [B*S, num_codes]
        indices = distances.argmin(dim=-1)         # [B*S]
        quantized = codebook[indices].view(B, S, codebook.shape[1])

        if self.training:
            one_hot = F.one_hot(indices, self.num_codes).type_as(flat_x)
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.T @ flat_x

            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * cluster_size
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * embed_sum

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.codebook.data.copy_(embed_normalized)

            if torch.rand(1).item() < 0.01:
                print(f"[EMA Codebook Usage] {self.cluster_size.int().tolist()}")

        return quantized, indices.view(B, S)


class TensorBus:
    def __init__(self, max_messages, batch_size, seq_len, symbolic_dim, latent_dim, device):
        self.max_messages = max_messages
        self.symbolic_dim = symbolic_dim
        self.latent_dim = latent_dim
        self.device = device

        self._init_buffers(batch_size, seq_len)

    def _init_buffers(self, batch_size, seq_len):
        self.B = batch_size
        self.S = seq_len
        self.ptr = 0
        self.valid = 0

        self.symbol_buffer = torch.zeros(self.max_messages, self.B, self.S, self.symbolic_dim, device=self.device)
        self.index_buffer = torch.zeros(self.max_messages, self.B, self.S, dtype=torch.long, device=self.device)
        self.output_buffer = torch.zeros(self.max_messages, self.B, self.S, self.latent_dim, device=self.device)
        self.message_mask = torch.zeros(self.max_messages, dtype=torch.bool, device=self.device)

    def maybe_resize(self, batch_size, seq_len):
        if batch_size != self.B or seq_len != self.S:
            self._init_buffers(batch_size, seq_len)

    def reset(self):
        with torch.no_grad():
            self.ptr = 0
            self.valid = 0
            self.symbol_buffer.zero_()
            self.index_buffer.zero_()
            self.output_buffer.zero_()
            self.message_mask.zero_()

    def append(self, symbol, symbol_idx, output):
        with torch.no_grad():
            if self.valid >= self.max_messages:
                self.symbol_buffer = torch.roll(self.symbol_buffer, shifts=-1, dims=0)
                self.index_buffer = torch.roll(self.index_buffer, shifts=-1, dims=0)
                self.output_buffer = torch.roll(self.output_buffer, shifts=-1, dims=0)
                self.message_mask = torch.roll(self.message_mask, shifts=-1, dims=0)
                self.ptr = self.max_messages - 1
                self.valid = self.max_messages - 1

            self.symbol_buffer[self.ptr] = symbol
            self.index_buffer[self.ptr] = symbol_idx
            self.output_buffer[self.ptr] = output
            self.message_mask[self.ptr] = True
            self.ptr += 1
            self.valid += 1

    def get_active(self, return_indices=False):
        act_mask = self.message_mask[:self.valid]
        indices = act_mask.nonzero(as_tuple=False).squeeze(1)
        outputs = (
            self.symbol_buffer[:self.valid][act_mask],
            self.index_buffer[:self.valid][act_mask],
            self.output_buffer[:self.valid][act_mask],
            act_mask.clone()
        )
        if return_indices:
            return *outputs, indices
        return outputs

    def init_global_mask(self):
        with torch.no_grad():
            return torch.ones(self.valid, dtype=torch.bool, device=self.device)

    def update_mask(self, global_mask, active_indices, updated_mask):
        with torch.no_grad():
            device = global_mask.device
            active_indices = active_indices.to(device)
            updated_mask = updated_mask.to(device)

            new_values = global_mask[active_indices] & updated_mask
            return global_mask.clone().scatter(0, active_indices, new_values)

    def mark_consumed(self, updated_mask):
        with torch.no_grad():
            if updated_mask.shape[0] != self.valid:
                raise ValueError(f"mask shape {updated_mask.shape} doesn't match valid messages {self.valid}")
            self.message_mask[:self.valid] = updated_mask


class BusNode(nn.Module):
    def __init__(self, latent_dim, symbolic_dim, num_codes, target_code_id):
        super().__init__()
        self.target_code_id = target_code_id
        self.symbol_proj = nn.Linear(latent_dim, symbolic_dim)
        self.query_proj = nn.Linear(symbolic_dim, 1)
        self.read_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.compute = nn.Sequential(
            nn.Linear(latent_dim + symbolic_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.quantizer = EMAQuantizer(num_codes=num_codes, code_dim=symbolic_dim)
        
    def forward(self, token_state, bus_symbols, bus_indices, bus_outputs, bus_mask):
        device = token_state.device
        bus_symbols = bus_symbols.to(device)
        bus_indices = bus_indices.to(device)
        bus_outputs = bus_outputs.to(device)
        bus_mask = bus_mask.to(device)
        B, S, _ = token_state.shape

        if bus_symbols is None or bus_symbols.shape[0] == 0:
            empty_symbol = torch.zeros(B, S, self.quantizer.code_dim, device=device)
            empty_index = torch.zeros(B, S, dtype=torch.long, device=device)
            return token_state, empty_symbol, empty_index, bus_mask

        T, B, S, D = bus_symbols.shape
        N = B * S

        bus_syms_flat = bus_symbols.view(T, N, -1)      # [T, N, sym_dim]
        bus_outs_flat = bus_outputs.view(T, N, -1)      # [T, N, latent_dim]

        relevance = self.query_proj(bus_syms_flat).squeeze(-1)  # [T, N]
        top_msg_idx = relevance.argmax(dim=0)                   # [N]

        # Gather best message outputs
        chosen_outs = bus_outs_flat[top_msg_idx, torch.arange(N)]  # [N, latent_dim]
        bus_context = chosen_outs.view(B, S, -1)                    # [B, S, latent_dim]

        z = torch.cat([token_state, bus_context], dim=-1)
        z_read = self.read_proj(z)
        raw_symbol = self.symbol_proj(z_read)
        quantized_symbol, symbol_indices = self.quantizer(raw_symbol)

        node_output = self.compute(torch.cat([z_read, quantized_symbol], dim=-1)) + token_state

        # Track which messages were used (consumed)
        used_counts = torch.bincount(top_msg_idx, minlength=T)  # [T]
        keep_mask = (used_counts == 0)                          # [T] — keep only unused messages

        return node_output, quantized_symbol, symbol_indices, keep_mask


class BusSynthesizer(nn.Module):
    def __init__(self, input_dim, latent_dim, symbolic_dim, num_nodes=4, num_codes=32,
                 max_ops=4, max_bus=32, halt_eps=1e-3, max_seq_len=256, max_batch_size=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.token_prompts = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))

        self.nodes = nn.ModuleList([
            BusNode(latent_dim, symbolic_dim, num_codes=num_codes, target_code_id=i)
            for i in range(num_nodes)
        ])
        self.max_ops = max_ops
        self.latent_dim = latent_dim
        self.symbolic_dim = symbolic_dim
        self.halt_eps = halt_eps
        self.max_bus = max_bus

        # Instantiate persistent bus
        self.bus = TensorBus(
            max_messages=max_bus,
            batch_size=max_batch_size,
            seq_len=max_seq_len,
            symbolic_dim=symbolic_dim,
            latent_dim=latent_dim,
            device=torch.device("cpu")  # This will be updated during forward
        )

    def forward(self, x, return_program=False):
        B, S, _ = x.shape
        device = x.device

        self.bus.maybe_resize(B, S)
        self.bus.device = device
        self.bus.reset()

        out = self.input_proj(x) + self.token_prompts[:, :S, :]
        prev_out = out.clone()
        active = torch.ones(B, S, dtype=torch.bool, device=device)
        token_tags = torch.zeros(B, S, self.symbolic_dim, device=device)
        program_trace = []

        self.bus.append(token_tags.clone(), torch.zeros(B, S, dtype=torch.long, device=device), out.clone())

        for t in range(self.max_ops):
            step_ids = torch.zeros(B, S, dtype=torch.long, device=device)

            bus_syms, bus_idxs, bus_outs, active_mask, active_indices = self.bus.get_active(return_indices=True)

            if active_indices.numel() == 0:
                break  # No active messages left

            masks = []  # Will collect (indices, local_mask) tuples

            for node_id, node in enumerate(self.nodes):
                assigned = active if t > 0 else torch.ones_like(active)

                node_out, symbol, symbol_idx, updated_mask = node(
                    out, bus_syms, bus_idxs, bus_outs, active_mask
                )

                delta = (node_out - prev_out).norm(dim=-1)
                halt = (delta < self.halt_eps) & active
                update_mask = assigned & ~halt

                out = torch.where(update_mask.unsqueeze(-1), node_out, out)
                token_tags = torch.where(update_mask.unsqueeze(-1), symbol, token_tags)
                step_ids = torch.where(update_mask, torch.full_like(step_ids, node_id), step_ids)

                self.bus.append(symbol, symbol_idx, node_out)
                masks.append((active_indices.clone(), updated_mask.clone()))

                active = active & ~halt
                prev_out = out.clone()
                print("Active messages on bus:", self.bus.message_mask[:self.bus.valid].sum().item())
            # After all appends, valid has increased — now safe to init full-length mask
            global_mask = self.bus.init_global_mask()
            for indices, mask in masks:
                global_mask = self.bus.update_mask(global_mask, indices, mask)

            self.bus.mark_consumed(global_mask)
            program_trace.append(step_ids)

            if not active.any():
                break

        if return_program:
            return out, torch.stack(program_trace, dim=-1)
        return out
