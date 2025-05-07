import torch
import torch.nn as nn
import time

class Synthesizer(nn.Module):
    def __init__(self, nodes, router, gem, input_dim, latent_dim, device='cuda'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim).to(device)
        self.nodes = nn.ModuleList(nodes)
        self.router = router
        self.gem = gem
        self.device = device
        self.last_symbolic = None
        self.last_programs = None

    def forward(self, x, max_ops=4):
        batch_size, dim = x.size()

        # Step 0: Project input to latent space
        z = self.input_proj(x)  # (batch, latent_dim)

        # Step 1: Emit ops + symbolic embeds
        op_list = []
        symbolic_embeds = []
        for node in self.nodes:
            op, sym_embed = node.emit_operator(z)
            op_list.append(op)
            symbolic_embeds.append(sym_embed)
        symbolic_embeds = torch.stack(symbolic_embeds, dim=1)  # (batch, num_nodes, sym_dim)

        # Step 2: Query GEM by symbolic embedding
        gem_embeds = None
        if len(self.gem.symbolic_embeds) > 0:
            with torch.no_grad():
                query_vec = symbolic_embeds.mean(dim=(0, 1)).detach().cpu().numpy()
                retrieved = self.gem.retrieve(query_vec)
                if retrieved:
                    gem_embeds = torch.stack([r['symbolic'].to(self.device) for r in retrieved], dim=0)

        # Step 3: Router selects program
        program_indices = self.router(z, symbolic_embeds, gem_embeds, max_ops)  # (batch, max_ops)

        # Step 4: Apply selected ops (batched by op index)
        out = z.clone()
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for hop in range(max_ops):
            hop_idx = program_indices[:, hop]
            still_active = active_mask & (hop_idx != self.router.num_nodes)

            if not still_active.any():
                break

            selected_ops = hop_idx[still_active]
            x_active = out[still_active]

            # NEW: group samples by op index
            unique_ops = selected_ops.unique()
            out_active = torch.empty_like(x_active)

            for idx in unique_ops:
                idx_mask = selected_ops == idx
                op_inputs = x_active[idx_mask]
                op = op_list[idx]
                op_outputs = op(op_inputs)  # batched forward
                out_active[idx_mask] = op_outputs

            # Write back to full output tensor
            out[still_active] = out_active

            active_mask = still_active

        # Cache symbolic embedding + programs for GEM update
        self.last_symbolic = symbolic_embeds.detach()
        self.last_programs = program_indices.detach()

        return out, program_indices, symbolic_embeds

    def update_gem(self, probs, target):
        start = time.time()
        batch_size = target.size(0)
        for b in range(batch_size):
            if self.last_programs[b].numel() > 0 and not (self.last_programs[b] == self.router.num_nodes).all():
                prob = probs[b, target[b]].item()  # probability assigned to correct class
                avg_symbolic = self.last_symbolic[b].mean(dim=0)  # (sym_dim,)
                self.gem.insert(avg_symbolic, self.last_programs[b], prob)
        print(f"Update gem time: {time.time() - start:.4f}s")
