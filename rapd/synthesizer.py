import torch
import torch.nn as nn
import numpy as np
import time

class Synthesizer(nn.Module):
    def __init__(self, nodes, router, gem, input_dim, latent_dim, device='cuda'):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim).to(device)
        self.nodes = nn.ModuleList(nodes)
        self.router = router
        self.gem = gem
        self.device = device

        # Buffers to collect batch updates
        self.collected_symbolic = []
        self.collected_programs = []

    def forward(self, x, max_ops=4):
        batch_size, _dim = x.size()

        z = self.input_proj(x)
        op_list, symbolic_embeds = [], []
        for node in self.nodes:
            op, sym_embed = node.emit_operator(z)
            op_list.append((op, sym_embed))
            symbolic_embeds.append(sym_embed)
        symbolic_embeds = torch.stack(symbolic_embeds, dim=1)

        gem_embeds = []

        # Retrieve nearest GEM neighbors
        if len(self.gem.symbolic_embeds) > 0:
            with torch.no_grad():
                query_vec = symbolic_embeds.mean(dim=(0, 1)).detach().cpu().numpy()
                retrieved = self.gem.retrieve(query_vec, 1000)
                if retrieved:
                    gem_embeds.append(
                        (torch.stack([r['symbolic'].to(self.device) for r in retrieved], dim=0),
                        torch.tensor([r['reward'] for r in retrieved], device=self.device).unsqueeze(1))
                    )

        # Inject top-k global high-reward programs
        top_global = self.gem.get_top_k(k=20)
        if top_global:
            gem_embeds.append(
                (torch.stack([r['symbolic'].to(self.device) for r in top_global], dim=0),
                torch.tensor([r['reward'] for r in top_global], device=self.device).unsqueeze(1))
            )

        # Inject mutated top programs directly
        mutated_programs = self.gem.mutate_top_programs(k=20, num_mutations=5, num_nodes=self.router.num_nodes, max_ops=max_ops)
        if mutated_programs:
            mut_embeds = []
            mut_rewards = []
            for mut_prog in mutated_programs:
                dummy_embed = torch.randn(self.router.symbolic_proj.in_features, device=self.device)
                mut_embeds.append(dummy_embed)
                mut_rewards.append(0.0)
                # Optionally: also prestore in GEM memory
                self.gem.symbolic_embeds.append(dummy_embed.cpu())
                self.gem.programs.append(mut_prog)
                self.gem.rewards.append(0.0)
            gem_embeds.append(
                (torch.stack(mut_embeds, dim=0), torch.tensor(mut_rewards, device=self.device).unsqueeze(1))
            )

        # Combine all GEM sources
        if gem_embeds:
            all_embeds = torch.cat([pair[0] for pair in gem_embeds], dim=0)  # (total_k, sym_dim)
            all_rewards = torch.cat([pair[1] for pair in gem_embeds], dim=0).squeeze(1)  # (total_k,)

            gem_embeds = (all_embeds, all_rewards)
        else:
            gem_embeds = None

        program_indices = self.router(z, symbolic_embeds, gem_embeds, max_ops)
        
        out = z.clone()
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for hop in range(max_ops):
            hop_idx = program_indices[:, hop]
            reached_sentinel = hop_idx == self.router.num_nodes
            done_mask |= reached_sentinel

            still_active = active_mask & ~done_mask
            if not still_active.any():
                break

            selected_ops = hop_idx[still_active]
            x_active = out[still_active]
            unique_ops = selected_ops.unique()
            out_active = torch.empty_like(x_active)

            for idx in unique_ops:
                idx_mask = selected_ops == idx
                op_inputs = x_active[idx_mask]

                # Unpack both the op and symbolic embedding
                op_fn, symbolic_batch = op_list[idx]

                # Select only the symbolic vectors for still-active + idx-matching samples
                symbolic_inputs = symbolic_batch[still_active][idx_mask]

                # Apply op (passing both latent and symbolic slice)
                op_outputs = op_fn(op_inputs, symbolic_inputs)

                out_active[idx_mask] = op_outputs

            out[still_active] = out_active

        if self.training:
            for i in range(batch_size):
                self.collected_symbolic.append(symbolic_embeds[i].detach().cpu())
                self.collected_programs.append(program_indices[i].detach().cpu())

        return out, program_indices, symbolic_embeds

    def update_gem(self, rewards):
        print("starting gem batch insert")
        start_time = time.time()
        all_symbolic = torch.stack(self.collected_symbolic, dim=0)  # (N, num_nodes, sym_dim)
        all_programs = torch.stack(self.collected_programs, dim=0)  # (N, max_ops)

        embeddings_np = []
        programs_list = []
        rewards_list = []

        total_entries = all_programs.size(0)
        decay = 0.001

        for b in range(total_entries):
            prog = all_programs[b]
            sentinel_idx = (prog == self.router.num_nodes).nonzero(as_tuple=True)
            if len(sentinel_idx[0]) > 0:
                prog = prog[:sentinel_idx[0][0]]
            if prog.numel() > 0:
                avg_symbolic = all_symbolic[b].mean(dim=0)
                program_length = prog.numel()
                base_reward = rewards[b].item()
                effective_reward = np.float32(base_reward * np.exp(-decay * (program_length - 1)))
                embeddings_np.append(avg_symbolic.numpy())
                rewards_list.append(effective_reward)
                programs_list.append(prog.tolist())

        if embeddings_np:
            embeddings_np = np.stack(embeddings_np, axis=0)
            self.gem.insert_batch(embeddings_np, programs_list, rewards_list)

        self.collected_symbolic = []
        self.collected_programs = []
        duration = time.time() - start_time
        print(f"batch insert finished in {duration:.4f}s")
