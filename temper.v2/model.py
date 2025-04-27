import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# --- TemperController ---
class TemperController(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, novelty, conflict, plasticity, moving_avg_reward, usage_count):
        device = self.policy_net[0].weight.device
        stats = torch.tensor(
            [novelty, conflict, plasticity, moving_avg_reward, usage_count / 1000.0],
            dtype=torch.float32
        ).to(device)
        return self.policy_net(stats)

# --- RoutingPolicy ---
class RoutingPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tempers + 1)  # +1 for STOP
        )

    def forward(self, input):
        logits = self.net(input)
        return logits

# --- Temper ---
class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim, id, max_ops=6):
        super().__init__()
        self.id = id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_ops = max_ops

        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(3)])
        self.routing_head = nn.Linear(hidden_dim, 12 + 1)
        self.controller = TemperController(hidden_dim)

    def make_operator(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        weights = F.softmax(torch.randn(len(self.operator_bank), device=x.device), dim=0)
        out = torch.zeros_like(x)

        for op, w in zip(self.operator_bank, weights):
            out += w * op(x)

        next_logits = self.routing_head(out)
        return out, next_logits

class TemperGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tempers=12, max_path_hops=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tempers = num_tempers
        self.max_path_hops = max_path_hops
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tempers = nn.ModuleList([Temper(input_dim, hidden_dim, id=i) for i in range(num_tempers)])
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.routing_policy = RoutingPolicy(input_dim, hidden_dim, num_tempers)
        self.routing_stats = []
        
    def forward(self, x):
        batch_size, _ = x.shape
        x_proj = self.input_proj(x)
        patch_size = self.hidden_dim
        num_patches = x_proj.shape[1] // patch_size
        patches = x_proj.view(batch_size * num_patches, patch_size)

        patch_states = patches 
        patch_states = patches
        patch_tempers = torch.randint(0, self.num_tempers, (patches.size(0),), device=x.device)
        patch_done = torch.zeros(patches.size(0), dtype=torch.bool, device=x.device)

        for hop in range(self.max_path_hops):
            active_mask = ~patch_done
            if active_mask.sum() == 0:
                break

            active_states = patch_states[active_mask]
            active_tempers = patch_tempers[active_mask]

            # Scatter patches to tempers
            sorted_tempers, sorted_indices = active_tempers.sort()
            sorted_states = active_states[sorted_indices]

            temper_boundaries = (sorted_tempers[1:] != sorted_tempers[:-1]).nonzero(as_tuple=False).squeeze(1) + 1
            temper_patch_batches = torch.tensor_split(sorted_states, temper_boundaries.tolist())
            temper_ids = sorted_tempers.unique()

            outputs = torch.zeros_like(active_states)
            logits = torch.zeros(active_states.size(0), self.num_tempers + 1, device=x.device)

            idx = 0
            for tid, patch_batch in zip(temper_ids, temper_patch_batches):
                out, next_logits = self.tempers[tid](patch_batch)
                outputs[idx:idx + patch_batch.size(0)] = out
                logits[idx:idx + patch_batch.size(0)] = next_logits
                idx += patch_batch.size(0)

            # Sample next tempers
            probs = F.softmax(logits, dim=-1)
            sampled_tempers = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # ⚡ Lightning Routing Debugger
            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

            done_now = (sampled_tempers == self.num_tempers)

            # Update global states
            patch_states[active_mask] = outputs
            patch_tempers[active_mask] = sampled_tempers.clamp(max=self.num_tempers - 1)
            patch_done[active_mask] |= done_now

        # Aggregate final outputs
        patch_states = patch_states.view(batch_size, num_patches, patch_size)
        final_output = patch_states.mean(dim=1)

        return self.readout(final_output)

    def print_epoch_summary(self, epoch, loss):
        if not self.routing_stats:
            print(f"Epoch {epoch} | No routing stats collected.")
            return

        print(f"\n=== Epoch {epoch} Routing Summary ===")

        stats = torch.stack(self.routing_stats)  # [num_hops, num_tempers+1]

        total_patches = stats[0].sum().item()
        stop_counts = stats[:, -1]
        active_counts = total_patches - stop_counts.cumsum(dim=0)

        print(f"Initial patches: {total_patches}")
        print(f"Final active patches after {len(stats)} hops: {active_counts[-1].item()}")
        
        for hop, (temper_counts, stop_count) in enumerate(zip(stats[:, :-1], stop_counts)):
            print(f" Hop {hop}: Active: {active_counts[hop].item()} | STOP: {stop_count.item()} | Temper distribution: {temper_counts.tolist()}")

        self.routing_stats.clear()

