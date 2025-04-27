import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Temper(nn.Module):
    def __init__(self, hidden_dim, id, num_ops=3):
        super().__init__()
        self.id = id
        self.hidden_dim = hidden_dim
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(num_ops)])

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
        return out

class PredictiveHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.predictor(x)

class TemperGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers=12, max_path_hops=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tempers = num_tempers
        self.max_path_hops = max_path_hops
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tempers = nn.ModuleList([Temper(hidden_dim, id=i) for i in range(num_tempers)])
        self.routing_policy = RoutingPolicy(hidden_dim, hidden_dim, num_tempers)
        self.predictive_head = PredictiveHead(hidden_dim, input_dim)

        self.routing_stats = []

    def forward(self, x):
        batch_size, input_dim = x.shape
        x_proj = self.input_proj(x)
        patch_size = self.hidden_dim
        num_patches = x_proj.shape[1] // patch_size
        patches = x_proj.view(batch_size * num_patches, patch_size)

        patch_states = patches
        patch_tempers = torch.randint(0, self.num_tempers, (patches.size(0),), device=x.device)
        patch_done = torch.zeros(patches.size(0), dtype=torch.bool, device=x.device)

        for _ in range(self.max_path_hops):
            active_mask = ~patch_done
            if active_mask.sum() == 0:
                break

            active_states = patch_states[active_mask]
            active_tempers = patch_tempers[active_mask]

            sorted_tempers, sorted_indices = active_tempers.sort()
            sorted_states = active_states[sorted_indices]

            temper_boundaries = (sorted_tempers[1:] != sorted_tempers[:-1]).nonzero(as_tuple=False).squeeze(1) + 1
            temper_patch_batches = torch.tensor_split(sorted_states, temper_boundaries.tolist())
            temper_ids = sorted_tempers.unique()

            outputs = torch.zeros_like(active_states)
            logits = torch.zeros(active_states.size(0), self.num_tempers + 1, device=x.device)

            idx = 0
            for tid, patch_batch in zip(temper_ids, temper_patch_batches):
                out = self.tempers[tid](patch_batch)
                outputs[idx:idx + patch_batch.size(0)] = out
                logits[idx:idx + patch_batch.size(0)] = self.routing_policy(out)
                idx += patch_batch.size(0)

            # Sample next tempers
            probs = F.softmax(logits, dim=-1)
            sampled_tempers = torch.multinomial(probs, num_samples=1).squeeze(-1)

            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

            done_now = (sampled_tempers == self.num_tempers)

            # Update states
            patch_states[active_mask] = outputs
            patch_tempers[active_mask] = sampled_tempers.clamp(max=self.num_tempers - 1)
            patch_done[active_mask] |= done_now

        # Aggregate
        patch_states = patch_states.view(batch_size, num_patches, patch_size)
        final_output = patch_states.mean(dim=1)

        # Predict next sensory latent
        predicted_next_latent = self.predictive_head(final_output)

        # Compute prediction error
        prediction_error = F.mse_loss(predicted_next_latent, x, reduction='none').mean(dim=-1)

        return predicted_next_latent, prediction_error

    def print_epoch_summary(self, epoch, loss):
        if not self.routing_stats:
            print(f"Epoch {epoch} | No routing stats collected.")
            return

        print(f"=== Epoch {epoch} Routing Summary ===")

        stats = torch.stack(self.routing_stats)  # [num_hops, num_tempers+1]

        total_patches = stats[0].sum().item()
        stop_counts = stats[:, -1]
        active_counts = total_patches - stop_counts.cumsum(dim=0)

        print(f"Initial patches: {total_patches}")
        print(f"Final active patches after {len(stats)} hops: {active_counts[-1].item()}")
        
        for hop, (temper_counts, stop_count) in enumerate(zip(stats[:, :-1], stop_counts)):
            print(f" Hop {hop}: Active: {active_counts[hop].item()} | STOP: {stop_count.item()} | Temper distribution: {temper_counts.tolist()}")
        print("")
        self.routing_stats.clear()
