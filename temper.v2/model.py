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
        self.baseline = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.decay = 0.99
        self.saved_log_probs = []

    def forward(self, input):
        logits = self.net(input)
        return logits

    def reinforce(self, reward):
        if not self.saved_log_probs:
            return torch.tensor(0.0, device=reward.device)

        advantage = reward - self.baseline
        self.baseline.data = self.decay * self.baseline.data + (1 - self.decay) * reward.detach()

        policy_loss = 0.0
        for log_prob in self.saved_log_probs:
            policy_loss += -log_prob.mean() * advantage
        return policy_loss

    def clear_log_probs(self):
        self.saved_log_probs.clear()

class PredictiveHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.predictor(x)

class IntrinsicGrowthController(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [grow_signal, prune_signal]
        )

        self.register_buffer('input_mean', torch.zeros(hidden_dim))
        self.register_buffer('reward_moving_avg', torch.tensor(0.0))
        self.register_buffer('reward_moving_var', torch.tensor(1.0))
        self.reward_beta = 0.99

        self.global_alpha = nn.Parameter(torch.tensor(0.01))

    def compute_intrinsic_signals(self, x, out, operator_usage):
        device = x.device

        self.input_mean = self.input_mean.to(device)
        self.reward_moving_avg = self.reward_moving_avg.to(device)
        self.reward_moving_var = self.reward_moving_var.to(device)

        operator_usage = operator_usage.to(device)

        noise = torch.randn_like(x) * 0.01
        out_noisy = out + torch.randn_like(out) * 0.01
        plasticity = ((out_noisy - out) ** 2).mean(dim=-1)

        novelty = ((x - self.input_mean) ** 2).mean(dim=-1)

        prediction_error = ((out - x) ** 2).mean(dim=-1)

        usage_probs = operator_usage.float() / (operator_usage.sum() + 1e-6)
        usage_entropy = -(usage_probs * usage_probs.clamp(min=1e-6).log()).sum()

        sparsity = out.abs().mean(dim=-1)

        reward_estimate = prediction_error.detach()
        reward_delta = self.reward_moving_avg - reward_estimate
        self.reward_moving_avg = self.reward_beta * self.reward_moving_avg + (1 - self.reward_beta) * reward_estimate.mean()

        reward_var = (reward_estimate - self.reward_moving_avg).pow(2).mean()

        batch_mean = x.detach().mean(dim=0)
        self.input_mean = 0.99 * self.input_mean + 0.01 * batch_mean

        signals = torch.stack([
            plasticity.mean(),
            novelty.mean(),
            prediction_error.mean(),
            usage_entropy,
            sparsity.mean(),
            reward_delta.mean(),
            reward_var
        ], dim=0)

        return signals

    def forward(self, x, out, operator_usage, global_signal=None):
        signals = self.compute_intrinsic_signals(x, out, operator_usage)

        if global_signal is not None:
            signals = signals + self.global_alpha * global_signal

        decision = self.net(signals.unsqueeze(0))
        grow_signal, prune_signal = decision.squeeze(0)
        return grow_signal, prune_signal

class Temper(nn.Module):
    def __init__(self, hidden_dim, id, num_ops=3):
        super().__init__()
        self.id = id
        self.hidden_dim = hidden_dim
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(num_ops)])
        self.operator_usage = torch.zeros(len(self.operator_bank), dtype=torch.long)
        self.growth_controller = IntrinsicGrowthController(hidden_dim)

    def make_operator(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

    def apply_operators(self, x):
        if len(self.operator_bank) == 0:
            return x

        weights = F.softmax(torch.randn(len(self.operator_bank), device=x.device), dim=0)
        out = torch.zeros_like(x)
        for idx, (op, w) in enumerate(zip(self.operator_bank, weights)):
            out += w * op(x)
            self.operator_usage[idx] += 1
        return out

    def forward(self, x, global_signal=None):
        out = self.apply_operators(x)
        grow_signal, prune_signal = self.growth_controller(x, out, self.operator_usage, global_signal)

        if grow_signal > 0:
            self.grow()
        if prune_signal > 0:
            self.prune()

        return out

    def grow(self):
        new_op = self.make_operator().to(next(self.parameters()).device)
        self.operator_bank.append(new_op)

        new_usage = torch.zeros(1, dtype=torch.long, device=self.operator_usage.device)
        self.operator_usage = torch.cat([self.operator_usage, new_usage], dim=0)

        print(f"[Temper {self.id}] 🌱 Grew new operator! Total now: {len(self.operator_bank)}")

    def prune(self, usage_threshold=5):
        keep_indices = [i for i, count in enumerate(self.operator_usage.tolist()) if count >= usage_threshold]
        if len(keep_indices) == len(self.operator_bank):
            return

        if not keep_indices:
            keep_indices = [torch.argmax(self.operator_usage).item()]

        self.operator_bank = nn.ModuleList([self.operator_bank[i] for i in keep_indices])
        self.operator_usage = torch.zeros(len(self.operator_bank), device=self.operator_bank[0][0].weight.device)

        print(f"[Temper {self.id}] 🧹 Pruned operators. Remaining: {len(self.operator_bank)}")

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

        self.routing_policy.clear_log_probs()

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

            outputs = torch.zeros_like(active_states, device=x.device)
            logits = torch.zeros(active_states.size(0), self.num_tempers + 1, device=x.device)

            idx = 0
            for tid, patch_batch in zip(temper_ids, temper_patch_batches):
                out = self.tempers[tid](patch_batch)  # No global signal yet, will apply later
                outputs[idx:idx + patch_batch.size(0)] = out
                logits[idx:idx + patch_batch.size(0)] = self.routing_policy(out)
                idx += patch_batch.size(0)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled_tempers = dist.sample()
            log_probs = dist.log_prob(sampled_tempers)
            self.routing_policy.saved_log_probs.append(log_probs)

            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

            done_now = (sampled_tempers == self.num_tempers)

            patch_states[active_mask] = outputs
            patch_tempers[active_mask] = sampled_tempers.clamp(max=self.num_tempers - 1)
            patch_done[active_mask] |= done_now

        patch_states = patch_states.view(batch_size, num_patches, patch_size)
        final_output = patch_states.mean(dim=1)

        predicted_next_latent = self.predictive_head(final_output)
        prediction_error = F.mse_loss(predicted_next_latent, x, reduction='none').mean(dim=-1)

        return predicted_next_latent, prediction_error

    def print_epoch_summary(self, epoch, loss):
        if not self.routing_stats:
            print(f"Epoch {epoch} | No routing stats collected.")
            return

        print(f"=== Epoch {epoch} Routing Summary ===")

        stats = torch.stack(self.routing_stats)

        total_patches = stats[0].sum().item()
        stop_counts = stats[:, -1]
        active_counts = total_patches - stop_counts.cumsum(dim=0)

        print(f"Initial patches: {total_patches}")
        print(f"Final active patches after {len(stats)} hops: {active_counts[-1].item()}")
        
        for hop, (temper_counts, stop_count) in enumerate(zip(stats[:, :-1], stop_counts)):
            print(f" Hop {hop}: Active: {active_counts[hop].item()} | STOP: {stop_count.item()} | Temper distribution: {temper_counts.tolist()}")
        
        print("\n--- Operator Usage Summary ---")
        for temper in self.tempers:
            usage = temper.operator_usage.tolist()
            print(f" Temper {temper.id}: ops={len(usage)} usage={usage}")
        
        print("")
        self.routing_stats.clear()
