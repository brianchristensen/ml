import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tempers + 1)
        )
        self.baseline = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.decay = 0.99
        self.saved_log_probs = []
        self.saved_rewards = []

    def forward(self, input):
        logits = self.net(input)
        return logits

    def reinforce(self):
        if not self.saved_log_probs:
            return torch.tensor(0.0, device=self.baseline.device)

        log_probs = torch.cat(self.saved_log_probs, dim=0)
        rewards = torch.cat(self.saved_rewards, dim=0)

        advantage = rewards - self.baseline
        self.baseline.data = self.decay * self.baseline.data + (1 - self.decay) * rewards.mean().detach()

        policy_loss = -(log_probs * advantage.detach()).mean()

        return policy_loss

    def clear_log_probs(self):
        self.saved_log_probs.clear()
        self.saved_rewards.clear()

class PredictiveHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.predictor = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        return self.predictor(x)

class IntrinsicGrowthController(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ReLU()
        )
        self.grow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.prune_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.register_buffer('input_mean', torch.zeros(hidden_dim))
        self.register_buffer('reward_moving_avg', torch.tensor(0.0))
        self.reward_beta = 0.99
        self.global_alpha = nn.Parameter(torch.tensor(0.01))

    def compute_intrinsic_signals(self, x, out, operator_usage):
        device = x.device
        self.input_mean = self.input_mean.to(device)
        self.reward_moving_avg = self.reward_moving_avg.to(device)
        operator_usage = operator_usage.to(device)

        out_noisy = out + torch.randn_like(out) * 0.01
        plasticity = ((out_noisy - out) ** 2).mean(dim=-1)

        novelty = ((x - self.input_mean) ** 2).mean(dim=-1)
        prediction_error = ((out - x) ** 2).mean(dim=-1)

        usage_probs = operator_usage.float() / (operator_usage.sum() + 1e-6)
        usage_entropy = -(usage_probs * usage_probs.clamp(min=1e-6).log()).sum()

        mean_usage = operator_usage.mean()
        max_usage = operator_usage.max()
        usage_std = operator_usage.std()
        used_fraction = (operator_usage > 0).float().mean()

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
            reward_var,
            mean_usage,
            max_usage,
            usage_std,
            used_fraction
        ], dim=0)

        return signals

    def forward(self, x, out, operator_usage, intrinsic_stats=None, global_signal=None):
        signals = self.compute_intrinsic_signals(x, out, operator_usage)

        if intrinsic_stats is not None:
            stats = torch.tensor([
                intrinsic_stats["mean_novelty"],
                intrinsic_stats["max_novelty"],
                intrinsic_stats["mean_plasticity"],
                intrinsic_stats["max_plasticity"],
            ], device=signals.device)
            signals = torch.cat([signals, stats], dim=0)

        if global_signal is not None:
            signals = signals + self.global_alpha * global_signal

        # 🔥 Pass through shared trunk first
        shared_out = self.shared(signals.unsqueeze(0))  # [1, hidden_dim]

        # 🔥 Then pass through grow_head and prune_head separately
        grow_signal = self.grow_head(shared_out).squeeze(0)  # [1] -> scalar
        prune_signal = self.prune_head(shared_out).squeeze(0)

        return grow_signal, prune_signal

class Temper(nn.Module):
    def __init__(self, hidden_dim, id, num_ops=3):
        super().__init__()
        self.id = id
        self.hidden_dim = hidden_dim
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(num_ops)])
        self.operator_logits = nn.Parameter(torch.zeros(len(self.operator_bank)))
        self.operator_usage = torch.zeros(len(self.operator_bank), dtype=torch.float)
        self.operator_freshness = torch.zeros(len(self.operator_bank), dtype=torch.float)

        self.rewrites_this_epoch = 0
        self.growth_controller = IntrinsicGrowthController(hidden_dim)
        self.pending_grow = False
        self.pending_prune = False

        # Intrinsic stats
        self.last_novelty = torch.tensor(0.0)
        self.mean_novelty = torch.tensor(0.0)
        self.max_novelty = torch.tensor(0.0)
        self.last_conflict = torch.tensor(0.0)
        self.last_plasticity = torch.tensor(0.0)
        self.mean_plasticity = torch.tensor(0.0)
        self.max_plasticity = torch.tensor(0.0)

        # Intrinsic reward tracking
        self.prev_inputs = None
        self.prev_outputs = None
        self.intrinsic_reward_weight = torch.nn.Parameter(torch.tensor(0.8), requires_grad=True)

    def make_operator(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=False),  # <-- IMPORTANT: no in-place ops!
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=False)   # <-- also here
        )

    def apply_operators(self, x):
        if len(self.operator_bank) == 0:
            return x

        device = x.device
        batch_size = x.size(0)

        self.operator_usage = self.operator_usage.to(device)
        self.operator_freshness = self.operator_freshness.to(device)

        weights = F.softmax(self.operator_logits, dim=0)
        operator_probs = weights.unsqueeze(0).expand(batch_size, -1)

        dist = torch.distributions.Categorical(operator_probs)
        chosen_ops = dist.sample()

        out = torch.zeros_like(x)

        # Track patchwise plasticity
        patch_plasticity = torch.zeros(batch_size, device=device)

        for idx, op in enumerate(self.operator_bank):
            mask = (chosen_ops == idx)
            if mask.any():
                selected = x[mask]
                result = op(selected)

                if self.operator_freshness[idx] > 0:
                    result = result + 0.01 * result
                    self.operator_freshness[idx] = torch.clamp(self.operator_freshness[idx] - 0.01, min=0.0)

                out[mask] = result
                self.operator_usage[idx] += mask.sum().float()

                # Patchwise plasticity only for the patches that this op handled
                patch_plasticity[mask] = ((result - selected) ** 2).mean(dim=1)

        # Updated intrinsic signals
        self.last_conflict = torch.std(weights)

        if self.prev_outputs is not None and self.prev_outputs.shape == out.shape:
            plasticity_per_patch = ((out - self.prev_outputs) ** 2).mean(dim=1)
            self.last_plasticity = plasticity_per_patch.mean()
            self.max_plasticity = plasticity_per_patch.max()
        else:
            self.last_plasticity = torch.tensor(0.0, device=out.device)
            self.max_plasticity = torch.tensor(0.0, device=out.device)

        if self.prev_inputs is not None and self.prev_inputs.shape == x.shape:
            novelty_per_patch = ((x - self.prev_inputs) ** 2).mean(dim=1)
            self.last_novelty = novelty_per_patch.mean()
            self.max_novelty = novelty_per_patch.max()
        else:
            self.last_novelty = torch.tensor(0.0, device=x.device)
            self.max_novelty = torch.tensor(0.0, device=x.device)

        self.prev_outputs = out.detach()
        self.prev_inputs = x.detach()

        return out

    def grow(self):
        new_op = self.make_operator().to(next(self.parameters()).device)
        self.operator_bank.append(new_op)

        new_logit = torch.tensor([1.0], device=self.operator_logits.device)
        self.operator_logits = nn.Parameter(torch.cat([self.operator_logits, new_logit]))

        new_usage = torch.zeros(1, device=self.operator_usage.device)
        self.operator_usage = torch.cat([self.operator_usage, new_usage])

        new_freshness = torch.ones(1, device=self.operator_usage.device) * 5.0
        self.operator_freshness = torch.cat([self.operator_freshness, new_freshness])

        print(f"[Temper {self.id}] 🌱 Grew new operator! Total now: {len(self.operator_bank)}")

    def prune(self):
        usage_threshold = 5.0
        freshness_threshold = 0  # protect operators that are still "fresh"

        # Only prune ops that are both low-usage AND not fresh anymore
        keep_indices = [
            i for i, (u, f) in enumerate(zip(self.operator_usage, self.operator_freshness))
            if (u > usage_threshold) or (f > freshness_threshold)
        ]

        if not keep_indices:
            keep_indices = [torch.argmax(self.operator_usage).item()]

        self.operator_bank = nn.ModuleList([self.operator_bank[i] for i in keep_indices])
        self.operator_logits = nn.Parameter(self.operator_logits[keep_indices])
        self.operator_usage = self.operator_usage[keep_indices]
        self.operator_freshness = self.operator_freshness[keep_indices]

        print(f"[Temper {self.id}] 🧹 Pruned operators. Remaining: {len(self.operator_bank)}")

    def reset_epoch(self):
        self.rewrites_this_epoch = 0
        self.operator_usage.zero_()
        self.operator_logits.data *= 0.96
        self.operator_freshness.clamp_(min=0.0)

        # Anneal exploration over time
        with torch.no_grad():
            self.intrinsic_reward_weight *= 0.995
            self.intrinsic_reward_weight.clamp_(min=0.1)

class TemperGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers=12, max_path_hops=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tempers = num_tempers
        self.max_path_hops = max_path_hops

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tempers = nn.ModuleList([Temper(hidden_dim, id=i) for i in range(num_tempers)])
        self.routing_policy = RoutingPolicy(hidden_dim + 7, hidden_dim, num_tempers)
        self.predictive_head = PredictiveHead(hidden_dim, input_dim)
        self.temper_id_embeddings = nn.Embedding(num_tempers, 4)
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

        # Initialize route hop counters
        patch_hop_counts = torch.zeros(patches.size(0), device=x.device, dtype=torch.int)

        for hop in range(self.max_path_hops):
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
            enriched_inputs = []

            idx = 0
            for tid, patch_batch in zip(temper_ids, temper_patch_batches):
                # === Call apply_operators
                out = self.tempers[tid].apply_operators(patch_batch)

                # === Call growth_controller manually (formerly in Temper.forward)
                grow_signal, prune_signal = self.tempers[tid].growth_controller(
                    patch_batch,
                    out,
                    operator_usage=self.tempers[tid].operator_usage,
                    intrinsic_stats={
                        "mean_novelty": self.tempers[tid].mean_novelty,
                        "max_novelty": self.tempers[tid].max_novelty,
                        "mean_plasticity": self.tempers[tid].mean_plasticity,
                        "max_plasticity": self.tempers[tid].max_plasticity,
                    },
                    global_signal=None,
                )

                if grow_signal > 0.5 and self.tempers[tid].rewrites_this_epoch < 2 and self.tempers[tid].operator_usage.mean() > 10:
                    self.tempers[tid].pending_grow = True
                    self.tempers[tid].rewrites_this_epoch += 1
                    self.tempers[tid].intrinsic_reward_weight.data = torch.tensor(0.8, device=out.device)

                if prune_signal > 0.65 and self.tempers[tid].rewrites_this_epoch < 2:
                    self.tempers[tid].pending_prune = True
                    self.tempers[tid].rewrites_this_epoch += 1

                # === Prepare enrichments
                id_embed = self.temper_id_embeddings(tid).expand(patch_batch.size(0), -1)
                novelty = self.tempers[tid].last_novelty.to(out.device).unsqueeze(0).expand(out.size(0), 1)
                conflict = self.tempers[tid].last_conflict.to(out.device).unsqueeze(0).expand(out.size(0), 1)
                plasticity = self.tempers[tid].last_plasticity.to(out.device).unsqueeze(0).expand(out.size(0), 1)

                enriched_input = torch.cat([out, id_embed, novelty, conflict, plasticity], dim=1)
                enriched_inputs.append(enriched_input)

                outputs[idx:idx + patch_batch.size(0)] = out
                idx += patch_batch.size(0)

            routing_inputs = torch.cat(enriched_inputs, dim=0)

            logits = self.routing_policy(routing_inputs)

            # 💥 Clamp aggressively to avoid extreme logits:
            logits = logits.clamp(min=-10.0, max=10.0)

            # 💥 Also shift logits for numerical stability:
            logits = logits - logits.max(dim=-1, keepdim=True).values  # subtract max for better softmax stability

            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1e-6, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            dist = torch.distributions.Categorical(probs)
            sampled_tempers = dist.sample()
            log_probs = dist.log_prob(sampled_tempers)

            self.routing_policy.saved_log_probs.append(log_probs)
            self.routing_policy.saved_rewards.append(torch.zeros_like(log_probs))  # fill later

            done_now = (sampled_tempers == self.num_tempers)

            patch_states[active_mask] = outputs
            patch_tempers[active_mask] = sampled_tempers.clamp(max=self.num_tempers - 1)
            patch_done[active_mask] |= done_now

            # 🆕 Count hop visits
            patch_hop_counts[active_mask] += 1

            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

        # After routing:
        patch_states = patch_states.view(batch_size, num_patches, patch_size)
        final_output = patch_states.mean(dim=1)

        predicted_next_latent = self.predictive_head(final_output)
        prediction_error = F.mse_loss(predicted_next_latent, x, reduction='none').mean(dim=-1)

        # Routing reward
        rewards = -prediction_error.detach()

        # 🆕 Add exploration bonus
        # Mean and std of operator usages across tempers
        temper_usage_std = torch.stack([
            temper.operator_usage.std() for temper in self.tempers
        ]).mean()

        # Normalize std to [0, 1] range roughly
        normalized_usage_std = torch.clamp(temper_usage_std / 10.0, 0.0, 1.0)

        # Dynamic exploration bonus
        exploration_bonus = (0.01 + 0.04 * normalized_usage_std) * patch_hop_counts.float()

        rewards = rewards + exploration_bonus

        for r in self.routing_policy.saved_rewards:
            r.copy_(rewards[:r.size(0)])

        return predicted_next_latent, prediction_error

    def reset_epoch(self):
        for temper in self.tempers:
            if temper.pending_grow:
                temper.grow()
                temper.pending_grow = False
            if temper.pending_prune:
                temper.prune()
                temper.pending_prune = False
            temper.reset_epoch()

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

        print(f"\n--- Intrinsic Stats ---")
        for temper in self.tempers:
            usage = [f"{u.item():.0f}" for u in temper.operator_usage.float()]
            print(f" Temper {temper.id}:")
            print(f"   novelty ~ {temper.max_novelty:.4f}")
            print(f"   conflict ~ {temper.last_conflict:.4f}")
            print(f"   plasticity ~ {temper.max_plasticity:.4f}")
            print(f"   usage ~ {usage}")
        print("")

        print(f"Final active patches after {len(stats)} hops: {active_counts[-1].item()}")
        for hop, (temper_counts, stop_count) in enumerate(zip(stats[:, :-1], stop_counts)):
            print(f" Hop {hop}: Active: {active_counts[hop].item()} | STOP: {stop_count.item()} | Temper distribution: {temper_counts.tolist()}")
        
        print("\n--- Operator Usage Summary ---")
        for temper in self.tempers:
            usage = temper.operator_usage.tolist()
            print(f" Temper {temper.id}: ops={len(usage)} usage={usage}")
        
        print("")
        self.routing_stats.clear()
