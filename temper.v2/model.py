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

class Temper(nn.Module):
    def __init__(self, hidden_dim, id, num_ops=3):
        super().__init__()
        self.id = id
        self.hidden_dim = hidden_dim
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(num_ops)])
        self.operator_logits = nn.Parameter(torch.zeros(len(self.operator_bank)))
        self.operator_usage = torch.zeros(len(self.operator_bank), dtype=torch.float)
        self.operator_freshness = torch.ones(len(self.operator_bank), dtype=torch.float) * 5.0

        self.rewrites_this_epoch = 0
        self.pending_grow = False
        self.pending_prune = False

        # === Hidden state + Predictive target
        self.hidden_state = nn.Parameter(torch.zeros(hidden_dim))
        self.prev_hidden_state = None  # <-- 🆕 Track previous hidden state

        # === Intrinsic reward network predicts next hidden
        self.intrinsic_reward_net = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim)  # 🛠 Outputs a predicted next hidden state
        )

        self.register_buffer('reward_moving_avg', torch.tensor(0.0))
        self.register_buffer('reward_step', torch.tensor(0))
        self.reward_beta = 0.99

    def make_operator(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=False)
        )

    def apply_operators(self, x):
        if len(self.operator_bank) == 0:
            return x

        device = x.device
        batch_size = x.size(0)

        self.operator_usage = self.operator_usage.to(device)
        self.operator_freshness = self.operator_freshness.to(device)

        # === 🆕 Save previous hidden state for predictive reward
        self.prev_hidden_state = self.hidden_state.detach().clone()

        weights = F.softmax(self.operator_logits, dim=0)
        operator_probs = weights.unsqueeze(0).expand(batch_size, -1)

        dist = torch.distributions.Categorical(operator_probs)
        chosen_ops = dist.sample()

        out = torch.zeros_like(x)

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

        # === Update hidden state based on outputs
        batch_mean = out.mean(dim=0)
        self.hidden_state.data = 0.95 * self.hidden_state.data + 0.05 * batch_mean

        return out

    def compute_intrinsic_reward(self):
        if self.prev_hidden_state is None:
            return torch.tensor(0.0, device=self.hidden_state.device)

        normed_prev = F.normalize(self.prev_hidden_state, dim=0)
        predicted_next_hidden = self.intrinsic_reward_net(normed_prev)

        mse = F.mse_loss(predicted_next_hidden, self.hidden_state.detach())

        reward = -mse  # Reward = negative prediction error

        # (optional) clamp reward to reasonable band
        reward = torch.clamp(reward, min=-1.0, max=1.0)

        return reward

    def grow(self):
        device = next(self.parameters()).device
        if len(self.operator_bank) >= 32:
            print(f"[Temper {self.id}] ❗ Max operators reached, skipping growth.")
            return

        if len(self.operator_bank) < 2:
            new_op = self.make_operator().to(device)
        else:
            usage = self.operator_usage.clone()
            top2 = usage.topk(2, largest=True).indices
            op1 = self.operator_bank[top2[0]]
            op2 = self.operator_bank[top2[1]]

            new_op = self.make_operator().to(device)
            with torch.no_grad():
                for p_new, p1, p2 in zip(new_op.parameters(), op1.parameters(), op2.parameters()):
                    p_new.copy_(0.5 * (p1 + p2))

        self.operator_bank.append(new_op)

        new_logit = torch.tensor([1.0], device=device)
        self.operator_logits = nn.Parameter(torch.cat([self.operator_logits, new_logit]))
        self.operator_usage = torch.cat([self.operator_usage, torch.zeros(1, device=device)])
        self.operator_freshness = torch.cat([self.operator_freshness, torch.ones(1, device=device) * 5.0])

        print(f"[Temper {self.id}] 🌱 Recombined new operator! Total now: {len(self.operator_bank)}")

    def prune(self):
        if len(self.operator_bank) <= 2:
            print(f"[Temper {self.id}] ⏩ Skipped pruning (minimum 2 ops).")
            return

        usage_threshold = 5.0
        freshness_threshold = 0

        keep_indices = [
            i for i, (u, f) in enumerate(zip(self.operator_usage, self.operator_freshness))
            if (u > usage_threshold) or (f > freshness_threshold)
        ]

        if len(keep_indices) <= 1:
            sorted_indices = torch.argsort(self.operator_usage, descending=True)
            keep_indices = sorted_indices[:2].tolist()

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

class TemperGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers=12, max_path_hops=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tempers = num_tempers
        self.max_path_hops = max_path_hops

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tempers = nn.ModuleList([Temper(hidden_dim, id=i) for i in range(num_tempers)])
        self.routing_policy = RoutingPolicy(hidden_dim + 4, hidden_dim, num_tempers)
        self.predictive_head = PredictiveHead(hidden_dim, input_dim)
        self.temper_id_embeddings = nn.Embedding(num_tempers, 4)
        self.routing_stats = []

        self.task_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # normalize for stability
            nn.Linear(hidden_dim, 10)  # 10 classes for FashionMNIST
        )

    def forward(self, x, targets=None):
        batch_size, _ = x.shape
        x_proj = self.input_proj(x)
        patch_size = self.hidden_dim
        num_patches = x_proj.shape[1] // patch_size
        patches = x_proj.view(batch_size * num_patches, patch_size)

        patch_states = patches
        patch_tempers = torch.randint(0, self.num_tempers, (patches.size(0),), device=x.device)
        patch_done = torch.zeros(patches.size(0), dtype=torch.bool, device=x.device)

        self.routing_policy.clear_log_probs()

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
                # === Apply operators
                out = self.tempers[tid].apply_operators(patch_batch)

                # === Prepare enrichments
                id_embed = self.temper_id_embeddings(tid).expand(patch_batch.size(0), -1)

                enriched_input = torch.cat([out, id_embed], dim=1)  # ONLY out + id_embed
                enriched_inputs.append(enriched_input)

                outputs[idx:idx + patch_batch.size(0)] = out
                idx += patch_batch.size(0)

            routing_inputs = torch.cat(enriched_inputs, dim=0)

            logits = self.routing_policy(routing_inputs)

            logits = logits.clamp(min=-10.0, max=10.0)
            logits = logits - logits.max(dim=-1, keepdim=True).values  # numerical stability

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

            patch_hop_counts[active_mask] += 1

            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

        # After routing:
        patch_states = patch_states.view(batch_size, num_patches, patch_size)
        final_output = patch_states.mean(dim=1)

        predicted_next_latent = self.predictive_head(final_output)
        prediction_error = F.mse_loss(predicted_next_latent, x, reduction='none').mean(dim=-1)

        # === Intrinsic rewards from tempers
        temper_rewards = torch.stack([
            temper.compute_intrinsic_reward() for temper in self.tempers
        ]).to(x.device)

        avg_intrinsic_reward = temper_rewards.mean()

        rewards = -prediction_error.detach() + 0.1 * avg_intrinsic_reward.detach()

        # === Add exploration bonus
        temper_usage_std = torch.stack([
            temper.operator_usage.std() for temper in self.tempers
        ]).mean()

        normalized_usage_std = torch.clamp(temper_usage_std / 10.0, 0.0, 1.0)
        exploration_bonus = (0.01 + 0.04 * normalized_usage_std) * patch_hop_counts.float()

        rewards = rewards + exploration_bonus

        # === Save rewards into routing policy
        for r in self.routing_policy.saved_rewards:
            r.copy_(rewards[:r.size(0)])

        # === New grow/prune decisions based on intrinsic reward moving average
        for temper in self.tempers:
            intrinsic_reward = temper.compute_intrinsic_reward()

            temper.reward_step += 1

            beta = 0.99
            temper.reward_moving_avg = beta * temper.reward_moving_avg + (1 - beta) * intrinsic_reward.detach()

            bias_correction = 1.0 - beta ** temper.reward_step.item()
            corrected_moving_avg = temper.reward_moving_avg / bias_correction

            reward_delta = intrinsic_reward.detach() - corrected_moving_avg

            print(f"Moving avg: {corrected_moving_avg:.6f}, Reward Delta: {reward_delta:.6f}")

            if reward_delta > 0.015 and temper.rewrites_this_epoch < 2:
                temper.pending_grow = True
                temper.rewrites_this_epoch += 1

            if reward_delta < -0.005 and temper.rewrites_this_epoch < 2:
                temper.pending_prune = True
                temper.rewrites_this_epoch += 1
        
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
            temper.reward_moving_avg = temper.reward_moving_avg * 0.995
        self.routing_stats.clear()

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
            usage = [f"{u.item():.0f}" for u in temper.operator_usage.float()]
            print(f" Temper {temper.id}: ops={len(usage)} usage={usage}")
        
        print("")
        self.routing_stats.clear()
