import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoutingPolicy(nn.Module):
    def __init__(self, input_dim, num_tempers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_tempers + 1)  # +1 for STOP
        )
        self.saved_log_probs = []

    def forward(self, x):
        return self.net(x)

    def clear_log_probs(self):
        self.saved_log_probs.clear()

    def reinforce(self, reward, patch_hop_counts=None):
        loss = torch.tensor(0.0, device=reward.device, requires_grad=True)

        total_hops = len(self.saved_log_probs)
        hop_idx = 0

        for log_prob, active_indices in self.saved_log_probs:
            num_active = log_prob.shape[0]

            if reward.ndim == 0:
                patch_rewards = reward.expand(num_active)
            else:
                batch_size = reward.shape[0]
                total_patches = patch_hop_counts.shape[0]
                patches_per_sample = total_patches // batch_size

                if total_patches % batch_size != 0:
                    raise ValueError(f"Cannot evenly split {total_patches} patches across {batch_size} samples!")

                expanded_rewards = reward.repeat_interleave(patches_per_sample)
                patch_rewards = expanded_rewards[active_indices]

            if patch_hop_counts is not None:
                patch_hop_counts = patch_hop_counts.to(patch_rewards.device)
                selected_patch_hop_counts = patch_hop_counts[active_indices]

                if selected_patch_hop_counts.shape[0] != patch_rewards.shape[0]:
                    raise ValueError(f"Mismatch between active patch_rewards ({patch_rewards.shape}) and selected_patch_hop_counts ({selected_patch_hop_counts.shape})!")

                hop_bonus = 1.0 + 0.6 * selected_patch_hop_counts.float()
                patch_rewards = patch_rewards * hop_bonus

            # 🎯 Apply early-hop reward boosting
            hop_decay_factor = 1.0 - (hop_idx / max(1, total_hops)) * 0.2  # 20% decay across hops
            patch_rewards = patch_rewards * hop_decay_factor

            loss = loss + (-log_prob * patch_rewards).mean()

            hop_idx += 1

        return loss

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
        self.embedding_dim = hidden_dim // 2
        self.operator_embeddings = nn.Embedding(num_ops, self.embedding_dim)
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
        self.reward_trend = []
        self.reward_trend_std = 1e-3

    def make_operator(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim),  # <-- Wider input
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

        self.prev_hidden_state = self.hidden_state.detach().clone()

        # Sample a single operator per patch
        logits = self.operator_logits
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        sampled_ops = dist.sample((batch_size,))  # [batch_size] of operator ids

        # Prepare augmented inputs
        op_embeds = self.operator_embeddings(sampled_ops)  # (batch_size, embedding_dim)
        x_augmented = torch.cat([x, op_embeds], dim=-1)    # (batch_size, hidden_dim + embedding_dim)

        out = torch.zeros_like(x)

        # === FAST grouped processing ===
        unique_ops = sampled_ops.unique()
        for op_idx in unique_ops:
            mask = (sampled_ops == op_idx)
            if mask.sum() == 0:
                continue

            selected_inputs = x_augmented[mask]
            op = self.operator_bank[op_idx]

            selected_outputs = op(selected_inputs)

            out[mask] = selected_outputs

            # Update usage and freshness
            self.operator_usage[op_idx] += mask.sum().item()
            if self.operator_freshness[op_idx] > 0:
                self.operator_freshness[op_idx] = torch.clamp(
                    self.operator_freshness[op_idx] - 0.1 * mask.sum().item(), min=0.0
                )

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

        operator_penalty = 0.001 * len(self.operator_bank)
        reward = reward - operator_penalty

        # (optional) clamp reward to reasonable band
        reward = torch.clamp(reward, min=-1.0, max=1.0)

        return reward

    def grow(self):
        device = next(self.parameters()).device
        if len(self.operator_bank) >= 3200:
            #print(f"[Temper {self.id}] ❗ Max operators reached, skipping growth.")
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

        # Grow logits, usage, freshness like before
        new_logit = torch.tensor([1.0], device=device)
        self.operator_logits = nn.Parameter(torch.cat([self.operator_logits, new_logit]))
        self.operator_usage = torch.cat([self.operator_usage, torch.zeros(1, device=device)])
        self.operator_freshness = torch.cat([self.operator_freshness, torch.ones(1, device=device) * 5.0])

        # Grow embeddings too! (dirty but effective re-init)
        new_embeddings = nn.Embedding(len(self.operator_bank), self.embedding_dim).to(device)
        with torch.no_grad():
            new_embeddings.weight[:-1].copy_(self.operator_embeddings.weight)
            new_embeddings.weight[-1].uniform_(-0.1, 0.1)  # small random init
        self.operator_embeddings = new_embeddings

        print(f"[Temper {self.id}] 🌱 Recombined new operator! Total now: {len(self.operator_bank)}")

    def prune(self):
        device = next(self.parameters()).device
        if len(self.operator_bank) <= 2:
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

        if len(keep_indices) == len(self.operator_bank):
            # No pruning actually needed
            return

        # Now apply pruning
        self.operator_bank = nn.ModuleList([self.operator_bank[i] for i in keep_indices])
        self.operator_logits = nn.Parameter(self.operator_logits[keep_indices])
        self.operator_usage = self.operator_usage[keep_indices]
        self.operator_freshness = self.operator_freshness[keep_indices]

        # Prune embeddings too
        new_embeddings = nn.Embedding(len(self.operator_bank), self.embedding_dim).to(device)
        with torch.no_grad():
            for new_idx, old_idx in enumerate(keep_indices):
                new_embeddings.weight[new_idx].copy_(self.operator_embeddings.weight[old_idx])
        self.operator_embeddings = new_embeddings

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

        self.tempers = nn.ModuleList([Temper(hidden_dim, id=i) for i in range(num_tempers)])
        self.routing_policy = RoutingPolicy(hidden_dim + 4, num_tempers)
        self.predictive_head = PredictiveHead(hidden_dim, input_dim)
        self.temper_id_embeddings = nn.Embedding(num_tempers, 4)
        self.routing_stats = []

        self.task_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # normalize for stability
            nn.Linear(hidden_dim, 10)  # 10 classes for FashionMNIST
        )

    def forward(self, x, targets=None):
        batch_size, input_dim = x.shape

        # Patching
        pad_size = (self.hidden_dim - (input_dim % self.hidden_dim)) % self.hidden_dim
        if pad_size > 0:
            x = F.pad(x, (0, pad_size), value=0.0)

        total_dim = x.shape[1]
        num_patches = total_dim // self.hidden_dim

        patches = x.view(batch_size, num_patches, self.hidden_dim)
        patches = patches.view(batch_size * num_patches, self.hidden_dim)

        patch_states = patches
        patch_tempers = torch.randint(0, self.num_tempers, (patch_states.size(0),), device=x.device)
        patch_done = torch.zeros(patch_states.size(0), dtype=torch.bool, device=x.device)

        self.routing_policy.clear_log_probs()
        patch_hop_counts = torch.zeros(patch_states.size(0), device=x.device, dtype=torch.int)

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
                out = self.tempers[tid].apply_operators(patch_batch)

                id_embed = self.temper_id_embeddings(tid).expand(patch_batch.size(0), -1)
                enriched_input = torch.cat([out, id_embed], dim=1)
                enriched_inputs.append(enriched_input)

                outputs[idx:idx + patch_batch.size(0)] = out
                idx += patch_batch.size(0)

            routing_inputs = torch.cat(enriched_inputs, dim=0)

            logits = self.routing_policy(routing_inputs)
            logits = logits.clamp(min=-10.0, max=10.0)
            logits = logits - logits.max(dim=-1, keepdim=True).values

            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1e-6, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            dist = torch.distributions.Categorical(probs)
            sampled_tempers = dist.sample()
            log_probs = dist.log_prob(sampled_tempers)

            # Calculate patch rewards at that moment
            if self.training:
                batch_size = x.shape[0]
                reward_per_patch = None
                if targets is not None:
                    patches_per_sample = patches.size(0) // batch_size
                    reward_per_patch = targets.repeat_interleave(patches_per_sample)

            self.routing_policy.saved_log_probs.append((log_probs.detach(), active_mask.nonzero(as_tuple=False).squeeze(1)))

            done_now = (sampled_tempers == self.num_tempers)

            patch_states[active_mask] = outputs
            patch_tempers[active_mask] = sampled_tempers.clamp(max=self.num_tempers - 1)
            patch_done[active_mask] |= done_now

            patch_hop_counts[active_mask] += 1

            with torch.no_grad():
                patch_counts = torch.bincount(sampled_tempers, minlength=self.num_tempers + 1)
                self.routing_stats.append(patch_counts.cpu())

        # === SAVE patch_hop_counts here ===
        self.latest_patch_hop_counts = patch_hop_counts.detach()

        # Continue...
        patch_states = patch_states.view(batch_size, num_patches, self.hidden_dim)
        latent_output = patch_states.mean(dim=1)

        predicted_next_latent = self.predictive_head(latent_output)
        prediction_error = F.mse_loss(predicted_next_latent, x[:, :self.input_dim], reduction='none').mean(dim=-1)

        # === Step 4: Intrinsic Rewards
        temper_rewards = []
        intrinsic_losses = []

        for temper in self.tempers:
            reward = temper.compute_intrinsic_reward()
            temper_rewards.append(reward)

            if temper.prev_hidden_state is not None:
                normed_prev = F.normalize(temper.prev_hidden_state, dim=0)
                predicted_next_hidden = temper.intrinsic_reward_net(normed_prev)
                mse = F.mse_loss(predicted_next_hidden, temper.hidden_state.detach())
                intrinsic_losses.append(mse)

        temper_rewards = torch.stack(temper_rewards).to(x.device)
        intrinsic_loss = torch.stack(intrinsic_losses).mean() if intrinsic_losses else torch.tensor(0.0, device=x.device)

        avg_intrinsic_reward = temper_rewards.mean()

        rewards = -prediction_error.detach() + 0.1 * avg_intrinsic_reward.detach()

        # Exploration bonus
        temper_usage_std = torch.stack([
            temper.operator_usage.std().to(x.device) for temper in self.tempers
        ]).mean()

        normalized_usage_std = torch.clamp(temper_usage_std / 10.0, 0.0, 1.0)
        exploration_bonus = (0.01 + 0.04 * normalized_usage_std) * patch_hop_counts.float()

        rewards = rewards + exploration_bonus

        self.adjust_operators()

        return latent_output, predicted_next_latent, prediction_error, intrinsic_loss

    def adjust_operators(self):
        for temper in self.tempers:
            intrinsic_reward = temper.compute_intrinsic_reward()

            temper.reward_step += 1
            beta = 0.99
            temper.reward_moving_avg = beta * temper.reward_moving_avg + (1 - beta) * intrinsic_reward.detach()

            bias_correction = 1.0 - beta ** temper.reward_step.item()
            corrected_moving_avg = temper.reward_moving_avg / bias_correction

            reward_delta = intrinsic_reward.detach() - corrected_moving_avg

            # === NEW DYNAMIC THRESHOLDS ===
            volatility = (temper.reward_trend_std + 1e-6) if hasattr(temper, 'reward_trend_std') else 1e-3
            dynamic_threshold = math.log(len(temper.operator_bank) + 1) * volatility * 0.5

            #print(f"[Temper {temper.id}] Volatility={volatility:.6f} | Reward Delta={reward_delta:.6f} | Grow@>{dynamic_threshold:.6f} | Prune@<-{dynamic_threshold:.6f}")

            if reward_delta > dynamic_threshold and temper.rewrites_this_epoch < 2:
                temper.pending_grow = True
                temper.rewrites_this_epoch += 1

            if reward_delta < -dynamic_threshold and temper.rewrites_this_epoch < 2:
                temper.pending_prune = True
                temper.rewrites_this_epoch += 1
                
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
            intrinsic_reward = temper.compute_intrinsic_reward().detach()
            temper.reward_trend.append(intrinsic_reward.item())

            if len(temper.reward_trend) > 100:  # Moving window
                temper.reward_trend.pop(0)

            # Update standard deviation
            trend_tensor = torch.tensor(temper.reward_trend)
            if len(trend_tensor) > 1:
                temper.reward_trend_std = trend_tensor.std().item()
            else:
                temper.reward_trend_std = 1e-3
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
            usage = [f"{u:.8f}" for u in temper.operator_usage.tolist()]
            print(f" Temper {temper.id}: ops={len(usage)} usage={usage}")
        
        print("")
        self.routing_stats.clear()
