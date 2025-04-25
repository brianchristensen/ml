import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import csv

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
        stats = torch.tensor([
            novelty,
            conflict,
            plasticity,
            moving_avg_reward,
            usage_count / 1000.0
        ], dtype=torch.float32)
        return self.policy_net(stats)

class RoutingPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tempers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tempers + 1)  # +1 for STOP
        )
        self.baseline = 0.0
        self.decay = 0.99
        self.saved_log_probs = []

    def forward(self, input):
        logits = self.net(input)
        return F.softmax(logits, dim=-1), logits

    def sample(self, input):
        probs, _ = self.forward(input.detach())
        dist = torch.distributions.Categorical(probs)
        choice = dist.sample()
        log_prob = dist.log_prob(choice)
        self.saved_log_probs.append(log_prob)
        return choice.item()

    def reinforce(self, reward):
        advantage = reward - self.baseline
        self.baseline = self.decay * self.baseline + (1 - self.decay) * reward

        device = self.saved_log_probs[0].device
        loss = torch.tensor(0.0, device=device)
        for log_prob in self.saved_log_probs:
            loss += -log_prob * advantage
        return loss
    
    def clear_log_probs(self):
        self.saved_log_probs.clear()

class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim, id, memory_size=100, max_ops=6):
        super().__init__()
        self.id = id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_ops = max_ops

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(3)])
        self.operator_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim)) for _ in range(3)
        ])
        self.operator_freshness = [0] * len(self.operator_bank)
        self.routing_logits = nn.Parameter(torch.randn(len(self.operator_bank)))

        self.memory = []
        self.memory_size = memory_size
        self.plasticity = 1.0
        self.success_count = 0
        self.usage_count = 0
        self.rewrites_this_epoch = 0
        self.usage_hist = [0] * len(self.operator_bank)
        self.max_memory = 50
        self.last_novelty = 0.0
        self.last_conflict = 0.0
        self.last_choice = -1
        self.moving_avg_value = 0.0
        self.baseline_decay = 0.95
        self.moving_avg_reward = 0.0
        self.is_fresh = False
        self.fresh_counter = 0

        self.controller = TemperController(hidden_dim)

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def make_operator(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if x.shape[-1] == self.input_dim:
            x_proj = self.input_proj(x)
        elif x.shape[-1] == self.hidden_dim:
            x_proj = x
        else:
            raise ValueError(f"Unexpected input shape for Temper {self.id}: {x.shape}")

        if self.memory:
            mem = torch.stack(self.memory)
            x_exp = x_proj.unsqueeze(1).expand(-1, mem.shape[0], -1)
            mem_exp = mem.unsqueeze(0).expand(x_proj.shape[0], -1, -1)
            dists = torch.norm(x_exp - mem_exp, dim=-1)
            novelty_std = dists.std().item()
        else:
            novelty_std = 1.0
        self.last_novelty = min(1.5, novelty_std)

        weights = F.softmax(self.routing_logits, dim=0)
        self.last_conflict = float(torch.std(weights).item())

        out = 0
        for i, (op, w) in enumerate(zip(self.operator_bank, weights)):
            result = op(x_proj)
            out = out + w * result
            self.usage_hist[i] += float(w.item())

            if self.operator_freshness[i] > 0:
                self.operator_freshness[i] -= 1

        self.last_choice = torch.argmax(weights).item()
        self.usage_count += 1

        for i in range(x_proj.shape[0]):
            self.memory.append(x_proj[i].detach())
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)

        control_signal = self.controller(
            self.last_novelty,
            self.last_conflict,
            self.plasticity,
            self.moving_avg_reward,
            self.usage_count
        )
        self.control_logic(control_signal)

        return out

    def control_logic(self, control_signal):
        expand_prob, prune_prob = control_signal

        # --- EXPANSION ---
        if expand_prob > 0.5 and len(self.operator_bank) < self.max_ops and self.rewrites_this_epoch < 2:
            self.operator_bank.append(self.make_operator())
            self.operator_embeddings.append(nn.Parameter(torch.randn(self.hidden_dim)))
            self.operator_freshness.append(5)
            new_logits = torch.cat([self.routing_logits.data, torch.tensor([0.0], device=self.routing_logits.device)])
            self.routing_logits = nn.Parameter(new_logits)
            self.usage_hist.append(0)
            print(f"[Temper {self.id}] Expanded to {len(self.operator_bank)} operators.")
            self.rewrites_this_epoch += 1

        # --- PRUNING ---
        if prune_prob > 0.5 and len(self.operator_bank) > 3 and self.rewrites_this_epoch < 2:
            self.maybe_prune_operators()
            self.rewrites_this_epoch += 1

    def update_plasticity(self, reward):
        self.success_count += int(reward > 0.0)
        self.moving_avg_value = self.baseline_decay * self.moving_avg_value + (1 - self.baseline_decay) * reward
        advantage = reward - self.moving_avg_value

        if advantage > 0:
            self.plasticity *= 0.99
        else:
            self.plasticity *= 1.01
        self.plasticity = min(max(self.plasticity, 0.01), 1.0)

    def maybe_prune_operators(self, min_usage=5, min_logit=-2.0, decay_thresh=0.001):
        if len(self.operator_bank) <= 3:
            return
        with torch.no_grad():
            self.routing_logits.data -= 0.01
        to_keep = [i for i, (u, l) in enumerate(zip(self.usage_hist, self.routing_logits))
                   if u > min_usage or l.item() > min_logit + decay_thresh]
        if len(to_keep) < len(self.operator_bank):
            self.operator_bank = nn.ModuleList([self.operator_bank[i] for i in to_keep])
            self.operator_embeddings = nn.ParameterList([self.operator_embeddings[i] for i in to_keep])
            self.routing_logits = nn.Parameter(torch.stack([self.routing_logits[i] for i in to_keep]))
            self.usage_hist = [self.usage_hist[i] for i in to_keep]
            print(f"[Temper {self.id}] Pruned to {len(self.operator_bank)} operators.")

    def logit_decay(self, factor=0.98):
        with torch.no_grad():
            self.routing_logits.data *= factor

    def observe_reward(self, reward: float):
        self.last_reward = reward
        self.moving_avg_reward = 0.95 * self.moving_avg_reward + 0.05 * reward

    def reset_epoch_counters(self):
        self.rewrites_this_epoch = 0
        self.usage_hist = [0 for _ in range(len(self.operator_bank))]
        self.logit_decay()

    def diagnostics(self):
        return {
            'id': self.id,
            'plasticity': self.plasticity,
            'novelty': self.last_novelty,
            'conflict': self.last_conflict,
            'usage': self.usage_count,
            'last_choice': self.last_choice,
            'rewrites': self.rewrites_this_epoch,
            'usage_hist': [int(round(u)) for u in self.usage_hist]
        }
    
class TemperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tempers=4):
        super().__init__()
        self.tempers = nn.ModuleList([Temper(input_dim, hidden_dim, id=i) for i in range(num_tempers)])
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.policy = RoutingPolicy(input_dim + hidden_dim + 3, hidden_dim, num_tempers)

        self.path_stats = {
            'start_counts': defaultdict(int),
            'hops': defaultdict(lambda: defaultdict(int)),
            'path_lengths': []
        }

    def forward(self, x, return_hidden=False):
        self.path_stats = {
            'start_counts': defaultdict(int),
            'hops': defaultdict(lambda: defaultdict(int)),
            'path_lengths': []
        }

        used_tempers = set()
        path = []
        prev_id = None
        current_output = x
        input_summary = x.mean(dim=0)

        while True:
            if len(path) == 0:
                temper_input = torch.zeros(3, device=x.device)
                temper_embed = torch.zeros_like(input_summary)
            else:
                last_temper = self.tempers[path[-1][0]]
                temper_input = torch.tensor([last_temper.last_novelty, last_temper.last_conflict, last_temper.plasticity],
                                            dtype=torch.float32, device=x.device)
                temper_embed = current_output.mean(dim=0)

            policy_input = torch.cat([input_summary, temper_embed, temper_input], dim=-1)
            choice = self.policy.sample(policy_input)

            if len(path) == 0 and choice == len(self.tempers):
                # Force at least one Temper per pass
                continue
            elif choice == len(self.tempers) or choice in used_tempers:
                break

            used_tempers.add(choice)
            temper = self.tempers[choice]
            current_output = temper(current_output)
            path.append((choice, current_output))

            if prev_id is not None:
                self.path_stats['hops'][prev_id][choice] += 1
            self.path_stats['start_counts'][path[0][0]] += 1
            prev_id = choice

        self.last_path_outputs = path
        self.path_stats['path_lengths'].append(len(path))

        if len(path) == 0:
            final_rep = torch.zeros_like(x)
        else:
            final_rep = path[-1][1]

        return final_rep if return_hidden else self.readout(final_rep)

    def update_tempers_with_local_rewards(self, y_pred, y_target):
        if not self.policy.saved_log_probs:
            return (y_pred.sum() * 0.0).clone().detach().requires_grad_()
        
        if y_pred.dim() == 3:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            B, T, V = y_pred.shape
            loss = loss_fn(y_pred.reshape(-1, V), y_target.reshape(-1)).reshape(B, T)
            full_loss = loss.mean(dim=1)
        else:
            loss_fn = nn.MSELoss(reduction='none')
            loss = loss_fn(y_pred, y_target)
            full_loss = loss.mean(dim=1) if loss.dim() > 1 else loss

        with torch.no_grad():
            reward = -full_loss.mean().item()

        # Count how many unique tempers were visited
        if self.last_path_outputs:
            tempers_used = {t[0] for t in self.last_path_outputs}
            diversity_bonus = len(tempers_used) * 0.2
        else:
            diversity_bonus = 0.0

        reward += diversity_bonus

        reinforce_loss = self.policy.reinforce(reward)

        for temper in self.tempers:
            temper.observe_reward(reward)
            fresh_bonus = sum(1 for f in temper.operator_freshness if f > 0) * 0.05
            reward += fresh_bonus

        return reinforce_loss

    def reset_batch(self):
        self.policy.clear_log_probs()

    def reset_epoch(self):
        for temper in self.tempers:
            temper.reset_epoch_counters()
            temper.maybe_prune_operators()

    def get_diagnostics(self):
        return [t.diagnostics() for t in self.tempers]

    def print_epoch_summary(self, epoch, total_task_loss, total_policy_loss, epoch_duration):
        print(f"=== Epoch {epoch} Summary - Task Loss: {total_task_loss:.3f} - Policy Loss: {total_policy_loss:.3f} - Duration: {epoch_duration:.2f}s ===")
        for diag in self.get_diagnostics():
            print(
                f" Temper {diag['id']} | plast: {diag['plasticity']:.3f} | "
                f"nov: {diag['novelty']:.3f} | conf: {diag['conflict']:.3f} | "
                f"used: {diag['usage']} | last op: {diag['last_choice']} | "
                f"rewrites: {diag['rewrites']} | usage: {diag['usage_hist']}"
            )

        print("\n--- Routing Summary ---")
        start_counts = self.path_stats['start_counts']
        hops = self.path_stats['hops']
        total_paths = sum(start_counts.values())

        print("Start Temper Counts:")
        for tid, count in sorted(start_counts.items()):
            perc = 100.0 * count / total_paths if total_paths > 0 else 0.0
            print(f"  Temper {tid}: {count} ({perc:.1f}%)")

        print("\nHop Frequencies:")
        for from_tid, to_dict in hops.items():
            for to_tid, count in sorted(to_dict.items()):
                print(f"  {from_tid} → {to_tid}: {count}")
    
    def print_routing_diagnostics(self):
        start_counts = dict(self.path_stats['start_counts'])
        hops = {k: dict(v) for k, v in self.path_stats['hops'].items()}
        path_lengths = self.path_stats['path_lengths']
        avg_len = sum(path_lengths) / max(len(path_lengths), 1)

        print("\n--- Routing Diagnostics ---")
        print("Start Temper Counts:")
        for tid, count in sorted(start_counts.items()):
            print(f"  Temper {tid}: {count}")

        print("\nHop Frequencies (who routes to whom):")
        for from_tid, to_dict in hops.items():
            for to_tid, count in sorted(to_dict.items()):
                print(f"  {from_tid} → {to_tid}: {count}")

        print(f"\nAverage Path Length: {avg_len:.2f}\n")

        self.path_stats = {
            'start_counts': defaultdict(int),
            'hops': defaultdict(lambda: defaultdict(int)),
            'path_lengths': []
        }

        return {
            'start_counts': start_counts,
            'hops': hops,
            'avg_path_length': avg_len
        }
    
    def dump_routing_summary(self, path="routing_summary.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["temper"] + [f"op{i}" for i in range(max(len(t.routing_logits) for t in self.tempers))])
            for i, temper in enumerate(self.tempers):
                weights = F.softmax(temper.routing_logits, dim=0).detach().cpu().numpy()
                writer.writerow([i] + [f"{w:.4f}" for w in weights])
