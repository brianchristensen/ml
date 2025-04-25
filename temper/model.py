import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import csv

class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim, id, memory_size=100, max_ops=6):
        super().__init__()
        self.id = id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_ops = max_ops

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(3)])
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

        # Meta-learned value network
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

        top2 = torch.topk(weights, k=2).indices
        idx_a = top2[0].item()
        idx_b = top2[1].item()
        op_a, op_b = self.operator_bank[idx_a], self.operator_bank[idx_b]

        x1 = op_a(x_proj)
        out = op_b(x1)

        self.usage_hist[idx_a] += 1
        self.usage_hist[idx_b] += 1
        self.last_choice = idx_b
        self.usage_count += 1

        for i in range(x_proj.shape[0]):
            self.memory.append(x_proj[i].detach())
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)

        return out

    def control_logic(self, gates):
        if self.usage_count >= 50 and self.rewrites_this_epoch < 2 and gates[0] > 0.5:
            worst_op_idx = torch.argmin(self.routing_logits).item()
            self.operator_bank[worst_op_idx] = self.make_operator()
            with torch.no_grad():
                self.routing_logits[worst_op_idx] = 1.5
            self.rewrites_this_epoch += 1

        if self.usage_count >= 200 and gates[1] > 0.5 and len(self.operator_bank) < self.max_ops:
            self.operator_bank.append(self.make_operator())
            new_logits = torch.cat([self.routing_logits.data, torch.tensor([0.0], device=self.routing_logits.device)])
            self.routing_logits = nn.Parameter(new_logits)
            self.usage_hist.append(0)
            print(f"[Temper {self.id}] Expanded to {len(self.operator_bank)} operators.")

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
            self.routing_logits.data -= 0.01  # decay all logits a little to encourage pruning
        to_keep = [i for i, (u, l) in enumerate(zip(self.usage_hist, self.routing_logits)) if u > min_usage or l.item() > min_logit + decay_thresh]
        if len(to_keep) < len(self.operator_bank):
            self.operator_bank = nn.ModuleList([self.operator_bank[i] for i in to_keep])
            self.routing_logits = nn.Parameter(torch.stack([self.routing_logits[i] for i in to_keep]))
            self.usage_hist = [self.usage_hist[i] for i in to_keep]
            print(f"[Temper {self.id}] Pruned to {len(self.operator_bank)} operators.")

    def logit_decay(self, factor=0.98):
        with torch.no_grad():
            self.routing_logits.data *= factor

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
            'usage_hist': list(self.usage_hist)
        }

class TemperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tempers=4):
        super().__init__()
        self.tempers = nn.ModuleList([Temper(input_dim, hidden_dim, id=i) for i in range(num_tempers)])
        self.readout = nn.Linear(hidden_dim, output_dim)

        self.embed_table = nn.Embedding(num_tempers, hidden_dim)
        self.routing_policy = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_tempers + 1),  # +1 for STOP
        )
        self.start_temper_logits = nn.Parameter(torch.randn(num_tempers))

        self.path_stats = {
            'start_counts': defaultdict(int),
            'hops': defaultdict(lambda: defaultdict(int)),
            'path_lengths': []
        }

    def forward(self, x, context=None):
        path = []
        self.path_stats = {
            'start_counts': defaultdict(int),
            'hops': defaultdict(lambda: defaultdict(int)),
            'path_lengths': []
        }
        used_tempers = set()

        start_probs = F.softmax(self.start_temper_logits, dim=0)
        start_id = torch.multinomial(start_probs, num_samples=1).item()
        self.path_stats['start_counts'][start_id] += 1

        current_output = x
        prev_id = None

        while True:
            if start_id in used_tempers:
                break
            used_tempers.add(start_id)

            temper = self.tempers[start_id]
            current_output = temper(current_output)
            path.append((start_id, current_output))

            if prev_id is not None:
                self.path_stats['hops'][prev_id][start_id] += 1
            prev_id = start_id

            # Routing logic
            stats = torch.tensor([temper.last_novelty, temper.last_conflict, temper.plasticity],
                                 dtype=torch.float32, device=x.device)
            temper_embed = self.embed_table(torch.tensor(start_id, device=x.device))
            gates = torch.sigmoid(torch.randn(2, device=x.device))  # Placeholder
            temper.control_logic(gates)

            policy_input = torch.cat([current_output.mean(dim=0), temper_embed], dim=-1)
            logits = self.routing_policy(policy_input)
            probs = F.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == len(self.tempers):
                break
            start_id = next_id

        self.last_path_outputs = path
        self.path_stats['path_lengths'].append(len(path))
        return self.readout(path[-1][1])

    def get_routing_diagnostics(self):
        return {
            'start_counts': dict(self.path_stats['start_counts']),
            'hops': {k: dict(v) for k, v in self.path_stats['hops'].items()},
            'avg_path_length': sum(self.path_stats['path_lengths']) / max(len(self.path_stats['path_lengths']), 1)
        }

    def update_tempers_with_local_rewards(self, y_pred, y_target):
        loss_fn = nn.MSELoss(reduction='none')
        full_loss = loss_fn(y_pred, y_target).mean()

        for i, (tid, out) in enumerate(self.last_path_outputs):
            alt_outputs = self.last_path_outputs.copy()
            alt_outputs[i] = (tid, torch.zeros_like(out))
            combined = alt_outputs[-1][1]
            alt_pred = self.readout(combined).detach()
            delta = (alt_pred - y_target).pow(2).mean() - full_loss
            reward = delta.item() * 5.0  # Scale is still tunable

            self.tempers[tid].update_plasticity(reward)

    def reset_epoch(self):
        for temper in self.tempers:
            temper.reset_epoch_counters()
            temper.maybe_prune_operators()

    def print_epoch_summary(self, epoch, total_loss):
        print(f"=== Epoch {epoch} Summary - Loss: {total_loss:.3f} ===")
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

    def get_diagnostics(self):
        return [t.diagnostics() for t in self.tempers]

    def dump_routing_summary(self, path="routing_summary.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["temper"] + [f"op{i}" for i in range(max(len(t.routing_logits) for t in self.tempers))])
            for i, temper in enumerate(self.tempers):
                weights = F.softmax(temper.routing_logits, dim=0).detach().cpu().numpy()
                writer.writerow([i] + [f"{w:.4f}" for w in weights])
