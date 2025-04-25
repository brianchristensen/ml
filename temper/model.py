import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim, id, memory_size=100):
        super().__init__()
        self.id = id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.operator_bank = nn.ModuleList([
            self.make_operator(),
            self.make_operator(),
            self.make_operator()
        ])
        self.routing_logits = nn.Parameter(torch.randn(len(self.operator_bank)))

        self.memory = []
        self.memory_size = memory_size

        self.plasticity = 1.0
        self.usage_count = 0
        self.success_count = 0
        self.rewrites_this_epoch = 0

        # Diagnostics
        self.last_novelty = 0.0
        self.last_conflict = 0.0
        self.last_choice = -1
        self.usage_hist = [0, 0, 0]
        self.max_memory = 50

    def make_operator(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, context=None):
        if self.memory:
            mem = torch.stack(self.memory)
            x_exp = x.unsqueeze(1).expand(-1, mem.shape[0], -1)
            mem_exp = mem.unsqueeze(0).expand(x.shape[0], -1, -1)
            novelty = torch.mean(torch.norm(x_exp - mem_exp, dim=-1)).item()
        else:
            novelty = 1.0
        self.last_novelty = min(1.5, novelty)

        weights = F.softmax(self.routing_logits, dim=0)
        self.last_conflict = float(torch.std(weights).item())

        # Get top 2 ops (compositional step!)
        top2 = torch.topk(weights, k=2)
        idx_a, idx_b = top2.indices[0].item(), top2.indices[1].item()
        op_a, op_b = self.operator_bank[idx_a], self.operator_bank[idx_b]

        # 🔧 Input projection added here!
        x_proj = self.input_proj(x)  # [batch, hidden_dim]

        x1 = op_a(x_proj)
        out = op_b(x1)

        # Log routing stats
        self.usage_hist[idx_a] += 1
        self.usage_hist[idx_b] += 1
        self.last_choice = idx_b  # store second op as "final" choice
        self.usage_count += 1

        for i in range(x.shape[0]):
            self.memory.append(x[i].detach())
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)

        self.maybe_rewrite_operator()
        return out

    def update_plasticity(self, reward):
        self.success_count += int(reward > 0.5)
        if self.success_count > 0:
            self.plasticity *= 0.99
        else:
            self.plasticity *= 1.01
        self.plasticity = min(max(self.plasticity, 0.01), 1.0)

    def maybe_rewrite_operator(self):
        if self.usage_count < 50 or self.rewrites_this_epoch >= 2:
            return False
        if self.plasticity > 0.8 and self.last_novelty > 0.6 and self.success_count < 3:
            worst_op_idx = torch.argmin(self.routing_logits).item()
            self.operator_bank[worst_op_idx] = self.make_operator()
            with torch.no_grad():
                self.routing_logits[worst_op_idx] = 0.0
            self.rewrites_this_epoch += 1
            return True
        return False

    def reset_epoch_counters(self):
        self.rewrites_this_epoch = 0
        self.usage_hist = [0 for _ in range(len(self.operator_bank))]

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
        self.readout = nn.Linear(hidden_dim * num_tempers, output_dim)

    def forward(self, x, context=None):
        self.last_outputs = [temper(x, context) for temper in self.tempers]
        cat = torch.cat(self.last_outputs, dim=-1)
        return self.readout(cat)

    def update_tempers_with_local_rewards(self, y_pred, y_target):
        loss_fn = nn.MSELoss(reduction='none')
        full_loss = loss_fn(y_pred, y_target).mean()

        for i, temper in enumerate(self.tempers):
            # Remove one Temper's contribution and recompute
            alt_outputs = self.last_outputs.copy()
            alt_outputs[i] = torch.zeros_like(alt_outputs[i])
            alt_input = torch.cat(alt_outputs, dim=-1)
            alt_pred = self.readout(alt_input).detach()

            delta = (alt_pred - y_target).pow(2).mean() - full_loss
            reward = 5.0 * delta.item()  # amplify local signal
            temper.update_plasticity(reward)

    def dump_routing_summary(self, path="routing_summary.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["temper", "op0", "op1", "op2"])
            for i, temper in enumerate(self.tempers):
                weights = F.softmax(temper.routing_logits, dim=0).detach().cpu().numpy()
                writer.writerow([i] + [f"{w:.4f}" for w in weights])

    def reset_epoch(self):
        for temper in self.tempers:
            temper.reset_epoch_counters()

    def get_diagnostics(self):
        return [t.diagnostics() for t in self.tempers]
