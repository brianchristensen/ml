import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim, id, memory_size=100, max_ops=6):
        super().__init__()
        self.id = id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_ops = max_ops

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.operator_bank = nn.ModuleList([self.make_operator() for _ in range(3)])
        self.routing_logits = nn.Parameter(torch.randn(len(self.operator_bank)))

        self.memory = []
        self.memory_size = memory_size
        self.plasticity = 1.0
        self.usage_count = 0
        self.success_count = 0
        self.rewrites_this_epoch = 0
        self.usage_hist = [0] * len(self.operator_bank)
        self.max_memory = 50
        self.last_novelty = 0.0
        self.last_conflict = 0.0
        self.last_choice = -1

    def make_operator(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, context=None):
        x_proj = self.input_proj(x)

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

        top2 = torch.topk(weights, k=2)
        idx_a, idx_b = top2.indices[0].item(), top2.indices[1].item()
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
                self.routing_logits[worst_op_idx] = 0.0
            self.rewrites_this_epoch += 1

        if self.usage_count >= 200 and gates[1] > 0.5 and len(self.operator_bank) < self.max_ops:
            self.operator_bank.append(self.make_operator())
            new_logits = torch.cat([self.routing_logits.data, torch.tensor([0.0], device=self.routing_logits.device)])
            self.routing_logits = nn.Parameter(new_logits)
            self.usage_hist.append(0)
            print(f"[Temper {self.id}] Expanded to {len(self.operator_bank)} operators.")

    def update_plasticity(self, reward):
        self.success_count += int(reward > 0.5)
        if self.success_count > 0:
            self.plasticity *= 0.99
        else:
            self.plasticity *= 1.01
        self.plasticity = min(max(self.plasticity, 0.01), 1.0)

    def maybe_prune_operators(self, min_usage=5, min_logit=-2.0):
        if len(self.operator_bank) <= 3:
            return
        to_keep = [i for i, (u, l) in enumerate(zip(self.usage_hist, self.routing_logits)) if u > min_usage or l.item() > min_logit]
        if len(to_keep) == len(self.operator_bank):
            return
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
        self.readout = nn.Linear(hidden_dim * num_tempers, output_dim)

        self.embed_table = nn.Embedding(num_tempers, hidden_dim)
        self.policy = nn.Sequential(
            nn.Linear(3 + hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x, context=None):
        self.last_outputs = []
        for i, temper in enumerate(self.tempers):
            out = temper(x, context)
            self.last_outputs.append(out)

            # Shared meta-control logic with learned temper identity
            input_stats = torch.tensor([
                temper.last_novelty,
                temper.last_conflict,
                temper.plasticity
            ], dtype=torch.float32, device=x.device)
            temper_embed = self.embed_table(torch.tensor(i, device=x.device))
            control_input = torch.cat([input_stats, temper_embed], dim=0)
            gates = self.policy(control_input)
            temper.control_logic(gates)

        cat = torch.cat(self.last_outputs, dim=-1)
        return self.readout(cat)

    def update_tempers_with_local_rewards(self, y_pred, y_target):
        loss_fn = nn.MSELoss(reduction='none')
        full_loss = loss_fn(y_pred, y_target).mean()
        for i, temper in enumerate(self.tempers):
            alt_outputs = self.last_outputs.copy()
            alt_outputs[i] = torch.zeros_like(alt_outputs[i])
            alt_input = torch.cat(alt_outputs, dim=-1)
            alt_pred = self.readout(alt_input).detach()
            delta = (alt_pred - y_target).pow(2).mean() - full_loss
            reward = 5.0 * delta.item()
            temper.update_plasticity(reward)

    def dump_routing_summary(self, path="routing_summary.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["temper"] + [f"op{i}" for i in range(max(len(t.routing_logits) for t in self.tempers))])
            for i, temper in enumerate(self.tempers):
                weights = F.softmax(temper.routing_logits, dim=0).detach().cpu().numpy()
                writer.writerow([i] + [f"{w:.4f}" for w in weights])

    def reset_epoch(self):
        for temper in self.tempers:
            temper.reset_epoch_counters()
            temper.maybe_prune_operators()

    def get_diagnostics(self):
        return [t.diagnostics() for t in self.tempers]
