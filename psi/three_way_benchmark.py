"""
3-Way Comparison: Clifford PSI vs OptimalPhasor vs Mamba
on Associative Recall Task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Model 1: Clifford PSI
# ============================================================================
from clifford_memory import OrthogonalModel as CliffordModel

# ============================================================================
# Model 2: Optimal Phasor
# ============================================================================
from phasor_optimal import OptimalPhasorModel

# ============================================================================
# Model 3: Mamba-style SSM
# ============================================================================
class MambaBlock(nn.Module):
    """Simplified Mamba-style SSM block."""
    def __init__(self, dim, d_state=16, d_conv=4):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.conv1d = nn.Conv1d(dim, dim, d_conv, padding=d_conv-1, groups=dim)
        self.x_proj = nn.Linear(dim, d_state + d_state + dim, bias=False)  # B, C, dt
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(dim, -1)
        self.register_buffer('A', -A)
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Project to get B, C, dt
        x_dbl = self.x_proj(x_conv)
        B_ssm = x_dbl[:, :, :self.d_state]
        C_ssm = x_dbl[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(x_dbl[:, :, 2*self.d_state:])

        # Discretize
        dA = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(-2)  # [B, L, D, N]

        # Sequential SSM scan
        y = torch.zeros_like(x_conv)
        h = torch.zeros(batch, dim, self.d_state, device=x.device)
        for t in range(seq_len):
            h = h * dA[:, t] + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y[:, t] = (h * C_ssm[:, t].unsqueeze(1)).sum(-1) + self.D * x_conv[:, t]

        y = y * F.silu(z)
        return self.out_proj(y)

class MambaModel(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=2, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.head(self.norm_out(h))

# ============================================================================
# Task: Associative Recall
# ============================================================================
def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + n_queries * 2
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    query_positions = []

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]
        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2
        query_indices = np.random.choice(n_pairs, n_queries, replace=False)
        for qi, query_idx in enumerate(query_indices):
            data[b, pos] = QUERY_TOKEN
            pos += 1
            query_k, query_v = pairs[query_idx]
            data[b, pos] = query_k
            targets[b, pos] = query_v
            if b == 0:
                query_positions.append(pos)
            pos += 1
    return data, targets, query_positions

def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=500, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        data, targets, positions = generate_multi_query_recall(64, n_pairs, n_queries, vocab_size, device)
        logits = model(data)
        loss = sum(criterion(logits[:, pos, :vocab_size], targets[:, pos]) for pos in positions) / len(positions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item()
                             for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc
    return best_acc

# ============================================================================
# Main Comparison
# ============================================================================
def main():
    print('=' * 70)
    print('3-WAY COMPARISON: Clifford vs OptimalPhasor vs Mamba')
    print('=' * 70)
    print()
    print(f'Device: {device}')

    vocab_size = 32
    n_queries = 4
    epochs = 500
    dim = 128

    results = {}

    for n_pairs in [4, 6, 8, 10, 12]:
        print(f'\n--- {n_pairs} pairs, {n_queries} queries ---')
        print(f'Random baseline: {100/vocab_size:.1f}%')

        # Clifford
        cliff = CliffordModel(vocab_size + 1, dim, 2, n_orthogonal_sets=4, planes_per_set=16, pos_planes=16).to(device)
        c_params = sum(p.numel() for p in cliff.parameters())
        c_acc = train_and_eval(cliff, vocab_size, n_pairs, n_queries, epochs)
        print(f'  Clifford:      {c_acc:5.1f}% ({c_params:,} params)')

        # Optimal Phasor
        phasor = OptimalPhasorModel(vocab_size + 1, dim=dim).to(device)
        p_params = sum(p.numel() for p in phasor.parameters())
        p_acc = train_and_eval(phasor, vocab_size, n_pairs, n_queries, epochs)
        print(f'  OptimalPhasor: {p_acc:5.1f}% ({p_params:,} params)')

        # Mamba
        mamba = MambaModel(vocab_size + 1, dim, 2, d_state=16).to(device)
        m_params = sum(p.numel() for p in mamba.parameters())
        m_acc = train_and_eval(mamba, vocab_size, n_pairs, n_queries, epochs)
        print(f'  Mamba:         {m_acc:5.1f}% ({m_params:,} params)')

        results[n_pairs] = {'clifford': c_acc, 'phasor': p_acc, 'mamba': m_acc}

    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()
    print(f"{'Pairs':>6} {'Clifford':>10} {'OptPhasor':>10} {'Mamba':>10} {'Winner':>12}")
    print('-' * 50)
    for n_pairs, r in results.items():
        winner = max(r, key=r.get)
        print(f"{n_pairs:>6} {r['clifford']:>9.1f}% {r['phasor']:>9.1f}% {r['mamba']:>9.1f}% {winner:>12}")

    # Count wins
    print()
    wins = {'clifford': 0, 'phasor': 0, 'mamba': 0}
    for r in results.values():
        winner = max(r, key=r.get)
        wins[winner] += 1
    print(f"Total wins: Clifford={wins['clifford']}, OptPhasor={wins['phasor']}, Mamba={wins['mamba']}")

if __name__ == "__main__":
    main()
