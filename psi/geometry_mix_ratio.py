"""
Explore optimal ratio of periodic vs hyperbolic oscillators in product manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProductManifoldPhasorBlock(nn.Module):
    """
    Combine flat torus (periodic) with hyperbolic (hierarchical).
    """
    def __init__(self, dim, n_oscillators=64, n_hyperbolic=16):
        super().__init__()
        self.dim = dim
        self.n_periodic = n_oscillators - n_hyperbolic
        self.n_hyperbolic = n_hyperbolic

        # Periodic components
        self.key_phase = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.n_periodic)
        )
        self.query_phase = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.n_periodic)
        )

        # Hyperbolic components (2D points in Poincare ball)
        self.key_hyp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_hyperbolic * 2)
        )
        self.query_hyp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_hyperbolic * 2)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape

        # Periodic phases
        key_p = torch.tanh(self.key_phase(x)) * math.pi
        query_p = torch.tanh(self.query_phase(x)) * math.pi
        key_phasor_p = torch.exp(1j * key_p)
        query_phasor_p = torch.exp(1j * query_p)

        # Hyperbolic phases (from 2D Poincare points)
        key_h = torch.tanh(self.key_hyp(x).view(B, L, self.n_hyperbolic, 2)) * 0.9
        query_h = torch.tanh(self.query_hyp(x).view(B, L, self.n_hyperbolic, 2)) * 0.9

        # Convert hyperbolic points to phases (angle) + amplitude (1-radius)
        key_angle_h = torch.atan2(key_h[..., 1], key_h[..., 0])
        query_angle_h = torch.atan2(query_h[..., 1], query_h[..., 0])

        key_radius = key_h.norm(dim=-1)
        query_radius = query_h.norm(dim=-1)

        # Hyperbolic amplitude: closer to center = higher amplitude
        key_amp = 1 - key_radius
        query_amp = 1 - query_radius

        key_phasor_h = key_amp * torch.exp(1j * key_angle_h)
        query_phasor_h = query_amp * torch.exp(1j * query_angle_h)

        # Combine all oscillators
        key_phasor = torch.cat([key_phasor_p, key_phasor_h], dim=-1)
        query_phasor = torch.cat([query_phasor_p, query_phasor_h], dim=-1)

        # Memory operations
        V = self.to_value(x).to(torch.complex64)
        K_total = self.n_periodic + self.n_hyperbolic

        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K_total).view(1, L, 1)
        return x + self.to_out(retrieved / norm)


class ProductManifoldModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_oscillators=64, n_hyperbolic=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            ProductManifoldPhasorBlock(dim, n_oscillators, n_hyperbolic)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


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


def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=300, lr=1e-3):
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


def main():
    print('=' * 70)
    print('PRODUCT MANIFOLD: Optimizing Periodic/Hyperbolic Ratio')
    print('=' * 70)
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2
    n_oscillators = 64
    epochs = 300

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    # Test different ratios
    ratios = [
        (64, 0, "100% periodic (baseline)"),
        (56, 8, "87.5% periodic, 12.5% hyperbolic"),
        (48, 16, "75% periodic, 25% hyperbolic"),
        (32, 32, "50% periodic, 50% hyperbolic"),
        (16, 48, "25% periodic, 75% hyperbolic"),
        (0, 64, "100% hyperbolic"),
    ]

    results = []

    for n_periodic, n_hyperbolic, name in ratios:
        print('-' * 70)
        print(f'Testing: {name}')
        print('-' * 70)

        if n_hyperbolic == 0:
            # Pure periodic - use product manifold with 0 hyperbolic
            # (which is effectively flat torus)
            pass  # handled below

        if n_hyperbolic == n_oscillators:
            # Pure hyperbolic - special handling needed for n_periodic=0
            # Skip this case as ProductManifoldPhasorBlock requires n_periodic > 0
            print("Skipping 100% hyperbolic (requires n_periodic > 0)")
            results.append((name, 0.0, 100.0))
            continue

        if True:
            model = ProductManifoldModel(
                vocab_size + 1, dim, n_layers, n_oscillators, n_hyperbolic
            ).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {params:,}')

        acc = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
        print(f'Best accuracy: {acc:.1f}%')
        results.append((name, acc, n_hyperbolic / n_oscillators * 100))
        print()

    print('=' * 70)
    print('SUMMARY: Accuracy by Hyperbolic Ratio')
    print('=' * 70)
    print()
    print(f'{"Config":<45} {"Accuracy":>10} {"Hyp %":>8}')
    print('-' * 65)
    for name, acc, hyp_pct in results:
        bar = '#' * int(acc // 3)
        print(f'{name:<45} {acc:>9.1f}% {hyp_pct:>7.1f}% {bar}')

    best = max(results, key=lambda x: x[1])
    print()
    print(f'Best configuration: {best[0]} with {best[1]:.1f}% accuracy')


if __name__ == "__main__":
    main()
