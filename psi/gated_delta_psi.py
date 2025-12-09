"""
Gated Delta PSI: Combining phase-based addressing with delta rule updates.

Key innovations from research:
1. Delta rule: Don't just accumulate, CORRECT existing associations
   S_t = α·S_{t-1} + β·(value - retrieved)·e^(iφ)
2. Learned gates α (decay) and β (update strength)
3. Content-based addressing via separate key/query phase encoders

This should dramatically improve associative recall over pure cumsum PSI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class GatedDeltaPhasorBlock(nn.Module):
    """
    Phase-based memory with gated delta rule updates.

    Instead of: S_t = cumsum(value · e^(iφ))  [pure accumulation]
    We use:     S_t = α·S_{t-1} + β·(value - retrieved)·e^(iφ)  [delta correction]
    """
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Separate key and query encoders for content-based addressing
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )

        # Value projection
        self.to_value = nn.Linear(dim, dim)

        # Gating networks
        self.to_alpha = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )  # Decay gate
        self.to_beta = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )  # Update gate

        # Output projection
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Encode keys and queries as phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, K]
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi  # [B, L, K]

        # Get values
        value = self.to_value(x)  # [B, L, D]

        # Get gates
        alpha = torch.sigmoid(self.to_alpha(x))  # [B, L, 1] - decay
        beta = torch.sigmoid(self.to_beta(x))    # [B, L, 1] - update strength

        # Initialize memory: [B, K, D] complex
        memory_real = torch.zeros(B, K, D, device=x.device, dtype=x.dtype)
        memory_imag = torch.zeros(B, K, D, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(L):
            # Current key/query phases: [B, K]
            key_t = key_phase[:, t, :]
            query_t = query_phase[:, t, :]

            # Current value: [B, D]
            value_t = value[:, t, :]

            # Gates for this timestep: [B, 1]
            alpha_t = alpha[:, t, :]
            beta_t = beta[:, t, :]

            # ========== RETRIEVAL ==========
            # Query memory with current query phase
            # retrieved = sum_k memory_k * e^(-i*query_k)
            cos_q = torch.cos(query_t)  # [B, K]
            sin_q = torch.sin(query_t)  # [B, K]

            # memory: [B, K, D], cos_q: [B, K] -> [B, K, 1]
            retrieved_real = (memory_real * cos_q.unsqueeze(-1) +
                            memory_imag * sin_q.unsqueeze(-1)).sum(dim=1)  # [B, D]
            retrieved_imag = (memory_imag * cos_q.unsqueeze(-1) -
                            memory_real * sin_q.unsqueeze(-1)).sum(dim=1)  # [B, D]

            # Normalize by oscillator count
            retrieved = retrieved_real / math.sqrt(K)  # Use real part

            # ========== DELTA UPDATE ==========
            # Correction: what we want vs what we retrieved
            correction = value_t - retrieved  # [B, D]

            # Encode correction with key phase
            cos_k = torch.cos(key_t)  # [B, K]
            sin_k = torch.sin(key_t)  # [B, K]

            # Update: [B, K, D]
            update_real = correction.unsqueeze(1) * cos_k.unsqueeze(-1)  # [B, K, D]
            update_imag = correction.unsqueeze(1) * sin_k.unsqueeze(-1)  # [B, K, D]

            # ========== GATED MEMORY UPDATE ==========
            # S_t = α·S_{t-1} + β·correction·e^(iφ)
            memory_real = alpha_t.unsqueeze(-1) * memory_real + beta_t.unsqueeze(-1) * update_real
            memory_imag = alpha_t.unsqueeze(-1) * memory_imag + beta_t.unsqueeze(-1) * update_imag

            outputs.append(retrieved)

        # Stack outputs: [B, L, D]
        output = torch.stack(outputs, dim=1)

        # Normalize by position (early positions have less accumulated)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)
        output = output / norm

        return x + self.to_out(output)


class GatedDeltaPSIModel(nn.Module):
    """Full model with Gated Delta Phasor blocks."""
    def __init__(self, vocab_size, dim=64, n_layers=2, n_oscillators=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GatedDeltaPhasorBlock(dim, n_oscillators) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


# ============================================================================
# Baseline PSI (pure cumsum, no delta rule) for comparison
# ============================================================================

class BaselinePhasorBlock(nn.Module):
    """Original PSI with content-based addressing but NO delta rule."""
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        value = self.to_value(x)

        # Create phasors
        key_phasor = torch.exp(1j * key_phase)  # [B, L, K]
        query_phasor = torch.exp(1j * query_phase)

        # Bind value to key phasor
        V = value.to(torch.complex64)  # [B, L, D]
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]

        # Cumsum memory (pure accumulation, no correction)
        memory = torch.cumsum(bound, dim=1)  # [B, L, K, D]

        # Retrieve with query
        retrieved = memory * query_phasor.conj().unsqueeze(-1)  # [B, L, K, D]
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


class BaselinePSIModel(nn.Module):
    """Baseline PSI model for comparison."""
    def __init__(self, vocab_size, dim=64, n_layers=2, n_oscillators=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            BaselinePhasorBlock(dim, n_oscillators) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


# ============================================================================
# Multi-Query Recall Task
# ============================================================================

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    """
    Generate multi-query associative recall task.
    Store n_pairs key-value pairs, then query n_queries of them.
    """
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
        # Store pairs
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        # Queries
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


def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=200, lr=1e-3):
    """Train model and return best accuracy."""
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

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item()
                             for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc
                print(f'    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%')

    return best_acc


def main():
    print('=' * 70)
    print('GATED DELTA PSI vs BASELINE PSI - Multi-Query Recall')
    print('=' * 70)
    print()
    print('Key difference:')
    print('  Baseline: S_t = cumsum(value * e^(i*phi))  [pure accumulation]')
    print('  Delta:    S_t = alpha*S_{t-1} + beta*(value - retrieved)*e^(i*phi)  [correction]')
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

    # ========== Baseline PSI ==========
    print('-' * 70)
    print('Training Baseline PSI (pure cumsum, no delta rule)...')
    print('-' * 70)

    baseline = BaselinePSIModel(vocab_size + 1, dim, n_layers, n_oscillators).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f'Parameters: {baseline_params:,}')

    baseline_acc = train_and_eval(baseline, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {baseline_acc:.1f}%')
    print()

    # ========== Gated Delta PSI ==========
    print('-' * 70)
    print('Training Gated Delta PSI (with delta rule corrections)...')
    print('-' * 70)

    delta = GatedDeltaPSIModel(vocab_size + 1, dim, n_layers, n_oscillators).to(device)
    delta_params = sum(p.numel() for p in delta.parameters())
    print(f'Parameters: {delta_params:,}')

    delta_acc = train_and_eval(delta, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {delta_acc:.1f}%')
    print()

    # ========== Summary ==========
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Random baseline:    {100/vocab_size:.1f}%')
    print(f'Baseline PSI:       {baseline_acc:.1f}%  ({baseline_params:,} params)')
    print(f'Gated Delta PSI:    {delta_acc:.1f}%  ({delta_params:,} params)')
    print()

    improvement = delta_acc - baseline_acc
    if improvement > 5:
        print(f'VERDICT: Delta rule improves recall by {improvement:.1f}%!')
        print('The correction mechanism helps overwrite/refine associations.')
    elif improvement > 0:
        print(f'VERDICT: Marginal improvement of {improvement:.1f}%')
    else:
        print(f'VERDICT: No improvement ({improvement:.1f}%)')
        print('Delta rule may need tuning or different initialization.')


if __name__ == "__main__":
    main()
