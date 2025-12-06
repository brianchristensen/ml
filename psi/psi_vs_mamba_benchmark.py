"""
PSI vs Mamba Benchmark

Compare Unified Phasor (PSI) against Mamba on:
1. Associative Recall (content-based retrieval)
2. Sequence Reproduction (position-based retrieval)
3. Structured Next-Token Prediction (patterns)
4. Lorenz Dynamics Prediction (continuous dynamics)

Both are O(n) linear-time architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from mambapy.mamba import Mamba, MambaConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# PSI Model (Unified Alternating Position/Content)
# ============================================================================

class PositionOnlyPhasorBlock(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        base_phases = torch.randn(max_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)
        value = self.to_value(x)
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)
        return x + self.to_out(retrieved)


class ContentOnlyPhasorBlock(nn.Module):
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
        self.to_amp = nn.Sequential(nn.Linear(dim, n_oscillators), nn.Softplus())
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        amp = self.to_amp(x) + 0.1
        key_phasor = amp * torch.exp(1j * key_phase)
        query_phasor = amp * torch.exp(1j * query_phase)
        V = self.to_value(x).to(torch.complex64)
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm
        return x + self.to_out(retrieved)


class PSIModel(nn.Module):
    """Unified PSI with alternating position/content layers."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_oscillators=64, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(PositionOnlyPhasorBlock(dim, max_len))
            else:
                self.blocks.append(ContentOnlyPhasorBlock(dim, n_oscillators))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


# ============================================================================
# Mamba Model Wrapper
# ============================================================================

class MambaModel(nn.Module):
    """Mamba wrapper for token prediction."""
    def __init__(self, vocab_size, dim=128, n_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)

        config = MambaConfig(
            d_model=dim,
            n_layers=n_layers,
            d_state=16,
            expand_factor=2,
            d_conv=4,
            pscan=True,  # Use parallel scan
            use_cuda=False  # Pure PyTorch
        )
        self.mamba = Mamba(config)
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.mamba(h)
        return self.head(self.norm_out(h))


# ============================================================================
# Task Generators
# ============================================================================

def generate_associative_recall(batch_size, n_pairs, vocab_size, device):
    """k1,v1,k2,v2,...,QUERY,k_i -> predict v_i"""
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + 2
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]
        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2
        data[b, pos] = QUERY_TOKEN
        pos += 1
        query_idx = np.random.randint(0, n_pairs)
        query_k, query_v = pairs[query_idx]
        data[b, pos] = query_k
        targets[b, pos] = query_v
    return data, targets


def generate_sequence_reproduction(batch_size, seq_len, vocab_size, device):
    """Position i should output token[i]."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return tokens, tokens


def generate_repeating_pattern(batch_size, seq_len, pattern_len, vocab_size, device):
    """[a,b,c,d,a,b,c,d,...] -> predict next in cycle."""
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        pattern = torch.randint(0, vocab_size, (pattern_len,))
        for i in range(seq_len):
            data[b, i] = pattern[i % pattern_len]
    return data


def generate_counting(batch_size, seq_len, mod_n, device, **kwargs):
    """[0,1,2,...,mod_n-1,0,1,2,...] -> predict next."""
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        start = np.random.randint(0, mod_n)
        for i in range(seq_len):
            data[b, i] = (start + i) % mod_n
    return data


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, data_gen_fn, epochs=500, batch_size=32, lr=1e-3, **gen_kwargs):
    """Generic training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    vocab_size = model.vocab_size

    for epoch in range(epochs):
        model.train()

        result = data_gen_fn(batch_size, device=device, **gen_kwargs)
        if isinstance(result, tuple):
            data, targets = result
        else:
            data = result
            targets = None

        if targets is None:
            # Next-token prediction
            inputs = data[:, :-1]
            targets = data[:, 1:]
        else:
            inputs = data

        logits = model(inputs)

        # Handle different target formats
        if targets.shape[1] != logits.shape[1]:
            targets = targets[:, :logits.shape[1]]

        loss = criterion(
            logits[:, :, :vocab_size].reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model


def evaluate_associative_recall(model, vocab_size, n_pairs=5, n_batches=10, batch_size=100):
    """Evaluate associative recall accuracy."""
    model.eval()
    correct = 0
    total = 0
    query_pos = n_pairs * 2 + 1

    with torch.no_grad():
        for _ in range(n_batches):
            data, targets = generate_associative_recall(batch_size, n_pairs, vocab_size, device)
            logits = model(data)
            preds = logits[:, query_pos, :vocab_size].argmax(dim=-1)
            correct += (preds == targets[:, query_pos]).sum().item()
            total += batch_size

    return correct / total * 100


def evaluate_reproduction(model, vocab_size, seq_len=32, n_batches=10, batch_size=100):
    """Evaluate sequence reproduction (position i -> token[i])."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            data, targets = generate_sequence_reproduction(batch_size, seq_len, vocab_size, device)
            logits = model(data)
            preds = logits[:, :, :vocab_size].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

    return correct / total * 100


def evaluate_ntp(model, data_gen_fn, vocab_size, n_batches=10, batch_size=100, **gen_kwargs):
    """Evaluate next-token prediction accuracy."""
    model.eval()
    correct = 0
    total = 0

    # Pass vocab_size to generator if it needs it
    if 'vocab_size' not in gen_kwargs:
        gen_kwargs['vocab_size'] = vocab_size

    with torch.no_grad():
        for _ in range(n_batches):
            data = data_gen_fn(batch_size, device=device, **gen_kwargs)
            inputs = data[:, :-1]
            targets = data[:, 1:]

            logits = model(inputs)
            preds = logits[:, :, :vocab_size].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

    return correct / total * 100


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("=" * 70)
    print("PSI vs MAMBA BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    vocab_size = 16
    dim = 128
    n_layers = 4

    results = {}

    # ========== Task 1: Associative Recall ==========
    print("-" * 70)
    print("Task 1: ASSOCIATIVE RECALL")
    print("  Format: k1,v1,k2,v2,...,QUERY,k_i -> predict v_i")
    print("-" * 70)

    n_pairs = 5

    # PSI
    print("\nTraining PSI...")
    psi = PSIModel(vocab_size + 1, dim, n_layers).to(device)
    psi_params = sum(p.numel() for p in psi.parameters())
    print(f"  Parameters: {psi_params:,}")

    start = time.time()
    train_model(psi, generate_associative_recall, epochs=500, n_pairs=n_pairs, vocab_size=vocab_size)
    psi_time = time.time() - start
    psi_acc = evaluate_associative_recall(psi, vocab_size, n_pairs)
    print(f"  Accuracy: {psi_acc:.1f}%  (time: {psi_time:.1f}s)")

    # Mamba
    print("\nTraining Mamba...")
    mamba = MambaModel(vocab_size + 1, dim, n_layers).to(device)
    mamba_params = sum(p.numel() for p in mamba.parameters())
    print(f"  Parameters: {mamba_params:,}")

    start = time.time()
    train_model(mamba, generate_associative_recall, epochs=500, n_pairs=n_pairs, vocab_size=vocab_size)
    mamba_time = time.time() - start
    mamba_acc = evaluate_associative_recall(mamba, vocab_size, n_pairs)
    print(f"  Accuracy: {mamba_acc:.1f}%  (time: {mamba_time:.1f}s)")

    results['Associative Recall'] = {'PSI': psi_acc, 'Mamba': mamba_acc}

    # ========== Task 2: Sequence Reproduction ==========
    print()
    print("-" * 70)
    print("Task 2: SEQUENCE REPRODUCTION")
    print("  Format: Position i -> output token[i]")
    print("-" * 70)

    seq_len = 32

    # PSI
    print("\nTraining PSI...")
    psi = PSIModel(vocab_size, dim, n_layers).to(device)
    train_model(psi, generate_sequence_reproduction, epochs=500, seq_len=seq_len, vocab_size=vocab_size)
    psi_acc = evaluate_reproduction(psi, vocab_size, seq_len)
    print(f"  Accuracy: {psi_acc:.1f}%")

    # Mamba
    print("\nTraining Mamba...")
    mamba = MambaModel(vocab_size, dim, n_layers).to(device)
    train_model(mamba, generate_sequence_reproduction, epochs=500, seq_len=seq_len, vocab_size=vocab_size)
    mamba_acc = evaluate_reproduction(mamba, vocab_size, seq_len)
    print(f"  Accuracy: {mamba_acc:.1f}%")

    results['Sequence Reproduction'] = {'PSI': psi_acc, 'Mamba': mamba_acc}

    # ========== Task 3: Repeating Pattern NTP ==========
    print()
    print("-" * 70)
    print("Task 3: REPEATING PATTERN (Next-Token)")
    print("  Format: [a,b,c,d,a,b,c,d,...] -> predict next")
    print("-" * 70)

    pattern_len = 4
    seq_len = 32

    # PSI
    print("\nTraining PSI...")
    psi = PSIModel(vocab_size, dim, n_layers).to(device)
    train_model(psi, generate_repeating_pattern, epochs=500, seq_len=seq_len,
                pattern_len=pattern_len, vocab_size=vocab_size)
    psi_acc = evaluate_ntp(psi, generate_repeating_pattern, vocab_size, seq_len=seq_len,
                          pattern_len=pattern_len)
    print(f"  Accuracy: {psi_acc:.1f}%")

    # Mamba
    print("\nTraining Mamba...")
    mamba = MambaModel(vocab_size, dim, n_layers).to(device)
    train_model(mamba, generate_repeating_pattern, epochs=500, seq_len=seq_len,
                pattern_len=pattern_len, vocab_size=vocab_size)
    mamba_acc = evaluate_ntp(mamba, generate_repeating_pattern, vocab_size, seq_len=seq_len,
                            pattern_len=pattern_len)
    print(f"  Accuracy: {mamba_acc:.1f}%")

    results['Repeating Pattern'] = {'PSI': psi_acc, 'Mamba': mamba_acc}

    # ========== Task 4: Counting NTP ==========
    print()
    print("-" * 70)
    print("Task 4: COUNTING MOD 10 (Next-Token)")
    print("  Format: [0,1,2,...,9,0,1,2,...] -> predict next")
    print("-" * 70)

    mod_n = 10
    seq_len = 32

    # PSI
    print("\nTraining PSI...")
    psi = PSIModel(vocab_size, dim, n_layers).to(device)
    train_model(psi, generate_counting, epochs=500, seq_len=seq_len, mod_n=mod_n)
    psi_acc = evaluate_ntp(psi, generate_counting, vocab_size, seq_len=seq_len, mod_n=mod_n)
    print(f"  Accuracy: {psi_acc:.1f}%")

    # Mamba
    print("\nTraining Mamba...")
    mamba = MambaModel(vocab_size, dim, n_layers).to(device)
    train_model(mamba, generate_counting, epochs=500, seq_len=seq_len, mod_n=mod_n)
    mamba_acc = evaluate_ntp(mamba, generate_counting, vocab_size, seq_len=seq_len, mod_n=mod_n)
    print(f"  Accuracy: {mamba_acc:.1f}%")

    results['Counting mod 10'] = {'PSI': psi_acc, 'Mamba': mamba_acc}

    # ========== Summary ==========
    print()
    print("=" * 70)
    print("SUMMARY: PSI vs MAMBA")
    print("=" * 70)
    print()
    print(f"{'Task':<25} {'PSI':>10} {'Mamba':>10} {'Winner':>10}")
    print("-" * 55)

    psi_wins = 0
    mamba_wins = 0

    for task, scores in results.items():
        psi_score = scores['PSI']
        mamba_score = scores['Mamba']

        if psi_score > mamba_score + 1:
            winner = "PSI"
            psi_wins += 1
        elif mamba_score > psi_score + 1:
            winner = "Mamba"
            mamba_wins += 1
        else:
            winner = "Tie"

        print(f"{task:<25} {psi_score:>9.1f}% {mamba_score:>9.1f}% {winner:>10}")

    print("-" * 55)
    print(f"Random baseline: {100/vocab_size:.1f}%")
    print()
    print(f"PSI wins: {psi_wins}")
    print(f"Mamba wins: {mamba_wins}")
    print()

    if psi_wins > mamba_wins:
        print("RESULT: PSI outperforms Mamba on these tasks!")
    elif mamba_wins > psi_wins:
        print("RESULT: Mamba outperforms PSI on these tasks")
    else:
        print("RESULT: PSI and Mamba are comparable on these tasks")


if __name__ == "__main__":
    main()
