"""
PSI vs Mamba FAIR Benchmark

Compares:
- Accuracy (with hyperparameter tuning for both)
- Memory usage
- Training time
- Inference time
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
import gc

from mambapy.mamba import Mamba, MambaConfig

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# PSI Model
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
# Mamba Model
# ============================================================================

class MambaModel(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        config = MambaConfig(
            d_model=dim,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=2,
            d_conv=4,
            pscan=True,
            use_cuda=False
        )
        self.mamba = Mamba(config)
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.mamba(h)
        return self.head(self.norm_out(h))


# ============================================================================
# Data Generation
# ============================================================================

def generate_associative_recall(batch_size, n_pairs, vocab_size, device):
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


# ============================================================================
# Measurement utilities
# ============================================================================

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def train_and_measure(model, vocab_size, n_pairs, epochs, lr, batch_size=32):
    """Train model and return metrics."""
    reset_memory()

    query_pos = n_pairs * 2 + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training
    start_time = time.time()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        data, targets = generate_associative_recall(batch_size, n_pairs, vocab_size, device)
        logits = model(data)
        loss = criterion(logits[:, query_pos, :vocab_size], targets[:, query_pos])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                data, targets = generate_associative_recall(500, n_pairs, vocab_size, device)
                logits = model(data)
                preds = logits[:, query_pos, :vocab_size].argmax(dim=-1)
                acc = (preds == targets[:, query_pos]).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc

    train_time = time.time() - start_time
    train_memory = get_gpu_memory_mb()

    # Inference timing
    model.eval()
    reset_memory()

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            data, _ = generate_associative_recall(100, n_pairs, vocab_size, device)
            _ = model(data)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(100):
            data, _ = generate_associative_recall(100, n_pairs, vocab_size, device)
            _ = model(data)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        infer_time = (time.time() - start_time) / 100  # per batch

    infer_memory = get_gpu_memory_mb()

    return {
        'accuracy': best_acc,
        'train_time': train_time,
        'train_memory_mb': train_memory,
        'infer_time_ms': infer_time * 1000,
        'infer_memory_mb': infer_memory
    }


def main():
    print("=" * 80)
    print("PSI vs MAMBA - FAIR BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    vocab_size = 16
    n_pairs = 5
    dim = 128
    n_layers = 4
    epochs = 1000

    results = {}

    # ========== PSI ==========
    print("-" * 80)
    print("PSI (Unified Phasor)")
    print("-" * 80)

    # Test multiple learning rates
    best_psi_result = None
    best_psi_lr = None

    for lr in [1e-3, 5e-4, 3e-4]:
        print(f"\n  Testing lr={lr}...")
        reset_memory()

        model = PSIModel(vocab_size + 1, dim, n_layers).to(device)
        params = sum(p.numel() for p in model.parameters())

        result = train_and_measure(model, vocab_size, n_pairs, epochs, lr)
        print(f"    Accuracy: {result['accuracy']:.1f}%")

        if best_psi_result is None or result['accuracy'] > best_psi_result['accuracy']:
            best_psi_result = result
            best_psi_lr = lr
            best_psi_result['params'] = params

        del model
        reset_memory()

    results['PSI'] = best_psi_result
    print(f"\n  Best PSI: lr={best_psi_lr}, accuracy={best_psi_result['accuracy']:.1f}%")

    # ========== Mamba with different configs ==========
    print()
    print("-" * 80)
    print("Mamba (with hyperparameter search)")
    print("-" * 80)

    mamba_configs = [
        {'lr': 1e-3, 'd_state': 16, 'name': 'Mamba (d_state=16, lr=1e-3)'},
        {'lr': 3e-4, 'd_state': 16, 'name': 'Mamba (d_state=16, lr=3e-4)'},
        {'lr': 1e-4, 'd_state': 16, 'name': 'Mamba (d_state=16, lr=1e-4)'},
        {'lr': 1e-3, 'd_state': 64, 'name': 'Mamba (d_state=64, lr=1e-3)'},
        {'lr': 3e-4, 'd_state': 64, 'name': 'Mamba (d_state=64, lr=3e-4)'},
    ]

    best_mamba_result = None
    best_mamba_config = None

    for cfg in mamba_configs:
        print(f"\n  Testing {cfg['name']}...")
        reset_memory()

        model = MambaModel(vocab_size + 1, dim, n_layers, d_state=cfg['d_state']).to(device)
        params = sum(p.numel() for p in model.parameters())

        result = train_and_measure(model, vocab_size, n_pairs, epochs, cfg['lr'])
        print(f"    Accuracy: {result['accuracy']:.1f}%")

        if best_mamba_result is None or result['accuracy'] > best_mamba_result['accuracy']:
            best_mamba_result = result
            best_mamba_config = cfg
            best_mamba_result['params'] = params

        del model
        reset_memory()

    results['Mamba'] = best_mamba_result
    print(f"\n  Best Mamba: {best_mamba_config['name']}, accuracy={best_mamba_result['accuracy']:.1f}%")

    # ========== Summary ==========
    print()
    print("=" * 80)
    print("SUMMARY: PSI vs Mamba (Best configs)")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'PSI':>15} {'Mamba':>15} {'Winner':>10}")
    print("-" * 65)

    # Accuracy
    psi_acc = results['PSI']['accuracy']
    mamba_acc = results['Mamba']['accuracy']
    winner = 'PSI' if psi_acc > mamba_acc + 1 else ('Mamba' if mamba_acc > psi_acc + 1 else 'Tie')
    print(f"{'Accuracy (%)':<25} {psi_acc:>14.1f}% {mamba_acc:>14.1f}% {winner:>10}")

    # Parameters
    psi_params = results['PSI']['params']
    mamba_params = results['Mamba']['params']
    winner = 'PSI' if psi_params < mamba_params else ('Mamba' if mamba_params < psi_params else 'Tie')
    print(f"{'Parameters':<25} {psi_params:>15,} {mamba_params:>15,} {winner:>10}")

    # Training time
    psi_train = results['PSI']['train_time']
    mamba_train = results['Mamba']['train_time']
    winner = 'PSI' if psi_train < mamba_train else ('Mamba' if mamba_train < psi_train else 'Tie')
    print(f"{'Train time (s)':<25} {psi_train:>15.1f} {mamba_train:>15.1f} {winner:>10}")

    # Training memory
    psi_tmem = results['PSI']['train_memory_mb']
    mamba_tmem = results['Mamba']['train_memory_mb']
    winner = 'PSI' if psi_tmem < mamba_tmem else ('Mamba' if mamba_tmem < psi_tmem else 'Tie')
    print(f"{'Train memory (MB)':<25} {psi_tmem:>15.1f} {mamba_tmem:>15.1f} {winner:>10}")

    # Inference time
    psi_infer = results['PSI']['infer_time_ms']
    mamba_infer = results['Mamba']['infer_time_ms']
    winner = 'PSI' if psi_infer < mamba_infer else ('Mamba' if mamba_infer < psi_infer else 'Tie')
    print(f"{'Inference time (ms)':<25} {psi_infer:>15.2f} {mamba_infer:>15.2f} {winner:>10}")

    # Inference memory
    psi_imem = results['PSI']['infer_memory_mb']
    mamba_imem = results['Mamba']['infer_memory_mb']
    winner = 'PSI' if psi_imem < mamba_imem else ('Mamba' if mamba_imem < psi_imem else 'Tie')
    print(f"{'Inference memory (MB)':<25} {psi_imem:>15.1f} {mamba_imem:>15.1f} {winner:>10}")

    print("-" * 65)
    print(f"Random baseline: {100/vocab_size:.1f}%")
    print()

    # Final verdict
    psi_wins = 0
    mamba_wins = 0

    if psi_acc > mamba_acc + 5: psi_wins += 2
    elif psi_acc > mamba_acc + 1: psi_wins += 1
    elif mamba_acc > psi_acc + 5: mamba_wins += 2
    elif mamba_acc > psi_acc + 1: mamba_wins += 1

    if psi_params < mamba_params * 0.8: psi_wins += 1
    elif mamba_params < psi_params * 0.8: mamba_wins += 1

    if psi_infer < mamba_infer * 0.8: psi_wins += 1
    elif mamba_infer < psi_infer * 0.8: mamba_wins += 1

    print(f"PSI advantages: {psi_wins}")
    print(f"Mamba advantages: {mamba_wins}")

    if psi_wins > mamba_wins:
        print("\nVERDICT: PSI outperforms Mamba on this task!")
    elif mamba_wins > psi_wins:
        print("\nVERDICT: Mamba outperforms PSI on this task")
    else:
        print("\nVERDICT: PSI and Mamba are comparable")


if __name__ == "__main__":
    main()
