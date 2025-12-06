"""
PSI vs Mamba - Hard Recall Tasks

Tasks where Mamba is known to struggle:
1. Long-distance associative recall (many pairs, query from far back)
2. Multi-query recall (retrieve multiple values in sequence)
3. Selective copy (copy only specific tokens based on markers)
4. Induction heads (A...B...A -> predict B)

These tasks require precise content-addressable memory, not just compression.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
import gc

from mambapy.mamba import Mamba, MambaConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self, vocab_size, dim=128, n_layers=4, n_oscillators=64, max_len=512):
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
# Task 1: Long-Distance Associative Recall (Many Pairs)
# ============================================================================

def generate_long_associative_recall(batch_size, n_pairs, vocab_size, device):
    """
    More pairs = harder for linear attention.
    Format: k1,v1,k2,v2,...,kN,vN,QUERY,k_i -> predict v_i

    With many pairs, the memory gets saturated.
    """
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
# Task 2: Multi-Query Recall (Multiple retrievals in sequence)
# ============================================================================

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    """
    After storing pairs, must answer multiple queries in sequence.
    Format: k1,v1,...,kN,vN,QUERY,q1,QUERY,q2,QUERY,q3,...

    Each query position should predict the corresponding value.
    This is HARD because model must maintain memory across multiple retrievals.
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
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        # Multiple queries
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


# ============================================================================
# Task 3: Induction Heads (A...B...A -> B)
# ============================================================================

def generate_induction_task(batch_size, seq_len, vocab_size, device):
    """
    Induction head task: learn to predict what follows a repeated token.
    Sequence contains: ...A B ... A ? -> predict B

    The model must recognize that A appeared before, and recall what followed.
    """
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Pick a random position for the pattern
        pattern_pos = np.random.randint(5, seq_len // 2)
        repeat_pos = np.random.randint(seq_len // 2 + 5, seq_len - 1)

        # A B at pattern_pos
        A = np.random.randint(0, vocab_size)
        B = np.random.randint(0, vocab_size)
        data[b, pattern_pos] = A
        data[b, pattern_pos + 1] = B

        # A at repeat_pos -> should predict B at repeat_pos + 1
        data[b, repeat_pos] = A
        targets[b, repeat_pos + 1] = B

    return data, targets


# ============================================================================
# Task 4: Selective Copy (copy only marked tokens)
# ============================================================================

def generate_selective_copy(batch_size, n_to_copy, total_len, vocab_size, device):
    """
    Only copy tokens that are preceded by a MARK token.
    Format: random, MARK, a, random, MARK, b, random, MARK, c, ..., COPY -> a, b, c

    This requires selective attention to marked positions.
    """
    MARK_TOKEN = vocab_size
    COPY_TOKEN = vocab_size + 1

    data = torch.randint(0, vocab_size, (batch_size, total_len + n_to_copy + 1), device=device)
    targets = torch.zeros(batch_size, total_len + n_to_copy + 1, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Place MARK tokens at random positions
        mark_positions = sorted(np.random.choice(range(0, total_len - 1), n_to_copy, replace=False))
        values_to_copy = []

        for pos in mark_positions:
            data[b, pos] = MARK_TOKEN
            val = np.random.randint(0, vocab_size)
            data[b, pos + 1] = val
            values_to_copy.append(val)

        # COPY token signals start of output
        data[b, total_len] = COPY_TOKEN

        # Target: the marked values in order
        for i, val in enumerate(values_to_copy):
            targets[b, total_len + 1 + i] = val

    return data, targets, list(range(total_len + 1, total_len + 1 + n_to_copy))


# ============================================================================
# Training and Evaluation
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


def train_and_eval(model_class, model_kwargs, task_fn, task_kwargs,
                   eval_positions, vocab_size, epochs=500, lr=1e-3):
    """Train and evaluate a model on a task."""
    reset_memory()

    model = model_class(**model_kwargs).to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        result = task_fn(**task_kwargs, device=device)
        if len(result) == 2:
            data, targets = result
            positions = eval_positions
        else:
            data, targets, positions = result

        logits = model(data)

        # Loss only at evaluation positions
        loss = 0
        for pos in positions:
            loss += criterion(logits[:, pos, :vocab_size], targets[:, pos])
        loss = loss / len(positions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                result = task_fn(**{**task_kwargs, 'batch_size': 500}, device=device)
                if len(result) == 2:
                    data, targets = result
                else:
                    data, targets, _ = result

                logits = model(data)
                correct = 0
                total = 0
                for pos in positions:
                    preds = logits[:, pos, :vocab_size].argmax(dim=-1)
                    correct += (preds == targets[:, pos]).sum().item()
                    total += data.shape[0]

                acc = correct / total * 100
                if acc > best_acc:
                    best_acc = acc

    train_time = time.time() - start_time
    train_mem = get_gpu_memory_mb()

    del model
    reset_memory()

    return {
        'accuracy': best_acc,
        'params': params,
        'train_time': train_time,
        'train_mem': train_mem
    }


def main():
    print("=" * 80)
    print("PSI vs MAMBA - HARD RECALL TASKS")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    vocab_size = 32  # Larger vocab for harder tasks
    dim = 64  # Smaller for speed
    n_layers = 2  # Fewer layers
    epochs = 200  # Faster training

    results = {}

    # ========== Task 1: Long Associative Recall (15 pairs) ==========
    print("-" * 80)
    print("Task 1: Long Associative Recall (15 pairs, vocab=32)")
    print("  Challenge: Store and retrieve from many key-value pairs")
    print("-" * 80)

    n_pairs = 15
    query_pos = n_pairs * 2 + 1

    print("\n  Training PSI...")
    psi_result = train_and_eval(
        PSIModel, {'vocab_size': vocab_size + 1, 'dim': dim, 'n_layers': n_layers, 'max_len': 64},
        generate_long_associative_recall,
        {'batch_size': 64, 'n_pairs': n_pairs, 'vocab_size': vocab_size},
        [query_pos], vocab_size, epochs=epochs
    )
    print(f"    PSI: {psi_result['accuracy']:.1f}%")

    print("\n  Training Mamba...")
    mamba_result = train_and_eval(
        MambaModel, {'vocab_size': vocab_size + 1, 'dim': dim, 'n_layers': n_layers, 'd_state': 16},
        generate_long_associative_recall,
        {'batch_size': 64, 'n_pairs': n_pairs, 'vocab_size': vocab_size},
        [query_pos], vocab_size, epochs=epochs
    )
    print(f"    Mamba: {mamba_result['accuracy']:.1f}%")

    results['Long AR'] = {
        'PSI': psi_result,
        'Mamba': mamba_result,
        'random': 100 / vocab_size
    }

    # ========== Task 2: Multi-Query Recall ==========
    print()
    print("-" * 80)
    print("Task 2: Multi-Query Recall (8 pairs, 4 queries)")
    print("  Challenge: Answer multiple queries without forgetting")
    print("-" * 80)

    n_pairs = 8
    n_queries = 4

    print("\n  Training PSI...")
    psi_result = train_and_eval(
        PSIModel, {'vocab_size': vocab_size + 1, 'dim': dim, 'n_layers': n_layers, 'max_len': 64},
        generate_multi_query_recall,
        {'batch_size': 64, 'n_pairs': n_pairs, 'n_queries': n_queries, 'vocab_size': vocab_size},
        None, vocab_size, epochs=epochs  # positions returned by task_fn
    )
    print(f"    PSI: {psi_result['accuracy']:.1f}%")

    print("\n  Training Mamba...")
    mamba_result = train_and_eval(
        MambaModel, {'vocab_size': vocab_size + 1, 'dim': dim, 'n_layers': n_layers, 'd_state': 16},
        generate_multi_query_recall,
        {'batch_size': 64, 'n_pairs': n_pairs, 'n_queries': n_queries, 'vocab_size': vocab_size},
        None, vocab_size, epochs=epochs
    )
    print(f"    Mamba: {mamba_result['accuracy']:.1f}%")

    results['Multi-Query'] = {
        'PSI': psi_result,
        'Mamba': mamba_result,
        'random': 100 / vocab_size
    }

    # ========== Task 3: Selective Copy ==========
    print()
    print("-" * 80)
    print("Task 3: Selective Copy (copy 5 marked tokens from 40)")
    print("  Challenge: Selectively attend to marked positions only")
    print("-" * 80)

    n_to_copy = 5
    total_len = 40

    print("\n  Training PSI...")
    psi_result = train_and_eval(
        PSIModel, {'vocab_size': vocab_size + 2, 'dim': dim, 'n_layers': n_layers, 'max_len': 64},
        generate_selective_copy,
        {'batch_size': 64, 'n_to_copy': n_to_copy, 'total_len': total_len, 'vocab_size': vocab_size},
        None, vocab_size, epochs=epochs
    )
    print(f"    PSI: {psi_result['accuracy']:.1f}%")

    print("\n  Training Mamba...")
    mamba_result = train_and_eval(
        MambaModel, {'vocab_size': vocab_size + 2, 'dim': dim, 'n_layers': n_layers, 'd_state': 16},
        generate_selective_copy,
        {'batch_size': 64, 'n_to_copy': n_to_copy, 'total_len': total_len, 'vocab_size': vocab_size},
        None, vocab_size, epochs=epochs
    )
    print(f"    Mamba: {mamba_result['accuracy']:.1f}%")

    results['Selective Copy'] = {
        'PSI': psi_result,
        'Mamba': mamba_result,
        'random': 100 / vocab_size
    }

    # ========== Summary ==========
    print()
    print("=" * 80)
    print("SUMMARY: Hard Recall Tasks")
    print("=" * 80)
    print()

    print(f"{'Task':<20} {'PSI':>12} {'Mamba':>12} {'Random':>10}")
    print("-" * 55)

    for task_name, task_results in results.items():
        psi_acc = task_results['PSI']['accuracy']
        mamba_acc = task_results['Mamba']['accuracy']
        random = task_results['random']
        print(f"{task_name:<20} {psi_acc:>11.1f}% {mamba_acc:>11.1f}% {random:>9.1f}%")

    print()
    print("Parameters:")
    print(f"  PSI:   {results['Long AR']['PSI']['params']:>10,}")
    print(f"  Mamba: {results['Long AR']['Mamba']['params']:>10,}")

    # Verdict
    print()
    psi_total = sum(r['PSI']['accuracy'] for r in results.values())
    mamba_total = sum(r['Mamba']['accuracy'] for r in results.values())

    print(f"Total accuracy across tasks:")
    print(f"  PSI:   {psi_total:.1f}%")
    print(f"  Mamba: {mamba_total:.1f}%")

    if psi_total > mamba_total * 1.05:
        print("\nVERDICT: PSI outperforms Mamba on hard recall tasks!")
    elif mamba_total > psi_total * 1.05:
        print("\nVERDICT: Mamba outperforms PSI on hard recall tasks")
    else:
        print("\nVERDICT: PSI and Mamba are comparable on hard recall tasks")


if __name__ == "__main__":
    main()
