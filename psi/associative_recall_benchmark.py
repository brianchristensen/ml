"""
Associative Recall Benchmark: Phasor O(n) vs Transformer O(n²)

Tests:
1. Copy/Recall: Remember tokens from earlier in sequence, retrieve when cued
2. Associative Recall: Learn key-value bindings and retrieve values given keys

Both tasks require associative memory - storing key→value bindings and retrieving by similarity.
Phasor does this in O(n) via cumsum accumulation, Transformer in O(n²) via attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from phasor import PhasorModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ============================================================================
# Transformer Baseline
# ============================================================================

class TransformerBaseline(nn.Module):
    """Standard transformer for sequence modeling"""
    def __init__(self, vocab_size, dim=64, n_layers=4, n_heads=4, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)

        return self.output(h)


# ============================================================================
# Task 1: Copy/Recall
# ============================================================================

def generate_copy_task(batch_size, seq_len, vocab_size, n_to_copy=8):
    """
    Copy task - test positional memory.

    Format: [tokens to copy] [zeros as delimiter] [should output copied tokens]

    First half: random tokens to memorize
    Second half: model must reproduce them
    """
    half = seq_len // 2

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        tokens = torch.randint(1, min(vocab_size, 32), (n_to_copy,))
        sequences[b, :n_to_copy] = tokens
        sequences[b, n_to_copy:half] = 0
        sequences[b, half:half + n_to_copy] = 0

        for i in range(n_to_copy):
            targets[b, half + i] = tokens[i]

    return sequences, targets


# ============================================================================
# Task 2: Associative Recall (Key-Value)
# ============================================================================

def generate_induction_task(batch_size, seq_len, vocab_size, n_pairs=8):
    """
    Associative recall task - test content-based memory.

    Format: [K1 V1 K2 V2 ...] [zeros] [K3 -> V3, K1 -> V1, ...]

    Store key-value pairs, then recall values given shuffled keys.
    """
    key_vocab = vocab_size // 2

    assert n_pairs <= key_vocab, f"Need n_pairs ({n_pairs}) <= key_vocab ({key_vocab})"

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randperm(key_vocab)[:n_pairs]
        vals = torch.randint(key_vocab, key_vocab + key_vocab, (n_pairs,))

        for i in range(n_pairs):
            sequences[b, i * 2] = keys[i]
            sequences[b, i * 2 + 1] = vals[i]

        pair_end = n_pairs * 2
        delim_end = pair_end + 10
        sequences[b, pair_end:delim_end] = 0

        query_order = torch.randperm(n_pairs)
        for i, idx in enumerate(query_order):
            pos = delim_end + i * 2
            if pos + 1 < seq_len:
                sequences[b, pos] = keys[idx]
                sequences[b, pos + 1] = vals[idx]
                targets[b, pos] = vals[idx]

    return sequences, targets


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, sequences, targets, optimizer, batch_size=64):
    model.train()
    n_samples = len(sequences)
    indices = np.random.permutation(n_samples)

    total_loss = 0
    total_correct = 0
    total_counted = 0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        seq = sequences[batch_idx].to(device)
        tgt = targets[batch_idx].to(device)

        optimizer.zero_grad()
        logits = model(seq)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt.view(-1),
            ignore_index=-100
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mask = tgt != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = (preds == tgt) & mask
            total_correct += correct.sum().item()
            total_counted += mask.sum().item()

        total_loss += loss.item()
        n_batches += 1

    acc = total_correct / total_counted if total_counted > 0 else 0
    return total_loss / n_batches, acc


def evaluate(model, sequences, targets, batch_size=256):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_counted = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            seq = sequences[i:i+batch_size].to(device)
            tgt = targets[i:i+batch_size].to(device)

            logits = model(seq)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                ignore_index=-100
            )

            mask = tgt != -100
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1)
                correct = (preds == tgt) & mask
                total_correct += correct.sum().item()
                total_counted += mask.sum().item()

            total_loss += loss.item()
            n_batches += 1

    acc = total_correct / total_counted if total_counted > 0 else 0
    return total_loss / n_batches, acc


def measure_inference_time(model, seq_lens, vocab_size, batch_size=4, n_runs=5):
    """Measure inference time for different sequence lengths"""
    model.eval()
    times = {}

    for seq_len in seq_lens:
        try:
            if device == 'cuda':
                torch.cuda.empty_cache()

            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            with torch.no_grad():
                _ = model(x)

            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            with torch.no_grad():
                for _ in range(n_runs):
                    _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - start) / n_runs

            times[seq_len] = elapsed

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at seq_len={seq_len}")
            times[seq_len] = float('inf')
            if device == 'cuda':
                torch.cuda.empty_cache()

    return times


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark():
    print("=" * 80)
    print("MEMORY BENCHMARK: Phasor O(n) vs Transformer O(n²)")
    print("=" * 80)
    print()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Config
    vocab_size = 64
    dim = 64
    n_layers = 4
    seq_len = 128
    n_train = 2000
    n_test = 500
    n_epochs = 20

    # Create models
    models = {
        'Transformer': TransformerBaseline(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=4
        ).to(device),
        'Phasor': PhasorModel(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_phases=32
        ).to(device),
    }

    # Print param counts
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n_params:,} parameters")
    print()

    results = {}

    # ========== Task 1: Copy/Recall ==========
    print("=" * 60)
    print("Task 1: COPY/RECALL")
    print("=" * 60)
    print("Format: [tokens] [zeros] -> [reproduce tokens]")
    print("Memorize 8 tokens, recall them after a gap")
    print()

    copy_train_seq, copy_train_tgt = generate_copy_task(n_train, seq_len, vocab_size)
    copy_test_seq, copy_test_tgt = generate_copy_task(n_test, seq_len, vocab_size)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        for epoch in range(n_epochs):
            loss, acc = train_epoch(model, copy_train_seq, copy_train_tgt, optimizer)
            if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
                val_loss, val_acc = evaluate(model, copy_test_seq, copy_test_tgt)
                print(f"  Epoch {epoch+1}: train_acc={acc:.3f}, val_acc={val_acc:.3f}")

        final_loss, final_acc = evaluate(model, copy_test_seq, copy_test_tgt)
        results[f'{name}_copy'] = {'loss': final_loss, 'acc': final_acc}

    # ========== Task 2: Associative Recall ==========
    print("\n" + "=" * 60)
    print("Task 2: ASSOCIATIVE RECALL (Key-Value)")
    print("=" * 60)
    print("Format: [K1 V1 K2 V2 ...] [zeros] [K3 -> V3, K1 -> V1, ...]")
    print("Store 8 key-value pairs, recall values given shuffled keys")
    print()

    ind_train_seq, ind_train_tgt = generate_induction_task(n_train, seq_len, vocab_size)
    ind_test_seq, ind_test_tgt = generate_induction_task(n_test, seq_len, vocab_size)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Reset model weights for fair comparison
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        for epoch in range(n_epochs):
            loss, acc = train_epoch(model, ind_train_seq, ind_train_tgt, optimizer)
            if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
                val_loss, val_acc = evaluate(model, ind_test_seq, ind_test_tgt)
                print(f"  Epoch {epoch+1}: train_acc={acc:.3f}, val_acc={val_acc:.3f}")

        final_loss, final_acc = evaluate(model, ind_test_seq, ind_test_tgt)
        results[f'{name}_induction'] = {'loss': final_loss, 'acc': final_acc}

    # ========== Timing Comparison ==========
    print("\n" + "=" * 60)
    print("INFERENCE TIME SCALING")
    print("=" * 60)
    print()

    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"{'Seq Len':<10}", end="")
    for name in models.keys():
        print(f"{name:>15}", end="")
    print(f"{'Ratio':>12}")
    print("-" * 52)

    timing_results = {}
    for name, model in models.items():
        timing_results[name] = measure_inference_time(model, seq_lens, vocab_size)

    for seq_len in seq_lens:
        print(f"{seq_len:<10}", end="")
        times = []
        for name in models.keys():
            t = timing_results[name][seq_len]
            times.append(t)
            if t == float('inf'):
                print(f"{'OOM':>15}", end="")
            else:
                print(f"{t*1000:>12.2f}ms", end="")
        if len(times) >= 2 and times[1] > 0 and times[1] != float('inf') and times[0] != float('inf'):
            ratio = times[0] / times[1]
            print(f"{ratio:>11.2f}x")
        elif len(times) >= 2 and times[0] == float('inf') and times[1] != float('inf'):
            print(f"{'Phasor wins':>14}")
        else:
            print(f"{'N/A':>14}")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Task':<20} {'Transformer':>14} {'Phasor':>14} {'Winner':>12}")
    print("-" * 60)

    copy_tf = results['Transformer_copy']['acc']
    copy_ph = results['Phasor_copy']['acc']
    copy_winner = 'Transformer' if copy_tf > copy_ph else 'Phasor' if copy_ph > copy_tf else 'Tie'
    print(f"{'Copy/Recall Acc':<20} {copy_tf:>13.1%} {copy_ph:>13.1%} {copy_winner:>12}")

    ind_tf = results['Transformer_induction']['acc']
    ind_ph = results['Phasor_induction']['acc']
    ind_winner = 'Transformer' if ind_tf > ind_ph else 'Phasor' if ind_ph > ind_tf else 'Tie'
    print(f"{'Assoc Recall Acc':<20} {ind_tf:>13.1%} {ind_ph:>13.1%} {ind_winner:>12}")

    # Timing scaling
    def format_time(t):
        if t == float('inf'):
            return "OOM"
        return f"{t*1000:.1f}ms"

    print(f"\n{'Timing':<20} {'Transformer':>14} {'Phasor':>14}")
    print("-" * 48)
    for sl in [64, 512, 8192]:
        t_tf = timing_results['Transformer'][sl]
        t_ph = timing_results['Phasor'][sl]
        print(f"{'@ ' + str(sl) + ' tokens':<20} {format_time(t_tf):>14} {format_time(t_ph):>14}")

    # Find fastest at 8192
    times_8192 = {k: v[8192] for k, v in timing_results.items() if v[8192] != float('inf')}
    if times_8192:
        fastest = min(times_8192, key=times_8192.get)
        slowest_time = max(times_8192.values())
        fastest_time = times_8192[fastest]
        print(f"\nAt 8192 tokens: {fastest} is {slowest_time/fastest_time:.1f}x faster than slowest")

    return results, timing_results


if __name__ == "__main__":
    results, timing = run_benchmark()
