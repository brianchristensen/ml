"""
Associative Recall Benchmark: Clifford O(n) vs Transformer O(n²)

Tests:
1. Copy/Recall: Remember tokens from earlier in sequence, retrieve when cued
2. Induction Heads: Learn A...B...A→B pattern (if A followed by B, later A predicts B)

Both tasks require associative memory - storing key→value bindings and retrieving by similarity.
Clifford should do this in O(n) via phasor binding, Transformer in O(n²) via attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from clifford_memory import OrthogonalModel
from phasor_optimal import OptimalPhasorModel

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
# Task 1: Copy/Recall (Simplified)
# ============================================================================

def generate_copy_task(batch_size, seq_len, vocab_size, n_to_copy=8):
    """
    Simplified copy task.

    Format: [tokens to copy] [zeros as delimiter] [should output copied tokens]

    First half: random tokens to memorize
    Second half: model must reproduce them

    Uses vocab 0..vocab_size-1 for content, position encodes the task structure.
    """
    half = seq_len // 2

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        # First half: tokens to copy (use smaller vocab for easier learning)
        tokens = torch.randint(1, min(vocab_size, 32), (n_to_copy,))
        sequences[b, :n_to_copy] = tokens
        # Rest of first half is zeros (delimiter)
        sequences[b, n_to_copy:half] = 0

        # Second half: model should output the tokens
        # Input is zeros (or we could use a RECALL token)
        sequences[b, half:half + n_to_copy] = 0

        # Target: predict the copied tokens
        # At position half+i, having seen tokens[0:n_to_copy] earlier, predict tokens[i]
        for i in range(n_to_copy):
            targets[b, half + i] = tokens[i]

    return sequences, targets


# ============================================================================
# Task 2: Induction Heads (Simplified)
# ============================================================================

def generate_induction_task(batch_size, seq_len, vocab_size, n_pairs=8):
    """
    Simplified induction head task.

    Format: [A B ... A ?] where ? should be B

    We create sequences where a bigram A→B appears, then later A appears again.
    The model should predict B after the second A.

    Uses a fixed pattern: sequence is pairs of (key, value) tokens,
    then later the keys repeat and model must predict values.
    """
    # Split vocab in half: keys from first half, values from second half
    key_vocab = vocab_size // 2  # Keys from 0..key_vocab-1
    val_vocab = vocab_size // 2  # Values from key_vocab..vocab_size-1

    assert n_pairs <= key_vocab, f"Need n_pairs ({n_pairs}) <= key_vocab ({key_vocab})"

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        # Generate random key-value pairs (unique keys)
        keys = torch.randperm(key_vocab)[:n_pairs]
        vals = torch.randint(key_vocab, key_vocab + val_vocab, (n_pairs,))

        # First section: present key-value pairs
        # Format: K1 V1 K2 V2 K3 V3 ...
        for i in range(n_pairs):
            sequences[b, i * 2] = keys[i]
            sequences[b, i * 2 + 1] = vals[i]

        # Delimiter section (zeros)
        pair_end = n_pairs * 2
        delim_end = pair_end + 10
        sequences[b, pair_end:delim_end] = 0

        # Query section: present keys again, model should predict values
        # Shuffle order to test true recall, not just sequence memory
        query_order = torch.randperm(n_pairs)
        for i, idx in enumerate(query_order):
            pos = delim_end + i * 2
            if pos + 1 < seq_len:
                sequences[b, pos] = keys[idx]
                sequences[b, pos + 1] = vals[idx]  # Ground truth in sequence
                targets[b, pos] = vals[idx]  # Target: predict value after seeing key

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

        # Flatten for loss computation
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt.view(-1),
            ignore_index=-100
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute accuracy on non-ignored positions
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
            # Clear cache before each test
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
            times[seq_len] = float('inf')  # Mark as OOM
            if device == 'cuda':
                torch.cuda.empty_cache()

    return times


# ============================================================================
# Main Experiment
# ============================================================================

def run_associative_recall_benchmark():
    print("=" * 80)
    print("ASSOCIATIVE RECALL BENCHMARK: Clifford O(n) vs Transformer O(n²)")
    print("=" * 80)
    print()

    # Set seeds
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
    n_epochs = 10

    # Create models
    models = {
        'Transformer': TransformerBaseline(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=4
        ).to(device),
        'Clifford': OrthogonalModel(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_orthogonal_sets=4,
            planes_per_set=8
        ).to(device),
        'PhasorOpt': OptimalPhasorModel(
            vocab_size=vocab_size,
            dim=dim,
            max_seq_len=16384
        ).to(device)
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

    # ========== Task 2: Induction Heads ==========
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
        if times[1] > 0 and times[1] != float('inf') and times[0] != float('inf'):
            ratio = times[0] / times[1]
            print(f"{ratio:>11.2f}x")
        elif times[0] == float('inf') and times[1] != float('inf'):
            print(f"{'Cliff wins':>14}")
        else:
            print(f"{'N/A':>14}")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Task':<20} {'Transformer':>12} {'Clifford':>12} {'PhasorOpt':>12} {'Winner':>12}")
    print("-" * 72)

    copy_tf = results['Transformer_copy']['acc']
    copy_cl = results['Clifford_copy']['acc']
    copy_po = results['PhasorOpt_copy']['acc']
    copy_scores = {'Transformer': copy_tf, 'Clifford': copy_cl, 'PhasorOpt': copy_po}
    copy_winner = max(copy_scores, key=copy_scores.get)
    print(f"{'Copy/Recall Acc':<20} {copy_tf:>11.1%} {copy_cl:>11.1%} {copy_po:>11.1%} {copy_winner:>12}")

    ind_tf = results['Transformer_induction']['acc']
    ind_cl = results['Clifford_induction']['acc']
    ind_po = results['PhasorOpt_induction']['acc']
    ind_scores = {'Transformer': ind_tf, 'Clifford': ind_cl, 'PhasorOpt': ind_po}
    ind_winner = max(ind_scores, key=ind_scores.get)
    print(f"{'Assoc Recall Acc':<20} {ind_tf:>11.1%} {ind_cl:>11.1%} {ind_po:>11.1%} {ind_winner:>12}")

    # Timing scaling
    def format_time(t):
        if t == float('inf'):
            return "OOM"
        return f"{t*1000:.1f}ms"

    print(f"\n{'Timing':<20} {'Transformer':>12} {'Clifford':>12} {'PhasorOpt':>12}")
    print("-" * 56)
    for sl in [64, 512, 8192]:
        t_tf = timing_results['Transformer'][sl]
        t_cl = timing_results['Clifford'][sl]
        t_po = timing_results['PhasorOpt'][sl]
        print(f"{'@ ' + str(sl) + ' tokens':<20} {format_time(t_tf):>12} {format_time(t_cl):>12} {format_time(t_po):>12}")

    # Find fastest at 8192
    times_8192 = {k: v[8192] for k, v in timing_results.items() if v[8192] != float('inf')}
    if times_8192:
        fastest = min(times_8192, key=times_8192.get)
        slowest_time = max(times_8192.values())
        fastest_time = times_8192[fastest]
        print(f"\nAt 8192 tokens: {fastest} is {slowest_time/fastest_time:.1f}x faster than slowest")

    return results, timing_results


def run_scaled_associative_test():
    """
    Scale up associative recall to stress-test addressing capacity.

    Key question: Where does each model break as we increase key-value pairs?
    Transformer should degrade due to attention dilution.
    Clifford should maintain performance due to exponential address space.
    """
    print("=" * 80)
    print("SCALED ASSOCIATIVE RECALL: Testing Address Space Capacity")
    print("=" * 80)
    print()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    vocab_size = 64  # Match main benchmark
    dim = 64
    n_layers = 4
    n_train = 2000
    n_test = 500
    n_epochs = 15  # More epochs for harder tasks

    # Test with increasing number of key-value pairs
    # Max 32 pairs with vocab_size=64 (32 keys + 32 values)
    kv_counts = [8, 16, 24, 32]

    results = {'Transformer': {}, 'Clifford': {}}

    for n_kv in kv_counts:
        seq_len = max(128, n_kv * 4 + 32)  # Minimum 128 to match main benchmark
        print(f"\n{'='*60}")
        print(f"Testing {n_kv} Key-Value Pairs (seq_len={seq_len})")
        print('='*60)

        # Generate data for this KV count
        train_seq, train_tgt = generate_induction_task(n_train, seq_len, vocab_size, n_pairs=n_kv)
        test_seq, test_tgt = generate_induction_task(n_test, seq_len, vocab_size, n_pairs=n_kv)

        for name, ModelClass, kwargs in [
            ('Transformer', TransformerBaseline, {'vocab_size': vocab_size, 'dim': dim, 'n_layers': n_layers}),
            ('Clifford', OrthogonalModel, {'vocab_size': vocab_size, 'dim': dim, 'n_layers': n_layers,
                                           'n_orthogonal_sets': 4, 'planes_per_set': 8}),
        ]:
            model = ModelClass(**kwargs).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n--- {name} ({n_params:,} params) ---")

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            best_acc = 0
            for epoch in range(n_epochs):
                train_loss, train_acc = train_epoch(model, train_seq, train_tgt, optimizer)

                if (epoch + 1) % 5 == 0:
                    val_loss, val_acc = evaluate(model, test_seq, test_tgt)
                    print(f"  Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                    best_acc = max(best_acc, val_acc)

            results[name][n_kv] = best_acc

            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SCALING SUMMARY: Associative Recall Accuracy vs # Key-Value Pairs")
    print("=" * 80)
    print()
    print(f"{'# KV Pairs':<12} {'Transformer':>15} {'Clifford':>15} {'Winner':>12}")
    print("-" * 54)

    for n_kv in kv_counts:
        t_acc = results['Transformer'][n_kv]
        c_acc = results['Clifford'][n_kv]
        winner = "Clifford" if c_acc > t_acc else "Transformer" if t_acc > c_acc else "Tie"
        print(f"{n_kv:<12} {t_acc*100:>14.1f}% {c_acc*100:>14.1f}% {winner:>12}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        results = run_scaled_associative_test()
    else:
        results, timing = run_associative_recall_benchmark()
