"""
Comprehensive Phasor Benchmark

Tests both:
1. Language modeling (enwik8) - compare to Transformer
2. Associative recall (copy, KV binding) - must not regress

Use this to iterate on Phasor architecture while ensuring we don't break
the associative memory capabilities that make it special.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from phasor import PhasorModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# =============================================================================
# Transformer Baseline
# =============================================================================

class TransformerLM(nn.Module):
    """Standard transformer for language modeling."""
    def __init__(self, vocab_size=256, dim=128, n_layers=4, n_heads=4, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.head(h)


# =============================================================================
# Synthetic Tasks (Copy + Associative Recall)
# =============================================================================

def generate_copy_task(batch_size, seq_len, vocab_size, n_to_copy=8):
    """Copy task - tests positional memory."""
    half = seq_len // 2
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        tokens = torch.randint(1, min(vocab_size, 32), (n_to_copy,))
        sequences[b, :n_to_copy] = tokens
        for i in range(n_to_copy):
            targets[b, half + i] = tokens[i]

    return sequences, targets


def generate_assoc_task(batch_size, seq_len, vocab_size, n_pairs=8):
    """Associative recall - tests content-based memory."""
    key_vocab = vocab_size // 2
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

        query_order = torch.randperm(n_pairs)
        for i, idx in enumerate(query_order):
            pos = delim_end + i * 2
            if pos + 1 < seq_len:
                sequences[b, pos] = keys[idx]
                sequences[b, pos + 1] = vals[idx]
                targets[b, pos] = vals[idx]

    return sequences, targets


def eval_synthetic(model, vocab_size=64):
    """Evaluate on synthetic tasks (quick check)."""
    model.eval()
    results = {}

    for task_name, task_fn in [('copy', generate_copy_task), ('assoc', generate_assoc_task)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                seq, tgt = task_fn(64, 128, vocab_size)
                seq, tgt = seq.to(device), tgt.to(device)

                logits = model(seq)
                mask = tgt != -100
                if mask.sum() > 0:
                    preds = logits.argmax(dim=-1)
                    correct += ((preds == tgt) & mask).sum().item()
                    total += mask.sum().item()

        results[task_name] = correct / total if total > 0 else 0

    return results


def train_synthetic(model, vocab_size=64, n_epochs=20):
    """Train on synthetic tasks to verify associative capabilities."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    for epoch in range(n_epochs):
        model.train()
        for task_fn in [generate_copy_task, generate_assoc_task]:
            for _ in range(25):
                seq, tgt = task_fn(64, 128, vocab_size)
                seq, tgt = seq.to(device), tgt.to(device)

                logits = model(seq)
                loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), ignore_index=-100)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    return eval_synthetic(model, vocab_size)


# =============================================================================
# Language Modeling (enwik8)
# =============================================================================

def load_enwik8(path='data/enwik8', max_bytes=1000000):
    """Load enwik8 data."""
    with open(path, 'rb') as f:
        data = list(f.read(max_bytes))

    split = int(len(data) * 0.9)
    return data[:split], data[split:]


def get_batch(data, batch_size, seq_len):
    """Get a batch for language modeling."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1]) for i in ix])
    return x.to(device), y.to(device)


def train_lm_epoch(model, train_data, batch_size=32, seq_len=128, n_batches=100):
    """Train for one epoch on language modeling."""
    model.train()
    total_loss = 0

    optimizer = getattr(model, '_optimizer', None)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model._optimizer = optimizer

    for _ in range(n_batches):
        x, y = get_batch(train_data, batch_size, seq_len)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


def eval_lm(model, val_data, batch_size=32, seq_len=128, n_batches=50):
    """Evaluate language model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size, seq_len)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / n_batches
    bpc = avg_loss / np.log(2)
    return bpc


def measure_speed(model, seq_len=128, batch_size=32, n_runs=20):
    """Measure inference speed."""
    model.eval()
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
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
    return elapsed * 1000  # ms


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark():
    print("=" * 70)
    print("PHASOR COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print()

    # Config
    dim = 128
    n_layers = 4
    vocab_size_lm = 256
    vocab_size_synth = 64

    # Load data
    print("Loading enwik8...")
    train_data, val_data = load_enwik8(max_bytes=1000000)
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    print()

    # Create models
    print("Creating models...")
    phasor = PhasorModel(
        vocab_size=vocab_size_lm,
        dim=dim,
        n_layers=n_layers,
        max_seq_len=256
    ).to(device)

    transformer = TransformerLM(
        vocab_size=vocab_size_lm,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        max_seq_len=256
    ).to(device)

    phasor_params = sum(p.numel() for p in phasor.parameters())
    tf_params = sum(p.numel() for p in transformer.parameters())
    print(f"Phasor: {phasor_params:,} parameters")
    print(f"Transformer: {tf_params:,} parameters")
    print()

    # ==========================================================================
    # Part 1: Language Modeling
    # ==========================================================================
    print("=" * 70)
    print("PART 1: LANGUAGE MODELING (enwik8)")
    print("=" * 70)
    print()

    n_epochs = 10

    print("Training Transformer...")
    for epoch in range(n_epochs):
        train_loss = train_lm_epoch(transformer, train_data)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            val_bpc = eval_lm(transformer, val_data)
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.3f}, val_BPC={val_bpc:.3f}")
    tf_bpc = eval_lm(transformer, val_data)

    print("\nTraining Phasor...")
    for epoch in range(n_epochs):
        train_loss = train_lm_epoch(phasor, train_data)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            val_bpc = eval_lm(phasor, val_data)
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.3f}, val_BPC={val_bpc:.3f}")
    phasor_bpc = eval_lm(phasor, val_data)

    print(f"\nLanguage Modeling Results:")
    print(f"  Transformer: {tf_bpc:.3f} BPC")
    print(f"  Phasor:      {phasor_bpc:.3f} BPC")
    print(f"  Gap:         {phasor_bpc - tf_bpc:+.3f} BPC")

    # ==========================================================================
    # Part 2: Synthetic Tasks (Associative Memory)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 2: ASSOCIATIVE MEMORY (Copy + KV Recall)")
    print("=" * 70)
    print()

    # Create fresh models for synthetic tasks (smaller vocab)
    phasor_synth = PhasorModel(
        vocab_size=vocab_size_synth,
        dim=64,
        n_layers=4,
        max_seq_len=256
    ).to(device)

    transformer_synth = TransformerLM(
        vocab_size=vocab_size_synth,
        dim=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=256
    ).to(device)

    print("Training Transformer on synthetic tasks...")
    tf_synth = train_synthetic(transformer_synth, vocab_size_synth)
    print(f"  Copy: {tf_synth['copy']:.1%}, Assoc: {tf_synth['assoc']:.1%}")

    print("\nTraining Phasor on synthetic tasks...")
    phasor_synth_results = train_synthetic(phasor_synth, vocab_size_synth)
    print(f"  Copy: {phasor_synth_results['copy']:.1%}, Assoc: {phasor_synth_results['assoc']:.1%}")

    # ==========================================================================
    # Part 3: Speed Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 3: INFERENCE SPEED")
    print("=" * 70)
    print()

    for seq_len in [128, 256, 512, 1024]:
        try:
            tf_speed = measure_speed(transformer, seq_len)
            phasor_speed = measure_speed(phasor, seq_len)
            ratio = tf_speed / phasor_speed
            print(f"  seq_len={seq_len}: TF={tf_speed:.1f}ms, Phasor={phasor_speed:.1f}ms, ratio={ratio:.2f}x")
        except Exception as e:
            print(f"  seq_len={seq_len}: Error - {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Transformer':>15} {'Phasor':>15} {'Winner':>10}")
    print("-" * 65)
    print(f"{'LM BPC (lower=better)':<25} {tf_bpc:>15.3f} {phasor_bpc:>15.3f} {'TF' if tf_bpc < phasor_bpc else 'Phasor':>10}")
    print(f"{'Copy Accuracy':<25} {tf_synth['copy']:>14.1%} {phasor_synth_results['copy']:>14.1%} {'TF' if tf_synth['copy'] > phasor_synth_results['copy'] else 'Phasor':>10}")
    print(f"{'Assoc Recall Accuracy':<25} {tf_synth['assoc']:>14.1%} {phasor_synth_results['assoc']:>14.1%} {'TF' if tf_synth['assoc'] > phasor_synth_results['assoc'] else 'Phasor':>10}")
    print(f"{'Parameters':<25} {tf_params:>15,} {phasor_params:>15,}")
    print(f"{'Complexity':<25} {'O(n^2)':>15} {'O(n)':>15}")

    print("\n" + "=" * 70)
    print("PASS/FAIL CRITERIA")
    print("=" * 70)

    lm_pass = phasor_bpc < tf_bpc + 0.5  # Within 0.5 BPC of transformer
    copy_pass = phasor_synth_results['copy'] > 0.95  # >95% on copy
    assoc_pass = phasor_synth_results['assoc'] > 0.90  # >90% on associative

    print(f"  LM within 0.5 BPC of Transformer: {'PASS' if lm_pass else 'FAIL'} ({phasor_bpc:.3f} vs {tf_bpc:.3f})")
    print(f"  Copy accuracy > 95%:              {'PASS' if copy_pass else 'FAIL'} ({phasor_synth_results['copy']:.1%})")
    print(f"  Associative recall > 90%:         {'PASS' if assoc_pass else 'FAIL'} ({phasor_synth_results['assoc']:.1%})")

    all_pass = lm_pass and copy_pass and assoc_pass
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return {
        'lm_bpc': {'transformer': tf_bpc, 'phasor': phasor_bpc},
        'copy': {'transformer': tf_synth['copy'], 'phasor': phasor_synth_results['copy']},
        'assoc': {'transformer': tf_synth['assoc'], 'phasor': phasor_synth_results['assoc']},
        'all_pass': all_pass
    }


if __name__ == "__main__":
    results = run_benchmark()
