"""
Quick test of iterative memory models on associative recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from iterative_memory import IterativeMemoryModel, IterativeRefinementModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def generate_associative_recall(batch_size, n_pairs, vocab_size, seq_len):
    """
    Generate associative recall task.
    Format: [K1 V1 K2 V2 ... Kn Vn] [zeros] [Ki ... -> Vi ...]
    """
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    half = seq_len // 2

    for b in range(batch_size):
        # Generate unique keys and random values
        keys = torch.randperm(vocab_size - 1)[:n_pairs] + 1
        values = torch.randint(1, vocab_size, (n_pairs,))

        # Store pairs in first half
        for i, (k, v) in enumerate(zip(keys, values)):
            sequences[b, i * 2] = k
            sequences[b, i * 2 + 1] = v

        # Query in second half (shuffled order)
        query_order = torch.randperm(n_pairs)
        for i, idx in enumerate(query_order):
            query_pos = half + i * 2
            if query_pos + 1 < seq_len:
                sequences[b, query_pos] = keys[idx]
                targets[b, query_pos + 1] = values[idx]

    return sequences, targets


def train_epoch(model, optimizer, n_batches=50, batch_size=64, n_pairs=8, vocab_size=64, seq_len=64):
    model.train()
    total_loss = 0
    total_correct = 0
    total_counted = 0

    for _ in range(n_batches):
        seq, tgt = generate_associative_recall(batch_size, n_pairs, vocab_size, seq_len)
        seq, tgt = seq.to(device), tgt.to(device)

        logits = model(seq)
        loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        mask = tgt != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = (preds == tgt) & mask
            total_correct += correct.sum().item()
            total_counted += mask.sum().item()

    return total_loss / n_batches, total_correct / total_counted if total_counted > 0 else 0


def evaluate(model, n_batches=20, batch_size=64, n_pairs=8, vocab_size=64, seq_len=64):
    model.eval()
    total_correct = 0
    total_counted = 0

    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = generate_associative_recall(batch_size, n_pairs, vocab_size, seq_len)
            seq, tgt = seq.to(device), tgt.to(device)

            logits = model(seq)
            mask = tgt != -100
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1)
                correct = (preds == tgt) & mask
                total_correct += correct.sum().item()
                total_counted += mask.sum().item()

    return total_correct / total_counted if total_counted > 0 else 0


def run_test():
    vocab_size = 64
    dim = 64
    n_epochs = 30

    configs = [
        ('IterRefine-3', {'n_iterations': 3}),
        ('IterRefine-5', {'n_iterations': 5}),
        ('IterRefine-7', {'n_iterations': 7}),
    ]

    print("=" * 70)
    print("ITERATIVE MEMORY: ASSOCIATIVE RECALL TEST")
    print("=" * 70)
    print(f"Task: Store {8} key-value pairs, recall values from shuffled keys")
    print()

    for name, kwargs in configs:
        print(f"\n--- {name} ---")
        model = IterativeRefinementModel(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=2,
            n_phases=32,
            **kwargs
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val = 0
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_epoch(model, optimizer)
            val_acc = evaluate(model)
            best_val = max(best_val, val_acc)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: train={train_acc:.1%}, val={val_acc:.1%}")

        print(f"  Best val: {best_val:.1%}")

    # Also test with more phases
    print("\n" + "=" * 70)
    print("VARYING NUMBER OF PHASES")
    print("=" * 70)

    for n_phases in [16, 32, 64, 128]:
        print(f"\n--- n_phases={n_phases} ---")
        model = IterativeRefinementModel(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=2,
            n_phases=n_phases,
            n_iterations=5
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val = 0
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_epoch(model, optimizer)
            val_acc = evaluate(model)
            best_val = max(best_val, val_acc)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}: train={train_acc:.1%}, val={val_acc:.1%}")

        print(f"  Best val: {best_val:.1%}")


if __name__ == "__main__":
    run_test()
