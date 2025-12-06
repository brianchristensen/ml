"""
Test Unified Product Phasor on Next-Token Prediction with STRUCTURED data.

Random sequences have no learnable pattern for next-token prediction.
Let's test on sequences WITH patterns:
1. Repeating patterns: [a,b,c,d,a,b,c,d,...]
2. Counting mod N: [0,1,2,3,...,N-1,0,1,2,...]
3. Delayed copy: token[i] = token[i-delay]
"""

import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from unified_product_phasor import UnifiedProductModel


def generate_repeating_pattern(batch_size, seq_len, pattern_len, vocab_size, device='cuda'):
    """Generate repeating patterns: [a,b,c,d,a,b,c,d,...]"""
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Random pattern for each batch
        pattern = torch.randint(0, vocab_size, (pattern_len,))
        for i in range(seq_len):
            data[b, i] = pattern[i % pattern_len]

    # For next-token prediction: input[:-1] predicts target[1:]
    return data


def generate_counting(batch_size, seq_len, mod_n, device='cuda'):
    """Generate counting sequences: [0,1,2,...,mod_n-1,0,1,2,...]"""
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        start = np.random.randint(0, mod_n)  # Random start
        for i in range(seq_len):
            data[b, i] = (start + i) % mod_n

    return data


def generate_delayed_copy(batch_size, seq_len, delay, vocab_size, device='cuda'):
    """Generate delayed copy: token[i] = token[i-delay] for i >= delay"""
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        # First 'delay' tokens are random
        for i in range(delay):
            data[b, i] = np.random.randint(0, vocab_size)
        # Rest copy from 'delay' positions back
        for i in range(delay, seq_len):
            data[b, i] = data[b, i - delay]

    return data


def test_next_token_prediction(model, data_generator, n_batches=10, batch_size=100, **gen_kwargs):
    """Test accuracy on next-token prediction."""
    model.eval()
    vocab_size = model.vocab_size

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            data = data_generator(batch_size, **gen_kwargs)

            # Input: all but last, Target: all but first
            inputs = data[:, :-1]
            targets = data[:, 1:]

            logits = model(inputs)
            preds = logits[:, :, :vocab_size].argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += targets.numel()

    return correct / total * 100


def train_on_task(model, data_generator, epochs=500, batch_size=32, seq_len=32, **gen_kwargs):
    """Train model on a specific task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    vocab_size = model.vocab_size

    for epoch in range(epochs):
        model.train()

        data = data_generator(batch_size, seq_len=seq_len, device=device, **gen_kwargs)

        inputs = data[:, :-1]
        targets = data[:, 1:]

        logits = model(inputs)
        loss = criterion(
            logits[:, :, :vocab_size].reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            acc = test_next_token_prediction(
                model, data_generator, n_batches=5, batch_size=100,
                seq_len=seq_len, device=device, **gen_kwargs
            )
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.1f}%")

    return test_next_token_prediction(
        model, data_generator, n_batches=10, batch_size=100,
        seq_len=seq_len, device=device, **gen_kwargs
    )


def main():
    print("=" * 70)
    print("UNIFIED PRODUCT MODEL - Next Token Prediction on STRUCTURED Data")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print()

    vocab_size = 16
    dim = 128
    n_layers = 4
    seq_len = 32

    # Task 1: Repeating patterns
    print("-" * 70)
    print("Task 1: Repeating Patterns (period=4)")
    print("  Pattern: [a,b,c,d,a,b,c,d,...] -> predict next in pattern")
    print("-" * 70)

    model1 = UnifiedProductModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_oscillators=64,
        max_len=64
    ).to(device)

    acc1 = train_on_task(
        model1, generate_repeating_pattern,
        epochs=500, batch_size=32, seq_len=seq_len,
        pattern_len=4, vocab_size=vocab_size
    )
    print(f"\nFinal Accuracy: {acc1:.1f}%")
    print()

    # Task 2: Counting mod N
    print("-" * 70)
    print("Task 2: Counting mod 10")
    print("  Sequence: [0,1,2,...,9,0,1,2,...] -> predict next number")
    print("-" * 70)

    model2 = UnifiedProductModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_oscillators=64,
        max_len=64
    ).to(device)

    acc2 = train_on_task(
        model2, generate_counting,
        epochs=500, batch_size=32, seq_len=seq_len,
        mod_n=10
    )
    print(f"\nFinal Accuracy: {acc2:.1f}%")
    print()

    # Task 3: Delayed copy
    print("-" * 70)
    print("Task 3: Delayed Copy (delay=3)")
    print("  Sequence: token[i] = token[i-3] for i >= 3")
    print("-" * 70)

    model3 = UnifiedProductModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_oscillators=64,
        max_len=64
    ).to(device)

    acc3 = train_on_task(
        model3, generate_delayed_copy,
        epochs=500, batch_size=32, seq_len=seq_len,
        delay=3, vocab_size=vocab_size
    )
    print(f"\nFinal Accuracy: {acc3:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY: Unified Product Model on Next-Token Prediction")
    print("=" * 70)
    print()
    print(f"Task 1 - Repeating Patterns (period=4): {acc1:.1f}%")
    print(f"Task 2 - Counting mod 10:               {acc2:.1f}%")
    print(f"Task 3 - Delayed Copy (delay=3):        {acc3:.1f}%")
    print()
    print(f"Random baseline: {100/vocab_size:.1f}%")
    print()

    if acc1 > 90 and acc2 > 90 and acc3 > 90:
        print("SUCCESS: Model learns structured next-token prediction!")
    elif acc1 > 50 or acc2 > 50 or acc3 > 50:
        print("PARTIAL: Model learns some structured patterns")
    else:
        print("FAILED: Model struggles with next-token prediction")


if __name__ == "__main__":
    main()
