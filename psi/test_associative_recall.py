"""
Associative Recall Benchmark: Phase Attention vs Transformer

Task: Store key-value pairs, then recall values by key.

Example:
Input:  [k1, v1, k2, v2, k3, v3, QUERY, k2, QUERY, k1]
Target: [0,  0,  0,  0,  0,  0,  0,     v2, 0,     v1]

Tests content-based attention - model must learn to:
1. Bind keys to their values during encoding
2. Retrieve correct value when key is queried
3. Ignore irrelevant key-value pairs

This is what transformers excel at! Can phase attention match it?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

from phase_binding_memory import PhaseBindingRecall


# Special tokens
PAD = 0
SOS = 1
EOS = 2
QUERY = 3
FIRST_TOKEN = 4  # Keys and values start from here

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Associative Recall Dataset
# ============================================================================

class AssociativeRecallDataset(Dataset):
    """
    Generate sequences with key-value pairs followed by queries.

    Format:
    [k1, v1, k2, v2, ..., kN, vN, QUERY, ki, QUERY, kj, ...]

    Target has value at position after QUERY token, 0 elsewhere.
    """

    def __init__(self, n_examples: int = 1000, n_pairs: int = 10,
                 n_queries: int = 3, vocab_size: int = 50):
        self.examples = []
        self.vocab_size = vocab_size
        self.n_pairs = n_pairs
        self.n_queries = n_queries

        # Available tokens for keys and values
        self.available_tokens = list(range(FIRST_TOKEN, vocab_size))

        for _ in range(n_examples):
            # Sample unique keys
            keys = np.random.choice(
                self.available_tokens,
                size=n_pairs,
                replace=False
            )

            # Sample values (can repeat)
            values = np.random.choice(
                self.available_tokens,
                size=n_pairs,
                replace=True
            )

            # Build input: interleave keys and values
            input_seq = []
            kv_dict = {}
            for k, v in zip(keys, values):
                input_seq.append(k)
                input_seq.append(v)
                kv_dict[k] = v

            # Add queries: randomly sample from keys
            target_seq = [PAD] * len(input_seq)  # No supervision for key-value pairs

            for _ in range(n_queries):
                query_key = np.random.choice(keys)
                input_seq.append(QUERY)
                input_seq.append(query_key)

                # Target: value should appear after QUERY token
                target_seq.append(PAD)  # No supervision for QUERY itself
                target_seq.append(kv_dict[query_key])  # Supervise the position after QUERY

            self.examples.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target_seq, dtype=torch.long),
                'length': len(input_seq)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad sequences to same length."""
    max_len = max([item['length'] for item in batch])

    inputs = []
    targets = []
    lengths = []

    for item in batch:
        inp = item['input']
        tgt = item['target']
        length = item['length']

        # Pad to max_len
        if length < max_len:
            inp = torch.cat([inp, torch.zeros(max_len - length, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(max_len - length, dtype=torch.long)])

        inputs.append(inp)
        targets.append(tgt)
        lengths.append(length)

    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets),
        'length': torch.tensor(lengths)
    }


# ============================================================================
# Baseline Transformer
# ============================================================================

class TransformerRecallModel(nn.Module):
    """Standard transformer for associative recall."""

    def __init__(self, vocab_size: int = 50, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 2, max_len: int = 500):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            batch_first=True
        )

        # Output
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_indices, target_indices=None):
        batch_size, seq_len = input_indices.shape

        # Embed inputs
        src = self.embedding(input_indices) + self.pos_encoding[:seq_len]

        if target_indices is not None:
            # Training mode - use target as decoder input
            tgt = self.embedding(target_indices) + self.pos_encoding[:seq_len]
        else:
            # Inference mode
            tgt = src

        # Generate causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_indices.device)

        # Forward
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        logits = self.output_head(output)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # Forward
        logits = model(inputs, targets)

        # Loss - only on non-PAD positions
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, device):
    """
    Evaluate recall accuracy: correct if predicted value matches target
    at positions immediately after QUERY tokens.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Forward WITHOUT teacher forcing during eval!
            logits = model(inputs)  # Don't pass targets!
            predictions = logits.argmax(dim=-1)

            # Check accuracy only at query positions (non-PAD targets)
            mask = (targets != PAD)
            if mask.sum() > 0:
                correct += (predictions[mask] == targets[mask]).sum().item()
                total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def measure_speed(model, batch_size, seq_len, n_pairs, n_queries,
                  vocab_size, device, n_iterations=50):
    """Measure inference speed."""
    model.eval()

    # Create dummy input matching the task format
    dummy_input = []
    for _ in range(batch_size):
        # Key-value pairs
        seq = []
        for _ in range(n_pairs):
            seq.append(np.random.randint(FIRST_TOKEN, vocab_size))  # key
            seq.append(np.random.randint(FIRST_TOKEN, vocab_size))  # value

        # Queries
        for _ in range(n_queries):
            seq.append(QUERY)
            seq.append(np.random.randint(FIRST_TOKEN, vocab_size))  # query key

        # Pad to seq_len
        while len(seq) < seq_len:
            seq.append(PAD)
        seq = seq[:seq_len]

        dummy_input.append(seq)

    dummy_input = torch.tensor(dummy_input, dtype=torch.long, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    time_per_seq = (elapsed / n_iterations) / batch_size * 1000  # ms

    return time_per_seq


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("=" * 80)
    print("Associative Recall Benchmark: Phase Attention vs Transformer")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    vocab_size = 50
    n_pairs = 20  # Number of key-value pairs to store
    n_queries = 5  # Number of queries per sequence
    batch_size = 32
    n_epochs = 60  # Train even longer for this challenging task

    seq_len = n_pairs * 2 + n_queries * 2  # keys + values + queries

    print(f"Task: Store {n_pairs} key-value pairs, recall {n_queries} values")
    print(f"Sequence length: ~{seq_len} tokens")
    print(f"Vocabulary size: {vocab_size}")
    print()

    # Create datasets
    print("Creating datasets...")
    train_dataset = AssociativeRecallDataset(
        n_examples=2000,
        n_pairs=n_pairs,
        n_queries=n_queries,
        vocab_size=vocab_size
    )
    val_dataset = AssociativeRecallDataset(
        n_examples=200,
        n_pairs=n_pairs,
        n_queries=n_queries,
        vocab_size=vocab_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print()

    # ========================================================================
    # Baseline: Transformer (COMMENTED OUT - we know the baseline)
    # ========================================================================

    # print("=" * 80)
    # print("Training Transformer Baseline")
    # print("=" * 80)
    # print()

    # transformer = TransformerRecallModel(
    #     vocab_size=vocab_size,
    #     d_model=256,
    #     nhead=4,
    #     num_layers=2,
    #     max_len=1000
    # ).to(device)

    # print(f"Parameters: {transformer.count_parameters():,}")
    # print()

    # optimizer_tf = optim.AdamW(transformer.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # for epoch in range(n_epochs):
    #     train_loss = train_epoch(transformer, train_loader, optimizer_tf, criterion, device)
    #     val_acc = evaluate(transformer, val_loader, device)
    #     print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.1%}")

    # print()

    # ========================================================================
    # Phase Attention Model
    # ========================================================================

    print("=" * 80)
    print("Training Phase Attention Model")
    print("=" * 80)
    print()

    phase_model = PhaseBindingRecall(
        vocab_size=vocab_size,
        dim=128,
        n_oscillators=64,
        query_token=QUERY  # QUERY=3 in this benchmark
    ).to(device)

    print(f"Parameters: {phase_model.count_parameters():,}")
    print()

    optimizer_phase = optim.AdamW(phase_model.parameters(), lr=3e-3)  # High LR for fast learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_phase, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(n_epochs):
        train_loss = train_epoch(phase_model, train_loader, optimizer_phase, criterion, device)
        val_acc = evaluate(phase_model, val_loader, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.1%} (Best: {best_val_acc:.1%})")

        # Early stopping if no improvement for 10 epochs
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print()

    # ========================================================================
    # Performance Comparison (COMMENTED OUT - no transformer baseline)
    # ========================================================================

    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print()

    # Final accuracy
    phase_acc = evaluate(phase_model, val_loader, device)

    print(f"{'Model':<25} {'Params':>10} {'Val Acc':>10}")
    print("-" * 50)
    print(f"{'Phase Attention':<25} {phase_model.count_parameters():>10,} {phase_acc:>9.1%}")
    print()

    # # Speed comparison
    # print("Speed Comparison (ms per sequence):")
    # print(f"{'Pairs':<10} {'Queries':<10} {'Transformer':>15} {'Phase Attn':>15} {'Speedup':>10}")
    # print("-" * 65)

    # test_configs = [
    #     (10, 3),
    #     (20, 5),
    #     (50, 10),
    #     (100, 20),
    # ]

    # for n_p, n_q in test_configs:
    #     seq_l = n_p * 2 + n_q * 2
    #     try:
    #         tf_time = measure_speed(transformer, 1, seq_l, n_p, n_q, vocab_size, device, n_iterations=20)
    #         phase_time = measure_speed(phase_model, 1, seq_l, n_p, n_q, vocab_size, device, n_iterations=20)
    #         speedup = tf_time / phase_time

    #         print(f"{n_p:<10} {n_q:<10} {tf_time:>15.2f} {phase_time:>15.2f} {speedup:>10.2f}x")
    #     except Exception as e:
    #         print(f"{n_p:<10} {n_q:<10} Error: {str(e)[:40]}")

    # print()

    # Test generalization to more pairs
    print("Generalization to more key-value pairs:")
    print(f"{'Pairs':<10} {'Queries':<10} {'Phase Attn':>15}")
    print("-" * 40)

    max_pairs = vocab_size - FIRST_TOKEN  # Can't have more unique keys than available tokens
    for n_p in [30, 40]:
        if n_p > max_pairs:
            continue
        n_q = max(5, n_p // 10)
        test_dataset = AssociativeRecallDataset(
            n_examples=100,
            n_pairs=n_p,
            n_queries=n_q,
            vocab_size=vocab_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn
        )

        # try:
        #     tf_acc = evaluate(transformer, test_loader, device)
        # except:
        #     tf_acc = 0.0

        try:
            phase_acc = evaluate(phase_model, test_loader, device)
        except:
            phase_acc = 0.0

        print(f"{n_p:<10} {n_q:<10} {phase_acc:>15.1%}")

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
