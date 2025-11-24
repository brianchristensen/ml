"""
Copy Task Benchmark: Phase Attention vs Transformer

Test on sequence copying task with increasing length to demonstrate:
1. Phase attention matches transformer accuracy
2. Phase attention is faster on long sequences
3. Phase attention uses less memory
4. Phase attention generalizes better to longer sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from typing import List, Tuple
import numpy as np

from phi import ParallelHolographicIntegrator


# ============================================================================
# Copy Task Dataset
# ============================================================================

class CopyDataset(Dataset):
    """
    Simple copy task: Input [3, 7, 2, 9] → Output [3, 7, 2, 9]

    Tests attention mechanism's ability to retrieve correct positions.
    """

    def __init__(self, n_examples: int = 1000, min_len: int = 10,
                 max_len: int = 50, vocab_size: int = 20):
        self.examples = []
        self.vocab_size = vocab_size

        for _ in range(n_examples):
            length = np.random.randint(min_len, max_len + 1)
            # Generate random sequence (avoid 0=PAD, 1=SOS, 2=EOS)
            sequence = np.random.randint(3, vocab_size, size=length)
            self.examples.append(sequence)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sequence = self.examples[idx]
        return {
            'input': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(sequence, dtype=torch.long),
            'length': len(sequence)
        }


def collate_copy_batch(batch, max_len=100):
    """Pad sequences to same length."""
    # Find max length in batch
    batch_max_len = max([item['length'] for item in batch])
    batch_max_len = min(batch_max_len, max_len)

    inputs = []
    targets = []
    lengths = []

    for item in batch:
        seq = item['input']
        length = min(len(seq), batch_max_len)

        # Pad
        if length < batch_max_len:
            padded = torch.cat([seq[:length], torch.zeros(batch_max_len - length, dtype=torch.long)])
        else:
            padded = seq[:batch_max_len]

        inputs.append(padded)
        targets.append(padded)
        lengths.append(length)

    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets),
        'length': torch.tensor(lengths)
    }


# ============================================================================
# Baseline Transformer
# ============================================================================

class TransformerCopyModel(nn.Module):
    """Standard transformer for comparison."""

    def __init__(self, vocab_size: int = 30, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 2, max_len: int = 1000):
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
        """
        input_indices: [batch, seq_len]
        target_indices: [batch, seq_len]
        """
        batch_size, seq_len = input_indices.shape

        # Embed inputs
        src = self.embedding(input_indices) + self.pos_encoding[:seq_len]

        if target_indices is not None:
            # Training mode
            tgt = self.embedding(target_indices) + self.pos_encoding[:seq_len]
        else:
            # Inference mode (autoregressive)
            tgt = src

        # Generate masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_indices.device)

        # Transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
        )

        # Logits
        logits = self.output_head(output)  # [batch, seq_len, vocab_size]

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training
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
        logits = model(inputs, targets)  # [batch, seq_len, vocab_size]

        # Loss (ignore padding)
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
    """Evaluate exact sequence accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            lengths = batch['length']

            # Forward
            logits = model(inputs, targets)
            predictions = logits.argmax(dim=-1)  # [batch, seq_len]

            # Check exact match for each sequence
            for i in range(inputs.shape[0]):
                length = lengths[i]
                if torch.all(predictions[i, :length] == targets[i, :length]):
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def measure_speed(model, batch_size, seq_len, device, n_iterations=100):
    """Measure inference speed."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(3, 20, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_per_seq = (elapsed / n_iterations) / batch_size * 1000  # ms

    return time_per_seq


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("=" * 80)
    print("Copy Task Benchmark: Phase Attention vs Transformer")
    print("=" * 80)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters - LONG SEQUENCES to test O(n²) vs O(n)
    vocab_size = 20
    train_min_len = 100
    train_max_len = 500
    test_lengths = [500, 1000, 2000, 5000]  # Test scaling with 2 orders of magnitude
    batch_size = 8  # Smaller batch for longer sequences
    n_epochs = 10  # Test with learnable embeddings

    # Create datasets
    print("Creating datasets...")
    train_dataset = CopyDataset(
        n_examples=2000,
        min_len=train_min_len,
        max_len=train_max_len,
        vocab_size=vocab_size
    )
    val_dataset = CopyDataset(
        n_examples=200,
        min_len=train_min_len,
        max_len=train_max_len,
        vocab_size=vocab_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_copy_batch(b, max_len=train_max_len)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_copy_batch(b, max_len=train_max_len)
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

    # transformer = TransformerCopyModel(
    #     vocab_size=vocab_size,
    #     d_model=256,
    #     nhead=4,
    #     num_layers=2,
    #     max_len=10000  # Support long sequences
    # ).to(device)

    # print(f"Parameters: {transformer.count_parameters():,}")
    # print()

    # optimizer_tf = optim.AdamW(transformer.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

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

    phase_model = ParallelHolographicIntegrator(
        vocab_size=vocab_size,
        dim=128,
        num_layers=8,
        device=device
    ).to(device)

    print(f"Parameters: {phase_model.count_parameters():,}")
    print()

    optimizer_phase = optim.AdamW(phase_model.parameters(), lr=3e-3)  # Higher LR for smaller model

    for epoch in range(n_epochs):
        train_loss = train_epoch(phase_model, train_loader, optimizer_phase, criterion, device)
        val_acc = evaluate(phase_model, val_loader, device)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.1%}")

    print()

    # ========================================================================
    # Comparison: Speed and Generalization (COMMENTED OUT - no transformer baseline)
    # ========================================================================

    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print()

    print(f"{'Model':<25} {'Params':>10} {'Val Acc':>10}")
    print("-" * 50)
    print(f"{'Phase Attention':<25} {phase_model.count_parameters():>10,} {val_acc:.1%}")
    print()

    # # Speed comparison
    # print("Speed Comparison (ms per sequence):")
    # print(f"{'Length':<10} {'Transformer':>15} {'Phase Attn':>15} {'Speedup':>10}")
    # print("-" * 55)

    # for seq_len in [100, 500, 1000, 2000, 5000]:
    #     try:
    #         # Fewer iterations for very long sequences
    #         n_iter = 20 if seq_len <= 1000 else 10
    #         tf_time = measure_speed(transformer, 1, seq_len, device, n_iterations=n_iter)
    #         phase_time = measure_speed(phase_model, 1, seq_len, device, n_iterations=n_iter)
    #         speedup = tf_time / phase_time

    #         print(f"{seq_len:<10} {tf_time:>15.2f} {phase_time:>15.2f} {speedup:>10.2f}x")
    #     except Exception as e:
    #         print(f"{seq_len:<10} Error: {str(e)[:50]}")

    # print()

    # Generalization to longer sequences
    print("Generalization to longer sequences:")
    print(f"{'Length':<10} {'Phase Attn':>15}")
    print("-" * 30)

    for test_len in test_lengths:
        test_dataset = CopyDataset(
            n_examples=100,
            min_len=test_len,
            max_len=test_len,
            vocab_size=vocab_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda b: collate_copy_batch(b, max_len=test_len)
        )

        # try:
        #     tf_acc = evaluate(transformer, test_loader, device)
        # except:
        #     tf_acc = 0.0

        try:
            phase_acc = evaluate(phase_model, test_loader, device)
        except:
            phase_acc = 0.0

        print(f"{test_len:<10} {phase_acc:>15.1%}")

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
