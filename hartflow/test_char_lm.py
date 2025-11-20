"""
Character-Level Language Modeling: Novel Attention vs Transformer

Dataset: enwik8 (Wikipedia dump, first 100M characters)
Task: Predict next character given previous context
Metric: Bits per character (BPC) - lower is better

Tests whether novel attention can model real language structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import os
import urllib.request
import zipfile

from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Download and Process enwik8
# ============================================================================

def download_enwik8(data_dir='./data'):
    """Download enwik8 dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, 'enwik8.zip')
    txt_path = os.path.join(data_dir, 'enwik8')

    if os.path.exists(txt_path):
        print(f"enwik8 already downloaded at {txt_path}")
        return txt_path

    if not os.path.exists(zip_path):
        print("Downloading enwik8 (35MB)...")
        url = 'http://mattmahoney.net/dc/enwik8.zip'
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    return txt_path


def load_enwik8(path, n_chars=10_000_000):
    """
    Load enwik8 dataset.

    Standard splits:
    - Train: first 90M chars
    - Val: next 5M chars
    - Test: final 5M chars

    For quick testing, use smaller n_chars.
    """
    with open(path, 'rb') as f:
        data = f.read()

    # Use only first n_chars for faster testing
    data = data[:n_chars]

    # Standard splits (proportional)
    n_train = int(len(data) * 0.9)
    n_val = int(len(data) * 0.05)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    return train_data, val_data, test_data


# ============================================================================
# Character-Level Dataset
# ============================================================================

class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, data, seq_len=256, stride=None):
        """
        Args:
            data: Raw bytes
            seq_len: Sequence length for training
            stride: Step size between sequences (default: seq_len for no overlap)
        """
        self.data = data
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Calculate number of sequences
        self.n_seqs = (len(data) - seq_len - 1) // self.stride

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len

        # Input: characters [start:end]
        # Target: characters [start+1:end+1] (next char prediction)
        input_seq = torch.tensor(list(self.data[start:end]), dtype=torch.long)
        target_seq = torch.tensor(list(self.data[start+1:end+1]), dtype=torch.long)

        return {
            'input': input_seq,
            'target': target_seq,
            'length': self.seq_len
        }


def collate_fn(batch):
    """Collate function for dataloaders."""
    return {
        'input': torch.stack([item['input'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'length': torch.tensor([item['length'] for item in batch])
    }


# ============================================================================
# Baseline Single-Layer Transformer
# ============================================================================

class SimpleTransformerLM(nn.Module):
    """Simple single-layer transformer for fair comparison."""

    def __init__(self, vocab_size=256, d_model=512, nhead=8,
                 dim_feedforward=2048, max_len=1024):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))

        # Single transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False  # Post-norm like original transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Output
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] input indices
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = x.shape[1]

        # Embed and add positional encoding
        x = self.embedding(x) + self.pos_encoding[:seq_len]

        # Causal mask (prevent attending to future)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)

        # Output logits
        logits = self.output_head(x)

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
    total_chars = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # Forward
        logits = model(inputs)

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        total_chars += inputs.numel()

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / total_chars
            bpc = avg_loss / math.log(2)
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {avg_loss:.4f} - BPC: {bpc:.4f}")

    avg_loss = total_loss / total_chars
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluate model and return bits per character (BPC).
    BPC = loss / log(2)
    """
    model.eval()
    total_loss = 0.0
    total_chars = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Forward
            logits = model(inputs)

            # Loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_chars += inputs.numel()

    avg_loss = total_loss / total_chars
    bpc = avg_loss / math.log(2)

    return bpc


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("=" * 80)
    print("Character-Level Language Modeling: Novel Attention vs Transformer")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    vocab_size = 256  # All possible bytes
    seq_len = 256     # Context length
    batch_size = 32
    n_epochs = 5     # Train longer
    n_chars = 10_000_000  # Use 10M chars for quick testing (full enwik8 = 100M)

    # Download and load data
    print("Loading enwik8 dataset...")
    enwik8_path = download_enwik8()
    train_data, val_data, test_data = load_enwik8(enwik8_path, n_chars=n_chars)

    print(f"Train: {len(train_data):,} chars")
    print(f"Val: {len(val_data):,} chars")
    print(f"Test: {len(test_data):,} chars")
    print()

    # Create datasets
    train_dataset = CharDataset(train_data, seq_len=seq_len, stride=seq_len)
    val_dataset = CharDataset(val_data, seq_len=seq_len, stride=seq_len)
    test_dataset = CharDataset(test_data, seq_len=seq_len, stride=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Define loss criterion (used by both models)
    criterion = nn.CrossEntropyLoss()

    # # ========================================================================
    # # Baseline: Single-Layer Transformer
    # # ========================================================================

    # print("=" * 80)
    # print("Training Single-Layer Transformer")
    # print("=" * 80)
    # print()

    # transformer = SimpleTransformerLM(
    #     vocab_size=vocab_size,
    #     d_model=512,
    #     nhead=8,
    #     dim_feedforward=2048,
    #     max_len=seq_len
    # ).to(device)

    # print(f"Parameters: {transformer.count_parameters():,}")
    # print()

    # optimizer_tf = optim.AdamW(transformer.parameters(), lr=1e-3)

    # best_val_bpc_tf = float('inf')
    # for epoch in range(n_epochs):
    #     print(f"Epoch {epoch+1}/{n_epochs}")
    #     train_loss = train_epoch(transformer, train_loader, optimizer_tf, criterion, device)
    #     val_bpc = evaluate(transformer, val_loader, device)

    #     train_bpc = train_loss / math.log(2)

    #     print(f"Train BPC: {train_bpc:.4f} - Val BPC: {val_bpc:.4f}")
    #     print()

    #     if val_bpc < best_val_bpc_tf:
    #         best_val_bpc_tf = val_bpc
    #         # Save best model
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': transformer.state_dict(),
    #             'optimizer_state_dict': optimizer_tf.state_dict(),
    #             'best_val_bpc': best_val_bpc_tf,
    #         }, 'transformer_charlm.pt')
    #         print(f"  → Saved best Transformer (Val BPC: {best_val_bpc_tf:.4f})")

    # # Test
    # test_bpc_tf = evaluate(transformer, test_loader, device)
    # print(f"Final Test BPC: {test_bpc_tf:.4f}")
    # print()

    # # Save final transformer model
    # torch.save({
    #     'epoch': n_epochs,
    #     'model_state_dict': transformer.state_dict(),
    #     'optimizer_state_dict': optimizer_tf.state_dict(),
    #     'best_val_bpc': best_val_bpc_tf,
    #     'test_bpc': test_bpc_tf,
    # }, 'transformer_charlm_final.pt')
    # print("Saved final Transformer to transformer_charlm_final.pt")
    # print()

    # ========================================================================
    # Novel Attention Model
    # ========================================================================

    print("=" * 80)
    print("Training Single-Layer Novel Attention")
    print("=" * 80)
    print()

    novel_model = NovelAttentionLM(
        vocab_size=vocab_size,
        dim=128,
        num_layers=20,
        device=device
    ).to(device)

    print(f"Parameters: {novel_model.count_parameters():,}")
    print()

    optimizer_novel = optim.AdamW(novel_model.parameters(), lr=3e-3)

    best_val_bpc_novel = float('inf')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss = train_epoch(novel_model, train_loader, optimizer_novel, criterion, device)
        val_bpc = evaluate(novel_model, val_loader, device)

        train_bpc = train_loss / math.log(2)

        print(f"Train BPC: {train_bpc:.4f} - Val BPC: {val_bpc:.4f}")
        print()

        if val_bpc < best_val_bpc_novel:
            best_val_bpc_novel = val_bpc
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': novel_model.state_dict(),
                'optimizer_state_dict': optimizer_novel.state_dict(),
                'best_val_bpc': best_val_bpc_novel,
            }, 'novel_attention_charlm.pt')
            print(f"  Saved best model (Val BPC: {best_val_bpc_novel:.4f})")

    # Test
    test_bpc_novel = evaluate(novel_model, test_loader, device)
    print(f"Final Test BPC: {test_bpc_novel:.4f}")
    print()

    # Save final model
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': novel_model.state_dict(),
        'optimizer_state_dict': optimizer_novel.state_dict(),
        'best_val_bpc': best_val_bpc_novel,
        'test_bpc': test_bpc_novel,
    }, 'novel_attention_charlm_final.pt')
    print("Saved final model to novel_attention_charlm_final.pt")
    print()

    # ========================================================================
    # Results Summary (COMMENTED OUT - no transformer baseline)
    # ========================================================================

    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()

    print(f"{'Model':<30} {'Parameters':>12} {'Best Val BPC':>14} {'Test BPC':>12}")
    print("-" * 80)
    # print(f"{'Transformer':<30} {transformer.count_parameters():>12,} {best_val_bpc_tf:>14.4f} {test_bpc_tf:>12.4f}")
    print(f"{'Novel Attention':<30} {novel_model.count_parameters():>12,} {best_val_bpc_novel:>14.4f} {test_bpc_novel:>12.4f}")
    print()

    print("Notes:")
    print("- BPC (bits per character): Lower is better")
    print("- Random baseline: 8.0 BPC (uniform over 256 chars)")
    print("- Good model: < 2.0 BPC")
    print("- State-of-art (deep models): ~1.0 BPC")
    print()

    # if test_bpc_novel < test_bpc_tf:
    #     print(f"Novel Attention WINS: {test_bpc_tf - test_bpc_novel:.4f} BPC better")
    # elif test_bpc_novel > test_bpc_tf:
    #     print(f"Transformer WINS: {test_bpc_novel - test_bpc_tf:.4f} BPC better")
    # else:
    #     print("TIE!")

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
