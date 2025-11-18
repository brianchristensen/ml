"""
Character-Level Language Modeling on text8 (Clean Wikipedia Text)

text8 is 100M characters of cleaned Wikipedia text - no XML markup,
just plain English. Better for learning actual language patterns.

This will train overnight with more epochs on the full dataset.
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
# Download and Process text8
# ============================================================================

def download_text8(data_dir='./data'):
    """Download text8 dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, 'text8.zip')
    txt_path = os.path.join(data_dir, 'text8')

    if os.path.exists(txt_path):
        print(f"text8 already downloaded at {txt_path}")
        return txt_path

    if not os.path.exists(zip_path):
        print("Downloading text8 (31MB)...")
        url = 'http://mattmahoney.net/dc/text8.zip'
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    return txt_path


def load_text8(path, n_chars=None):
    """
    Load text8 dataset.

    Standard splits:
    - Train: first 90M chars
    - Val: next 5M chars
    - Test: final 5M chars

    For full training, use n_chars=None (all 100M).
    """
    with open(path, 'rb') as f:
        data = f.read()

    if n_chars is not None:
        data = data[:n_chars]

    # Standard splits
    n_train = int(len(data) * 0.9)
    n_val = int(len(data) * 0.05)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    print(f"Total: {len(data):,} chars")
    print(f"Train: {len(train_data):,} chars")
    print(f"Val: {len(val_data):,} chars")
    print(f"Test: {len(test_data):,} chars")

    return train_data, val_data, test_data


# ============================================================================
# Character-Level Dataset
# ============================================================================

class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, data, seq_len=256, stride=None):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.n_seqs = (len(data) - seq_len - 1) // self.stride

    def __len__(self):
        return self.n_seqs

    def _byte_to_idx(self, byte_val):
        """Convert text8 byte to index (0-26)."""
        # text8 uses: lowercase a-z (bytes 97-122) + space (byte 32)
        if byte_val == 32:  # space
            return 26
        elif 97 <= byte_val <= 122:  # a-z
            return byte_val - 97  # Map to 0-25
        else:
            # Unknown chars (shouldn't happen in text8, but just in case)
            return 26  # Map to space

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len

        # Convert raw bytes to indices (0-26)
        input_bytes = list(self.data[start:end])
        target_bytes = list(self.data[start+1:end+1])

        input_seq = torch.tensor([self._byte_to_idx(b) for b in input_bytes], dtype=torch.long)
        target_seq = torch.tensor([self._byte_to_idx(b) for b in target_bytes], dtype=torch.long)

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
# Baseline Transformer
# ============================================================================

class SimpleTransformerLM(nn.Module):
    """Simple single-layer transformer for fair comparison."""

    def __init__(self, vocab_size=256, d_model=512, nhead=8,
                 dim_feedforward=2048, max_len=1024):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.output_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, n_epochs):
    """Train one epoch with progress reporting."""
    model.train()
    total_loss = 0.0
    total_chars = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        logits = model(inputs)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        total_chars += inputs.numel()

        if (batch_idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / total_chars
            bpc = avg_loss / math.log(2)
            chars_per_sec = total_chars / elapsed
            print(f"  Epoch {epoch}/{n_epochs} - Batch {batch_idx+1}/{len(dataloader)} - "
                  f"BPC: {bpc:.4f} - {chars_per_sec:.0f} chars/sec")

    avg_loss = total_loss / total_chars
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model and return BPC."""
    model.eval()
    total_loss = 0.0
    total_chars = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            logits = model(inputs)
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
# Main Training
# ============================================================================

def main():
    print("=" * 80)
    print("Character-Level Language Modeling on text8")
    print("Clean Wikipedia Text - Overnight Training")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    vocab_size = 27  # text8 has only lowercase a-z + space
    seq_len = 256
    batch_size = 64  # Larger batch for faster training
    n_epochs = 20  # More epochs for overnight training
    use_full_dataset = True  # Set False for quick testing

    # Download and load data
    print("Loading text8 dataset...")
    text8_path = download_text8()

    if use_full_dataset:
        n_chars = None  # All 100M chars
        print("Using FULL dataset (100M chars) - this will take hours!")
    else:
        n_chars = 10_000_000  # 10M for testing
        print("Using subset (10M chars) for quick testing")

    train_data, val_data, test_data = load_text8(text8_path, n_chars=n_chars)
    print()

    # Create datasets
    train_dataset = CharDataset(train_data, seq_len=seq_len, stride=seq_len)
    val_dataset = CharDataset(val_data, seq_len=seq_len, stride=seq_len)
    test_dataset = CharDataset(test_data, seq_len=seq_len, stride=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    criterion = nn.CrossEntropyLoss()

    # ========================================================================
    # Transformer Baseline (COMMENTED OUT - focusing on novel architecture)
    # ========================================================================

    # print("=" * 80)
    # print("Training Transformer Baseline")
    # print("=" * 80)
    # print()
    #
    # transformer = SimpleTransformerLM(
    #     vocab_size=vocab_size,
    #     d_model=512,
    #     nhead=8,
    #     dim_feedforward=2048,
    #     max_len=seq_len
    # ).to(device)
    #
    # print(f"Parameters: {transformer.count_parameters():,}")
    # print()
    #
    # optimizer_tf = optim.AdamW(transformer.parameters(), lr=1e-3)
    # scheduler_tf = optim.lr_scheduler.CosineAnnealingLR(optimizer_tf, n_epochs)
    #
    # best_val_bpc_tf = float('inf')
    # for epoch in range(1, n_epochs + 1):
    #     train_loss = train_epoch(transformer, train_loader, optimizer_tf,
    #                             criterion, device, epoch, n_epochs)
    #     val_bpc = evaluate(transformer, val_loader, device)
    #     train_bpc = train_loss / math.log(2)
    #
    #     scheduler_tf.step()
    #
    #     print(f"Epoch {epoch}/{n_epochs} - Train BPC: {train_bpc:.4f} - Val BPC: {val_bpc:.4f} - "
    #           f"LR: {optimizer_tf.param_groups[0]['lr']:.2e}")
    #
    #     if val_bpc < best_val_bpc_tf:
    #         best_val_bpc_tf = val_bpc
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': transformer.state_dict(),
    #             'optimizer_state_dict': optimizer_tf.state_dict(),
    #             'best_val_bpc': best_val_bpc_tf,
    #         }, 'transformer_text8.pt')
    #         print(f"  → Saved best (Val BPC: {best_val_bpc_tf:.4f})")
    #     print()
    #
    # test_bpc_tf = evaluate(transformer, test_loader, device)
    # print(f"Transformer Test BPC: {test_bpc_tf:.4f}\n")

    # ========================================================================
    # Trajectory-Based Attention
    # ========================================================================

    print("=" * 80)
    print("Training Trajectory-Based Attention")
    print("=" * 80)
    print()

    novel_model = NovelAttentionLM(
        vocab_size=vocab_size,
        dim=512,
        hidden_dim=512,
        n_layers=1,
        n_heads=4,
        n_neurons=512,
        max_len=seq_len,
        device=device
    ).to(device)

    print(f"Parameters: {novel_model.count_parameters():,}")
    print()

    optimizer_novel = optim.AdamW(novel_model.parameters(), lr=3e-3)
    scheduler_novel = optim.lr_scheduler.CosineAnnealingLR(optimizer_novel, n_epochs)

    best_val_bpc_novel = float('inf')
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(novel_model, train_loader, optimizer_novel,
                                criterion, device, epoch, n_epochs)
        val_bpc = evaluate(novel_model, val_loader, device)
        train_bpc = train_loss / math.log(2)

        scheduler_novel.step()

        print(f"Epoch {epoch}/{n_epochs} - Train BPC: {train_bpc:.4f} - Val BPC: {val_bpc:.4f} - "
              f"LR: {optimizer_novel.param_groups[0]['lr']:.2e}")

        if val_bpc < best_val_bpc_novel:
            best_val_bpc_novel = val_bpc
            torch.save({
                'epoch': epoch,
                'model_state_dict': novel_model.state_dict(),
                'optimizer_state_dict': optimizer_novel.state_dict(),
                'best_val_bpc': best_val_bpc_novel,
            }, 'novel_attention_text8.pt')
            print(f"  → Saved best (Val BPC: {best_val_bpc_novel:.4f})")
        print()

    test_bpc_novel = evaluate(novel_model, test_loader, device)
    print(f"Trajectory-Based Attention Test BPC: {test_bpc_novel:.4f}\n")

    # ========================================================================
    # Final Results
    # ========================================================================

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"{'Model':<30} {'Params':>12} {'Best Val BPC':>14} {'Test BPC':>12}")
    print("-" * 75)
    print(f"{'Trajectory-Based Attention':<30} {novel_model.count_parameters():>12,} "
          f"{best_val_bpc_novel:>14.4f} {test_bpc_novel:>12.4f}")
    print()
    print(f"Target: BPC < 1.5 (competitive with state-of-art on text8)")
    print()
    print("Training complete! Run generate_text8.py to test generation quality.")


if __name__ == "__main__":
    main()
