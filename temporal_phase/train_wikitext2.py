"""
WikiText-2 Language Modeling with TPI

Dataset: WikiText-2 (~2M tokens - 50x smaller than WikiText-103!)
Tokenization: BPE with GPT-2 tokenizer
Context: 512 tokens
Goal: FAST iteration for architecture testing

WikiText-2 is perfect for:
- Ultra-fast iteration (~6 min per epoch)
- Same format as WikiText-103 (easy comparison)
- Testing architectural changes quickly
- Then validate on WikiText-103 when ready

Expected training time: ~2 hours (20 epochs × ~6 min)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import os
from pathlib import Path

from datasets import load_dataset
import tiktoken

from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tokenizer():
    """Get BPE tokenizer."""
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


def load_wikitext2(cache_dir='./data/wikitext2_cache'):
    """Load WikiText-2 dataset (~2M tokens)."""
    print("Loading WikiText-2 dataset...")

    # Load from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)

    # Get tokenizer
    tokenizer = get_tokenizer()

    print(f"Tokenizer vocab size: {tokenizer.n_vocab}")
    print()

    # Tokenize all splits
    def tokenize_split(split_name):
        print(f"Tokenizing {split_name} split...")
        split = dataset[split_name]

        # Concatenate all text
        text = "\n\n".join(split['text'])

        # Remove empty lines
        text = "\n".join(line for line in text.split("\n") if line.strip())

        # Tokenize
        tokens = tokenizer.encode(text)

        print(f"  {split_name}: {len(tokens):,} tokens")
        return np.array(tokens, dtype=np.int32)

    train_tokens = tokenize_split('train')
    val_tokens = tokenize_split('validation')
    test_tokens = tokenize_split('test')

    print()
    return train_tokens, val_tokens, test_tokens, tokenizer


class TokenDataset(Dataset):
    """Dataset for language modeling with fixed-length sequences."""

    def __init__(self, tokens, seq_len=512):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = len(tokens) // (seq_len + 1)
        self.tokens = tokens[:self.n_seqs * (seq_len + 1)]

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]

        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)

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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

        # Progress every 20 batches (dataset is small)
        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed

            print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                  f"Loss: {avg_loss:.4f} - PPL: {ppl:.2f} - "
                  f"Speed: {tokens_per_sec:.0f} tok/s")

    avg_loss = total_loss / total_tokens
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model and return perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

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
            total_tokens += inputs.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def main():
    print("=" * 80)
    print("WikiText-2 Language Modeling with TPI (FAST ITERATION)")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    seq_len = 512
    batch_size = 16
    n_epochs = 3
    learning_rate = 1e-3
    warmup_steps = 0

    # Model config
    dim = 256
    num_layers = 8

    print("Hyperparameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Load data
    train_tokens, val_tokens, test_tokens, tokenizer = load_wikitext2()
    vocab_size = tokenizer.n_vocab

    # Create datasets
    train_dataset = TokenDataset(train_tokens, seq_len=seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len=seq_len)
    test_dataset = TokenDataset(test_tokens, seq_len=seq_len)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print()
    print(f"Expected time per epoch: ~6 minutes")
    print(f"Expected total time: ~{n_epochs * 6} minutes ({n_epochs * 6 / 60:.1f} hours)")
    print()

    # Create model
    print("Creating model...")
    model = NovelAttentionLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        max_len=seq_len,
        device=device
    ).to(device)

    print(f"Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    print()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * n_epochs, eta_min=learning_rate * 0.1
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    best_val_ppl = float('inf')
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()

        print(f"Epoch {epoch+1}/{n_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        train_ppl = math.exp(train_loss)

        # Evaluate
        val_ppl = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = n_epochs - (epoch + 1)
        est_remaining_time = avg_epoch_time * remaining_epochs

        print()
        print(f"Train PPL: {train_ppl:.2f} - Val PPL: {val_ppl:.2f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Epoch time: {epoch_time/60:.1f} min - Est. remaining: {est_remaining_time/60:.1f} min")
        print()

        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'config': {
                    'vocab_size': vocab_size,
                    'dim': dim,
                    'num_layers': num_layers,
                    'seq_len': seq_len
                }
            }, 'wikitext2_best.pt')
            print(f"  → Saved best model (Val PPL: {val_ppl:.2f})")
            print()

    # Final test
    print("=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)
    print()

    checkpoint = torch.load('wikitext2_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ppl = evaluate(model, test_loader, device)
    print(f"Test Perplexity: {test_ppl:.2f}")
    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Test perplexity: {test_ppl:.2f}")
    print()
    print("Next steps:")
    print("  - Test generation: python generate_wikitext103.py --checkpoint wikitext2_best.pt")
    print("  - If results good, scale to WikiText-103")
    print()


if __name__ == "__main__":
    main()
