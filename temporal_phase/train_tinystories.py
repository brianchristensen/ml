"""
TinyStories Language Modeling with TPI

Dataset: TinyStories (~5M tokens of simple children's stories)
Tokenization: BPE with GPT-2 tokenizer (~50k vocab)
Context: 256 tokens (stories are short)
Goal: Fast iteration for architecture testing

This dataset is perfect for:
- Quick architecture validation (~30 min training)
- Testing if model can learn coherent narratives
- Clean evaluation (no Wikipedia artifacts)
- Small model friendly (designed for <100M params)

Expected results:
- Good model: Coherent 2-3 sentence stories
- Bad model: Topic drift, incoherence
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

# HuggingFace datasets and tokenizers
from datasets import load_dataset
import tiktoken

from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Tokenization
# ============================================================================

def get_tokenizer():
    """Get BPE tokenizer (same as WikiText-103)."""
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


# ============================================================================
# TinyStories Dataset
# ============================================================================

def load_tinystories(cache_dir='./data/tinystories_cache'):
    """
    Load TinyStories dataset.

    Returns tokenized train/val splits.
    """
    print("Loading TinyStories dataset...")

    # Load from HuggingFace
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)

    # Get tokenizer
    tokenizer = get_tokenizer()

    print(f"Tokenizer vocab size: {tokenizer.n_vocab}")
    print()

    # Tokenize splits
    def tokenize_split(split_name):
        print(f"Tokenizing {split_name} split...")
        split = dataset[split_name]

        # Concatenate all stories with separator
        texts = [story['text'] for story in split if story['text'].strip()]
        text = "\n\n".join(texts)

        # Tokenize
        tokens = tokenizer.encode(text)

        print(f"  {split_name}: {len(tokens):,} tokens ({len(texts):,} stories)")
        return np.array(tokens, dtype=np.int32)

    train_tokens = tokenize_split('train')
    val_tokens = tokenize_split('validation')

    print()
    return train_tokens, val_tokens, tokenizer


# ============================================================================
# Sequence Dataset
# ============================================================================

class TokenDataset(Dataset):
    """
    Dataset for language modeling with fixed-length sequences.
    """

    def __init__(self, tokens, seq_len=256):
        self.tokens = tokens
        self.seq_len = seq_len

        # Number of full sequences
        self.n_seqs = len(tokens) // (seq_len + 1)

        # Truncate to fit full sequences
        self.tokens = tokens[:self.n_seqs * (seq_len + 1)]

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]

        # Input: [0:seq_len], Target: [1:seq_len+1]
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


# ============================================================================
# Training and Evaluation
# ============================================================================

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

        # Forward
        logits = model(inputs)

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

        # Progress (every 50 batches for faster dataset)
        if (batch_idx + 1) % 50 == 0:
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

            # Forward
            logits = model(inputs)

            # Loss
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


def generate_samples(model, tokenizer, prompts, max_length=100, temperature=0.8, top_k=50):
    """Generate text samples from prompts."""
    model.eval()

    generations = []

    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)

            # Generate
            generated = model.generate(
                input_ids,
                max_length=len(tokens) + max_length,
                temperature=temperature,
                top_k=top_k
            )

            # Decode
            generated_tokens = generated[0].cpu().numpy().tolist()
            generated_text = tokenizer.decode(generated_tokens)

            generations.append(generated_text)

    return generations


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("=" * 80)
    print("TinyStories Language Modeling with TPI")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    seq_len = 256          # Shorter context (stories are short)
    batch_size = 32        # Larger batch (shorter sequences = less memory)
    n_epochs = 20          # More epochs (dataset is small)
    learning_rate = 1e-3   # Same as WikiText-103
    warmup_steps = 0       # No warmup

    # Model config
    dim = 256              # Start with same size as WikiText-103 baseline
    num_layers = 8         # Same depth

    print("Hyperparameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Load data
    train_tokens, val_tokens, tokenizer = load_tinystories()
    vocab_size = tokenizer.n_vocab

    # Create datasets
    train_dataset = TokenDataset(train_tokens, seq_len=seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len=seq_len)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
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

    # Cosine annealing
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

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        train_ppl = math.exp(train_loss)

        # Evaluate
        val_ppl = evaluate(model, val_loader, device)

        print()
        print(f"Train PPL: {train_ppl:.2f} - Val PPL: {val_ppl:.2f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
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
            }, 'tinystories_best.pt')
            print(f"  → Saved best model (Val PPL: {val_ppl:.2f})")
            print()

        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("-" * 80)
            print("Generation Samples:")
            print("-" * 80)

            prompts = [
                "Once upon a time",
                "There was a little",
                "One day, a cat",
            ]

            samples = generate_samples(model, tokenizer, prompts, max_length=80, temperature=0.8)

            for prompt, sample in zip(prompts, samples):
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {sample}")

            print()
            print("=" * 80)
            print()

    # Final evaluation
    print("=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    print()

    # Load best model
    checkpoint = torch.load('tinystories_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    val_ppl = evaluate(model, val_loader, device)
    print(f"Best Validation Perplexity: {val_ppl:.2f}")
    print()

    # Final generation samples
    print("-" * 80)
    print("Final Generation Samples:")
    print("-" * 80)

    prompts = [
        "Once upon a time, there was a",
        "One day, a little girl named",
        "The dog was happy because",
        "In the forest, they found",
    ]

    samples = generate_samples(model, tokenizer, prompts, max_length=100, temperature=0.8)

    for prompt, sample in zip(prompts, samples):
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print(sample)

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print()
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print()
    print("Expected results:")
    print("  Good model: ~30-50 PPL, coherent 2-3 sentence stories")
    print("  Your model: See generation samples above")
    print()


if __name__ == "__main__":
    main()
