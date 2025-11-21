"""
WikiText-103 Language Modeling with TPI

Dataset: WikiText-103 (word-level Wikipedia)
Tokenization: BPE with ~50k vocab (GPT-2 tokenizer)
Context: 512 tokens (can extend to 1024+)
Metrics: Perplexity + generation quality

This properly tests TPI's strengths:
- Long-range dependencies (512+ token context)
- Semantic content (word/subword level)
- Clean text (no XML markup like enwik8)
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
import tiktoken  # OpenAI's BPE tokenizer (fast, robust)

from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Tokenization
# ============================================================================

def get_tokenizer():
    """
    Get BPE tokenizer.

    Using tiktoken (OpenAI's tokenizer) with GPT-2 encoding:
    - 50,257 vocab size
    - Robust BPE merges
    - Fast encoding/decoding
    """
    # GPT-2 BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


# ============================================================================
# WikiText-103 Dataset
# ============================================================================

def load_wikitext103(cache_dir='./data/wikitext103_cache'):
    """
    Load WikiText-103 dataset.

    Returns tokenized train/val/test splits.
    """
    print("Loading WikiText-103 dataset...")

    # Load from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

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

        # Remove empty lines and extra whitespace
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


# ============================================================================
# Sequence Dataset
# ============================================================================

class TokenDataset(Dataset):
    """
    Dataset for language modeling with fixed-length sequences.

    Creates non-overlapping sequences of length seq_len.
    """

    def __init__(self, tokens, seq_len=512):
        self.tokens = tokens
        self.seq_len = seq_len

        # Number of full sequences
        self.n_seqs = len(tokens) // (seq_len + 1)  # +1 for target

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

        # Progress
        if (batch_idx + 1) % 100 == 0:
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
    """
    Evaluate model and return perplexity.
    PPL = exp(loss)
    """
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
    """
    Generate text samples from prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of text prompts
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        List of generated texts
    """
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
    print("WikiText-103 Language Modeling with TPI")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    seq_len = 512          # Context length (can increase to 1024+)
    batch_size = 16        # Increased for faster training
    n_epochs = 10           # Reduced for quick testing
    learning_rate = 1e-3   # Higher LR, no warmup initially
    warmup_steps = 0       # Disabled warmup to test if model learns

    # Model config
    dim = 256              # Embedding dimension
    num_layers = 8         # Number of TPI blocks

    # Use subset of data for faster iteration
    use_subset = False      # Set to False for full training
    subset_size = 10_000_000  # 10M tokens instead of 100M+

    print("Hyperparameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Load data
    train_tokens, val_tokens, test_tokens, tokenizer = load_wikitext103()
    vocab_size = tokenizer.n_vocab

    # Use subset for faster iteration
    if use_subset:
        print(f"Using subset of data: {subset_size:,} tokens")
        train_tokens = train_tokens[:subset_size]
        print(f"Train tokens (subset): {len(train_tokens):,}")
        print()

    # Create datasets
    train_dataset = TokenDataset(train_tokens, seq_len=seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len=seq_len)
    test_dataset = TokenDataset(test_tokens, seq_len=seq_len)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    print(f"Test sequences: {len(test_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows, 2-4 for Linux
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
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

    # Learning rate scheduler
    if warmup_steps > 0:
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (len(train_loader) * n_epochs - warmup_steps)
                return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # No warmup, just cosine decay
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
            }, 'wikitext103_best.pt')
            print(f"  → Saved best model (Val PPL: {val_ppl:.2f})")
            print()

        # Generate samples every epoch
        if (epoch + 1) % 1 == 0:
            print("-" * 80)
            print("Generation Samples:")
            print("-" * 80)

            prompts = [
                "The history of",
                "In computer science,",
                "The theory of",
            ]

            samples = generate_samples(model, tokenizer, prompts, max_length=50, temperature=0.8)

            for prompt, sample in zip(prompts, samples):
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {sample}")

            print()
            print("=" * 80)
            print()

    # Final test evaluation
    print("=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)
    print()

    # Load best model
    checkpoint = torch.load('wikitext103_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ppl = evaluate(model, test_loader, device)
    print(f"Test Perplexity: {test_ppl:.2f}")
    print()

    # Final generation samples
    print("-" * 80)
    print("Final Generation Samples (longer):")
    print("-" * 80)

    prompts = [
        "The",
        "In the year 2024,",
        "Scientists have discovered",
        "The president announced",
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
    print(f"Test perplexity: {test_ppl:.2f}")
    print()


if __name__ == "__main__":
    main()
