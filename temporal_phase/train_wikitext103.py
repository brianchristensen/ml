"""
WikiText-103 Language Modeling with TPI - CHARACTER LEVEL

Dataset: WikiText-103 (~103M BPE tokens → ~400M+ characters!)
Tokenization: CHARACTER LEVEL (vocab=256)
Context: 512 characters
Metrics: Bits-per-character + generation quality

THE BIG TEST: Can TPI scale character-level to 400M+ characters?

This tests TPI's unique strength:
- Parallelizable (unlike RNNs)
- Character-level efficient (O(n), unlike transformers' O(n²))
- Sequential composition learning (characters → words → semantics)
- 99.5% of params in computation (not embeddings)

Expected advantages:
- No embedding overhead (256 vocab vs 50K)
- Better compositional generalization
- More parameters for actual learning
- Handles rare words via spelling

WikiText-2 character-level success (12L/256D, 12.7M params):
- Train: 3.16 PPL, Val: 3.62 PPL (1.66 BPC)
- Learned proper spelling, grammar, semantics in 20 epochs
- No train/val gap

Now scaling to 50x more data!
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

from phi import ParallelHolographicIntegrator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Character-Level Tokenization
# ============================================================================

class CharacterTokenizer:
    """
    Simple character-level tokenizer with vocab size 256.
    Uses raw bytes (UTF-8 encoding).
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        """Convert text to list of byte values (0-255)."""
        return list(text.encode('utf-8'))

    def decode(self, tokens):
        """Convert list of byte values back to text."""
        return bytes(tokens).decode('utf-8', errors='replace')

    @property
    def n_vocab(self):
        return self.vocab_size


def get_tokenizer():
    """Get character-level tokenizer."""
    return CharacterTokenizer()


# ============================================================================
# WikiText-103 Dataset (Character-Level)
# ============================================================================

def load_wikitext103(cache_dir='./data/wikitext103_cache'):
    """
    Load WikiText-103 dataset with character-level tokenization.

    Returns tokenized train/val/test splits.
    """
    print("Loading WikiText-103 dataset (CHARACTER LEVEL)...")

    # Load from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

    # Get tokenizer
    tokenizer = get_tokenizer()

    print(f"Tokenizer vocab size: {tokenizer.n_vocab} (character-level)")
    print()

    # Tokenize all splits
    def tokenize_split(split_name):
        print(f"Tokenizing {split_name} split...")
        split = dataset[split_name]

        # Concatenate all text
        text = "\n\n".join(split['text'])

        # Remove empty lines and extra whitespace
        text = "\n".join(line for line in text.split("\n") if line.strip())

        # Tokenize (character-level = bytes)
        tokens = tokenizer.encode(text)

        print(f"  {split_name}: {len(tokens):,} characters")
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
    print("WikiText-103 Language Modeling with TPI - CHARACTER LEVEL")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters (CHARACTER-LEVEL WITH CURRICULUM LEARNING)
    batch_size = 16        # Same batch size across all stages
    learning_rate = 1e-3   # Same learning rate
    warmup_steps = 0

    # CURRICULUM LEARNING: Gradually increase sequence length
    # Start short (learn basic patterns fast) → extend long (learn coherence)
    # Each stage builds on the previous one's learned representations
    curriculum_stages = [
        {'name': 'Stage 1: Characters→Words', 'seq_len': 128, 'epochs': 5},
        {'name': 'Stage 2: Words→Phrases', 'seq_len': 256, 'epochs': 5},
        {'name': 'Stage 3: Phrases→Sentences', 'seq_len': 512, 'epochs': 5},
        {'name': 'Stage 4: Sentences→Paragraphs', 'seq_len': 1024, 'epochs': 10},
    ]

    # Model config (CHARACTER-LEVEL needs more depth!)
    # Character→word composition requires learning lower-level patterns
    # TPI's cheap layers make deeper networks affordable
    # WikiText-2 success: 12L/256D learned proper language
    dim = 256              # Moderate width
    num_layers = 12        # More depth for character-level (vs 8 for BPE)

    # Use subset of data for faster iteration
    # Full WikiText-103 is ~400M+ characters (huge!)
    use_subset = True      # Set to False for full training
    subset_size = 40_000_000  # 10M characters for initial test

    print("=" * 80)
    print("CURRICULUM LEARNING SCHEDULE:")
    print("=" * 80)
    total_epochs = sum(stage['epochs'] for stage in curriculum_stages)
    for i, stage in enumerate(curriculum_stages, 1):
        print(f"{i}. {stage['name']}")
        print(f"   - Sequence length: {stage['seq_len']} characters")
        print(f"   - Epochs: {stage['epochs']}")
    print(f"\nTotal epochs: {total_epochs}")
    print("=" * 80)
    print()

    print("Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D (CHAR + TIED EMBEDDINGS)")
    print()

    # Load data
    train_tokens, val_tokens, test_tokens, tokenizer = load_wikitext103()
    vocab_size = tokenizer.n_vocab

    # Use subset for faster iteration
    if use_subset:
        print(f"Using subset of data: {subset_size:,} characters")
        train_tokens = train_tokens[:subset_size]
        print(f"Train characters (subset): {len(train_tokens):,}")
        print()

    # Create model with max_len = longest sequence in curriculum
    max_seq_len = max(stage['seq_len'] for stage in curriculum_stages)
    print("Creating model...")
    model = ParallelHolographicIntegrator(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        max_len=max_seq_len,
        device=device
    ).to(device)

    print(f"Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    print()

    # Optimizer (reused across curriculum stages)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    # Curriculum training loop
    print("=" * 80)
    print("CURRICULUM TRAINING")
    print("=" * 80)
    print()

    best_val_ppl = float('inf')
    total_epoch_counter = 0

    for stage_idx, stage in enumerate(curriculum_stages, 1):
        stage_name = stage['name']
        seq_len = stage['seq_len']
        n_epochs = stage['epochs']

        print("=" * 80)
        print(f"{stage_name}")
        print(f"Sequence length: {seq_len}, Epochs: {n_epochs}")
        print("=" * 80)
        print()

        # Create datasets for this stage
        train_dataset = TokenDataset(train_tokens, seq_len=seq_len)
        val_dataset = TokenDataset(val_tokens, seq_len=seq_len)

        print(f"Train sequences: {len(train_dataset):,}")
        print(f"Val sequences: {len(val_dataset):,}")
        print()

        # Create dataloaders for this stage
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

        # Learning rate scheduler for this stage
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * n_epochs, eta_min=learning_rate * 0.1
        )

        # Training loop for this stage
        for epoch in range(n_epochs):
            total_epoch_counter += 1
            print(f"Stage {stage_idx}/{len(curriculum_stages)}, Epoch {epoch+1}/{n_epochs} (Total: {total_epoch_counter}/{total_epochs})")
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
                    'total_epoch': total_epoch_counter,
                    'stage': stage_idx,
                    'stage_name': stage_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ppl': val_ppl,
                    'config': {
                        'vocab_size': vocab_size,
                        'dim': dim,
                        'num_layers': num_layers,
                        'seq_len': seq_len,
                        'max_len': max_seq_len
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

        print()
        print(f"Stage {stage_idx} complete!")
        print(f"Best validation PPL so far: {best_val_ppl:.2f}")
        print()
        print("=" * 80)
        print()

    # Final test evaluation
    print("=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)
    print()

    # Save final model state (after all curriculum stages complete)
    torch.save({
        'total_epoch': total_epoch_counter,
        'stage': len(curriculum_stages),
        'stage_name': 'Final (all stages complete)',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'dim': dim,
            'num_layers': num_layers,
            'max_len': max_seq_len
        }
    }, 'wikitext103_final.pt')

    # Test with FINAL model (not "best" which might be from early stage!)
    print("Testing FINAL model (after all curriculum stages)...")
    final_seq_len = curriculum_stages[-1]['seq_len']

    # DIAGNOSTIC: Print data sizes
    print(f"Data sizes:")
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")
    print(f"  Test tokens: {len(test_tokens):,}")
    print()

    test_dataset = TokenDataset(test_tokens, seq_len=final_seq_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Test sequences ({final_seq_len} seq_len): {len(test_dataset):,}")
    print()

    # DIAGNOSTIC: Also evaluate on validation set with test code
    print("=" * 80)
    print("DIAGNOSTIC: Re-evaluating validation set with test code...")
    print("(Should match final stage validation PPL ~3.30 if code is correct)")
    print("=" * 80)
    val_dataset_test = TokenDataset(val_tokens, seq_len=final_seq_len)
    val_loader_test = DataLoader(
        val_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_ppl_test = evaluate(model, val_loader_test, device)
    print(f"Validation PPL (re-evaluated): {val_ppl_test:.2f}")
    print()

    test_ppl_final = evaluate(model, test_loader, device)
    print(f"Test Perplexity (FINAL model): {test_ppl_final:.2f}")
    print()

    if abs(val_ppl_test - 3.30) > 0.5:
        print("⚠️  WARNING: Val PPL re-evaluation doesn't match! Possible evaluation bug.")
    if test_ppl_final > 10.0:
        print("⚠️  WARNING: Test PPL is very high! Possible data distribution issue.")
    print()

    # Also test "best" model for comparison
    print("-" * 80)
    print("Testing BEST validation model (for comparison)...")
    checkpoint = torch.load('wikitext103_best.pt')
    print(f"  Best model from: {checkpoint.get('stage_name', 'unknown')}")
    print(f"  Best model seq_len: {checkpoint['config'].get('seq_len', 'unknown')}")
    print(f"  Best validation PPL: {best_val_ppl:.2f}")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ppl_best = evaluate(model, test_loader, device)
    print(f"Test Perplexity (BEST model on {final_seq_len} seq_len): {test_ppl_best:.2f}")
    print()

    print("=" * 80)
    print("IMPORTANT: If test PPL is much worse for 'BEST' model,")
    print("it means best model was from early curriculum stage!")
    print("Always use FINAL model for deployment.")
    print("=" * 80)
    print()

    # Final generation samples (using FINAL model, which is currently loaded)
    # Note: Currently loaded model is "best" from above, reload final
    final_checkpoint = torch.load('wikitext103_final.pt')
    model.load_state_dict(final_checkpoint['model_state_dict'])

    print("-" * 80)
    print("Final Generation Samples (using FINAL model after all stages):")
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
    print("CURRICULUM TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Best validation PPL (across all stages): {best_val_ppl:.2f}")
    print(f"Final model test PPL: {test_ppl_final:.2f}")
    print(f"Best model test PPL: {test_ppl_best:.2f}")
    print()
    print("=" * 80)
    print("Use 'wikitext103_final.pt' for generation/deployment")
    print("(not 'wikitext103_best.pt' which may be from early stage)")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
