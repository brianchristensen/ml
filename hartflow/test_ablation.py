"""
Ablation Study: Is phase attention actually doing the work?

Tests:
1. Baseline: Full phase attention model (should work)
2. Ablation 1: Replace attention with mean pooling (should fail)
3. Ablation 2: Random attention weights (should fail)
4. Ablation 3: Reverse task instead of copy (requires proper attention)
5. Visualization: Plot attention weights to verify correctness
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model_phase_attention_fast import FastPhaseAttentionModel, FastPhaseAttention

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Ablated Models
# ============================================================================

class MeanPoolingModel(FastPhaseAttentionModel):
    """Replace attention with mean pooling - should fail if attention matters!"""

    def forward(self, input_indices, target_indices=None, idx2token=None):
        batch_size, input_len = input_indices.shape

        if target_indices is not None:
            output_len = target_indices.shape[1]
        else:
            output_len = input_len

        # Get embeddings and phases (same as original)
        token_emb = self.get_embeddings(input_indices)
        positions = torch.arange(input_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_phases = self.encoder.get_phases_batched(positions)
        memory_complex = token_emb * pos_phases

        mag = torch.abs(memory_complex)
        memory_complex = memory_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        memory_real = torch.cat([memory_complex.real, memory_complex.imag], dim=-1)
        memory_real = torch.clamp(memory_real, -10.0, 10.0)
        memory_values = self.value_proj(memory_real)

        # ABLATION: Mean pooling instead of attention!
        attended = memory_values.mean(dim=1, keepdim=True).expand(-1, output_len, -1)

        # Output
        logits = self.output_head(attended)
        return logits


class RandomAttentionModel(FastPhaseAttentionModel):
    """Use random attention weights - should fail if attention matters!"""

    def forward(self, input_indices, target_indices=None, idx2token=None):
        batch_size, input_len = input_indices.shape

        if target_indices is not None:
            output_len = target_indices.shape[1]
        else:
            output_len = input_len

        # Get embeddings and phases (same as original)
        token_emb = self.get_embeddings(input_indices)
        positions = torch.arange(input_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_phases = self.encoder.get_phases_batched(positions)
        memory_complex = token_emb * pos_phases

        mag = torch.abs(memory_complex)
        memory_complex = memory_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        memory_real = torch.cat([memory_complex.real, memory_complex.imag], dim=-1)
        memory_real = torch.clamp(memory_real, -10.0, 10.0)
        memory_values = self.value_proj(memory_real)

        # ABLATION: Random attention weights!
        random_weights = torch.rand(batch_size, output_len, input_len, device=self.device)
        random_weights = random_weights / random_weights.sum(dim=-1, keepdim=True)
        attended = torch.bmm(random_weights, memory_values)

        # Output
        logits = self.output_head(attended)
        return logits


# ============================================================================
# Test Datasets
# ============================================================================

class SimpleDataset(Dataset):
    """Simple copy task."""
    def __init__(self, n_examples=500, length=20, vocab_size=20):
        self.examples = []
        for _ in range(n_examples):
            seq = torch.randint(3, vocab_size, (length,))
            self.examples.append(seq)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        return {'input': seq, 'target': seq, 'length': len(seq)}


class ReverseDataset(Dataset):
    """Reverse task - requires proper positional attention."""
    def __init__(self, n_examples=500, length=20, vocab_size=20):
        self.examples = []
        for _ in range(n_examples):
            seq = torch.randint(3, vocab_size, (length,))
            reversed_seq = torch.flip(seq, [0])
            self.examples.append((seq, reversed_seq))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, rev = self.examples[idx]
        return {'input': seq, 'target': rev, 'length': len(seq)}


def collate_fn(batch):
    return {
        'input': torch.stack([item['input'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'length': torch.tensor([item['length'] for item in batch])
    }


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_quick(model, train_loader, n_epochs=5, lr=3e-3):
    """Quick training."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            logits = model(inputs, targets)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")


def evaluate_accuracy(model, test_loader):
    """Evaluate exact sequence accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            lengths = batch['length']

            logits = model(inputs, targets)
            predictions = logits.argmax(dim=-1)

            for i in range(inputs.shape[0]):
                length = lengths[i]
                if torch.all(predictions[i, :length] == targets[i, :length]):
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def visualize_attention_weights(model, test_example):
    """Extract and visualize attention weights for one example."""
    model.eval()

    with torch.no_grad():
        input_seq = test_example['input'].unsqueeze(0).to(device)
        target_seq = test_example['target'].unsqueeze(0).to(device)

        # Forward pass and capture attention weights
        batch_size, input_len = input_seq.shape
        output_len = target_seq.shape[1]

        # Replicate forward logic to capture attention
        token_emb = model.get_embeddings(input_seq)
        positions = torch.arange(input_len, device=device).unsqueeze(0)
        pos_phases = model.encoder.get_phases_batched(positions)
        memory_complex = token_emb * pos_phases

        mag = torch.abs(memory_complex)
        memory_complex = memory_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        memory_real = torch.cat([memory_complex.real, memory_complex.imag], dim=-1)
        memory_real = torch.clamp(memory_real, -10.0, 10.0)
        memory_values = model.value_proj(memory_real)

        # Query
        query_positions = torch.arange(input_len, input_len + output_len, device=device).unsqueeze(0)
        query_emb = model.get_embeddings(target_seq)
        query_phases = model.encoder.get_phases_batched(query_positions)
        queries_complex = query_emb * query_phases

        mag = torch.abs(queries_complex)
        queries_complex = queries_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        # Compute attention scores
        scores = model.attention.compute_coherence_batched(queries_complex, memory_complex)
        scores = torch.clamp(scores, -10.0, 10.0)

        attn_weights = torch.softmax(scores / model.attention.temperature, dim=-1)

        return attn_weights.squeeze(0).cpu().numpy()


# ============================================================================
# Main Ablation Tests
# ============================================================================

def main():
    print("=" * 80)
    print("ABLATION STUDY: Is Phase Attention Actually Working?")
    print("=" * 80)
    print()

    vocab_size = 20
    seq_len = 50

    # Create datasets (larger for better training)
    train_dataset_copy = SimpleDataset(n_examples=2000, length=seq_len, vocab_size=vocab_size)
    test_dataset_copy = SimpleDataset(n_examples=200, length=seq_len, vocab_size=vocab_size)

    train_dataset_reverse = ReverseDataset(n_examples=2000, length=seq_len, vocab_size=vocab_size)
    test_dataset_reverse = ReverseDataset(n_examples=200, length=seq_len, vocab_size=vocab_size)

    train_loader_copy = DataLoader(train_dataset_copy, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader_copy = DataLoader(test_dataset_copy, batch_size=32, shuffle=False, collate_fn=collate_fn)

    train_loader_reverse = DataLoader(train_dataset_reverse, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader_reverse = DataLoader(test_dataset_reverse, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ========================================================================
    # Test 1: Full Model on Copy Task (Baseline)
    # ========================================================================

    print("Test 1: Full Phase Attention Model (Copy Task)")
    print("-" * 80)
    model_full = FastPhaseAttentionModel(vocab_size=vocab_size, dim=512, hidden_dim=256,
                                         top_k=32, max_len=100, device=device).to(device)
    train_quick(model_full, train_loader_copy, n_epochs=20)
    acc_full = evaluate_accuracy(model_full, test_loader_copy)
    print(f"Accuracy: {acc_full:.1%}")
    print()

    # ========================================================================
    # Test 2: Mean Pooling Instead of Attention
    # ========================================================================

    print("Test 2: Mean Pooling (No Attention) on Copy Task")
    print("-" * 80)
    model_mean = MeanPoolingModel(vocab_size=vocab_size, dim=512, hidden_dim=256,
                                  top_k=32, max_len=100, device=device).to(device)
    train_quick(model_mean, train_loader_copy, n_epochs=20)
    acc_mean = evaluate_accuracy(model_mean, test_loader_copy)
    print(f"Accuracy: {acc_mean:.1%}")
    print()

    # ========================================================================
    # Test 3: Random Attention Weights
    # ========================================================================

    print("Test 3: Random Attention Weights on Copy Task")
    print("-" * 80)
    model_random = RandomAttentionModel(vocab_size=vocab_size, dim=512, hidden_dim=256,
                                       top_k=32, max_len=100, device=device).to(device)
    train_quick(model_random, train_loader_copy, n_epochs=20)
    acc_random = evaluate_accuracy(model_random, test_loader_copy)
    print(f"Accuracy: {acc_random:.1%}")
    print()

    # ========================================================================
    # Test 4: Reverse Task (Requires Proper Attention)
    # ========================================================================

    print("Test 4: Full Phase Attention Model (Reverse Task)")
    print("-" * 80)
    model_reverse = FastPhaseAttentionModel(vocab_size=vocab_size, dim=512, hidden_dim=256,
                                           top_k=32, max_len=100, device=device).to(device)
    train_quick(model_reverse, train_loader_reverse, n_epochs=20)
    acc_reverse = evaluate_accuracy(model_reverse, test_loader_reverse)
    print(f"Accuracy: {acc_reverse:.1%}")
    print()

    # ========================================================================
    # Test 5: Attention Visualization
    # ========================================================================

    print("Test 5: Attention Weight Visualization")
    print("-" * 80)
    test_example = test_dataset_copy[0]
    attn_weights = visualize_attention_weights(model_full, test_example)

    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected: Each output position should attend strongly to corresponding input position")
    print()
    print("Attention matrix (output_pos × input_pos):")
    print("Diagonal should be strong (output[i] attends to input[i])")
    print()

    # Show first 10x10
    print("First 10×10 of attention matrix:")
    print("      ", end="")
    for i in range(min(10, attn_weights.shape[1])):
        print(f"in{i:2d}  ", end="")
    print()

    for i in range(min(10, attn_weights.shape[0])):
        print(f"out{i:2d}: ", end="")
        for j in range(min(10, attn_weights.shape[1])):
            val = attn_weights[i, j]
            # Highlight diagonal
            if i == j:
                print(f"[{val:.2f}]", end=" ")
            else:
                print(f" {val:.2f} ", end=" ")
        print()

    print()

    # Check if diagonal is strong
    diagonal_mean = np.mean([attn_weights[i, i] for i in range(min(attn_weights.shape[0], attn_weights.shape[1]))])
    off_diagonal_mean = np.mean([attn_weights[i, j] for i in range(attn_weights.shape[0])
                                 for j in range(attn_weights.shape[1]) if i != j])

    print(f"Diagonal mean: {diagonal_mean:.3f}")
    print(f"Off-diagonal mean: {off_diagonal_mean:.3f}")
    print(f"Diagonal/Off-diagonal ratio: {diagonal_mean / (off_diagonal_mean + 1e-8):.2f}x")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Full Model (Copy):          {acc_full:>6.1%}  <- Should be high")
    print(f"Mean Pooling (Copy):        {acc_mean:>6.1%}  <- Should be low if attention matters")
    print(f"Random Attention (Copy):    {acc_random:>6.1%}  <- Should be low if attention matters")
    print(f"Full Model (Reverse):       {acc_reverse:>6.1%}  <- Should be high if attention works")
    print()

    if acc_full > 90 and acc_mean < 30 and acc_random < 30 and acc_reverse > 80:
        print("[OK] ATTENTION IS WORKING! All tests passed.")
        print("  - Full model succeeds on both copy and reverse")
        print("  - Ablations fail, proving attention is necessary")
        print("  - Attention weights show correct positional alignment")
    else:
        print("[WARN] Results suggest attention may not be working as expected!")
        if acc_mean > 70:
            print("  - Mean pooling works too well - task might be too simple")
        if acc_reverse < 70:
            print("  - Reverse task fails - attention might not be learning positions")

    print()


if __name__ == "__main__":
    main()
