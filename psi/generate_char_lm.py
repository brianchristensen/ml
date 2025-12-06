"""
Autoregressive Text Generation with Novel Attention

Loads a trained novel attention model and generates text character-by-character
to verify it has learned coherent language patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from phase_binding_memory import PhaseBindingLanguageModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Transformer Model (copied from test_char_lm.py for generation)
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
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Output
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.output_head(x)
        return logits


def generate_text(model, seed_text: str, length: int = 500, temperature: float = 1.0):
    """
    Generate text autoregressively from a seed.

    Args:
        model: Trained PhaseSpaceIntegrator
        seed_text: Initial text to condition on
        length: Number of characters to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated text string
    """
    model.eval()

    # Convert seed text to bytes
    seed_bytes = seed_text.encode('utf-8', errors='ignore')
    current_sequence = list(seed_bytes)

    print(f"Seed: {seed_text}")
    print(f"Generating {length} characters...\n")
    print("=" * 80)

    with torch.no_grad():
        for i in range(length):
            # Take last 256 chars as context (model's max context window)
            context = current_sequence[-256:] if len(current_sequence) > 256 else current_sequence

            # Pad if needed
            if len(context) < 256:
                context = [0] * (256 - len(context)) + context

            # Convert to tensor
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            # Forward pass
            logits = model(input_tensor)  # [1, seq_len, vocab_size]

            # Get logits for the last position (predicting next char)
            next_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            next_logits = next_logits / temperature

            # Sample from distribution
            probs = torch.softmax(next_logits, dim=0)
            next_char = torch.multinomial(probs, num_samples=1).item()

            # Add to sequence
            current_sequence.append(next_char)

            # Print generated character (if it's valid UTF-8)
            try:
                char = bytes([next_char]).decode('utf-8')
                print(char, end='', flush=True)
            except:
                print('?', end='', flush=True)

    print("\n" + "=" * 80)

    # Convert full sequence to text
    try:
        generated_text = bytes(current_sequence).decode('utf-8', errors='replace')
    except:
        generated_text = str(current_sequence)

    return generated_text


def generate_with_top_k(model, seed_text: str, length: int = 500,
                        temperature: float = 1.0, top_k: int = 40):
    """Generate text with top-k sampling for more coherent output."""
    model.eval()

    seed_bytes = seed_text.encode('utf-8', errors='ignore')
    current_sequence = list(seed_bytes)

    print(f"Seed: {seed_text}")
    print(f"Generating {length} characters (top-k={top_k})...\n")
    print("=" * 80)

    with torch.no_grad():
        for i in range(length):
            context = current_sequence[-256:] if len(current_sequence) > 256 else current_sequence

            if len(context) < 256:
                context = [0] * (256 - len(context)) + context

            input_tensor = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_logits = logits[0, -1, :]

            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_logits, k=top_k)
            top_k_logits = top_k_logits / temperature
            top_k_probs = torch.softmax(top_k_logits, dim=0)

            # Sample from top-k
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_char = top_k_indices[sampled_idx].item()

            current_sequence.append(next_char)

            try:
                char = bytes([next_char]).decode('utf-8')
                print(char, end='', flush=True)
            except:
                print('?', end='', flush=True)

    print("\n" + "=" * 80)

    try:
        generated_text = bytes(current_sequence).decode('utf-8', errors='replace')
    except:
        generated_text = str(current_sequence)

    return generated_text


def test_generation():
    """Test generation with various seeds, comparing Transformer vs Novel Attention."""
    print("=" * 80)
    print("Side-by-Side Generation Comparison: Transformer vs Novel Attention")
    print("=" * 80)
    print()

    # Load Novel Attention model
    print("Loading Novel Attention model...")

    try:
        checkpoint = torch.load('tempo_charlm_final.pt', map_location=device)

        novel_model = PhaseBindingLanguageModel(
            vocab_size=256,
            dim=128,
            num_layers=8,
            device=device
        ).to(device)

        novel_model.load_state_dict(checkpoint['model_state_dict'])
        novel_bpc = checkpoint.get('best_val_bpc', '?')
        print(f"Loaded Novel Attention (Val BPC: {novel_bpc:.4f})")
    except FileNotFoundError:
        print("No Novel Attention model found!")
        novel_model = None

    # Load Transformer model
    print("Loading Transformer model...")
    transformer = SimpleTransformerLM(
        vocab_size=256,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        max_len=256
    ).to(device)

    try:
        checkpoint = torch.load('transformer_charlm.pt', map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        tf_bpc = checkpoint.get('best_val_bpc', '?')
        print(f"Loaded Transformer (Val BPC: {tf_bpc:.4f})")
    except FileNotFoundError:
        print("No Transformer model found!")
        transformer = None

    if novel_model is None and transformer is None:
        print("\nERROR: No models found! Train them first with test_char_lm.py")
        return

    print()
    print("=" * 80)
    print()

    # Test seeds
    test_seeds = [
        "The capital of France is",
        "In the year 1776",
        "Albert Einstein was",
    ]

    for i, seed in enumerate(test_seeds):
        print("\n" + "=" * 80)
        print(f"Test {i + 1}/{len(test_seeds)}: {seed}")
        print("=" * 80)
        print()

        if transformer is not None:
            print("TRANSFORMER:")
            print("-" * 80)
            generate_with_top_k(transformer, seed, length=150, temperature=0.8, top_k=40)
            print()

        if novel_model is not None:
            print("PHASE ATTENTION:")
            print("-" * 80)
            generate_with_top_k(novel_model, seed, length=150, temperature=0.8, top_k=40)
            print()

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print()
    print("Analysis:")
    print("- Compare coherence: Which produces more readable text?")
    print("- Compare patterns: Which learned better letter/word combinations?")
    print("- Compare creativity: Which has more variety vs repetition?")
    print()
    if novel_model and transformer:
        print(f"Novel Attention BPC: {novel_bpc:.4f}")
        print(f"Transformer BPC: {tf_bpc:.4f}")
        if isinstance(novel_bpc, float) and isinstance(tf_bpc, float):
            ratio = tf_bpc / novel_bpc
            print(f"BPC Ratio: {ratio:.1f}× (does generation quality match?)")


if __name__ == "__main__":
    test_generation()
