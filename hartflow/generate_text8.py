"""
Text Generation for text8 Models

Loads models trained on text8 and generates clean English text.
text8 has vocab_size=27 (lowercase a-z + space), so generation
should be more readable than enwik8.
"""

import torch
import torch.nn as nn
import numpy as np
from phase_attention import PhaseAttentionLM, MultiHeadPhaseAttentionLM


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Transformer Model
# ============================================================================

class SimpleTransformerLM(nn.Module):
    """Simple single-layer transformer."""

    def __init__(self, vocab_size=27, d_model=512, nhead=8,
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


# ============================================================================
# Text8 Character Mapping
# ============================================================================

# text8 uses: lowercase a-z (0-25) + space (26)
def char_to_idx(c):
    """Convert character to index."""
    if c == ' ':
        return 26
    elif 'a' <= c <= 'z':
        return ord(c) - ord('a')
    else:
        return 26  # Unknown chars become space

def idx_to_char(idx):
    """Convert index to character."""
    if idx == 26:
        return ' '
    elif 0 <= idx <= 25:
        return chr(ord('a') + idx)
    else:
        return ' '


# ============================================================================
# Generation Functions
# ============================================================================

def generate_text(model, seed_text: str, length: int = 500,
                  temperature: float = 1.0, top_k: int = 40):
    """Generate text autoregressively."""
    model.eval()

    # Convert seed to indices
    seed_text = seed_text.lower()
    current_sequence = [char_to_idx(c) for c in seed_text]

    print(f"Seed: '{seed_text}'")
    print(f"Generating {length} characters...\n")
    print("=" * 80)
    print(seed_text, end='', flush=True)

    with torch.no_grad():
        for i in range(length):
            # Context window (last 256 chars)
            context = current_sequence[-256:] if len(current_sequence) > 256 else current_sequence

            # Pad if needed
            if len(context) < 256:
                context = [26] * (256 - len(context)) + context  # Pad with space

            # Convert to tensor
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            # Forward pass
            logits = model(input_tensor)
            next_logits = logits[0, -1, :]

            # Top-k sampling (cap at vocab size)
            k = min(top_k, next_logits.shape[0])
            top_k_logits, top_k_indices = torch.topk(next_logits, k=k)
            top_k_logits = top_k_logits / temperature
            top_k_probs = torch.softmax(top_k_logits, dim=0)

            # Sample
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_char_idx = top_k_indices[sampled_idx].item()

            # Add to sequence
            current_sequence.append(next_char_idx)

            # Print
            char = idx_to_char(next_char_idx)
            print(char, end='', flush=True)

    print("\n" + "=" * 80)

    # Return full text
    return ''.join([idx_to_char(idx) for idx in current_sequence])


# ============================================================================
# Side-by-Side Comparison
# ============================================================================

def test_generation():
    """Compare Transformer vs Phase Attention on text8."""
    print("=" * 80)
    print("text8 Generation Comparison: Transformer vs Phase Attention")
    print("=" * 80)
    print()

    vocab_size = 27

    # Load Phase Attention
    print("Loading Phase Attention model...")
    phase_model = PhaseAttentionLM(
        vocab_size=vocab_size,
        dim=512,
        hidden_dim=512,
        max_len=256,
        device=device
    ).to(device)

    try:
        checkpoint = torch.load('phase_attention_text8.pt', map_location=device)
        phase_model.load_state_dict(checkpoint['model_state_dict'])
        phase_bpc = checkpoint.get('best_val_bpc', '?')
        phase_epoch = checkpoint.get('epoch', '?')
        print(f"✓ Loaded Phase Attention (Epoch {phase_epoch}, Val BPC: {phase_bpc:.4f})")
    except FileNotFoundError:
        print("✗ No Phase Attention model found!")
        phase_model = None

    # Load Transformer
    print("Loading Transformer model...")
    transformer = SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        max_len=256
    ).to(device)

    try:
        checkpoint = torch.load('transformer_text8.pt', map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        tf_bpc = checkpoint.get('best_val_bpc', '?')
        tf_epoch = checkpoint.get('epoch', '?')
        print(f"✓ Loaded Transformer (Epoch {tf_epoch}, Val BPC: {tf_bpc:.4f})")
    except FileNotFoundError:
        print("✗ No Transformer model found!")
        transformer = None

    if phase_model is None and transformer is None:
        print("\nERROR: No models found! Train them first with train_text8.py")
        return

    print()
    print("=" * 80)
    print()

    # Test seeds (clean English prompts)
    test_seeds = [
        "the history of",
        "in the year one thousand",
        "the scientific method",
        "once upon a time",
        "the capital city",
    ]

    for i, seed in enumerate(test_seeds):
        print("\n" + "=" * 80)
        print(f"Test {i + 1}/{len(test_seeds)}")
        print("=" * 80)
        print()

        if transformer is not None:
            print("🔷 TRANSFORMER:")
            print("-" * 80)
            generate_text(transformer, seed, length=200, temperature=0.8, top_k=20)
            print()

        if phase_model is not None:
            print("🔶 PHASE ATTENTION:")
            print("-" * 80)
            generate_text(phase_model, seed, length=200, temperature=0.8, top_k=20)
            print()

    # Final comparison
    print("\n" + "=" * 80)
    print("Generation Comparison Complete!")
    print("=" * 80)
    print()

    if phase_model and transformer:
        print("Metrics:")
        print(f"  Transformer:       {tf_bpc:.4f} BPC (Epoch {tf_epoch})")
        print(f"  Phase Attention:   {phase_bpc:.4f} BPC (Epoch {phase_epoch})")
        print()

        print("Analysis:")
        print("- Which model generates more coherent English?")
        print("- Which has better word boundaries and grammar?")
        print("- Which shows more creativity vs repetition?")
        print()
        print("Remember: Generation quality matters more than BPC!")
        print("If phase attention has lower BPC but worse generation,")
        print("it means BPC is being 'gamed' by high confidence on common patterns.")


if __name__ == "__main__":
    test_generation()
