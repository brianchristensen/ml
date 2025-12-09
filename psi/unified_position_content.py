"""
Unified Phasor: Position as Content

Key insight: Instead of separate position/content phases, we INJECT position
information INTO the content BEFORE phase encoding.

This mirrors how transformers work:
  - Transformer: content + pos_embed -> Q, K, V
  - This model: content + pos_embed -> phase encoding

The phase encoder then learns to use BOTH position and content for addressing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class PositionContentEncoder(nn.Module):
    """
    Encodes content+position into phases.

    Position is injected as learnable embeddings BEFORE phase computation.
    This lets the network learn to attend by position, content, or both.
    """
    def __init__(self, dim, n_oscillators=64, max_len=1024):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Learnable position embeddings (like transformer)
        self.pos_embed = nn.Parameter(torch.randn(max_len, dim) * 0.02)

        # Phase encoder takes position-augmented content
        self.to_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_oscillators)
        )

        # Amplitude encoder
        self.to_amp = nn.Sequential(
            nn.Linear(dim, n_oscillators),
            nn.Softplus()
        )

    def forward(self, x, for_key=True):
        """
        x: [B, L, D]
        for_key: if True, add position embedding (keys need position)
                 if False, don't add (queries match on content)
        """
        B, L, D = x.shape

        if for_key:
            # Keys: content + position
            x_aug = x + self.pos_embed[:L].unsqueeze(0)
        else:
            # Queries: just content (or with position for position-based retrieval)
            x_aug = x + self.pos_embed[:L].unsqueeze(0)  # Both need position!

        phase = torch.tanh(self.to_phase(x_aug)) * math.pi
        amp = self.to_amp(x_aug) + 0.1

        return amp * torch.exp(1j * phase)


class UnifiedPhasorBlock(nn.Module):
    """
    Single phasor block with position-content unified encoding.

    O(n) via cumsum, but position info is embedded in the phases.
    """
    def __init__(self, dim, n_oscillators=64, max_len=1024):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Separate key and query encoders
        self.key_encoder = PositionContentEncoder(dim, n_oscillators, max_len)
        self.query_encoder = PositionContentEncoder(dim, n_oscillators, max_len)

        # Value projection
        self.to_value = nn.Linear(dim, dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Encode keys and queries (both include position info)
        key_phasors = self.key_encoder(x, for_key=True)    # [B, L, K]
        query_phasors = self.query_encoder(x, for_key=False)  # [B, L, K]

        # Values
        V = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Bind: modulate values by key phasors
        # key_phasors: [B, L, K] -> [B, L, K, 1]
        # V: [B, L, D] -> [B, L, 1, D]
        bound = key_phasors.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]

        # Accumulate causally
        memory = torch.cumsum(bound, dim=1)  # [B, L, K, D]

        # Unbind: demodulate with query phasors
        # query_phasors.conj(): [B, L, K] -> [B, L, K, 1]
        retrieved = memory * query_phasors.conj().unsqueeze(-1)  # [B, L, K, D]

        # Sum over oscillators, take real part
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize by position
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm

        return x + self.to_out(retrieved)


class UnifiedPhasorModel(nn.Module):
    """
    Stacked unified phasor blocks for sequence modeling.
    """
    def __init__(self, vocab_size, dim=128, n_layers=4, n_oscillators=64, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            UnifiedPhasorBlock(dim, n_oscillators, max_len)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """x: [B, L] token indices -> [B, L, vocab_size] logits"""
        h = self.embed(x)

        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))

        return self.head(self.norm_out(h))


# ============================================================================
# Test on both tasks
# ============================================================================

def generate_copy_data(batch_size, seq_len, vocab_size, device='cuda'):
    """Generate copy task: input sequence, then retrieve by position."""
    # Random tokens
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # For copy task, target is the same as input (autoregressive prediction)
    # But we test: given position i, predict token at position i
    return tokens, tokens


def generate_associative_recall_data(batch_size, n_pairs, vocab_size, device='cuda'):
    """Generate associative recall: k1,v1,k2,v2,...,QUERY,k_i -> predict v_i"""
    QUERY_TOKEN = vocab_size  # Special token
    seq_len = n_pairs * 2 + 2  # pairs + QUERY + query_key

    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Generate unique key-value pairs
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]

        # Fill sequence: k1, v1, k2, v2, ...
        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        # QUERY token
        data[b, pos] = QUERY_TOKEN
        pos += 1

        # Query key
        query_idx = np.random.randint(0, n_pairs)
        query_k, query_v = pairs[query_idx]
        data[b, pos] = query_k
        targets[b, pos] = query_v

    return data, targets


def test_copy_task(model, vocab_size, device='cuda'):
    """Test copy/retrieval by position."""
    model.eval()

    # Test: can model predict next token given context?
    # This tests positional memory
    seq_len = 32
    batch_size = 100

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(10):
            tokens, _ = generate_copy_data(batch_size, seq_len, vocab_size, device)

            # Predict next token at each position
            logits = model(tokens)  # [B, L, V]

            # Check if prediction at position i-1 matches token at position i
            preds = logits[:, :-1, :].argmax(dim=-1)  # [B, L-1]
            targets = tokens[:, 1:]  # [B, L-1]

            correct += (preds == targets).sum().item()
            total += preds.numel()

    return correct / total * 100


def test_associative_recall(model, vocab_size, n_pairs=5, device='cuda'):
    """Test associative recall."""
    model.eval()

    correct = 0
    total = 0
    query_pos = n_pairs * 2 + 1  # Position of query key

    with torch.no_grad():
        for _ in range(10):
            data, targets = generate_associative_recall_data(100, n_pairs, vocab_size, device)

            logits = model(data)  # [B, L, V]

            # Check prediction at query position
            preds = logits[:, query_pos, :vocab_size].argmax(dim=-1)  # [B]
            target_vals = targets[:, query_pos]  # [B]

            correct += (preds == target_vals).sum().item()
            total += preds.shape[0]

    return correct / total * 100


def train_unified_model():
    print("=" * 70)
    print("UNIFIED PHASOR: Position as Content")
    print("=" * 70)
    print()
    print("Key idea: Inject position embeddings BEFORE phase encoding")
    print("This lets phases capture both position and content information")
    print()

    vocab_size = 16
    n_pairs = 5
    dim = 128
    n_layers = 4
    n_oscillators = 64

    # Model with vocab_size + 1 for QUERY token
    model = UnifiedPhasorModel(
        vocab_size=vocab_size + 1,
        dim=dim,
        n_layers=n_layers,
        n_oscillators=n_oscillators,
        max_len=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layers} layers, dim={dim}, oscillators={n_oscillators}")
    print(f"Parameters: {params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training - alternate between tasks
    print("Training on BOTH tasks simultaneously...")
    print("-" * 70)

    best_copy = 0
    best_recall = 0

    for epoch in range(1000):
        model.train()

        # Batch 1: Copy task (next-token prediction)
        tokens, _ = generate_copy_data(32, 32, vocab_size, device)
        logits = model(tokens)
        loss_copy = criterion(
            logits[:, :-1, :vocab_size].reshape(-1, vocab_size),
            tokens[:, 1:].reshape(-1)
        )

        # Batch 2: Associative recall
        data, targets = generate_associative_recall_data(32, n_pairs, vocab_size, device)
        logits = model(data)
        query_pos = n_pairs * 2 + 1
        loss_recall = criterion(
            logits[:, query_pos, :vocab_size],
            targets[:, query_pos]
        )

        # Combined loss
        loss = loss_copy + loss_recall

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 100 == 0:
            copy_acc = test_copy_task(model, vocab_size, device)
            recall_acc = test_associative_recall(model, vocab_size, n_pairs, device)

            best_copy = max(best_copy, copy_acc)
            best_recall = max(best_recall, recall_acc)

            print(f"Epoch {epoch+1:4d}: Copy={copy_acc:5.1f}% (best={best_copy:.1f}%), "
                  f"Recall={recall_acc:5.1f}% (best={best_recall:.1f}%), "
                  f"Loss={loss.item():.4f}")

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"Copy Task:         {best_copy:.1f}%")
    print(f"Associative Recall: {best_recall:.1f}%")
    print()

    random_baseline = 100 / vocab_size
    print(f"Random baseline: {random_baseline:.1f}%")
    print()

    if best_copy > 50 and best_recall > 50:
        print("SUCCESS: Model learned both tasks!")
    elif best_copy > 50:
        print("PARTIAL: Only learned copy task")
    elif best_recall > 50:
        print("PARTIAL: Only learned recall task")
    else:
        print("FAILED: Neither task learned well")

    return model, best_copy, best_recall


if __name__ == "__main__":
    train_unified_model()
