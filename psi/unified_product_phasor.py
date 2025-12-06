"""
Unified Phasor: Product Binding (Position × Content)

Key insight from transformer analysis:
- Transformers can do both because position becomes part of content
- Simple addition dilutes both signals
- PRODUCT binding preserves both - you need BOTH matching for retrieval

This implements: phase = position_phase + content_phase (in complex domain = product)
Retrieval requires matching BOTH position AND content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProductPhasorBlock(nn.Module):
    """
    Phasor with PRODUCT binding of position and content.

    Write: value × exp(i × (pos_phase + content_phase))
    Read: memory × exp(-i × (pos_phase + content_phase))

    The conjugate multiplication means you need BOTH to match for retrieval.
    """
    def __init__(self, dim, n_oscillators=64, max_len=1024):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # FIXED position phases (like phasor_optimal)
        # Shape: [max_len, n_oscillators]
        pos_phases = torch.randn(max_len, n_oscillators) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Content phase encoder (like phase_binding_memory)
        self.content_phase_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_oscillators)
        )

        # Amplitude encoder
        self.amp_encoder = nn.Sequential(
            nn.Linear(dim, n_oscillators),
            nn.Softplus()
        )

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

        # Position phases [L, K] -> [1, L, K]
        pos_phase = self.pos_phases[:L].unsqueeze(0)

        # Content phases [B, L, K]
        content_phase = torch.tanh(self.content_phase_encoder(x)) * math.pi

        # PRODUCT: total_phase = pos_phase + content_phase
        # In complex exponential: exp(i*(a+b)) = exp(i*a) * exp(i*b)
        total_phase = pos_phase + content_phase  # [B, L, K]

        # Amplitude
        amp = self.amp_encoder(x) + 0.1  # [B, L, K]

        # Create phasor
        phasor = amp * torch.exp(1j * total_phase)  # [B, L, K]

        # Value
        V = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Bind: modulate value by phasor
        # phasor: [B, L, K] -> [B, L, K, 1]
        # V: [B, L, D] -> [B, L, 1, D]
        bound = phasor.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]

        # Accumulate causally
        memory = torch.cumsum(bound, dim=1)  # [B, L, K, D]

        # Unbind with conjugate (negates phase)
        retrieved = memory * phasor.conj().unsqueeze(-1)  # [B, L, K, D]

        # Sum over oscillators, take real part
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm

        return x + self.to_out(retrieved)


class PositionOnlyPhasorBlock(nn.Module):
    """
    Pure position-based phasor (like phasor_optimal).
    Uses ONLY fixed position phases, no content.
    """
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim

        # Fixed random phases per (position, dimension)
        base_phases = torch.randn(max_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)  # [1, L, D]

        value = self.to_value(x)  # [B, L, D]

        # Bind using real arithmetic (more stable)
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)

        # Cumsum
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind with same phase
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)

        return x + self.to_out(retrieved)


class ContentOnlyPhasorBlock(nn.Module):
    """
    Pure content-based phasor (like phase_binding_memory).
    Uses ONLY learned content phases, no position.
    """
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_oscillators)
        )
        self.to_amp = nn.Sequential(
            nn.Linear(dim, n_oscillators),
            nn.Softplus()
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Separate key and query encoders
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        amp = self.to_amp(x) + 0.1

        key_phasor = amp * torch.exp(1j * key_phase)
        query_phasor = amp * torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)

        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm

        return x + self.to_out(retrieved)


class UnifiedProductModel(nn.Module):
    """
    Model with ALTERNATING position-only and content-only layers.

    This lets the model learn to use position OR content as needed,
    without forcing them to combine within a single layer.
    """
    def __init__(self, vocab_size, dim=128, n_layers=4, n_oscillators=64, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embed = nn.Embedding(vocab_size, dim)

        # Alternate between position and content layers
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(PositionOnlyPhasorBlock(dim, max_len))
            else:
                self.blocks.append(ContentOnlyPhasorBlock(dim, n_oscillators))

        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)

        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))

        return self.head(self.norm_out(h))


# ============================================================================
# Test
# ============================================================================

def generate_copy_data(batch_size, seq_len, vocab_size, device='cuda'):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return tokens, tokens


def generate_associative_recall_data(batch_size, n_pairs, vocab_size, device='cuda'):
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + 2

    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]

        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        data[b, pos] = QUERY_TOKEN
        pos += 1

        query_idx = np.random.randint(0, n_pairs)
        query_k, query_v = pairs[query_idx]
        data[b, pos] = query_k
        targets[b, pos] = query_v

    return data, targets


def test_copy_task(model, vocab_size, device='cuda'):
    model.eval()
    seq_len = 32
    batch_size = 100
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(10):
            tokens, _ = generate_copy_data(batch_size, seq_len, vocab_size, device)
            logits = model(tokens)
            preds = logits[:, :-1, :vocab_size].argmax(dim=-1)
            targets = tokens[:, 1:]
            correct += (preds == targets).sum().item()
            total += preds.numel()

    return correct / total * 100


def test_associative_recall(model, vocab_size, n_pairs=5, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    query_pos = n_pairs * 2 + 1

    with torch.no_grad():
        for _ in range(10):
            data, targets = generate_associative_recall_data(100, n_pairs, vocab_size, device)
            logits = model(data)
            preds = logits[:, query_pos, :vocab_size].argmax(dim=-1)
            target_vals = targets[:, query_pos]
            correct += (preds == target_vals).sum().item()
            total += preds.shape[0]

    return correct / total * 100


def train_model():
    print("=" * 70)
    print("UNIFIED PRODUCT PHASOR: Alternating Position/Content Layers")
    print("=" * 70)
    print()
    print("Architecture: Position layer -> Content layer -> Position -> Content")
    print("Each layer type can specialize for its addressing mode")
    print()

    vocab_size = 16
    n_pairs = 5
    dim = 128
    n_layers = 4  # 2 position + 2 content
    n_oscillators = 64

    model = UnifiedProductModel(
        vocab_size=vocab_size + 1,
        dim=dim,
        n_layers=n_layers,
        n_oscillators=n_oscillators,
        max_len=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layers} layers (alternating pos/content)")
    print(f"Parameters: {params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Training on BOTH tasks simultaneously...")
    print("-" * 70)

    best_copy = 0
    best_recall = 0

    for epoch in range(1000):
        model.train()

        # Copy task
        tokens, _ = generate_copy_data(32, 32, vocab_size, device)
        logits = model(tokens)
        loss_copy = criterion(
            logits[:, :-1, :vocab_size].reshape(-1, vocab_size),
            tokens[:, 1:].reshape(-1)
        )

        # Associative recall
        data, targets = generate_associative_recall_data(32, n_pairs, vocab_size, device)
        logits = model(data)
        query_pos = n_pairs * 2 + 1
        loss_recall = criterion(
            logits[:, query_pos, :vocab_size],
            targets[:, query_pos]
        )

        loss = loss_copy + loss_recall

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
    train_model()
