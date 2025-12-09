"""
Phasor with Stronger Orthogonality Enforcement

The issue: ortho_loss isn't decreasing much.
Let's try:
1. Stronger regularization
2. Different orthogonality formulation
3. Pre-normalize phases to unit circle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class StrongOrthoPhasor(nn.Module):
    """
    Phasor with stronger orthogonality enforcement.

    Key changes:
    1. Normalize phase vectors before computing similarity
    2. Use cosine similarity directly on the complex phasors
    3. Penalize similarity more aggressively
    """
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        # Simpler phase projection
        self.to_key_phase = nn.Linear(dim, dim)
        self.to_query_phase = nn.Linear(dim, dim)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        self.ortho_loss = 0.0

    def forward(self, x, store_phases=False):
        batch_size, seq_len, dim = x.shape

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        # Get phases (will be in radians via learned projection)
        key_phase = self.to_key_phase(key)
        query_phase = self.to_query_phase(query)

        # Create unit complex numbers from phases
        # phasor = exp(i * phase) = cos(phase) + i*sin(phase)
        key_cos = torch.cos(key_phase)
        key_sin = torch.sin(key_phase)
        query_cos = torch.cos(query_phase)
        query_sin = torch.sin(query_phase)

        # Compute orthogonality loss on the PHASORS (not raw phases)
        # Two phasors are orthogonal if their dot product is 0
        # For complex: real part of conj(a) * b = 0
        self.ortho_loss = self._compute_phasor_ortho_loss(key_cos, key_sin)

        if store_phases:
            self.last_key_phase = key_phase.detach()
            self.last_query_phase = query_phase.detach()

        # Bind: value * exp(i * key_phase)
        bound_real = value * key_cos
        bound_imag = value * key_sin

        # Chunked cumsum
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            bound_real = F.pad(bound_real, (0, 0, 0, pad_len))
            bound_imag = F.pad(bound_imag, (0, 0, 0, pad_len))
            query_cos = F.pad(query_cos, (0, 0, 0, pad_len))
            query_sin = F.pad(query_sin, (0, 0, 0, pad_len))

        padded_len = bound_real.shape[1]
        num_chunks = padded_len // self.chunk_size

        br = bound_real.view(batch_size, num_chunks, self.chunk_size, dim)
        bi = bound_imag.view(batch_size, num_chunks, self.chunk_size, dim)
        qc = query_cos.view(batch_size, num_chunks, self.chunk_size, dim)
        qs = query_sin.view(batch_size, num_chunks, self.chunk_size, dim)

        # Cumsum
        mem_real = torch.cumsum(br, dim=2)
        mem_imag = torch.cumsum(bi, dim=2)

        # Unbind: multiply by conj of query phasor
        # conj(exp(i*q)) = exp(-i*q) = cos(q) - i*sin(q)
        # real part of (mem_r + i*mem_i) * (cos_q - i*sin_q)
        # = mem_r * cos_q + mem_i * sin_q
        retrieved = mem_real * qc + mem_imag * qs

        retrieved = retrieved.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        # Scale by expected magnitude
        retrieved = retrieved / math.sqrt(seq_len)

        out = self.to_out(retrieved)
        return x + out

    def _compute_phasor_ortho_loss(self, cos_p, sin_p):
        """
        Compute orthogonality loss on complex phasors.

        For unit complex numbers z_i = cos(θ_i) + i*sin(θ_i),
        orthogonality means Re(conj(z_i) * z_j) = cos(θ_i - θ_j) ≈ 0
        for i ≠ j.
        """
        batch_size, seq_len, dim = cos_p.shape

        # Sample positions
        n_samples = min(32, seq_len)
        idx = torch.randperm(seq_len, device=cos_p.device)[:n_samples]

        cos_s = cos_p[:, idx, :]  # (batch, n_samples, dim)
        sin_s = sin_p[:, idx, :]

        # Pairwise dot product of complex numbers
        # Re(conj(z_i) * z_j) = cos_i * cos_j + sin_i * sin_j
        # This equals cos(phase_i - phase_j)
        dot = (cos_s.unsqueeze(2) * cos_s.unsqueeze(1) +
               sin_s.unsqueeze(2) * sin_s.unsqueeze(1))  # (batch, n, n, dim)

        # Mask diagonal
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=cos_p.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)

        # We want dot product ≈ 0, so minimize |dot|
        # Using squared loss
        ortho_loss = (dot ** 2 * mask).sum() / (mask.sum() * dim + 1e-6)

        return ortho_loss


class StrongOrthoBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor = StrongOrthoPhasor(dim, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, store_phases=False):
        x = self.phasor(self.norm1(x), store_phases)
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def ortho_loss(self):
        return self.phasor.ortho_loss


class StrongOrthoModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            StrongOrthoBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, store_phases=False):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, store_phases)
        return self.head(self.norm(h))

    def get_ortho_loss(self):
        return sum(block.ortho_loss for block in self.blocks)


def test_strong_ortho():
    print("=" * 60)
    print("TEST: Strong Orthogonality Enforcement")
    print("=" * 60)

    vocab_size = 10
    seq_len = 64
    dim = 64

    def generate_batch(batch_size=32):
        first_tokens = torch.randint(0, vocab_size, (batch_size,))
        middle = torch.randint(0, vocab_size, (batch_size, seq_len - 2))
        x = torch.cat([
            first_tokens.unsqueeze(1),
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long)
        ], dim=1)
        y = torch.cat([
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long),
            first_tokens.unsqueeze(1)
        ], dim=1)
        return x, y, first_tokens

    for ortho_weight in [0.0, 1.0, 10.0, 100.0]:
        print(f"\n--- ortho_weight = {ortho_weight} ---")

        model = StrongOrthoModel(vocab_size, dim=dim, num_layers=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(150):
            x, y, _ = generate_batch(32)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            ortho_loss = model.get_ortho_loss()

            loss = ce_loss + ortho_weight * ortho_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: CE={ce_loss.item():.4f}, Ortho={ortho_loss.item():.6f}")

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(10):
                x, _, first_tokens = generate_batch(32)
                x = x.to(device)
                first_tokens = first_tokens.to(device)
                logits = model(x)
                pred = logits[:, -1].argmax(dim=-1)
                correct += (pred == first_tokens).sum().item()
                total += 32

        acc = correct / total * 100
        print(f"  Result: {acc:.1f}% (random: 10%)")


def analyze_phase_distribution():
    """Check if phases are actually becoming orthogonal."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Phase Distribution After Training")
    print("=" * 60)

    vocab_size = 10
    seq_len = 64
    dim = 64

    def generate_batch(batch_size=32):
        first_tokens = torch.randint(0, vocab_size, (batch_size,))
        middle = torch.randint(0, vocab_size, (batch_size, seq_len - 2))
        x = torch.cat([
            first_tokens.unsqueeze(1),
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long)
        ], dim=1)
        y = torch.cat([
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long),
            first_tokens.unsqueeze(1)
        ], dim=1)
        return x, y, first_tokens

    model = StrongOrthoModel(vocab_size, dim=dim, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train with strong ortho
    ortho_weight = 10.0
    model.train()
    for epoch in range(150):
        x, y, _ = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        ortho_loss = model.get_ortho_loss()
        loss = ce_loss + ortho_weight * ortho_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Analyze phases
    model.eval()
    x, _, _ = generate_batch(1)
    x = x.to(device)
    with torch.no_grad():
        _ = model(x, store_phases=True)

    # Get phases from first block
    key_phase = model.blocks[0].phasor.last_key_phase[0]  # (seq_len, dim)

    # Compute pairwise phase differences
    phase_diff = key_phase.unsqueeze(0) - key_phase.unsqueeze(1)  # (seq, seq, dim)
    cos_diff = torch.cos(phase_diff)

    # Average cosine similarity across dimensions
    avg_cos = cos_diff.mean(dim=-1)  # (seq, seq)

    print(f"Pairwise cosine similarity stats (should be ~0 for orthogonal):")
    print(f"  Mean (off-diagonal): {avg_cos[~torch.eye(64, dtype=bool, device=device)].mean().item():.4f}")
    print(f"  Std: {avg_cos[~torch.eye(64, dtype=bool, device=device)].std().item():.4f}")
    print(f"  Max: {avg_cos[~torch.eye(64, dtype=bool, device=device)].max().item():.4f}")
    print(f"  Min: {avg_cos[~torch.eye(64, dtype=bool, device=device)].min().item():.4f}")

    # Check specific positions
    print(f"\n  Similarity between pos 0 and pos 63: {avg_cos[0, 63].item():.4f}")
    print(f"  Similarity between pos 0 and pos 1: {avg_cos[0, 1].item():.4f}")


if __name__ == "__main__":
    test_strong_ortho()
    analyze_phase_distribution()
