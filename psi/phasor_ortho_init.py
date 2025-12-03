"""
Phasor with Orthogonal Base Phases

New idea: Start with RANDOM (thus nearly orthogonal) phases,
then let the network learn to modulate them.

This gives us the orthogonality guarantee from random initialization,
plus the flexibility of learned adjustments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OrthoInitPhasor(nn.Module):
    """
    Phasor with orthogonal random base phases.

    Key idea: Use random phases as base, add learned modulation.
    Random phases in high-ish dimension are nearly orthogonal.
    """
    def __init__(self, dim, max_seq_len=128, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len

        # Random base phases for each position - these are FIXED (not learned)
        # Random phases are nearly orthogonal in expectation
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        # Small learned modulation on top of base phases
        self.key_mod = nn.Linear(dim, dim)
        self.query_mod = nn.Linear(dim, dim)

        self.mod_scale = nn.Parameter(torch.ones(1) * 0.1)  # Start small

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        # Base phases from random initialization (position-based)
        base_phase = self.base_phases[:seq_len].unsqueeze(0)  # (1, seq, dim)

        # Learned modulation (content-based)
        key_modulation = self.key_mod(key) * self.mod_scale
        query_modulation = self.query_mod(query) * self.mod_scale

        # Final phases = base + modulation
        key_phase = base_phase + key_modulation
        query_phase = base_phase + query_modulation

        # Bind
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # Chunked cumsum
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            bound_real = F.pad(bound_real, (0, 0, 0, pad_len))
            bound_imag = F.pad(bound_imag, (0, 0, 0, pad_len))
            query_phase = F.pad(query_phase, (0, 0, 0, pad_len))

        padded_len = bound_real.shape[1]
        num_chunks = padded_len // self.chunk_size

        br = bound_real.view(batch_size, num_chunks, self.chunk_size, dim)
        bi = bound_imag.view(batch_size, num_chunks, self.chunk_size, dim)
        qp = query_phase.view(batch_size, num_chunks, self.chunk_size, dim)

        # Cumsum
        mem_real = torch.cumsum(br, dim=2)
        mem_imag = torch.cumsum(bi, dim=2)

        # Unbind
        retrieved = mem_real * torch.cos(qp) + mem_imag * torch.sin(qp)

        retrieved = retrieved.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out


class OrthoInitBlock(nn.Module):
    def __init__(self, dim, max_seq_len=128, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor = OrthoInitPhasor(dim, max_seq_len, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.phasor(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OrthoInitModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=4, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthoInitBlock(dim, max_seq_len) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def verify_base_orthogonality():
    """Verify that random base phases are approximately orthogonal."""
    print("=" * 60)
    print("VERIFICATION: Random Base Phase Orthogonality")
    print("=" * 60)

    for dim in [64, 128, 256, 512]:
        phases = torch.randn(64, dim) * math.pi  # 64 positions

        # Compute pairwise cos similarity
        cos_p = torch.cos(phases)
        sin_p = torch.sin(phases)

        # Phasor dot product
        dot = (cos_p.unsqueeze(0) * cos_p.unsqueeze(1) +
               sin_p.unsqueeze(0) * sin_p.unsqueeze(1))  # (64, 64, dim)

        avg_dot = dot.mean(dim=-1)  # (64, 64)

        off_diag = avg_dot[~torch.eye(64, dtype=bool)]
        print(f"dim={dim:3d}: mean={off_diag.mean().item():.4f}, std={off_diag.std().item():.4f}, max={off_diag.abs().max().item():.4f}")

    print("\n(For good orthogonality: mean~0, std small, max<0.3)")


def test_ortho_init():
    print("\n" + "=" * 60)
    print("TEST: Long-Range Copy with Orthogonal Init Phases")
    print("=" * 60)

    vocab_size = 10
    seq_len = 64

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

    for dim in [64, 128, 256]:
        print(f"\n--- dim = {dim} ---")

        model = OrthoInitModel(vocab_size, dim=dim, num_layers=4, max_seq_len=128).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(150):
            x, y, _ = generate_batch(32)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")

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


def test_pure_position_retrieval():
    """
    Sanity check: Can we retrieve by position alone?
    (No content modulation, just base phases)
    """
    print("\n" + "=" * 60)
    print("TEST: Pure Position-Based Retrieval")
    print("=" * 60)
    print("Using ONLY base phases (no learning) to see if retrieval works")

    dim = 256
    seq_len = 64

    # Random base phases
    base_phases = torch.randn(seq_len, dim) * math.pi

    # Random values at each position
    values = torch.randn(seq_len, dim)

    # Bind each value to its position's phase
    bound_real = values * torch.cos(base_phases)
    bound_imag = values * torch.sin(base_phases)

    # Accumulate (simulate cumsum at last position)
    mem_real = bound_real.sum(dim=0)  # (dim,)
    mem_imag = bound_imag.sum(dim=0)

    # Try to retrieve value at position 0
    query_phase = base_phases[0]  # Use phase from position 0
    retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)

    # Compare to original value
    original = values[0]
    cos_sim = F.cosine_similarity(retrieved.unsqueeze(0), original.unsqueeze(0)).item()

    print(f"Cosine similarity between retrieved and original: {cos_sim:.4f}")
    print(f"(Perfect retrieval = 1.0, random = ~0)")

    # Try a few more positions
    print("\nRetrieval accuracy at different positions:")
    for pos in [0, 10, 30, 63]:
        query_phase = base_phases[pos]
        retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)
        original = values[pos]
        cos_sim = F.cosine_similarity(retrieved.unsqueeze(0), original.unsqueeze(0)).item()
        print(f"  Position {pos}: cos_sim = {cos_sim:.4f}")


if __name__ == "__main__":
    verify_base_orthogonality()
    test_pure_position_retrieval()
    test_ortho_init()
