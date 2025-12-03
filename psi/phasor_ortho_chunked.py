"""
Orthogonal Phasor with Cross-Chunk Memory

Key idea:
- Random orthogonal phases for associative retrieval
- Carry holographic memory state across chunks
- But normalize/stabilize within each chunk

The memory at position i contains ALL tokens 0..i, not just within-chunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OrthoCrossChunkPhasor(nn.Module):
    """
    Orthogonal phasor with memory that spans across chunks.

    Key insight: The holographic memory (sum of bound values) should
    accumulate across the full sequence. We just need to stabilize
    the magnitudes without losing the cross-chunk information.
    """
    def __init__(self, dim, max_seq_len=1024, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        # Random orthogonal base phases for each position
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        self.key_mod = nn.Linear(dim, dim)
        self.query_mod = nn.Linear(dim, dim)
        self.mod_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        base_phase = self.base_phases[:seq_len].unsqueeze(0)

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_modulation = self.key_mod(key) * self.mod_scale
        query_modulation = self.query_mod(query) * self.mod_scale

        key_phase = base_phase + key_modulation
        query_phase = base_phase + query_modulation

        # Bind values to key phases
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # FULL sequence cumsum (cross-chunk memory)
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Normalize by sqrt of position to prevent magnitude growth
        # This is gentler than chunk-reset
        positions = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype)
        positions = positions.view(1, -1, 1)
        norm_factor = torch.sqrt(positions)

        mem_real = mem_real / norm_factor
        mem_imag = mem_imag / norm_factor

        # Unbind with query phase
        retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)

        # Scale by sqrt(dim)
        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out


class OrthoCrossChunkBlock(nn.Module):
    def __init__(self, dim, max_seq_len=1024, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor = OrthoCrossChunkPhasor(dim, max_seq_len, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.phasor(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OrthoCrossChunkModel(nn.Module):
    def __init__(self, vocab_size, dim=256, num_layers=4, max_seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthoCrossChunkBlock(dim, max_seq_len) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def test_long_range_copy():
    print("=" * 70)
    print("TEST: Long-Range Copy with Cross-Chunk Memory")
    print("=" * 70)
    print("Using sqrt(position) normalization instead of chunk reset")
    print()

    vocab_size = 10
    dim = 256

    def generate_batch(batch_size, seq_len):
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

    results = {}

    for seq_len in [64, 128, 256, 512]:
        print(f"\n--- seq_len = {seq_len} ---")

        model = OrthoCrossChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=1024).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(300):
            x, y, _ = generate_batch(32, seq_len)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                # Test
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for _ in range(5):
                        x, _, first_tokens = generate_batch(32, seq_len)
                        x = x.to(device)
                        first_tokens = first_tokens.to(device)
                        logits = model(x)
                        pred = logits[:, -1].argmax(dim=-1)
                        correct += (pred == first_tokens).sum().item()
                        total += 32

                acc = correct / total * 100
                print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}, acc = {acc:.1f}%")
                model.train()

                if acc >= 95:
                    results[seq_len] = (epoch + 1, acc)
                    break
        else:
            # Final test
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(10):
                    x, _, first_tokens = generate_batch(32, seq_len)
                    x = x.to(device)
                    first_tokens = first_tokens.to(device)
                    logits = model(x)
                    pred = logits[:, -1].argmax(dim=-1)
                    correct += (pred == first_tokens).sum().item()
                    total += 32

            acc = correct / total * 100
            results[seq_len] = (300, acc)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for seq_len, (epochs, acc) in results.items():
        print(f"  seq_len={seq_len:3d}: {acc:.1f}% in {epochs} epochs")

    return results


def test_extrapolation():
    """Train on short, test on long."""
    print("\n" + "=" * 70)
    print("TEST: Extrapolation (Train on 64, Test on Longer)")
    print("=" * 70)

    vocab_size = 10
    dim = 256
    train_len = 64

    def generate_batch(batch_size, seq_len):
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

    print(f"Training on seq_len={train_len} for 200 epochs...")
    model = OrthoCrossChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(200):
        x, y, _ = generate_batch(32, train_len)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\nTesting extrapolation:")
    model.eval()

    for test_len in [64, 128, 256, 512, 1024]:
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(10):
                x, _, first_tokens = generate_batch(32, test_len)
                x = x.to(device)
                first_tokens = first_tokens.to(device)
                logits = model(x)
                pred = logits[:, -1].argmax(dim=-1)
                correct += (pred == first_tokens).sum().item()
                total += 32

        acc = correct / total * 100
        status = "TRAIN" if test_len == train_len else "EXTRAP"
        print(f"  seq_len={test_len:4d} ({status}): {acc:.1f}%")


def analyze_signal_at_positions():
    """Check if signal degrades with position."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Signal Magnitude at Different Positions")
    print("=" * 70)

    dim = 256

    for seq_len in [64, 256, 1024]:
        # Create a simple test: can we retrieve position 0?
        base_phases = torch.randn(seq_len, dim) * math.pi
        values = torch.randn(seq_len, dim)

        # Bind
        bound_real = values * torch.cos(base_phases)
        bound_imag = values * torch.sin(base_phases)

        # Cumsum
        mem_real = torch.cumsum(bound_real, dim=0)
        mem_imag = torch.cumsum(bound_imag, dim=0)

        # Normalize by sqrt(position)
        positions = torch.arange(1, seq_len + 1).float().unsqueeze(1)
        mem_real = mem_real / torch.sqrt(positions)
        mem_imag = mem_imag / torch.sqrt(positions)

        # Retrieve position 0 from last position
        query_phase = base_phases[0]
        retrieved = mem_real[-1] * torch.cos(query_phase) + mem_imag[-1] * torch.sin(query_phase)

        # Compare to original
        original = values[0]
        cos_sim = F.cosine_similarity(retrieved.unsqueeze(0), original.unsqueeze(0)).item()

        print(f"seq_len={seq_len:4d}: cos_sim(retrieved, original) = {cos_sim:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("ORTHOGONAL PHASOR WITH CROSS-CHUNK MEMORY")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    analyze_signal_at_positions()
    test_long_range_copy()
    test_extrapolation()
