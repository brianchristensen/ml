"""
Orthogonal Phasor WITHOUT Chunking

Hypothesis: Orthogonal phases prevent noise growth because
non-matching signals destructively interfere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class OrthoNoChunkPhasor(nn.Module):
    """Orthogonal phasor with full sequence cumsum (no chunking)."""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Random base phases - FIXED
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

        if seq_len > self.max_seq_len:
            extra = torch.randn(seq_len - self.max_seq_len, dim, device=x.device) * math.pi
            base_phase = torch.cat([self.base_phases, extra], dim=0)[:seq_len]
        else:
            base_phase = self.base_phases[:seq_len]

        base_phase = base_phase.unsqueeze(0)

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_modulation = self.key_mod(key) * self.mod_scale
        query_modulation = self.query_mod(query) * self.mod_scale

        key_phase = base_phase + key_modulation
        query_phase = base_phase + query_modulation

        # Bind
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # FULL cumsum - no chunking!
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind
        retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)

        # Scale by sqrt(dim) for stable magnitudes
        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out


class OrthoNoChunkBlock(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor = OrthoNoChunkPhasor(dim, max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.phasor(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OrthoNoChunkModel(nn.Module):
    def __init__(self, vocab_size, dim=256, num_layers=4, max_seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthoNoChunkBlock(dim, max_seq_len) for _ in range(num_layers)
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
    print("TEST: Long-Range Copy WITHOUT Chunking")
    print("=" * 70)

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

    for seq_len in [32, 64, 128, 256, 512]:
        print(f"\n--- seq_len = {seq_len} ---")

        model = OrthoNoChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=1024).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(150):
            x, y, _ = generate_batch(32, seq_len)
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
                x, _, first_tokens = generate_batch(32, seq_len)
                x = x.to(device)
                first_tokens = first_tokens.to(device)
                logits = model(x)
                pred = logits[:, -1].argmax(dim=-1)
                correct += (pred == first_tokens).sum().item()
                total += 32

        acc = correct / total * 100
        results[seq_len] = acc
        print(f"  Result: {acc:.1f}%")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for seq_len, acc in results.items():
        bar = "#" * int(acc / 2)
        print(f"  seq_len={seq_len:3d}: {acc:5.1f}% {bar}")

    return results


def test_extrapolation():
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

    print(f"Training on seq_len={train_len}...")
    model = OrthoNoChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(150):
        x, y, _ = generate_batch(32, train_len)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")

    print("\nTesting extrapolation:")
    model.eval()

    for test_len in [64, 128, 256, 512]:
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
        print(f"  seq_len={test_len:3d} ({status}): {acc:.1f}%")


def analyze_signal_magnitude():
    """Check if signal magnitude explodes without chunking."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Signal Magnitude at Different Positions")
    print("=" * 70)

    dim = 256
    model = OrthoNoChunkPhasor(dim, max_seq_len=1024).to(device)

    for seq_len in [64, 256, 512, 1024]:
        x = torch.randn(1, seq_len, dim, device=device)

        with torch.no_grad():
            out = model(x)

        print(f"seq_len={seq_len:4d}: input_mag={x.abs().mean():.4f}, output_mag={out.abs().mean():.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("ORTHOGONAL PHASOR WITHOUT CHUNKING")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    analyze_signal_magnitude()
    test_long_range_copy()
    test_extrapolation()
