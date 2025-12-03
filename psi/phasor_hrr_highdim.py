"""
High-Dimensional Holographic Phasor

Traditional HRR uses 10,000+ dimensions for reliable retrieval.
We were using 64. Let's test if higher dimensions enable associative recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HighDimHRRPhasor(nn.Module):
    """HRR Phasor with configurable (high) dimensionality."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        # Content-based projections
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        # Phase projections
        self.to_key_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.to_query_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        self.phase_scale = nn.Parameter(torch.ones(dim) * math.pi)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_phase = self.to_key_phase(key) * self.phase_scale
        query_phase = self.to_query_phase(query) * self.phase_scale

        # Bind: value * exp(i * key_phase)
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # Chunked cumsum (NO normalization - let interference work)
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            bound_real = F.pad(bound_real, (0, 0, 0, pad_len))
            bound_imag = F.pad(bound_imag, (0, 0, 0, pad_len))
            query_phase = F.pad(query_phase, (0, 0, 0, pad_len))

        padded_len = bound_real.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape for parallel processing
        br = bound_real.view(batch_size, num_chunks, self.chunk_size, dim)
        bi = bound_imag.view(batch_size, num_chunks, self.chunk_size, dim)
        qp = query_phase.view(batch_size, num_chunks, self.chunk_size, dim)

        # Cumsum WITHOUT dividing by position
        mem_real = torch.cumsum(br, dim=2)
        mem_imag = torch.cumsum(bi, dim=2)

        # Unbind with query phase
        cos_q = torch.cos(qp)
        sin_q = torch.sin(qp)
        retrieved_real = mem_real * cos_q + mem_imag * sin_q

        # Reshape back
        retrieved = retrieved_real.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        # Scale by sqrt(dim) to normalize interference
        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out


class HighDimHRRBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.hrr = HighDimHRRPhasor(dim, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.hrr(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HighDimHRRModel(nn.Module):
    def __init__(self, vocab_size, dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            HighDimHRRBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def test_with_dimension(dim, vocab_size=10, seq_len=64):
    """Test long-range copy with given dimension."""
    print(f"\nTesting dim={dim}...")

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

    model = HighDimHRRModel(vocab_size, dim=dim, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    model.train()
    for epoch in range(100):
        x, y, _ = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
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
    print(f"  Result: {correct}/{total} = {acc:.1f}%")
    return acc


if __name__ == "__main__":
    print("=" * 60)
    print("HIGH-DIMENSIONAL HRR PHASOR")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    print("Hypothesis: Higher dimensions enable better interference")
    print("Traditional HRR uses 10,000+ dims")
    print("Random baseline: 10%")
    print()

    results = {}
    for dim in [64, 128, 256, 512, 1024]:
        acc = test_with_dimension(dim)
        results[dim] = acc

    print("\n" + "=" * 60)
    print("SUMMARY: Dimension vs Accuracy")
    print("=" * 60)
    for dim, acc in results.items():
        bar = "#" * int(acc / 2)
        print(f"dim={dim:4d}: {acc:5.1f}% {bar}")

    print(f"\nRandom baseline: 10%")

    best_dim = max(results, key=results.get)
    if results[best_dim] > 50:
        print(f"\nSUCCESS! dim={best_dim} enables associative recall!")
    elif results[best_dim] > 20:
        print(f"\nPARTIAL improvement at dim={best_dim}")
    else:
        print(f"\nHigher dimension alone doesn't solve it")
