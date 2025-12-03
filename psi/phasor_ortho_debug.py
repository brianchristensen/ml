"""
Debug: Why does seq_len >= 128 fail?

Hypotheses:
1. Signal-to-noise ratio degrades with more tokens
2. Learning problem - need more epochs or different LR
3. Dimensionality too low for 128+ positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OrthoNoChunkPhasor(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

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

    def forward(self, x, return_internals=False):
        batch_size, seq_len, dim = x.shape

        base_phase = self.base_phases[:seq_len].unsqueeze(0)

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_modulation = self.key_mod(key) * self.mod_scale
        query_modulation = self.query_mod(query) * self.mod_scale

        key_phase = base_phase + key_modulation
        query_phase = base_phase + query_modulation

        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)
        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)

        if return_internals:
            return x + out, {
                'mem_real': mem_real,
                'mem_imag': mem_imag,
                'retrieved': retrieved,
                'key_phase': key_phase,
                'query_phase': query_phase,
                'value': value,
            }

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


def analyze_snr():
    """Analyze signal-to-noise ratio at different positions."""
    print("=" * 70)
    print("ANALYSIS 1: Signal-to-Noise Ratio")
    print("=" * 70)
    print("When retrieving position 0 at position N, what's the SNR?")
    print()

    dim = 256

    for seq_len in [32, 64, 128, 256, 512]:
        # Random base phases
        base_phases = torch.randn(seq_len, dim) * math.pi

        # Random values
        values = torch.randn(seq_len, dim)

        # Bind each value to its phase
        bound_real = values * torch.cos(base_phases)
        bound_imag = values * torch.sin(base_phases)

        # Memory at last position (sum of all)
        mem_real = bound_real.sum(dim=0)
        mem_imag = bound_imag.sum(dim=0)

        # Try to retrieve value at position 0
        query_phase = base_phases[0]
        retrieved = mem_real * torch.cos(query_phase) + mem_imag * torch.sin(query_phase)

        # The "signal" is the original value at position 0
        signal = values[0]

        # Measure similarity
        cos_sim = F.cosine_similarity(retrieved.unsqueeze(0), signal.unsqueeze(0)).item()

        # Theoretical SNR: signal = 1, noise = sqrt(N-1) random phases
        # Expected cos_sim ~ 1/sqrt(N)
        theoretical = 1.0 / math.sqrt(seq_len)

        print(f"seq_len={seq_len:3d}: cos_sim={cos_sim:.4f}, theoretical~{theoretical:.4f}")


def test_higher_dim():
    """Test if higher dimension helps longer sequences."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Does Higher Dimension Help?")
    print("=" * 70)

    vocab_size = 10
    seq_len = 128  # The failing case

    def generate_batch(batch_size):
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

    for dim in [256, 512, 1024]:
        print(f"\n--- dim = {dim} ---")

        model = OrthoNoChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(200):
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
        print(f"  Result: {acc:.1f}%")


def test_more_epochs():
    """Test if more training helps."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: More Training Epochs")
    print("=" * 70)

    vocab_size = 10
    seq_len = 128
    dim = 256

    def generate_batch(batch_size):
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

    model = OrthoNoChunkModel(vocab_size, dim=dim, num_layers=4, max_seq_len=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(500):
        x, y, _ = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            # Test accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(5):
                    x, _, first_tokens = generate_batch(32)
                    x = x.to(device)
                    first_tokens = first_tokens.to(device)
                    logits = model(x)
                    pred = logits[:, -1].argmax(dim=-1)
                    correct += (pred == first_tokens).sum().item()
                    total += 32

            acc = correct / total * 100
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}, acc = {acc:.1f}%")
            model.train()


if __name__ == "__main__":
    analyze_snr()
    test_higher_dim()
    test_more_epochs()
