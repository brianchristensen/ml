"""
Phasor with Learned Orthogonal Phases

Key insight: HRR works because random high-dim vectors are near-orthogonal.
We want to LEARN orthogonality so we don't need huge dimensions.

Approach: Add orthogonality regularization to push phases apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class OrthogonalPhasor(nn.Module):
    """
    Phasor that learns orthogonal key phases.

    The key insight: if key phases are orthogonal, then unbinding with
    a matching query will constructively interfere while non-matching
    queries destructively interfere.

    We enforce orthogonality by:
    1. Computing pairwise similarity of key phases within a sequence
    2. Adding a loss term that penalizes high similarity
    """
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        # Phase projections - key is that these should produce orthogonal outputs
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

        # Store orthogonality loss for training
        self.ortho_loss = 0.0

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_phase = self.to_key_phase(key) * self.phase_scale
        query_phase = self.to_query_phase(query) * self.phase_scale

        # Compute orthogonality loss on key phases
        # We want different positions to have orthogonal phases
        self.ortho_loss = self._compute_ortho_loss(key_phase)

        # Bind value to key phase
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

        # Unbind with query
        cos_q = torch.cos(qp)
        sin_q = torch.sin(qp)
        retrieved = mem_real * cos_q + mem_imag * sin_q

        retrieved = retrieved.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out

    def _compute_ortho_loss(self, phases):
        """
        Compute loss that encourages orthogonal phases.

        For phases to be "orthogonal" in the interference sense,
        we want: cos(phase_i - phase_j) ≈ 0 for i ≠ j

        This means phase differences should be near ±π/2.
        """
        batch_size, seq_len, dim = phases.shape

        # Sample positions to avoid O(n²) computation
        n_samples = min(32, seq_len)
        idx = torch.randperm(seq_len, device=phases.device)[:n_samples]
        sampled = phases[:, idx, :]  # (batch, n_samples, dim)

        # Compute pairwise phase differences
        # (batch, n_samples, 1, dim) - (batch, 1, n_samples, dim)
        diff = sampled.unsqueeze(2) - sampled.unsqueeze(1)  # (batch, n, n, dim)

        # We want cos(diff) ≈ 0, i.e., diff ≈ ±π/2
        # Loss = mean(cos(diff)²) for off-diagonal elements
        cos_diff = torch.cos(diff)

        # Mask diagonal (self-similarity is 1, that's fine)
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=phases.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, n, n, 1)

        # Mean squared cosine similarity (want this to be 0)
        ortho_loss = (cos_diff ** 2 * mask).sum() / (mask.sum() * dim)

        return ortho_loss


class OrthoPhasorBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor = OrthogonalPhasor(dim, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.phasor(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def ortho_loss(self):
        return self.phasor.ortho_loss


class OrthoPhasorModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthoPhasorBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    def get_ortho_loss(self):
        return sum(block.ortho_loss for block in self.blocks)


def test_with_ortho_loss():
    """Test long-range copy WITH orthogonality regularization."""
    print("=" * 60)
    print("TEST: Long-Range Copy with Orthogonality Loss")
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

    # Test different ortho_weight values
    for ortho_weight in [0.0, 0.1, 1.0, 10.0]:
        print(f"\n--- ortho_weight = {ortho_weight} ---")

        model = OrthoPhasorModel(vocab_size, dim=dim, num_layers=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(100):
            x, y, _ = generate_batch(32)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            ortho_loss = model.get_ortho_loss()

            loss = ce_loss + ortho_weight * ortho_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                print(f"  Epoch {epoch+1}: CE={ce_loss.item():.4f}, Ortho={ortho_loss.item():.4f}")

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


def test_ortho_plus_matching_query():
    """
    Even better: ensure query at position j learns to match key at position i
    when we want to retrieve token i at position j.

    For the copy task: query at last position should match key at first position.
    """
    print("\n" + "=" * 60)
    print("TEST: Orthogonality + Query-Key Alignment Loss")
    print("=" * 60)
    print("Additional loss: query[last] should align with key[first]")

    vocab_size = 10
    seq_len = 64
    dim = 64

    class AlignedOrthoPhasor(OrthogonalPhasor):
        def __init__(self, dim, chunk_size=64):
            super().__init__(dim, chunk_size)
            self.align_loss = 0.0

        def forward(self, x, compute_align_loss=False):
            batch_size, seq_len, dim = x.shape

            key = self.to_key(x)
            value = self.to_value(x)
            query = self.to_query(x)

            key_phase = self.to_key_phase(key) * self.phase_scale
            query_phase = self.to_query_phase(query) * self.phase_scale

            # Ortho loss: different positions should have orthogonal key phases
            self.ortho_loss = self._compute_ortho_loss(key_phase)

            # Alignment loss: query at last pos should match key at first pos
            if compute_align_loss:
                key_first = key_phase[:, 0, :]  # (batch, dim)
                query_last = query_phase[:, -1, :]  # (batch, dim)
                # Want cos(key_first - query_last) ≈ 1
                diff = key_first - query_last
                self.align_loss = (1 - torch.cos(diff).mean())
            else:
                self.align_loss = 0.0

            # Rest is same as before...
            bound_real = value * torch.cos(key_phase)
            bound_imag = value * torch.sin(key_phase)

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

            mem_real = torch.cumsum(br, dim=2)
            mem_imag = torch.cumsum(bi, dim=2)

            cos_q = torch.cos(qp)
            sin_q = torch.sin(qp)
            retrieved = mem_real * cos_q + mem_imag * sin_q

            retrieved = retrieved.view(batch_size, padded_len, dim)
            if pad_len > 0:
                retrieved = retrieved[:, :seq_len]

            retrieved = retrieved / math.sqrt(dim)

            out = self.to_out(retrieved)
            return x + out

    class AlignedModel(nn.Module):
        def __init__(self, vocab_size, dim=64, num_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                self._make_block(dim) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size)

        def _make_block(self, dim):
            block = OrthoPhasorBlock(dim)
            block.phasor = AlignedOrthoPhasor(dim)
            return block

        def forward(self, x, compute_align_loss=False):
            h = self.embed(x)
            for block in self.blocks:
                block.phasor.forward.__func__(block.phasor, block.norm1(h), compute_align_loss)
                h = block(h)
            return self.head(self.norm(h))

        def get_ortho_loss(self):
            return sum(block.phasor.ortho_loss for block in self.blocks)

        def get_align_loss(self):
            return sum(block.phasor.align_loss for block in self.blocks)

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

    model = AlignedModel(vocab_size, dim=dim, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ortho_weight = 1.0
    align_weight = 1.0

    model.train()
    for epoch in range(150):
        x, y, _ = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x, compute_align_loss=True)
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        ortho_loss = model.get_ortho_loss()
        align_loss = model.get_align_loss()

        loss = ce_loss + ortho_weight * ortho_loss + align_weight * align_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}: CE={ce_loss.item():.4f}, Ortho={ortho_loss.item():.4f}, Align={align_loss.item():.4f}")

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
    print(f"\nResult: {acc:.1f}% (random: 10%)")

    if acc > 50:
        print("SUCCESS! Orthogonality + alignment enables retrieval!")
    elif acc > 20:
        print("PARTIAL improvement")
    else:
        print("Still not working")


if __name__ == "__main__":
    test_with_ortho_loss()
    test_ortho_plus_matching_query()
