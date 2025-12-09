"""
Phasor with Holographic Reduced Representations (HRR)

The key insight: use content-based keys for binding/unbinding,
not position-based phases. This should enable associative recall
while keeping O(n) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class HolographicPhasor(nn.Module):
    """
    Phasor with true holographic key-value binding.

    Instead of: encode with phi_i, retrieve with phi_i (same phase)
    We do: encode with key_phase, retrieve with query_phase (content-based)
    """
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        # Content-based projections (like Q, K, V in attention)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        # Project to phase angles (content-addressable)
        self.to_key_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),  # Bound the phase
        )
        self.to_query_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        # Scale for phase (learned, like temperature)
        self.phase_scale = nn.Parameter(torch.ones(dim) * math.pi)

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Content-based projections
        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        # Compute phases from content (not position!)
        key_phase = self.to_key_phase(key) * self.phase_scale
        query_phase = self.to_query_phase(query) * self.phase_scale

        # Bind value to key using complex multiplication
        # bound = value * exp(i * key_phase)
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # Chunked accumulation (for extrapolation stability)
        memory_real, memory_imag = self._chunked_cumsum(bound_real, bound_imag, seq_len)

        # Retrieve using query phase (DIFFERENT from key phase!)
        # retrieved = memory * exp(-i * query_phase)  (conjugate for unbinding)
        retrieved_real = memory_real * torch.cos(query_phase) + memory_imag * torch.sin(query_phase)
        retrieved_imag = memory_imag * torch.cos(query_phase) - memory_real * torch.sin(query_phase)

        # Combine real and imaginary (or just use real)
        retrieved = retrieved_real  # The imaginary part could also be used

        # Output projection with residual
        out = self.to_out(retrieved)

        return x + out

    def _chunked_cumsum(self, real, imag, seq_len):
        """Chunked cumsum with normalization for stable extrapolation."""
        batch_size = real.shape[0]
        dim = real.shape[2]

        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            real = F.pad(real, (0, 0, 0, pad_len))
            imag = F.pad(imag, (0, 0, 0, pad_len))

        padded_len = real.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape for parallel processing
        real_chunked = real.view(batch_size, num_chunks, self.chunk_size, dim)
        imag_chunked = imag.view(batch_size, num_chunks, self.chunk_size, dim)

        # Cumsum within chunks
        mem_real = torch.cumsum(real_chunked, dim=2)
        mem_imag = torch.cumsum(imag_chunked, dim=2)

        # Normalize by position within chunk
        pos = torch.arange(1, self.chunk_size + 1, device=real.device, dtype=real.dtype)
        pos = pos.view(1, 1, -1, 1)
        mem_real = mem_real / pos
        mem_imag = mem_imag / pos

        # Reshape back
        mem_real = mem_real.view(batch_size, padded_len, dim)
        mem_imag = mem_imag.view(batch_size, padded_len, dim)

        if pad_len > 0:
            mem_real = mem_real[:, :seq_len]
            mem_imag = mem_imag[:, :seq_len]

        return mem_real, mem_imag


class HolographicPhasorBlock(nn.Module):
    """Full block with HRR + FFN."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.hrr = HolographicPhasor(dim, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.hrr(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HolographicPhasorModel(nn.Module):
    """Full model for testing."""
    def __init__(self, vocab_size, dim=64, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            HolographicPhasorBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


# =============================================================================
# Test: Can HRR Phasor do associative recall?
# =============================================================================

def test_long_range_copy():
    """The test that original Phasor failed: copy first token to last position."""
    print("=" * 60)
    print("TEST: Long-Range Copy (Associative Recall)")
    print("=" * 60)
    print("Pattern: First token determines last token")
    print("This requires remembering a specific token across many positions")
    print()

    vocab_size = 10
    seq_len = 64

    def generate_copy_data(n_samples=300):
        data = []
        for _ in range(n_samples):
            first_token = torch.randint(0, vocab_size, (1,)).item()
            middle = torch.randint(0, vocab_size, (seq_len - 2,)).tolist()
            x = [first_token] + middle + [0]
            y = middle + [0, first_token]
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_copy_data(300)
    test_data = generate_copy_data(50)

    # Train HRR Phasor
    print("Training Holographic Phasor...")
    model = HolographicPhasorModel(vocab_size, dim=64, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_data):.4f}")

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits[0, -1].argmax()
            target = y[0, -1]
            if pred == target:
                correct += 1
            total += 1

    print(f"\nHolographic Phasor: {correct}/{total} = {correct/total*100:.1f}% copy accuracy")
    print(f"(Random baseline: {100/vocab_size:.0f}%)")

    return correct / total


def test_key_value_retrieval():
    """More explicit test: store key-value pairs, retrieve by key."""
    print("\n" + "=" * 60)
    print("TEST: Explicit Key-Value Retrieval")
    print("=" * 60)
    print("Pattern: 'K1 V1 K2 V2 ... K1 ?' -> should predict V1")
    print()

    vocab_size = 20
    n_pairs = 4

    def generate_kv_data(n_samples=300):
        data = []
        for _ in range(n_samples):
            # Generate key-value pairs
            keys = torch.randperm(vocab_size // 2)[:n_pairs].tolist()
            values = torch.randperm(vocab_size // 2)[:n_pairs].tolist()
            values = [v + vocab_size // 2 for v in values]  # Values in upper half

            # Build sequence: K1 V1 K2 V2 K3 V3 K4 V4 K_query
            seq = []
            for k, v in zip(keys, values):
                seq.extend([k, v])

            # Query: repeat one of the keys
            query_idx = torch.randint(0, n_pairs, (1,)).item()
            query_key = keys[query_idx]
            expected_value = values[query_idx]

            seq.append(query_key)

            x = torch.tensor([seq[:-1] + [0]])  # Input
            y = torch.tensor([seq[1:] + [expected_value]])  # Target (last is the retrieved value)

            data.append((x, y, expected_value))
        return data

    train_data = generate_kv_data(400)
    test_data = generate_kv_data(100)

    print("Training Holographic Phasor on key-value retrieval...")
    model = HolographicPhasorModel(vocab_size, dim=64, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(150):
        total_loss = 0
        for x, y, _ in train_data:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/len(train_data):.4f}")

    # Test retrieval accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, expected in test_data:
            x = x.to(device)
            logits = model(x)
            pred = logits[0, -1].argmax().item()
            if pred == expected:
                correct += 1
            total += 1

    print(f"\nHolographic Phasor: {correct}/{total} = {correct/total*100:.1f}% retrieval accuracy")
    print(f"(Random baseline: {100/(vocab_size//2):.0f}%)")

    return correct / total


def compare_with_original_phasor():
    """Compare HRR Phasor vs original Phasor on long-range copy."""
    print("\n" + "=" * 60)
    print("COMPARISON: HRR Phasor vs Original Phasor")
    print("=" * 60)

    from psi import PSIBlock  # Original

    vocab_size = 10
    seq_len = 64

    def generate_copy_data(n_samples=300):
        data = []
        for _ in range(n_samples):
            first_token = torch.randint(0, vocab_size, (1,)).item()
            middle = torch.randint(0, vocab_size, (seq_len - 2,)).tolist()
            x = [first_token] + middle + [0]
            y = middle + [0, first_token]
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_copy_data(300)
    test_data = generate_copy_data(50)

    results = {}

    # Original Phasor
    class OriginalPhasorModel(nn.Module):
        def __init__(self, vocab_size, dim=64, num_layers=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, x):
            h = self.embed(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.norm(h))

    for name, ModelClass in [
        ("Original Phasor", OriginalPhasorModel),
        ("HRR Phasor", HolographicPhasorModel),
    ]:
        print(f"\nTraining {name}...")
        model = ModelClass(vocab_size, dim=64, num_layers=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(100):
            for x, y in train_data:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits[0, -1].argmax()
                if pred == y[0, -1]:
                    correct += 1

        acc = correct / len(test_data)
        results[name] = acc
        print(f"{name}: {acc*100:.1f}%")

    print(f"\nImprovement: {results['HRR Phasor']/max(results['Original Phasor'], 0.1):.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("HOLOGRAPHIC PHASOR: Testing Associative Recall")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    test_long_range_copy()
    test_key_value_retrieval()
    compare_with_original_phasor()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
If HRR Phasor succeeds where original failed, it means:
- Content-based key/query phases enable associative recall
- We can have O(n) complexity AND associative memory
- The architecture becomes much more useful for language

If it still fails, the cumsum averaging may be fundamentally
incompatible with precise retrieval.
""")
