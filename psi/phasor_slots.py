"""
Slot-Based Memory Phasor

Idea: Instead of cumsum averaging everything together, use a fixed number
of learned "slots" that tokens can write to and read from.

If num_slots is fixed (e.g., 64), this is O(n) memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SlotMemoryPhasor(nn.Module):
    """
    Phasor with slot-based memory for associative recall.

    Each token:
    1. Computes a key to decide which slot(s) to write to
    2. Writes its value to those slots (soft attention over slots)
    3. Computes a query to decide which slot(s) to read from
    4. Reads from those slots

    Memory is O(batch * num_slots * dim) = O(1) w.r.t. sequence length
    """
    def __init__(self, dim, num_slots=64, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.chunk_size = chunk_size

        # Learned slot addresses (like learned positional queries)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, dim) * 0.02)

        # Projections
        self.to_key = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)

        # Temperature for soft attention
        self.scale = nn.Parameter(torch.ones(1) * (dim ** -0.5))

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Project to key, query, value
        key = self.to_key(x)      # For writing: which slot to write to
        query = self.to_query(x)  # For reading: which slot to read from
        value = self.to_value(x)  # What to write

        # Pad to chunk size
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            key = F.pad(key, (0, 0, 0, pad_len))
            query = F.pad(query, (0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, pad_len))

        padded_len = key.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape for chunked processing
        key = key.view(batch_size, num_chunks, self.chunk_size, dim)
        query = query.view(batch_size, num_chunks, self.chunk_size, dim)
        value = value.view(batch_size, num_chunks, self.chunk_size, dim)

        # Compute write weights: which slot does each token write to?
        # (batch, chunks, chunk_size, dim) @ (num_slots, dim).T -> (batch, chunks, chunk_size, num_slots)
        write_scores = torch.einsum('bctd,sd->bcts', key, self.slot_keys) * self.scale
        write_weights = F.softmax(write_scores, dim=-1)  # Soft assignment to slots

        # Cumulative write to slots (causal within chunk)
        # For each position t, slot s accumulates: sum_{i<=t} write_weights[i,s] * value[i]
        # Shape: (batch, chunks, chunk_size, num_slots, dim)
        contributions = write_weights.unsqueeze(-1) * value.unsqueeze(-2)  # (b,c,t,s,d)
        memory = torch.cumsum(contributions, dim=2)  # Causal accumulation

        # Read from slots using query
        read_scores = torch.einsum('bctd,sd->bcts', query, self.slot_keys) * self.scale
        read_weights = F.softmax(read_scores, dim=-1)

        # Weighted read from memory
        # memory: (b,c,t,s,d), read_weights: (b,c,t,s)
        retrieved = torch.einsum('bcts,bctsd->bctd', read_weights, memory)

        # Reshape back
        retrieved = retrieved.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        # Output with residual
        out = self.to_out(retrieved)
        return x + out


class SlotPhasorBlock(nn.Module):
    """Full block with slot memory + FFN."""
    def __init__(self, dim, num_slots=64, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.slot_mem = SlotMemoryPhasor(dim, num_slots, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.slot_mem(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SlotPhasorModel(nn.Module):
    """Full model for testing."""
    def __init__(self, vocab_size, dim=64, num_layers=4, num_slots=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SlotPhasorBlock(dim, num_slots) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


# =============================================================================
# Test: Long-Range Copy with Slot Memory
# =============================================================================

def test_long_range_copy():
    """Test if slot memory enables associative recall."""
    print("=" * 60)
    print("TEST: Long-Range Copy with Slot Memory")
    print("=" * 60)
    print("Pattern: First token determines last token")
    print("num_slots = chunk_size = 64")
    print()

    vocab_size = 10
    seq_len = 64
    num_slots = 64  # As user suggested

    # Generate batched data
    def generate_batch(batch_size=64):
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

    # Create model
    print(f"Training Slot Memory Phasor (num_slots={num_slots})...")
    model = SlotPhasorModel(vocab_size, dim=64, num_layers=4, num_slots=num_slots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training
    model.train()
    for epoch in range(100):
        x, y, _ = generate_batch(64)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")

    # Test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(10):  # 10 test batches
            x, y, first_tokens = generate_batch(64)
            x = x.to(device)
            first_tokens = first_tokens.to(device)

            logits = model(x)
            pred = logits[:, -1].argmax(dim=-1)

            correct += (pred == first_tokens).sum().item()
            total += 64

    accuracy = correct / total * 100
    print(f"\nSlot Memory Phasor: {correct}/{total} = {accuracy:.1f}%")
    print(f"Random baseline: {100/vocab_size:.0f}%")

    if accuracy > 50:
        print("\nSUCCESS! Slot memory enables associative recall!")
    elif accuracy > 20:
        print("\nPARTIAL - Better than random but not reliable")
    else:
        print("\nFAILED - No better than random guessing")

    return accuracy


def test_key_value_retrieval():
    """More explicit KV test."""
    print("\n" + "=" * 60)
    print("TEST: Explicit Key-Value Retrieval")
    print("=" * 60)
    print("Pattern: K1 V1 K2 V2 K3 V3 K4 V4 K1 ? -> should predict V1")
    print()

    vocab_size = 20
    num_slots = 64
    n_pairs = 4

    def generate_batch(batch_size=64):
        batch_x = []
        batch_y = []
        expected_values = []

        for _ in range(batch_size):
            keys = torch.randperm(vocab_size // 2)[:n_pairs].tolist()
            values = (torch.randperm(vocab_size // 2)[:n_pairs] + vocab_size // 2).tolist()

            seq = []
            for k, v in zip(keys, values):
                seq.extend([k, v])

            query_idx = torch.randint(0, n_pairs, (1,)).item()
            query_key = keys[query_idx]
            expected_value = values[query_idx]

            seq.append(query_key)

            x = seq[:-1] + [0]
            y = seq[1:] + [expected_value]

            batch_x.append(x)
            batch_y.append(y)
            expected_values.append(expected_value)

        return (
            torch.tensor(batch_x),
            torch.tensor(batch_y),
            torch.tensor(expected_values)
        )

    print(f"Training Slot Memory Phasor on KV retrieval...")
    model = SlotPhasorModel(vocab_size, dim=64, num_layers=4, num_slots=num_slots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(150):
        x, y, _ = generate_batch(64)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")

    # Test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(10):
            x, _, expected = generate_batch(64)
            x = x.to(device)
            expected = expected.to(device)

            logits = model(x)
            pred = logits[:, -1].argmax(dim=-1)

            correct += (pred == expected).sum().item()
            total += 64

    accuracy = correct / total * 100
    print(f"\nSlot Memory Phasor: {correct}/{total} = {accuracy:.1f}%")
    print(f"Random baseline: {100/(vocab_size//2):.0f}%")

    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("SLOT-BASED MEMORY PHASOR")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    print("Hypothesis: Fixed slots with soft write/read attention")
    print("should enable O(n) associative recall")
    print()

    copy_acc = test_long_range_copy()
    kv_acc = test_key_value_retrieval()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Long-range copy: {copy_acc:.1f}%")
    print(f"KV retrieval: {kv_acc:.1f}%")
    print()

    if copy_acc > 50 or kv_acc > 50:
        print("Slot memory shows promise for associative recall!")
    else:
        print("""
Slot memory also failed. The issue is that cumsum within chunks
still averages contributions together. Each slot accumulates ALL
tokens that wrote to it, diluting specific values.

To retrieve a SPECIFIC value, you need either:
1. O(n^2) direct access (attention)
2. O(vocab) dedicated slots (one per possible value)
3. Sparse/hard assignment (breaks differentiability)
""")
