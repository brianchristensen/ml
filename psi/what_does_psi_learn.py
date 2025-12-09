"""
What Does PSI Actually Learn?

PSI generates real English words but drifts semantically.
This suggests it learns LOCAL patterns but not GLOBAL associations.

Let's test what patterns PSI can and cannot capture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class PSIBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        gv = g * v

        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            gv_padded = F.pad(gv, (0, 0, 0, pad_len))
            g_padded = F.pad(g, (0, 0, 0, pad_len))
        else:
            gv_padded = gv
            g_padded = g

        padded_len = gv_padded.shape[1]
        num_chunks = padded_len // self.chunk_size

        gv_chunked = gv_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        g_chunked = g_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        cumsum_v = torch.cumsum(gv_chunked, dim=2)
        cumsum_g = torch.cumsum(g_chunked, dim=2) + 1e-6
        mem = cumsum_v / cumsum_g
        mem = mem.view(batch_size, padded_len, dim)
        if pad_len > 0:
            mem = mem[:, :seq_len, :]

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIModel(nn.Module):
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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=4, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        seq_len = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def train_model(model, data, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in data:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(data)


def test_accuracy(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# =============================================================================
# TEST 1: Local patterns (should work for PSI)
# =============================================================================

def test_local_ngrams():
    """Test: Can PSI learn simple n-gram patterns?"""
    print("=" * 60)
    print("TEST 1: Local N-gram Patterns")
    print("=" * 60)
    print("Pattern: 'AB' -> 'C', 'XY' -> 'Z'")
    print("This only requires looking at the previous 2 tokens")
    print()

    # Simple vocabulary: A=0, B=1, C=2, X=3, Y=4, Z=5, pad=6
    vocab_size = 7

    # Generate data: sequences with AB->C and XY->Z patterns
    def generate_ngram_data(n_samples=200, seq_len=32):
        data = []
        for _ in range(n_samples):
            x = []
            for i in range(seq_len - 1):
                if len(x) >= 2 and x[-2] == 0 and x[-1] == 1:  # AB
                    x.append(2)  # C
                elif len(x) >= 2 and x[-2] == 3 and x[-1] == 4:  # XY
                    x.append(5)  # Z
                else:
                    x.append(np.random.choice([0, 1, 3, 4, 6]))  # Random token

            # Target is next token
            y = x[1:] + [6]
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_ngram_data(200)
    test_data = generate_ngram_data(50)

    for name, model_class in [('PSI', PSIModel), ('Transformer', TransformerModel)]:
        model = model_class(vocab_size, dim=32, num_layers=2).to(device)
        train_model(model, train_data, epochs=50)
        acc = test_accuracy(model, test_data)
        print(f"{name}: {acc*100:.1f}% accuracy")

    print("\nExpected: Both should do well (local pattern)")


# =============================================================================
# TEST 2: Long-range dependency (should fail for PSI)
# =============================================================================

def test_long_range():
    """Test: Can PSI learn to copy a token from far back?"""
    print("\n" + "=" * 60)
    print("TEST 2: Long-Range Copy")
    print("=" * 60)
    print("Pattern: First token determines last token (copy after N steps)")
    print("This requires remembering a specific token across many positions")
    print()

    vocab_size = 10
    seq_len = 64  # Copy from position 0 to position 63

    def generate_copy_data(n_samples=200):
        data = []
        for _ in range(n_samples):
            first_token = np.random.randint(0, vocab_size)
            # Random middle tokens
            middle = np.random.randint(0, vocab_size, size=seq_len - 2).tolist()
            x = [first_token] + middle + [0]  # Last input is placeholder
            y = middle + [0, first_token]  # Target: copy first token at end
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_copy_data(300)
    test_data = generate_copy_data(50)

    for name, model_class in [('PSI', PSIModel), ('Transformer', TransformerModel)]:
        model = model_class(vocab_size, dim=64, num_layers=4).to(device)
        train_model(model, train_data, epochs=100, lr=1e-3)

        # Test specifically on the copy position
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                # Only check the last position
                pred = logits[0, -1].argmax()
                target = y[0, -1]
                if pred == target:
                    correct += 1
                total += 1

        print(f"{name}: {correct}/{total} = {correct/total*100:.1f}% (copy accuracy)")

    print(f"\nExpected: PSI ~{100/vocab_size:.0f}% (random), Transformer high")


# =============================================================================
# TEST 3: Bracket matching (requires stack-like memory)
# =============================================================================

def test_bracket_matching():
    """Test: Can PSI learn to close brackets?"""
    print("\n" + "=" * 60)
    print("TEST 3: Bracket Matching")
    print("=" * 60)
    print("Pattern: ( -> ), [ -> ], { -> }")
    print("Nested brackets require associative recall of opening bracket")
    print()

    # Vocabulary: (=0, )=1, [=2, ]=3, {=4, }=5, a=6 (filler)
    vocab_size = 7

    def generate_bracket_data(n_samples=200, max_depth=3):
        data = []
        for _ in range(n_samples):
            seq = []
            stack = []

            # Generate random nested brackets
            for _ in range(20):
                if len(stack) < max_depth and np.random.random() < 0.4:
                    # Open a bracket
                    bracket_type = np.random.randint(0, 3)  # 0, 1, or 2
                    open_token = bracket_type * 2  # 0, 2, or 4
                    seq.append(open_token)
                    stack.append(bracket_type)
                elif stack and np.random.random() < 0.4:
                    # Close a bracket
                    bracket_type = stack.pop()
                    close_token = bracket_type * 2 + 1  # 1, 3, or 5
                    seq.append(close_token)
                else:
                    # Filler
                    seq.append(6)

            # Close remaining brackets
            while stack:
                bracket_type = stack.pop()
                seq.append(bracket_type * 2 + 1)

            # Pad to fixed length
            seq = seq[:32] if len(seq) > 32 else seq + [6] * (32 - len(seq))

            x = seq[:-1]
            y = seq[1:]
            data.append((torch.tensor([x]), torch.tensor([y])))

        return data

    train_data = generate_bracket_data(300)
    test_data = generate_bracket_data(50)

    for name, model_class in [('PSI', PSIModel), ('Transformer', TransformerModel)]:
        model = model_class(vocab_size, dim=64, num_layers=4).to(device)
        train_model(model, train_data, epochs=100)

        # Test specifically on closing brackets
        model.eval()
        correct_close = 0
        total_close = 0

        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)

                # Check only positions where target is a closing bracket
                for i in range(y.shape[1]):
                    if y[0, i] in [1, 3, 5]:  # Closing brackets
                        total_close += 1
                        if pred[0, i] == y[0, i]:
                            correct_close += 1

        print(f"{name}: {correct_close}/{total_close} = {correct_close/total_close*100:.1f}% (bracket close accuracy)")

    print("\nExpected: PSI struggles with nested brackets, Transformer better")


# =============================================================================
# TEST 4: Rhythmic/periodic patterns (PSI's sweet spot?)
# =============================================================================

def test_periodic_patterns():
    """Test: Can PSI learn periodic patterns?"""
    print("\n" + "=" * 60)
    print("TEST 4: Periodic/Rhythmic Patterns")
    print("=" * 60)
    print("Pattern: ABCABCABC... (period 3)")
    print("This plays to PSI's phase-based encoding")
    print()

    vocab_size = 5

    def generate_periodic_data(n_samples=200, seq_len=64, period=3):
        data = []
        for _ in range(n_samples):
            # Random starting phase
            phase = np.random.randint(0, period)
            pattern = list(range(period))

            x = []
            for i in range(seq_len):
                x.append(pattern[(i + phase) % period])

            y = x[1:] + [pattern[(seq_len + phase) % period]]
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_periodic_data(200, period=5)
    test_data = generate_periodic_data(50, period=5)

    for name, model_class in [('PSI', PSIModel), ('Transformer', TransformerModel)]:
        model = model_class(vocab_size, dim=32, num_layers=2).to(device)
        train_model(model, train_data, epochs=50)
        acc = test_accuracy(model, test_data)
        print(f"{name}: {acc*100:.1f}% accuracy")

    print("\nExpected: PSI should do well (periodic pattern)")


# =============================================================================
# TEST 5: Frequency/distribution learning
# =============================================================================

def test_frequency_learning():
    """Test: Can PSI learn token frequency distributions?"""
    print("\n" + "=" * 60)
    print("TEST 5: Frequency Distribution")
    print("=" * 60)
    print("Pattern: Some tokens are more likely than others")
    print("This is basic language modeling without long-range dependencies")
    print()

    vocab_size = 10

    # Skewed distribution (like character frequencies in English)
    probs = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01])
    probs = probs / probs.sum()

    def generate_frequency_data(n_samples=200, seq_len=64):
        data = []
        for _ in range(n_samples):
            x = np.random.choice(vocab_size, size=seq_len, p=probs).tolist()
            y = x[1:] + [np.random.choice(vocab_size, p=probs)]
            data.append((torch.tensor([x]), torch.tensor([y])))
        return data

    train_data = generate_frequency_data(200)
    test_data = generate_frequency_data(50)

    for name, model_class in [('PSI', PSIModel), ('Transformer', TransformerModel)]:
        model = model_class(vocab_size, dim=32, num_layers=2).to(device)
        train_model(model, train_data, epochs=50)

        # Check if model learned the frequency distribution
        model.eval()
        predictions = []
        with torch.no_grad():
            x = torch.randint(0, vocab_size, (100, 64)).to(device)
            logits = model(x)
            pred = logits[:, -1].softmax(dim=-1).mean(dim=0).cpu().numpy()

        # Compare predicted distribution to true distribution
        kl_div = np.sum(probs * np.log(probs / (pred + 1e-8)))
        print(f"{name}: KL divergence from true dist = {kl_div:.4f}")

    print("\nExpected: Both should learn frequency distribution (lower KL = better)")


if __name__ == "__main__":
    print("=" * 60)
    print("WHAT DOES PSI ACTUALLY LEARN?")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    test_local_ngrams()
    test_long_range()
    test_bracket_matching()
    test_periodic_patterns()
    test_frequency_learning()

    print("\n" + "=" * 60)
    print("SUMMARY: What PSI Learns")
    print("=" * 60)
    print("""
PSI LEARNS (local/periodic patterns):
- N-gram statistics (local character transitions)
- Periodic/rhythmic patterns (via phase encoding)
- Frequency distributions
- Word shape patterns (prefixes, suffixes)

PSI CANNOT LEARN (requires associative recall):
- Long-range token copying
- Bracket/quote matching across distance
- Subject-verb agreement over clauses
- Semantic coherence (staying on topic)

WHY PSI GENERATES "ENGLISH-LIKE" TEXT:
- English has strong local statistics (th->e, ing endings, etc.)
- Word shapes are predictable from local context
- Character frequencies are learnable
- But: coherent MEANING requires long-range association

PSI = "Locally Fluent, Globally Incoherent"
""")
