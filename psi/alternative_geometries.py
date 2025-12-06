"""
Alternative Geometric Structures for Content-Addressable Memory
Comparing: Phase, VSA, HRR, Walsh-Hadamard

All maintain O(n) parallel cumsum while providing different binding/unbinding mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 1. PHASE-BASED (Baseline - your current approach)
# ============================================================================

class PhasePhasorBlock(nn.Module):
    """Phase-based binding: value * exp(i*theta_key), retrieve with exp(-i*theta_query)"""
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Encode phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, K]
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        # Complex phasors
        key_phasor = torch.exp(1j * key_phase)  # [B, L, K]
        query_phasor = torch.exp(1j * query_phase)

        # Value
        V = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Bind: [B, L, K, D]
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)

        # Cumsum memory
        memory = torch.cumsum(bound, dim=1)

        # Retrieve with conjugate query
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# 2. VECTOR SYMBOLIC ARCHITECTURE (VSA) - Element-wise binding
# ============================================================================

class VSAPhasorBlock(nn.Module):
    """
    VSA binding: key * value (element-wise product)
    Unbinding: memory * key (self-inverse for unit vectors)

    Key insight: Unit-normalized keys are approximately self-inverse:
    k * k approx 1 (when ||k|| = 1)
    """
    def __init__(self, dim, hd_dim=256):
        super().__init__()
        self.dim = dim
        self.hd_dim = hd_dim

        # Project to high-dimensional space
        self.to_hd = nn.Linear(dim, hd_dim)
        self.from_hd = nn.Linear(hd_dim, dim)

        # Key encoder (output normalized)
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, hd_dim),
            nn.Tanh()
        )

        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape

        # Map to HD space
        hd_values = self.to_hd(x)  # [B, L, HD]
        hd_keys = self.key_encoder(x)  # [B, L, HD]

        # Normalize keys to unit vectors
        hd_keys = F.normalize(hd_keys, dim=-1)

        # Bind via element-wise product
        bound = hd_values * hd_keys  # [B, L, HD]

        # Cumsum for memory
        memory = torch.cumsum(bound, dim=1)  # [B, L, HD]

        # Unbind (keys are approximately self-inverse when unit norm)
        retrieved_hd = memory * hd_keys  # [B, L, HD]

        # Project back
        retrieved = self.from_hd(retrieved_hd)  # [B, L, D]

        # Normalize by position
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# 3. HOLOGRAPHIC REDUCED REPRESENTATIONS (HRR) - Circular convolution
# ============================================================================

class HRRPhasorBlock(nn.Module):
    """
    HRR binding: circular convolution (via FFT)
    Unbinding: circular correlation (conjugate in frequency domain)

    F(x conv y) = F(x) * F(y)
    F(x corr y) = F(x) * conj(F(y))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.key_encoder = nn.Linear(dim, dim)
        self.value_encoder = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def circular_conv(self, x, y):
        """Circular convolution via FFT"""
        X = torch.fft.rfft(x, dim=-1)
        Y = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(X * Y, n=self.dim, dim=-1)

    def circular_corr(self, x, y):
        """Circular correlation (approximate unbinding)"""
        X = torch.fft.rfft(x, dim=-1)
        Y = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(X * Y.conj(), n=self.dim, dim=-1)

    def forward(self, x):
        B, L, D = x.shape

        keys = F.normalize(self.key_encoder(x), dim=-1)  # [B, L, D]
        values = self.value_encoder(x)  # [B, L, D]

        # Bind via circular convolution
        bound = self.circular_conv(keys, values)  # [B, L, D]

        # Bundle via cumsum
        memory = torch.cumsum(bound, dim=1)  # [B, L, D]

        # Unbind via circular correlation
        retrieved = self.circular_corr(memory, keys)  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# 4. WALSH-HADAMARD CODES - Binary orthogonal binding
# ============================================================================

class WalshHadamardPhasorBlock(nn.Module):
    """
    Walsh-Hadamard binding: value * code where code in {-1, +1}
    Unbinding: same operation (codes are self-inverse)

    Uses learned soft codes that are pushed toward binary via tanh.
    """
    def __init__(self, dim):
        super().__init__()
        # Ensure dim is power of 2 for fast WHT (or pad)
        self.dim = dim
        self.padded_dim = 2 ** math.ceil(math.log2(dim))

        # Learnable code selection
        self.code_selector = nn.Linear(dim, self.padded_dim)
        self.value_proj = nn.Linear(dim, self.padded_dim)
        self.out_proj = nn.Linear(self.padded_dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # Temperature for soft-to-hard codes
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, L, D = x.shape

        # Generate soft binary codes {-1, +1} via tanh
        # Higher temperature = harder codes
        codes = torch.tanh(self.code_selector(x) * self.temp)  # [B, L, padded_dim]

        # Values
        values = self.value_proj(x)  # [B, L, padded_dim]

        # Bind (element-wise, codes are approximately self-inverse)
        bound = values * codes  # [B, L, padded_dim]

        # Cumsum memory
        memory = torch.cumsum(bound, dim=1)

        # Unbind (codes are self-inverse: c * c = 1 for c in {-1, +1})
        retrieved = memory * codes  # [B, L, padded_dim]

        # Project back
        retrieved = self.out_proj(retrieved)  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# 5. QUATERNION BINDING - 4D rotations
# ============================================================================

class QuaternionPhasorBlock(nn.Module):
    """
    Quaternion binding: q = a + bi + cj + dk
    Non-commutative multiplication preserves more structure.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for quaternions"
        self.dim = dim
        self.n_quat = dim // 4

        self.key_encoder = nn.Linear(dim, dim)
        self.value_encoder = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def quat_mul(self, q1, q2):
        """Quaternion multiplication: [B, L, N, 4] x [B, L, N, 4]"""
        r1, i1, j1, k1 = q1.unbind(dim=-1)
        r2, i2, j2, k2 = q2.unbind(dim=-1)

        r = r1*r2 - i1*i2 - j1*j2 - k1*k2
        i = r1*i2 + i1*r2 + j1*k2 - k1*j2
        j = r1*j2 - i1*k2 + j1*r2 + k1*i2
        k = r1*k2 + i1*j2 - j1*i2 + k1*r2

        return torch.stack([r, i, j, k], dim=-1)

    def quat_conj(self, q):
        """Quaternion conjugate"""
        r, i, j, k = q.unbind(dim=-1)
        return torch.stack([r, -i, -j, -k], dim=-1)

    def forward(self, x):
        B, L, D = x.shape

        # Reshape to quaternions [B, L, N, 4]
        keys = self.key_encoder(x).reshape(B, L, self.n_quat, 4)
        keys = F.normalize(keys, dim=-1)  # Unit quaternions
        values = self.value_encoder(x).reshape(B, L, self.n_quat, 4)

        # Bind via quaternion multiplication
        bound = self.quat_mul(values, keys)  # [B, L, N, 4]

        # Cumsum
        memory = torch.cumsum(bound, dim=1)

        # Unbind via conjugate (q * q* = |q|^2, so for unit q: q * q* = 1)
        retrieved = self.quat_mul(memory, self.quat_conj(keys))

        # Flatten back
        retrieved = retrieved.reshape(B, L, D)

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# Model wrappers
# ============================================================================

def create_model(geometry, vocab_size, dim=64, n_layers=2):
    """Create a model with the specified geometry."""

    if geometry == 'phase':
        BlockClass = PhasePhasorBlock
    elif geometry == 'vsa':
        BlockClass = VSAPhasorBlock
    elif geometry == 'hrr':
        BlockClass = HRRPhasorBlock
    elif geometry == 'walsh':
        BlockClass = WalshHadamardPhasorBlock
    elif geometry == 'quaternion':
        BlockClass = QuaternionPhasorBlock
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    class GeometryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([BlockClass(dim) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
            self.norm_out = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, x):
            h = self.embed(x)
            for norm, block in zip(self.norms, self.blocks):
                h = block(norm(h))
            return self.head(self.norm_out(h))

    return GeometryModel()


# ============================================================================
# Multi-Query Associative Recall Task
# ============================================================================

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    """
    Store n_pairs key-value pairs, then query n_queries of them.
    """
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + n_queries * 2
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    query_positions = []

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]

        pos = 0
        # Store pairs
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        # Queries
        query_indices = np.random.choice(n_pairs, n_queries, replace=False)
        for qi, query_idx in enumerate(query_indices):
            data[b, pos] = QUERY_TOKEN
            pos += 1
            query_k, query_v = pairs[query_idx]
            data[b, pos] = query_k
            targets[b, pos] = query_v
            if b == 0:
                query_positions.append(pos)
            pos += 1

    return data, targets, query_positions


def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=300, lr=1e-3):
    """Train model and return best accuracy."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        data, targets, positions = generate_multi_query_recall(64, n_pairs, n_queries, vocab_size, device)
        logits = model(data)
        loss = sum(criterion(logits[:, pos, :vocab_size], targets[:, pos]) for pos in positions) / len(positions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item()
                             for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc
                print(f'    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%')

    return best_acc


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    print('=' * 70)
    print('ALTERNATIVE GEOMETRIES FOR CONTENT-ADDRESSABLE MEMORY')
    print('=' * 70)
    print()
    print('All methods use O(n) cumsum for memory accumulation.')
    print('Comparing different binding/unbinding mechanisms.')
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2
    epochs = 300

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    geometries = ['phase', 'vsa', 'hrr', 'walsh', 'quaternion']
    results = {}

    for geom in geometries:
        print('-' * 70)
        print(f'Testing: {geom.upper()}')
        print('-' * 70)

        try:
            model = create_model(geom, vocab_size + 1, dim, n_layers)
            params = sum(p.numel() for p in model.parameters())
            print(f'Parameters: {params:,}')

            acc = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
            results[geom] = (acc, params)
            print(f'Best accuracy: {acc:.1f}%')
        except Exception as e:
            print(f'Error: {e}')
            results[geom] = (0, 0)
        print()

    # Summary
    print('=' * 70)
    print('SUMMARY: Geometry Comparison')
    print('=' * 70)
    print()
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()
    print(f'{"Geometry":<15} {"Accuracy":>10} {"Params":>12}')
    print('-' * 40)

    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for geom, (acc, params) in sorted_results:
        bar = '#' * int(acc // 5)
        print(f'{geom:<15} {acc:>9.1f}% {params:>11,}  {bar}')

    print()

    # Winner analysis
    winner = sorted_results[0]
    baseline_acc = results.get('phase', (0, 0))[0]

    if winner[0] == 'phase':
        print('VERDICT: Phase encoding remains best!')
        print('Complex exponentials provide optimal interference-based retrieval.')
    else:
        improvement = winner[1][0] - baseline_acc
        print(f'VERDICT: {winner[0].upper()} beats phase by {improvement:.1f}%!')
        print(f'Consider switching from phase to {winner[0]} geometry.')


if __name__ == "__main__":
    main()
