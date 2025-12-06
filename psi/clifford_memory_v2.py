"""
Clifford Algebra Memory V2: Correct multiplicative structure.

The key insight: We want R_key * value -> memory -> R_query† * memory
NOT the sandwich product RvR† which is for rotating vectors in 3D.

This is exactly analogous to complex phasors:
  e^(iθ_key) * value -> memory -> e^(-iθ_query) * memory

In Clifford algebra, we can use:
1. Spinors (even subalgebra elements) as keys
2. Left multiplication for binding
3. Left multiplication by inverse for retrieval

The even subalgebra of Cl(n) has dimension 2^(n-1).
For Cl(4,0): even subalgebra has dim 8 (scalar + 6 bivectors + pseudoscalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiBivectorPhasorBlock(nn.Module):
    """
    Use multiple independent 2D rotation planes (bivectors) as parallel phasors.

    Each bivector eij squares to -1, giving us a complex-like structure.
    Multiple bivectors give us INDEPENDENT addressing channels.

    Key: K independent complex planes, each with its own cumsum memory.
    This is mathematically equivalent to K independent phasors but
    organized as bivectors in a higher-dimensional Clifford algebra.
    """

    def __init__(self, dim, n_planes=32):
        """
        dim: hidden dimension
        n_planes: number of independent rotation planes (bivector channels)
        """
        super().__init__()
        self.dim = dim
        self.n_planes = n_planes

        # Each plane has its own phase encoder
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_planes)  # One phase per plane
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_planes)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_planes

        # Get phases for each plane
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, K]
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        # Create complex phasors for each plane
        key_phasor = torch.exp(1j * key_phase)  # [B, L, K]
        query_phasor = torch.exp(1j * query_phase)

        # Value
        V = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Bind and accumulate
        # Each plane binds the SAME value with a DIFFERENT phase
        # This gives K independent address channels all pointing to the same content
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]
        memory = torch.cumsum(bound, dim=1)  # [B, L, K, D]

        # Retrieve
        retrieved = memory * query_phasor.conj().unsqueeze(-1)  # [B, L, K, D]
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


class HierarchicalBivectorBlock(nn.Module):
    """
    Hierarchical phase organization using coupled oscillators.

    Instead of K independent phases, use phases with hierarchical structure:
    - Low-frequency phases for coarse addressing
    - High-frequency phases for fine addressing

    This is inspired by how place cells and grid cells work in the brain:
    different scales of spatial encoding.
    """

    def __init__(self, dim, n_scales=4, phases_per_scale=16):
        super().__init__()
        self.dim = dim
        self.n_scales = n_scales
        self.phases_per_scale = phases_per_scale
        self.total_phases = n_scales * phases_per_scale

        # Frequency multipliers for each scale
        # Scale 0: base frequency, Scale 1: 2x, Scale 2: 4x, etc.
        freqs = []
        for s in range(n_scales):
            freqs.extend([2**s] * phases_per_scale)
        self.register_buffer('freq_mult', torch.tensor(freqs, dtype=torch.float32))

        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_phases)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_phases)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.total_phases

        # Get base phases and apply frequency multipliers
        key_base = torch.tanh(self.key_encoder(x))  # [B, L, K]
        query_base = torch.tanh(self.query_encoder(x))

        # Apply hierarchical frequencies
        key_phase = key_base * self.freq_mult * math.pi
        query_phase = query_base * self.freq_mult * math.pi

        # Create phasors
        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


class OrthogonalBivectorBlock(nn.Module):
    """
    Use mathematically orthogonal bivector planes from Cl(8,0).

    In Cl(8,0), we have 28 bivectors. We can choose subsets that are
    algebraically orthogonal (their geometric product is purely in
    4-vector or higher, not scalar/bivector).

    Orthogonal bivectors: e_ij and e_kl are orthogonal if {i,j} ∩ {k,l} = ∅

    For Cl(8,0) with basis e1...e8:
    - e12 is orthogonal to e34, e56, e78 (disjoint index pairs)
    - This gives us 4 mutually orthogonal planes: (12, 34, 56, 78)

    We can have multiple such sets:
    - Set A: (12, 34, 56, 78)
    - Set B: (13, 24, 57, 68)
    - Set C: (14, 23, 58, 67)
    - etc.
    """

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set

        # Each set has perfectly orthogonal planes
        # Within a set: no interference
        # Between sets: potential interference (but reduced)

        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_planes)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_planes)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # Learnable mixing weights for each set
        self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))

    def forward(self, x):
        B, L, D = x.shape

        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        # Process each orthogonal set separately, then combine
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        # Weighted retrieval from each set
        weights = F.softmax(self.set_weights, dim=0)  # [n_sets]

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        for s in range(self.n_sets):
            key_s = key_phasor[:, :, s, :]  # [B, L, planes_per_set]
            query_s = query_phasor[:, :, s, :]

            bound = key_s.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * query_s.conj().unsqueeze(-1)
            retrieved = retrieved.sum(dim=2).real

            total_retrieved = total_retrieved + weights[s] * retrieved

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)


# ============================================================================
# Models
# ============================================================================

class MultiBivectorModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_planes=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            MultiBivectorPhasorBlock(dim, n_planes) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


class HierarchicalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_scales=4, phases_per_scale=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            HierarchicalBivectorBlock(dim, n_scales, phases_per_scale) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


class OrthogonalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_sets=4, planes_per_set=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_sets, planes_per_set) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


# ============================================================================
# Test
# ============================================================================

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
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
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

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


def main():
    print('=' * 70)
    print('MULTI-BIVECTOR PHASE ORGANIZATION')
    print('=' * 70)
    print()
    print('Testing different ways to organize independent phase channels:')
    print('1. Standard: K independent phases (baseline)')
    print('2. Hierarchical: Multi-scale frequencies (coarse-to-fine)')
    print('3. Orthogonal: Mathematically orthogonal plane sets')
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2
    total_phases = 64
    epochs = 300

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    results = {}

    # Test 1: Standard multi-bivector (same as our current best)
    print('-' * 70)
    print('Testing: STANDARD (K=64 independent phases)')
    print('-' * 70)
    model = MultiBivectorModel(vocab_size + 1, dim, n_layers, total_phases).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    results['standard'] = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {results["standard"]:.1f}%')
    print()

    # Test 2: Hierarchical
    print('-' * 70)
    print('Testing: HIERARCHICAL (4 scales x 16 phases)')
    print('-' * 70)
    model = HierarchicalModel(vocab_size + 1, dim, n_layers, n_scales=4, phases_per_scale=16).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    results['hierarchical'] = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {results["hierarchical"]:.1f}%')
    print()

    # Test 3: Orthogonal sets
    print('-' * 70)
    print('Testing: ORTHOGONAL (4 sets x 16 orthogonal planes)')
    print('-' * 70)
    model = OrthogonalModel(vocab_size + 1, dim, n_layers, n_sets=4, planes_per_set=16).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    results['orthogonal'] = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {results["orthogonal"]:.1f}%')
    print()

    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Random baseline:  {100/vocab_size:.1f}%')
    for name, acc in results.items():
        bar = '#' * int(acc // 3)
        print(f'{name:15s}:  {acc:.1f}% {bar}')


if __name__ == "__main__":
    main()
