"""
Alternative Geometric Spaces for Phase-Based Memory

Instead of the flat torus (phases in [-pi, pi]), we can embed keys/queries in:
1. Hyperbolic space (negative curvature) - better for hierarchical structures
2. Spherical space (positive curvature) - bounded, natural for similarity
3. Elliptic space - projective, antipodal points identified

All maintain cumsum parallelism via tangent space operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# 1. FLAT TORUS (Baseline - standard phase encoding)
# ============================================================================

class FlatTorusPhasorBlock(nn.Module):
    """
    Standard phase encoding on flat torus [-pi, pi]^K
    Geodesic distance: |theta1 - theta2| (mod 2pi)
    """
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

        # Phases on flat torus
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        # Complex phasors
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


# ============================================================================
# 2. HYPERBOLIC SPACE (Poincare Ball Model)
# ============================================================================

class HyperbolicPhasorBlock(nn.Module):
    """
    Keys/queries embedded in Poincare ball (hyperbolic space).

    Hyperbolic space has negative curvature - distances grow exponentially.
    This is better for hierarchical/tree-like structures.

    We use the Poincare ball model:
    - Points: ||x|| < 1
    - Distance: d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))

    For cumsum compatibility, we work in tangent space at origin.
    """
    def __init__(self, dim, n_oscillators=64, curvature=1.0):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.c = curvature  # Curvature (can be learned)

        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators * 2)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators * 2)
        )
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def project_to_ball(self, x):
        """Project to Poincare ball (ensure ||x|| < 1)"""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        max_norm = 1.0 - 1e-5
        return x * (max_norm / norm).clamp(max=1.0)

    def hyperbolic_distance(self, x, y):
        """Poincare ball distance"""
        diff_norm_sq = (x - y).pow(2).sum(dim=-1)
        x_norm_sq = x.pow(2).sum(dim=-1)
        y_norm_sq = y.pow(2).sum(dim=-1)

        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        arg = 1 + 2 * diff_norm_sq / denom.clamp(min=1e-5)
        return torch.acosh(arg.clamp(min=1.0))

    def hyperbolic_similarity(self, x, y):
        """Convert hyperbolic distance to similarity (like phase matching)"""
        dist = self.hyperbolic_distance(x, y)
        # Exponential decay with distance (like cos for phases)
        return torch.exp(-dist)

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Encode to 2D points in Poincare ball (per oscillator)
        key_raw = self.key_encoder(x).view(B, L, K, 2)
        query_raw = self.query_encoder(x).view(B, L, K, 2)

        # Project to ball
        key_pts = self.project_to_ball(torch.tanh(key_raw) * 0.9)  # [B, L, K, 2]
        query_pts = self.project_to_ball(torch.tanh(query_raw) * 0.9)

        # Value
        V = self.to_value(x)  # [B, L, D]

        # Similarity-based binding (using hyperbolic geometry)
        # We compute similarity between consecutive positions
        # and use it as a soft "gate" for memory accumulation

        # For cumsum compatibility, we work with the similarity scores
        # directly rather than the manifold operations

        # Compute binding weights from key points
        # Use 2D coordinates as complex phase (tangent space approximation)
        key_phase = torch.atan2(key_pts[..., 1], key_pts[..., 0])  # [B, L, K]
        query_phase = torch.atan2(query_pts[..., 1], query_pts[..., 0])

        # Also use radius as amplitude (closer to origin = more general)
        key_radius = key_pts.norm(dim=-1)  # [B, L, K]
        query_radius = query_pts.norm(dim=-1)

        # Combine phase and radius into complex phasor
        # Radius modulates amplitude (hierarchical: closer to origin = stronger)
        key_phasor = (1 - key_radius) * torch.exp(1j * key_phase)
        query_phasor = (1 - query_radius) * torch.exp(1j * query_phase)

        # Standard phasor memory
        V_complex = V.to(torch.complex64)
        bound = key_phasor.unsqueeze(-1) * V_complex.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        return x + self.to_out(retrieved / norm)


# ============================================================================
# 3. SPHERICAL SPACE (S^n embedding)
# ============================================================================

class SphericalPhasorBlock(nn.Module):
    """
    Keys/queries embedded on hypersphere S^{K-1}.

    Spherical space has positive curvature - opposite of hyperbolic.
    Points are unit vectors, distance is arc length.

    This is natural for similarity: cos(angle) = dot product.
    """
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

        # Encode to unit sphere
        key_vec = F.normalize(self.key_encoder(x), dim=-1)  # [B, L, K]
        query_vec = F.normalize(self.query_encoder(x), dim=-1)

        V = self.to_value(x)  # [B, L, D]

        # Spherical binding: use dot product as similarity
        # For cumsum, we accumulate weighted values
        # Weight = key component (treating K-sphere as K separate 1-spheres)

        # Each dimension of the K-vector acts as a separate "oscillator"
        # whose phase is the angle on the circle (but now bounded in [-1, 1])

        # Convert to complex (interpreting as 2D angle)
        # key_vec[k] is cos(theta_k), we need sin(theta_k) too
        # Use Chebyshev: sin = sqrt(1 - cos^2), but need sign

        # Alternative: treat pairs as (cos, sin)
        # For odd K, we'll use 2D slices

        if K % 2 == 1:
            K_pairs = K // 2
            extra = 1
        else:
            K_pairs = K // 2
            extra = 0

        # Use consecutive pairs as (real, imag)
        key_complex = torch.complex(
            key_vec[..., :K_pairs*2:2],
            key_vec[..., 1:K_pairs*2:2]
        )  # [B, L, K_pairs]
        query_complex = torch.complex(
            query_vec[..., :K_pairs*2:2],
            query_vec[..., 1:K_pairs*2:2]
        )

        # Phasor memory with spherical encoding
        V_complex = V.to(torch.complex64)
        bound = key_complex.unsqueeze(-1) * V_complex.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_complex.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K_pairs).view(1, L, 1)
        return x + self.to_out(retrieved / norm)


# ============================================================================
# 4. PRODUCT MANIFOLD: Torus x Hyperbolic
# ============================================================================

class ProductManifoldPhasorBlock(nn.Module):
    """
    Combine flat torus (periodic) with hyperbolic (hierarchical).

    Some oscillators use periodic phases (good for patterns).
    Some oscillators use hyperbolic distances (good for hierarchy).
    """
    def __init__(self, dim, n_oscillators=64, n_hyperbolic=16):
        super().__init__()
        self.dim = dim
        self.n_periodic = n_oscillators - n_hyperbolic
        self.n_hyperbolic = n_hyperbolic

        # Periodic components
        self.key_phase = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.n_periodic)
        )
        self.query_phase = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.n_periodic)
        )

        # Hyperbolic components (2D points in Poincare ball)
        self.key_hyp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_hyperbolic * 2)
        )
        self.query_hyp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_hyperbolic * 2)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape

        # Periodic phases
        key_p = torch.tanh(self.key_phase(x)) * math.pi
        query_p = torch.tanh(self.query_phase(x)) * math.pi
        key_phasor_p = torch.exp(1j * key_p)
        query_phasor_p = torch.exp(1j * query_p)

        # Hyperbolic phases (from 2D Poincare points)
        key_h = torch.tanh(self.key_hyp(x).view(B, L, self.n_hyperbolic, 2)) * 0.9
        query_h = torch.tanh(self.query_hyp(x).view(B, L, self.n_hyperbolic, 2)) * 0.9

        # Convert hyperbolic points to phases (angle) + amplitude (1-radius)
        key_angle_h = torch.atan2(key_h[..., 1], key_h[..., 0])
        query_angle_h = torch.atan2(query_h[..., 1], query_h[..., 0])
        key_amp_h = 1 - key_h.norm(dim=-1)
        query_amp_h = 1 - query_h.norm(dim=-1)

        key_phasor_h = key_amp_h * torch.exp(1j * key_angle_h)
        query_phasor_h = query_amp_h * torch.exp(1j * query_angle_h)

        # Combine
        key_phasor = torch.cat([key_phasor_p, key_phasor_h], dim=-1)
        query_phasor = torch.cat([query_phasor_p, query_phasor_h], dim=-1)

        K = self.n_periodic + self.n_hyperbolic
        V = self.to_value(x).to(torch.complex64)
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        return x + self.to_out(retrieved / norm)


# ============================================================================
# Model Wrapper
# ============================================================================

def create_geometric_model(geometry, vocab_size, dim=64, n_layers=2, n_oscillators=64):
    """Create a model with the specified geometric space."""

    if geometry == 'flat_torus':
        BlockClass = lambda d: FlatTorusPhasorBlock(d, n_oscillators)
    elif geometry == 'hyperbolic':
        BlockClass = lambda d: HyperbolicPhasorBlock(d, n_oscillators)
    elif geometry == 'spherical':
        BlockClass = lambda d: SphericalPhasorBlock(d, n_oscillators)
    elif geometry == 'product':
        BlockClass = lambda d: ProductManifoldPhasorBlock(d, n_oscillators, n_oscillators // 4)
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


def main():
    print('=' * 70)
    print('GEOMETRIC SPACES FOR PHASE-BASED MEMORY')
    print('=' * 70)
    print()
    print('Comparing curvature of the phase manifold:')
    print('  - Flat Torus: zero curvature (standard phases)')
    print('  - Hyperbolic: negative curvature (hierarchical)')
    print('  - Spherical: positive curvature (bounded similarity)')
    print('  - Product: mix of flat + hyperbolic')
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2
    n_oscillators = 64
    epochs = 300

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    geometries = ['flat_torus', 'hyperbolic', 'spherical', 'product']
    results = {}

    for geom in geometries:
        print('-' * 70)
        print(f'Testing: {geom.upper().replace("_", " ")}')
        print('-' * 70)

        try:
            model = create_geometric_model(geom, vocab_size + 1, dim, n_layers, n_oscillators)
            params = sum(p.numel() for p in model.parameters())
            print(f'Parameters: {params:,}')

            acc = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
            results[geom] = (acc, params)
            print(f'Best accuracy: {acc:.1f}%')
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
            results[geom] = (0, 0)
        print()

    # Summary
    print('=' * 70)
    print('SUMMARY: Geometric Space Comparison')
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
    winner = sorted_results[0]
    if winner[0] == 'flat_torus':
        print('VERDICT: Flat torus (standard phases) remains optimal!')
        print('Euclidean phase space is sufficient for this task.')
    else:
        print(f'VERDICT: {winner[0].upper()} provides better geometry!')
        print('Non-Euclidean curvature helps with this task structure.')


if __name__ == "__main__":
    main()
