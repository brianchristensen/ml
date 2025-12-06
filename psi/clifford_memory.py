"""
Clifford Algebra Memory: Using geometric algebra for phase-based addressing.

Key idea: Instead of K independent complex phases, use bivector rotations
in Clifford algebra. Bivector planes are algebraically orthogonal -
no interference between planes by construction.

We implement Cl(4,0) which gives us 6 independent bivector planes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Cl(4,0) Implementation - 16-dimensional algebra
# ============================================================================

# Basis: 1, e1, e2, e3, e4, e12, e13, e14, e23, e24, e34, e123, e124, e134, e234, e1234
# Indices: 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,   11,   12,   13,   14,    15

# We focus on the bivector subspace (indices 5-10) for rotation planes
BIVECTOR_INDICES = [5, 6, 7, 8, 9, 10]  # e12, e13, e14, e23, e24, e34
N_BIVECTORS = 6


def geometric_product_cl4(a, b):
    """
    Geometric product in Cl(4,0).
    a, b: [..., 16] tensors representing multivectors
    Returns: [..., 16] tensor

    This is a simplified version focusing on what we need for rotors.
    Full implementation would have 16x16 multiplication table.
    """
    # For efficiency, we'll implement the key operations we need:
    # 1. Bivector exponential (for creating rotors)
    # 2. Rotor-vector-rotor_reverse (for rotation)
    # 3. Addition (trivial)

    # For now, implement via explicit multiplication table
    # This is the expensive but correct way

    result = torch.zeros_like(a)

    # Multiplication rules for Cl(4,0):
    # ei * ei = 1 (positive signature)
    # ei * ej = -ej * ei for i != j

    # This would be a large lookup table - let's use a more elegant approach
    # by working in the even subalgebra (rotors) for our use case

    raise NotImplementedError("Full geometric product is complex - using specialized rotor ops")


def bivector_exp(B):
    """
    Exponential of a bivector: exp(B) where B = sum_i theta_i * e_ij

    B: [..., 6] tensor of bivector coefficients (for e12, e13, e14, e23, e24, e34)
    Returns: [..., 16] tensor representing the rotor

    For a pure bivector B with B² = -|B|², we have:
    exp(B) = cos(|B|) + B/|B| * sin(|B|)

    But for multiple bivector components, we need to be more careful.
    In Cl(4,0), bivectors don't generally commute, so exp(B1+B2) ≠ exp(B1)*exp(B2)

    For simplicity, we'll use a first-order approximation for small angles,
    or factorize into commuting parts.
    """
    # [..., 6] -> [..., 16]
    batch_shape = B.shape[:-1]

    # For small angles, exp(B) ≈ 1 + B + B²/2 + ...
    # Let's use the exact formula for each bivector plane independently
    # This is valid when we're rotating in orthogonal planes

    # Magnitude of each bivector component
    theta = B  # [..., 6] - each is an angle in its plane

    # Initialize rotor as scalar 1
    rotor = torch.zeros(*batch_shape, 16, device=B.device, dtype=B.dtype)
    rotor[..., 0] = 1.0  # Scalar part

    # For independent planes, we can multiply the rotors
    # exp(θ₁₂ e₁₂) * exp(θ₁₃ e₁₃) * ... = exp(θ₁₂ e₁₂ + θ₁₃ e₁₃ + ...) when planes commute

    # Actually for Cl(4,0), the bivectors e12, e34 commute (share no indices)
    # But e12, e13 don't commute (share e1)

    # Let's use the simple approximation: treat as if all commute
    # rotor = cos(|θ|) + sin(|θ|)/|θ| * B

    theta_mag = theta.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [..., 1]

    cos_theta = torch.cos(theta_mag)  # [..., 1]
    sinc_theta = torch.sin(theta_mag) / theta_mag  # [..., 1]

    rotor[..., 0] = cos_theta.squeeze(-1)  # Scalar part
    rotor[..., 5:11] = sinc_theta * theta  # Bivector parts

    return rotor


def rotor_reverse(R):
    """
    Reverse of a rotor: R† = R̃
    For a rotor R = a + B (scalar + bivector), R̃ = a - B

    R: [..., 16] multivector
    Returns: [..., 16] reversed multivector
    """
    R_rev = R.clone()
    # Bivectors change sign under reversal
    R_rev[..., 5:11] = -R[..., 5:11]
    # Trivectors change sign
    R_rev[..., 11:15] = -R[..., 11:15]
    # Scalar and pseudoscalar (4-vector) unchanged
    return R_rev


def apply_rotor_to_vector(R, v, R_rev=None):
    """
    Apply rotor to vector: v' = R v R†

    R: [..., 16] rotor
    v: [..., 4] vector (in e1, e2, e3, e4 basis)
    R_rev: [..., 16] reverse of R (computed if not provided)

    Returns: [..., 4] rotated vector
    """
    if R_rev is None:
        R_rev = rotor_reverse(R)

    # For efficiency, we compute R v R† directly for vector v
    # This is a sandwhich product

    # A vector v = v1*e1 + v2*e2 + v3*e3 + v4*e4 sits in indices 1-4
    # We need to compute the full product R * v * R†

    # For a rotor R = cos(θ) + sin(θ)*B where B is a unit bivector,
    # R v R† rotates v in the plane defined by B

    # Let's implement this for the case where R is in even subalgebra
    # (scalar + bivector + pseudoscalar)

    # Extract components
    r0 = R[..., 0:1]  # Scalar
    r_biv = R[..., 5:11]  # Bivectors [e12, e13, e14, e23, e24, e34]

    # For small rotations, v' ≈ v + 2*(B × v) where × is the commutator
    # For exact rotation, we need the full sandwich product

    # Let's use a linearized version for efficiency:
    # This captures the first-order effect of each bivector on each vector component

    # Bivector action on vectors:
    # e12 rotates in e1-e2 plane: e1 -> e2, e2 -> -e1
    # e13 rotates in e1-e3 plane: e1 -> e3, e3 -> -e1
    # etc.

    v1, v2, v3, v4 = v[..., 0:1], v[..., 1:2], v[..., 2:3], v[..., 3:4]

    # Each bivector contributes a rotation
    # For rotor R = exp(θ*B/2), the rotation is by angle θ
    # The bivector coefficient in R is sin(θ/2) * B_unit

    # For small angle: sin(θ/2) ≈ θ/2, so rotation ≈ θ
    # For exact: rotation angle = 2 * arcsin(|bivector_coeffs|)

    theta12, theta13, theta14 = r_biv[..., 0:1], r_biv[..., 1:2], r_biv[..., 2:3]
    theta23, theta24, theta34 = r_biv[..., 3:4], r_biv[..., 4:5], r_biv[..., 5:6]

    # Apply rotations (using sin/cos for exact rotation)
    # For rotor with bivector B, rotation angle is 2*arctan2(|B|, scalar)

    # Simplified: use small angle approximation
    # Full rotation matrix would be more complex

    # Rotation contributions (linearized):
    # e12 plane: v1' += -2*θ12*v2, v2' += 2*θ12*v1
    # e13 plane: v1' += -2*θ13*v3, v3' += 2*θ13*v1
    # etc.

    # For exact treatment with scalar part:
    s = r0  # scalar = cos(θ/2)

    # Full rotation formula: v' = (s² - |B|²)v + 2(B·v)B + 2s(B×v)
    # where B·v is the inner product and B×v is related to commutator

    # For now, use approximate formula that works well for small-medium angles
    v1_new = v1 * (s**2 - theta12**2 - theta13**2 - theta14**2) + \
             v2 * 2 * theta12 * s + \
             v3 * 2 * theta13 * s + \
             v4 * 2 * theta14 * s

    v2_new = v2 * (s**2 - theta12**2 - theta23**2 - theta24**2) + \
             v1 * (-2 * theta12 * s) + \
             v3 * 2 * theta23 * s + \
             v4 * 2 * theta24 * s

    v3_new = v3 * (s**2 - theta13**2 - theta23**2 - theta34**2) + \
             v1 * (-2 * theta13 * s) + \
             v2 * (-2 * theta23 * s) + \
             v4 * 2 * theta34 * s

    v4_new = v4 * (s**2 - theta14**2 - theta24**2 - theta34**2) + \
             v1 * (-2 * theta14 * s) + \
             v2 * (-2 * theta24 * s) + \
             v3 * (-2 * theta34 * s)

    return torch.cat([v1_new, v2_new, v3_new, v4_new], dim=-1)


# ============================================================================
# Simpler approach: Use bivector phases directly without full GA
# ============================================================================

class BivectorPhasorBlock(nn.Module):
    """
    Use 6 bivector "phases" as independent addressing channels.

    Instead of K complex phases (each 2D rotation), we use 6 bivectors
    which give 6 orthogonal rotation planes in 4D space.

    Key insight: We don't need the full Clifford algebra machinery.
    We just need to ensure our 6 "phases" don't interfere.

    We represent values in 4D (e1, e2, e3, e4) and rotate them.
    """

    def __init__(self, dim, n_4d_blocks=16):
        """
        dim: hidden dimension
        n_4d_blocks: how many independent 4D spaces to use (dim = 4 * n_4d_blocks)
        """
        super().__init__()
        self.dim = dim
        self.n_4d_blocks = n_4d_blocks
        assert dim % 4 == 0, "dim must be divisible by 4 for 4D blocks"
        self.n_4d_blocks = dim // 4

        # Encode content to 6 bivector angles per 4D block
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_4d_blocks * 6)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_4d_blocks * 6)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_4d_blocks

        # Get bivector angles for keys and queries
        key_angles = torch.tanh(self.key_encoder(x)) * (math.pi / 2)  # [B, L, K*6]
        query_angles = torch.tanh(self.query_encoder(x)) * (math.pi / 2)  # [B, L, K*6]

        key_angles = key_angles.view(B, L, K, 6)
        query_angles = query_angles.view(B, L, K, 6)

        # Get values and reshape to K 4D blocks
        value = self.to_value(x).view(B, L, K, 4)  # [B, L, K, 4]

        # Create rotors from bivector angles
        key_rotors = bivector_exp(key_angles)  # [B, L, K, 16]
        query_rotors = bivector_exp(query_angles)

        # Apply key rotor to values: v' = R_key * v * R_key†
        rotated_value = apply_rotor_to_vector(key_rotors, value)  # [B, L, K, 4]

        # Cumsum in the rotated space
        memory = torch.cumsum(rotated_value, dim=1)  # [B, L, K, 4]

        # Apply query rotor (inverse) to retrieve
        query_rev = rotor_reverse(query_rotors)
        retrieved = apply_rotor_to_vector(query_rev, memory, query_rotors)  # [B, L, K, 4]

        # Reshape back to [B, L, D]
        retrieved = retrieved.view(B, L, D)

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)


# ============================================================================
# Even simpler: Quaternion pairs (two independent rotation planes)
# ============================================================================

class QuaternionPairPhasorBlock(nn.Module):
    """
    Use pairs of quaternions for independent 3D rotations.

    A quaternion q = w + xi + yj + zk represents a 3D rotation.
    We can use multiple independent quaternion spaces.

    Key property: Two quaternion rotations in independent spaces don't interfere.
    """

    def __init__(self, dim, n_quat_pairs=16):
        super().__init__()
        self.dim = dim
        # Each quaternion pair uses 8 dimensions (2 quaternions x 4)
        # We'll use dim/4 quaternions total
        self.n_quats = dim // 4

        # Encode to quaternion parameters (axis-angle representation)
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_quats * 4)  # 4 params per quaternion
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_quats * 4)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_quats

        # Get quaternion parameters
        key_params = self.key_encoder(x).view(B, L, K, 4)  # [B, L, K, 4]
        query_params = self.query_encoder(x).view(B, L, K, 4)

        # Normalize to unit quaternions
        key_q = F.normalize(key_params, dim=-1)
        query_q = F.normalize(query_params, dim=-1)

        # Get values as quaternions (pure quaternion: w=0, xyz=value)
        value = self.to_value(x).view(B, L, K, 4)

        # Quaternion rotation: q * v * q†
        # For pure quaternion v and unit quaternion q:
        # rotated = q * v * conj(q)

        rotated = self.quat_rotate(key_q, value)  # [B, L, K, 4]

        # Cumsum
        memory = torch.cumsum(rotated, dim=1)

        # Inverse rotation with query
        query_conj = self.quat_conjugate(query_q)
        retrieved = self.quat_rotate(query_conj, memory)

        # Reshape
        retrieved = retrieved.view(B, L, D)

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)

        return x + self.to_out(retrieved / norm)

    def quat_conjugate(self, q):
        """Quaternion conjugate: (w, x, y, z) -> (w, -x, -y, -z)"""
        return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)

    def quat_multiply(self, q1, q2):
        """Hamilton product of quaternions."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)

    def quat_rotate(self, q, v):
        """Rotate vector v by quaternion q: q * v * q†"""
        q_conj = self.quat_conjugate(q)
        return self.quat_multiply(self.quat_multiply(q, v), q_conj)


# ============================================================================
# Models
# ============================================================================

class CliffordMemoryModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2):
        super().__init__()
        # Ensure dim is divisible by 4
        dim = (dim // 4) * 4
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            BivectorPhasorBlock(dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


class QuaternionMemoryModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2):
        super().__init__()
        dim = (dim // 4) * 4
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            QuaternionPairPhasorBlock(dim) for _ in range(n_layers)
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
    print('CLIFFORD/QUATERNION MEMORY vs COMPLEX PHASOR')
    print('=' * 70)
    print()
    print('Hypothesis: Higher-dimensional rotation algebras provide')
    print('more independent addressing channels without interference.')
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

    # Test Quaternion-based memory
    print('-' * 70)
    print('Testing: QUATERNION MEMORY')
    print('-' * 70)

    model = QuaternionMemoryModel(vocab_size + 1, dim, n_layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')

    quat_acc = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {quat_acc:.1f}%')
    print()

    # Compare with standard complex phasor
    print('-' * 70)
    print('Testing: STANDARD COMPLEX PHASOR (baseline)')
    print('-' * 70)

    from geometric_spaces import FlatTorusPhasorBlock

    class ComplexPhasorModel(nn.Module):
        def __init__(self, vocab_size, dim=64, n_layers=2, n_oscillators=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                FlatTorusPhasorBlock(dim, n_oscillators) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
            self.norm_out = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, x):
            h = self.embed(x)
            for norm, block in zip(self.norms, self.blocks):
                h = block(norm(h))
            return self.head(self.norm_out(h))

    model = ComplexPhasorModel(vocab_size + 1, dim, n_layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')

    complex_acc = train_and_eval(model, vocab_size, n_pairs, n_queries, epochs)
    print(f'Best accuracy: {complex_acc:.1f}%')
    print()

    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Random baseline:     {100/vocab_size:.1f}%')
    print(f'Complex Phasor:      {complex_acc:.1f}%')
    print(f'Quaternion Memory:   {quat_acc:.1f}%')
    print()

    if quat_acc > complex_acc + 2:
        print('VERDICT: Quaternion rotations provide better addressing!')
    elif complex_acc > quat_acc + 2:
        print('VERDICT: Complex phasors still better (quaternion overhead?)')
    else:
        print('VERDICT: Comparable performance')


if __name__ == "__main__":
    main()
