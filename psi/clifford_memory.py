"""
Clifford Algebra Memory

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
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    Option B: Dedicated positional phase plane
    - One set uses fixed sinusoidal phases based on position
    - Provides temporal/positional addressing alongside content addressing
    """

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4,
                 use_positional_plane=False, max_len=None, pos_planes=16):
        # Note: max_len is deprecated and ignored - positional phases computed on-the-fly
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.use_positional_plane = use_positional_plane
        self.pos_planes = pos_planes

        # Content-based phase encoders
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

        # Learnable mixing weights for content sets
        self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))

        # Positional phase plane (Option B)
        if use_positional_plane:
            # Store frequencies for on-the-fly computation (supports any sequence length)
            freqs = torch.exp(torch.arange(0, pos_planes).float() *
                            (-math.log(10000.0) / pos_planes))
            self.register_buffer('pos_freqs', freqs)

            # Learnable weight for positional set (relative to content sets)
            self.pos_weight = nn.Parameter(torch.ones(1))

            # Nonlinear mixing: content and position interact through gating
            # Gate determines how much position influences the final address
            self.content_pos_gate = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1] to blend content vs position
            )

    def forward(self, x):
        B, L, D = x.shape

        # Content-based phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        # Process content sets
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        weights = F.softmax(self.set_weights, dim=0)

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        for s in range(self.n_sets):
            key_s = key_phasor[:, :, s, :]
            query_s = query_phasor[:, :, s, :]

            bound = key_s.unsqueeze(-1) * V.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * query_s.conj().unsqueeze(-1)
            retrieved = retrieved.sum(dim=2).real

            total_retrieved = total_retrieved + weights[s] * retrieved

        # Compute positional phases on-the-fly for any sequence length
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        pos_phase = pos * self.pos_freqs * 2 * math.pi  # [L, pos_planes]
        pos_phasor = torch.exp(1j * pos_phase)  # [L, pos_planes]

        # Expand for batch
        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)  # [B, L, pos_planes]
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)

        # Bind and retrieve with positional phases
        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, pos_planes, D]
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real  # [B, L, D]

        # Nonlinear gating: how much does position matter at each timestep?
        gate = self.content_pos_gate(x)  # [B, L, 1]

        # Weighted position contribution
        pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos

        # Gate blends content-retrieval with position-retrieval
        # High gate = more content, low gate = more position
        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

class OrthogonalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_orthogonal_sets=4, planes_per_set=16, max_len=512, pos_planes=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_orthogonal_sets, planes_per_set, max_len, pos_planes)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))
