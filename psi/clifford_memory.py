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
    Hybrid Clifford memory with separate content and positional planes.

    Content planes: Learned phases from input for associative key->value retrieval
    Positional planes: Fixed random phases per position for exact positional recall

    The key insight from PhasorOpt: random phases in high dimensions are approximately
    orthogonal, enabling O(n) positional retrieval via interference patterns.

    Content addressing: query encodes "what content am I looking for?"
    Positional addressing: query encodes "what position am I looking for?"

    Both use cumsum for O(n) complexity, but:
    - Content uses learned phases (content-dependent)
    - Position uses fixed random phases (position-dependent only)
    """

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.pos_planes = pos_planes

        # Content-based phase encoders (linear + orthogonal init for speed)
        self.key_encoder = nn.Linear(dim, self.total_planes)
        nn.init.orthogonal_(self.key_encoder.weight)
        self.query_encoder = nn.Linear(dim, self.total_planes)
        nn.init.orthogonal_(self.query_encoder.weight)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # Learnable mixing weights for content sets
        self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))

        # FIXED random phases for positional addressing (like PhasorOpt)
        # Each position gets a unique random signature that's approximately orthogonal
        # to other positions in high dimensions
        pos_phases = torch.randn(max_seq_len, pos_planes) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Learnable weight for positional set (relative to content sets)
        self.pos_weight = nn.Parameter(torch.ones(1))

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

        # Process all content sets in parallel
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        weights = F.softmax(self.set_weights, dim=0)

        # Bind all sets at once: [B, L, n_sets, planes_per_set, D]
        bound_all = key_phasor.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)

        # Reshape for single cumsum: [B, L, n_sets * planes_per_set, D]
        bound_flat = bound_all.view(B, L, self.n_sets * self.planes_per_set, D)
        memory_flat = torch.cumsum(bound_flat, dim=1)
        memory_all = memory_flat.view(B, L, self.n_sets, self.planes_per_set, D)

        # Retrieve with conjugate queries: [B, L, n_sets, planes_per_set, D]
        retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)

        # Sum over planes, take real: [B, L, n_sets, D]
        retrieved_per_set = retrieved_all.sum(dim=3).real

        # Apply weights and sum over sets: [B, L, D]
        total_retrieved = (retrieved_per_set * weights.view(1, 1, -1, 1)).sum(dim=2)

        # Positional addressing with FIXED random phases (like PhasorOpt)
        # Each position has a unique random phase signature that's approximately
        # orthogonal to other positions in high dimensions
        pos_phase = self.pos_phases[:L]  # [L, pos_planes]

        # Bind: value * e^(i*phase) for each position
        # Using real arithmetic for efficiency (equivalent to complex phasor)
        V_real = V.real  # [B, L, D]
        cos_phase = torch.cos(pos_phase).unsqueeze(0)  # [1, L, pos_planes]
        sin_phase = torch.sin(pos_phase).unsqueeze(0)  # [1, L, pos_planes]

        # Bind value with positional phase: [B, L, pos_planes, D]
        bound_real = V_real.unsqueeze(2) * cos_phase.unsqueeze(-1)
        bound_imag = V_real.unsqueeze(2) * sin_phase.unsqueeze(-1)

        # Cumsum accumulates memory: [B, L, pos_planes, D]
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind: same phase retrieves original value at that position
        # retrieved = mem_real * cos + mem_imag * sin (conjugate unbind)
        retrieved_pos = (mem_real * cos_phase.unsqueeze(-1) +
                        mem_imag * sin_phase.unsqueeze(-1))
        retrieved_pos = retrieved_pos.sum(dim=2)  # [B, L, D] - sum over pos_planes

        # Gating between content and positional
        gate = self.content_pos_gate(x)  # [B, L, 1]
        pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos
        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

class OrthogonalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_orthogonal_sets=4, planes_per_set=16, pos_planes=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_orthogonal_sets, planes_per_set, pos_planes)
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
