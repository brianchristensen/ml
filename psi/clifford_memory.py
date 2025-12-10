"""
Clifford Algebra Memory

Linear attention's problem isn't the compression - it's the addressing.

Standard linear attention does:
memory += key * value  (cumsum)
output = query @ memory

The keys and queries are just learned linear projections - they don't have any special
structure that prevents collisions. So you get interference and the memory "fills up"
quickly.

Clifford addressing fixes this:
memory += phasor(key) * value  (cumsum)
output = phasor(query)† @ memory

The phasor binding with cross-bank products creates an exponentially large address
space (combinatorial in the number of banks), even though the memory itself is still
linearly compressed.

We want R_key * value -> memory -> R_query† * memory
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

        # === RESONANCE-BASED RETRIEVAL (Optimized) ===
        # Flatten all planes: [B, L, total_planes]
        total_planes = self.n_sets * self.planes_per_set
        key_phasor_flat = key_phasor.view(B, L, total_planes)
        query_phasor_flat = query_phasor.view(B, L, total_planes)

        # Store: bind value with key phasor for each plane
        bound = key_phasor_flat.unsqueeze(-1) * V.unsqueeze(2)  # [B, L, total_planes, D]

        # Accumulate memory
        memory = torch.cumsum(bound, dim=1)  # [B, L, total_planes, D]

        # Raw retrieval per plane
        retrieved_per_plane = memory * query_phasor_flat.conj().unsqueeze(-1)  # [B, L, total_planes, D]

        # === FAST RESONANCE via phase coherence ===
        # Instead of comparing retrieval vectors, measure phase alignment directly
        # When phases align: key_phasor * query_phasor.conj() ≈ 1 (real, positive)
        # When misaligned: key_phasor * query_phasor.conj() ≈ random phase

        # Phase alignment per plane: [B, L, total_planes]
        phase_alignment = (key_phasor_flat * query_phasor_flat.conj()).real

        # Resonance = mean alignment across planes (1 = perfect, 0 = random, -1 = anti)
        resonance = phase_alignment.mean(dim=-1, keepdim=True)  # [B, L, 1]

        # Sum retrieval across planes (interference pattern)
        summed_retrieval = retrieved_per_plane.sum(dim=2).real  # [B, L, D]

        # Amplify by resonance: high coherence = strong signal
        # Scale factor: resonance in [-1, 1], we want positive amplification
        resonance_gain = F.softplus(resonance + 0.5)  # [B, L, 1]

        total_retrieved = summed_retrieval * resonance_gain / total_planes

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
