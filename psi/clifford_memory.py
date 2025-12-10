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
    Clifford memory with D-dimensional phase addressing.

    Key insight: Use D phases (one per dimension) for BOTH content and position.
    This gives us D parallel "planes" that all run through a single [B, L, D] cumsum.

    Content addressing: learned D-dimensional phases from input (key/query)
    Positional addressing: fixed random D-dimensional phases per position

    The D dimensions act like D independent memory banks (bloom filter style),
    but computed in parallel via a single cumsum operation.

    n_heads: number of independent attention heads (each gets its own phase projection)
    """

    def __init__(self, dim, max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Content-based phase encoders: project to D phases (one per dimension)
        # Multiple heads give us multiple independent addressing patterns
        self.key_phase = nn.Linear(dim, dim)  # Output D phases
        self.query_phase = nn.Linear(dim, dim)
        nn.init.orthogonal_(self.key_phase.weight)
        nn.init.orthogonal_(self.query_phase.weight)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # FIXED random phases for positional addressing: [max_seq_len, D]
        # Each position gets D random phases - approximately orthogonal in high D
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Learnable gate: blend content vs positional retrieval
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, D = x.shape

        # === CONTENT ADDRESSING with D-dimensional phases ===
        key_phases = torch.tanh(self.key_phase(x)) * math.pi    # [B, L, D]
        query_phases = torch.tanh(self.query_phase(x)) * math.pi  # [B, L, D]

        V = self.to_value(x)  # [B, L, D]
        V = self.dropout(V)  # Dropout on values

        # Phasor binding with real arithmetic (faster than complex on GPU)
        cos_key = torch.cos(key_phases)
        sin_key = torch.sin(key_phases)
        cos_query = torch.cos(query_phases)
        sin_query = torch.sin(query_phases)

        # Bind: V * e^(i*key_phase)
        bound_real = V * cos_key
        bound_imag = V * sin_key

        # Cumsum memory
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind: mem * e^(-i*query_phase) = mem * (cos_q - i*sin_q)
        # Real part: mem_real*cos_q + mem_imag*sin_q
        content_retrieved = mem_real * cos_query + mem_imag * sin_query

        # === POSITIONAL ADDRESSING with fixed D-dimensional phases ===
        pos_phases = self.pos_phases[:L]  # [L, D]
        cos_pos = torch.cos(pos_phases)
        sin_pos = torch.sin(pos_phases)

        pos_bound_real = V * cos_pos
        pos_bound_imag = V * sin_pos

        pos_mem_real = torch.cumsum(pos_bound_real, dim=1)
        pos_mem_imag = torch.cumsum(pos_bound_imag, dim=1)

        pos_retrieved = pos_mem_real * cos_pos + pos_mem_imag * sin_pos

        # === COMBINE with learned gate ===
        gate = self.gate(x)  # [B, L, 1]
        combined = gate * content_retrieved + (1 - gate) * pos_retrieved

        # Normalize by sqrt(position)
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions).view(1, L, 1)

        # Dropout on output
        return x + self.dropout(self.to_out(combined / norm))

class OrthogonalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, max_seq_len=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, max_seq_len)
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
