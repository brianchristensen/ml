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


def decayed_cumsum(x, retention=None):
    """
    Compute cumulative sum along sequence dimension.

    Note: retention parameter is deprecated and ignored.
    Always uses efficient torch.cumsum.
    """
    return torch.cumsum(x, dim=1)


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

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16):
        # Note: LTM, cross-bank binding, surprise gating, and positional planes are always enabled
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
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

        # MIRAS-inspired learnable retention gates (per set)
        # Initialized to ~0.9999 retention (effectively no decay) via sigmoid(9.2)
        # This ensures we use the fast cumsum path by default
        self.retention_logits = nn.Parameter(torch.ones(n_orthogonal_sets) * 9.2)
        # Positional memory retention
        self.pos_retention_logit = nn.Parameter(torch.tensor(9.2))

        # Positional phase plane - provides temporal/positional addressing
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

        # Long-Term Memory: Titans-style surprise-gated memory
        # Key insight from Titans (Google, 2025):
        # - Surprise = prediction error, NOT a learned gate
        # - surprise_t = ||M_{t-1}(k_t) - v_t||^2  (what memory predicts vs actual)
        # - High surprise -> more write to LTM (consolidate unexpected information)
        # - Low surprise -> less write (redundant, already know this)
        self.ltm_planes = pos_planes  # Use same number as positional planes

        # LTM key encoder (separate from working memory keys for task separation)
        self.ltm_key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.ltm_planes)
        )

        # LTM value projection
        self.ltm_value_proj = nn.Linear(dim, dim)

        # Learnable surprise scaling (how much surprise affects write gate)
        # surprise_gate = sigmoid(surprise_scale * surprise + surprise_bias)
        self.surprise_scale = nn.Parameter(torch.tensor(1.0))
        self.surprise_bias = nn.Parameter(torch.tensor(0.0))

        # LTM retrieval weight (how much LTM contributes to output)
        self.ltm_weight = nn.Parameter(torch.tensor(0.5))

        # Momentum for surprise (Titans uses momentum for stable surprise metric)
        self.surprise_momentum = nn.Parameter(torch.tensor(0.9))

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

        if self.n_sets > 1:
            # Cross-bank binding: multiply phasors across banks to create joint address
            # key_phasor: [B, L, n_sets, planes_per_set]

            # PARALLEL: Compute joint key/query by taking product across all banks (dim=2)
            joint_key = torch.prod(key_phasor, dim=2)  # [B, L, planes_per_set]
            joint_query = torch.prod(query_phasor, dim=2)  # [B, L, planes_per_set]

            # Get value in real space for surprise computation
            V_real = self.to_value(x)  # [B, L, D] - real values

            # SURPRISE-GATED WRITING for cross-bank memory via PHASE RESONANCE
            # Key insight: Use the same phase interference that makes retrieval work!
            # Query memory with current key - if key matches stored keys, phases align
            # High resonance = familiar, low resonance = novel

            # Accumulate joint keys (shifted to exclude current timestep)
            # key_memory[t] = sum of joint_keys from 0..t-1
            key_memory_shifted = torch.zeros_like(joint_key)  # [B, L, planes_per_set]
            key_memory_shifted[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)

            # Query the key memory with current key (conjugate for retrieval)
            # resonance = joint_key† · Σjoint_keys = Σ(joint_key† · joint_key_i)
            # When current key matches a stored key: joint_key† · joint_key_i ≈ 1
            # When keys differ: phases cancel out (destructive interference)
            key_resonance_cb = key_memory_shifted * joint_key.conj()  # [B, L, planes_per_set]

            # Sum across planes and take magnitude
            # High magnitude = phases aligned = familiar key
            # Low magnitude = phases cancelled = novel key
            total_resonance_cb = key_resonance_cb.sum(dim=-1)  # [B, L] complex
            resonance_magnitude_cb = total_resonance_cb.abs()  # [B, L]

            # Normalize by number of items in memory (position) and sqrt(planes)
            positions_cb = torch.arange(L, device=x.device, dtype=torch.float32)
            positions_cb = positions_cb.clamp(min=1.0)  # Avoid div by 0
            normalized_resonance_cb = resonance_magnitude_cb / (positions_cb * math.sqrt(self.planes_per_set))

            # Clamp to [0, 1] for stability
            familiarity_cb = normalized_resonance_cb.clamp(0.0, 1.0).unsqueeze(-1)  # [B, L, 1]

            # Surprise = 1 - familiarity
            # Novel key -> low familiarity -> high surprise -> write more
            # Seen key -> high familiarity -> low surprise -> write less
            surprise = 1.0 - familiarity_cb  # [B, L, 1]

            # Convert to write gate: sigmoid centered at 0.5 surprise
            # Higher surprise -> higher gate -> more writing
            write_gate = torch.sigmoid(self.surprise_scale * (surprise - 0.5) + self.surprise_bias)

            # Gate the binding (gate real values, then convert to complex)
            V_real_gated = V_real * write_gate
            V_gated = V_real_gated.to(torch.complex64)

            # Bind with joint key (using gated values)
            bound = joint_key.unsqueeze(-1) * V_gated.unsqueeze(-2)  # [B, L, planes_per_set, D]
            memory = decayed_cumsum(bound, None)
            retrieved = memory * joint_query.conj().unsqueeze(-1)
            cross_bank_retrieved = retrieved.sum(dim=2).real  # [B, L, D]

            # PARALLEL: Also compute individual bank retrievals (also gated)
            bound_all = key_phasor.unsqueeze(-1) * V_gated.unsqueeze(2).unsqueeze(3)  # [B, L, n_sets, planes_per_set, D]

            # Cumsum along sequence for each bank (need to reshape for cumsum)
            B, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B, L_seq, n_s * pp, D_dim)  # [B, L, n_sets*planes_per_set, D]
            memory_flat = decayed_cumsum(bound_flat, None)  # [B, L, n_sets*planes_per_set, D]
            memory_all = memory_flat.view(B, L_seq, n_s, pp, D_dim)  # [B, L, n_sets, planes_per_set, D]

            # Retrieve with conjugate queries
            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)  # [B, L, n_sets, planes_per_set, D]
            retrieved_per_bank = retrieved_all.sum(dim=3).real  # [B, L, n_sets, D]

            # Apply weights and sum across banks
            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)  # [B, L, n_sets, D]
            total_retrieved = weighted_retrieved.sum(dim=2)  # [B, L, D]

            # Add cross-bank contribution (weighted equally with individual banks)
            cross_weight = 1.0 / (self.n_sets + 1)  # Equal weight to cross-bank term
            total_retrieved = cross_weight * (total_retrieved + cross_bank_retrieved)
        else:
            # PARALLEL: Process all banks at once
            # key_phasor: [B, L, n_sets, planes_per_set], V: [B, L, D]
            # Bind all banks: [B, L, n_sets, planes_per_set, D]
            bound_all = key_phasor.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)

            # Cumsum along sequence (flatten banks/planes, then reshape)
            B, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B, L_seq, n_s * pp, D_dim)
            memory_flat = decayed_cumsum(bound_flat, None)
            memory_all = memory_flat.view(B, L_seq, n_s, pp, D_dim)

            # Retrieve with conjugate queries
            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)  # [B, L, n_sets, planes_per_set, D]
            retrieved_per_bank = retrieved_all.sum(dim=3).real  # [B, L, n_sets, D]

            # Apply weights and sum across banks
            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)  # [B, L, D]

        # Compute positional phases on-the-fly for any sequence length
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        pos_phase = pos * self.pos_freqs * 2 * math.pi  # [L, pos_planes]
        pos_phasor = torch.exp(1j * pos_phase)  # [L, pos_planes]

        # Expand for batch
        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)  # [B, L, pos_planes]
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)

        # Bind and retrieve with positional phases
        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, pos_planes, D]
        # Use learnable retention rate for positional memory
        pos_retention = torch.sigmoid(self.pos_retention_logit)
        memory_pos = decayed_cumsum(bound_pos, pos_retention)
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

        # LTM: Titans-style surprise-gated phase memory
        # Encode LTM keys and values
        ltm_key_phase = torch.tanh(self.ltm_key_encoder(x)) * math.pi  # [B, L, ltm_planes]
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = self.ltm_value_proj(x)  # [B, L, D] - keep real for surprise computation

        # PHASE RESONANCE SURPRISE
        # Key insight: Use the RETRIEVAL MAGNITUDE as familiarity signal.
        # When current key matches stored keys, phase interference is constructive
        # -> high retrieval magnitude = familiar. Low magnitude = novel.
        #
        # This leverages the same phase interference that makes retrieval work!

        # Build a "key-only" memory to measure key familiarity
        # Store just the keys (unit phasors), query with current key
        # High resonance = seen this key before

        # Accumulate keys (shifted to exclude current)
        # key_memory[t] = sum of keys from 0..t-1
        key_memory_shifted = torch.zeros_like(ltm_key_phasor)  # [B, L, ltm_planes]
        key_memory_shifted[:, 1:, :] = torch.cumsum(ltm_key_phasor[:, :-1, :], dim=1)

        # Query the key memory with current key (conjugate)
        # resonance = key† · Σkeys = Σ(key† · key_i)
        # When current key matches a stored key: key† · key_i ≈ 1 (phases align)
        # When keys differ: phases cancel out (interference)
        key_resonance = key_memory_shifted * ltm_key_phasor.conj()  # [B, L, ltm_planes]

        # Sum across planes and take magnitude
        # High magnitude = phases aligned = familiar key
        # Low magnitude = phases cancelled = novel key
        total_resonance = key_resonance.sum(dim=-1)  # [B, L] complex
        resonance_magnitude = total_resonance.abs()  # [B, L]

        # Normalize by number of items in memory (position - 1)
        # This gives us "average match strength" rather than absolute
        positions_shifted = torch.arange(L, device=x.device, dtype=torch.float32)
        positions_shifted = positions_shifted.clamp(min=1.0)  # Avoid div by 0
        normalized_resonance = resonance_magnitude / (positions_shifted * math.sqrt(self.ltm_planes))

        # normalized_resonance is now in roughly [0, 1] where:
        # - 1.0 = perfect match (current key identical to a stored key)
        # - 0.0 = no match (random interference)
        # - Values > 1 possible if multiple similar keys stored

        # Clamp to [0, 1] for stability
        familiarity = normalized_resonance.clamp(0.0, 1.0).unsqueeze(-1)  # [B, L, 1]

        # Surprise = 1 - familiarity
        # Novel key -> low familiarity -> high surprise -> write more
        # Seen key -> high familiarity -> low surprise -> write less
        ltm_surprise = 1.0 - familiarity  # [B, L, 1]

        # Convert surprise to write gate
        # Scale so that 0.5 surprise gives 0.5 gate (centered sigmoid)
        ltm_write_gate = torch.sigmoid(self.surprise_scale * (ltm_surprise - 0.5) + self.surprise_bias)

        # Bind with write gate: gated_binding = gate * key * value
        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)
        # gated_binding: [B, L, ltm_planes, D]

        # Cumulative LTM (accumulates bindings over sequence)
        ltm_memory = torch.cumsum(gated_binding, dim=1)  # [B, L, ltm_planes, D]

        # Retrieve from LTM using same key (content-addressable)
        ltm_retrieved = ltm_memory * ltm_key_phasor.conj().unsqueeze(-1)
        ltm_retrieved = ltm_retrieved.sum(dim=2).real  # [B, L, D]

        # Normalize by position
        ltm_norm = torch.sqrt(positions * self.ltm_planes).view(1, L, 1)
        ltm_retrieved = ltm_retrieved / ltm_norm

        # Add LTM contribution
        total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

    def get_ltm_params(self):
        """Return LTM parameters for selective gradient scaling"""
        return list(self.ltm_key_encoder.parameters()) + \
               list(self.ltm_value_proj.parameters()) + \
               [self.surprise_scale, self.surprise_bias, self.ltm_weight, self.surprise_momentum]

class OrthogonalModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_orthogonal_sets=4, planes_per_set=16,
                 pos_planes=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_orthogonal_sets, planes_per_set, pos_planes=pos_planes)
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

    def get_ltm_params(self):
        """Return all LTM parameters from blocks for selective gradient scaling"""
        params = []
        for block in self.blocks:
            params.extend(block.get_ltm_params())
        return params


class ContinuousDynamicsModel(nn.Module):
    """Clifford model for continuous dynamics (float inputs rather than tokens)"""
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Project continuous input to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(hidden_dim, n_orthogonal_sets, planes_per_set,
                                   pos_planes=pos_planes)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Output projection back to input dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, L, input_dim] continuous state sequence
        Returns:
            [B, L, input_dim] predicted states
        """
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.output_proj(h)

    def get_ltm_params(self):
        """Return all LTM parameters from blocks for selective gradient scaling"""
        params = []
        for block in self.blocks:
            params.extend(block.get_ltm_params())
        return params


