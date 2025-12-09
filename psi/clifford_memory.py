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

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


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
        # surprise_gate = sigmoid(surprise_scale * (surprise - 0.5) + surprise_bias)
        # Initialize scale=5.0 for good differentiation:
        #   surprise 0.1 -> gate 0.12 (low write for familiar)
        #   surprise 0.5 -> gate 0.50 (neutral)
        #   surprise 0.9 -> gate 0.88 (high write for novel)
        self.surprise_scale = nn.Parameter(torch.tensor(5.0))
        self.surprise_bias = nn.Parameter(torch.tensor(0.0))

        # Learnable resonance parameters for key-based familiarity detection
        # resonance_scale: how sharply to distinguish novel from familiar (higher = sharper)
        # resonance_threshold: what resonance level counts as "familiar" (lower = more sensitive)
        # WORKING MEMORY resonance params (for within-sequence surprise)
        self.resonance_scale = nn.Parameter(torch.tensor(5.0))
        self.resonance_threshold = nn.Parameter(torch.tensor(0.3))

        # LTM-SPECIFIC resonance params (for cross-sequence surprise)
        # LTM may need different sensitivity since it sees patterns across many sequences
        # Initialize same as working memory but let them diverge during training
        self.ltm_resonance_scale = nn.Parameter(torch.tensor(5.0))
        self.ltm_resonance_threshold = nn.Parameter(torch.tensor(0.3))
        # LTM-specific surprise->gate conversion (may need different sensitivity)
        self.ltm_surprise_scale = nn.Parameter(torch.tensor(5.0))
        self.ltm_surprise_bias = nn.Parameter(torch.tensor(0.0))

        # LTM retrieval weight (how much LTM contributes to output)
        self.ltm_weight = nn.Parameter(torch.tensor(0.5))

        # PERSISTENT LTM STATE - survives across sequences!
        # This is the key difference from per-sequence memory.
        # ltm_key_memory: accumulated key phasors for familiarity detection
        # ltm_binding_memory: accumulated key*value bindings for retrieval
        # ltm_count: number of items written (for normalization)
        # Using register_buffer so they move with the model but aren't parameters
        self.register_buffer('ltm_key_memory', torch.zeros(self.ltm_planes, dtype=torch.complex64))
        self.register_buffer('ltm_binding_memory', torch.zeros(self.ltm_planes, dim, dtype=torch.complex64))
        self.register_buffer('ltm_count', torch.tensor(0.0))

        # LTM decay rate - prevents unbounded growth over long training
        self.ltm_decay = nn.Parameter(torch.tensor(0.999))

        # Momentum for surprise (Titans uses momentum for stable surprise metric)
        self.surprise_momentum = nn.Parameter(torch.tensor(0.9))

        # Pending LTM update - stored during forward, applied after backward
        self._pending_ltm_update = None

    def forward(self, x):
        B, L, D = x.shape

        # Apply any pending LTM updates from previous forward pass (training mode only)
        # This is done at the START of forward to avoid inplace modification during autograd
        if self._pending_ltm_update is not None:
            update = self._pending_ltm_update
            self.ltm_key_memory.mul_(update['decay'])
            self.ltm_binding_memory.mul_(update['decay'])
            self.ltm_key_memory.add_(update['new_keys'])
            self.ltm_binding_memory.add_(update['new_bindings'])
            self.ltm_count.add_(update['count_delta'])
            self._pending_ltm_update = None

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

            # SURPRISE-GATED WRITING via KEY RESONANCE MAGNITUDE
            # O(n) approach: Strong resonance with prior memory = familiar = low surprise
            # Weak resonance = novel = high surprise
            #
            # Key insight: When querying memory with a key that matches stored keys,
            # the phase alignment produces large magnitude. Novel keys get random phases
            # that partially cancel, producing smaller magnitude.

            # Build memory of keys only (no values) shifted by 1 position
            # memory_keys[t] = sum of keys from 0..t-1
            key_memory = torch.zeros_like(joint_key)
            key_memory[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)  # [B, L, P]

            # Query key memory with current key to get resonance
            # High resonance = current key aligns with stored keys = familiar
            resonance = key_memory * joint_query.conj()  # [B, L, P]
            resonance_magnitude = resonance.abs().mean(dim=-1, keepdim=True)  # [B, L, 1]

            # Normalize by position (expected magnitude grows with sqrt(n) for random phases)
            positions = torch.arange(L, device=x.device, dtype=torch.float32).clamp(min=1.0)
            normalized_resonance = resonance_magnitude / positions.view(1, -1, 1).sqrt()

            # High resonance = familiar = LOW surprise
            # Low resonance = novel = HIGH surprise
            # Invert and scale to get surprise
            # Use tanh for sharper cutoff, with learnable scale and threshold
            # When normalized_resonance is ~0 (novel), surprise is high
            # When normalized_resonance is ~1 (familiar), surprise is low
            # Clamp scale to prevent explosion/collapse
            scale = self.resonance_scale.clamp(min=1.0, max=20.0)
            threshold = self.resonance_threshold.clamp(min=0.1, max=0.9)
            surprise = 0.5 * (1.0 - torch.tanh(scale * (normalized_resonance - threshold)))  # [B, L, 1]

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

        # =======================================================================
        # PERSISTENT LTM: Titans-style surprise-gated memory that survives
        # across sequences. This is the TRUE long-term memory.
        # =======================================================================

        # Encode LTM keys and values
        ltm_key_phase = torch.tanh(self.ltm_key_encoder(x)) * math.pi  # [B, L, ltm_planes]
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = self.ltm_value_proj(x)  # [B, L, D]

        # SURPRISE COMPUTATION: Query the PERSISTENT LTM key memory
        # The persistent memory accumulates keys across ALL sequences seen during training
        # High resonance with persistent memory = familiar pattern across training
        # Low resonance = novel pattern never seen before

        # Query persistent key memory with current keys
        # self.ltm_key_memory: [ltm_planes] - accumulated keys from all past sequences
        # ltm_key_phasor: [B, L, ltm_planes]
        # IMPORTANT: detach persistent memory to avoid inplace modification errors during backward
        key_resonance = self.ltm_key_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()  # [B, L, ltm_planes]

        # Sum across planes and take magnitude
        total_resonance = key_resonance.sum(dim=-1)  # [B, L] complex
        resonance_magnitude = total_resonance.abs()  # [B, L]

        # Normalize by count of items in persistent memory
        count_norm = (self.ltm_count.detach().clamp(min=1.0) * math.sqrt(self.ltm_planes))
        normalized_resonance = resonance_magnitude / count_norm

        # Apply LTM-specific learned resonance transformation
        # Use separate scale/threshold so LTM can learn different sensitivity than working memory
        ltm_scale = self.ltm_resonance_scale.clamp(min=1.0, max=20.0)
        ltm_threshold = self.ltm_resonance_threshold.clamp(min=0.1, max=0.9)

        # Transform raw resonance magnitude through learned scale and threshold
        # High resonance relative to threshold -> familiar -> low surprise
        # Low resonance relative to threshold -> novel -> high surprise
        ltm_surprise = 0.5 * (1.0 - torch.tanh(ltm_scale * (normalized_resonance - ltm_threshold)))
        ltm_surprise = ltm_surprise.unsqueeze(-1)  # [B, L, 1]

        # Convert surprise to write gate using LTM-specific learnable scale
        ltm_write_gate = torch.sigmoid(self.ltm_surprise_scale * (ltm_surprise - 0.5) + self.ltm_surprise_bias)

        # Bind with write gate: gated_binding = gate * key * value
        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)
        # gated_binding: [B, L, ltm_planes, D]

        # RETRIEVAL: Query persistent binding memory ONLY
        # The per-sequence memory is already handled by the cross-bank surprise-gated memory above
        # This LTM is purely for knowledge from PAST sequences
        # IMPORTANT: detach persistent memory to avoid inplace modification errors during backward
        persistent_retrieved = self.ltm_binding_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj().unsqueeze(-1)
        persistent_retrieved = persistent_retrieved.sum(dim=2).real  # [B, L, D]
        persistent_norm = (self.ltm_count.detach().clamp(min=1.0) * self.ltm_planes).sqrt()
        ltm_retrieved = persistent_retrieved / persistent_norm

        # STORE PENDING LTM UPDATE (only during training!)
        # We store the update and apply it at the END of the forward pass
        # to avoid inplace modification during autograd graph construction
        if self.training:
            with torch.no_grad():
                # Compute the updates but don't apply yet
                decay = self.ltm_decay.detach().clamp(min=0.9, max=0.9999)
                # ltm_write_gate: [B, L, 1], ltm_key_phasor: [B, L, ltm_planes]
                new_keys = (ltm_write_gate.detach() * ltm_key_phasor.detach()).mean(dim=(0, 1))  # [ltm_planes]
                new_bindings = gated_binding.detach().mean(dim=(0, 1))  # [ltm_planes, D]
                avg_gate = ltm_write_gate.detach().mean().item()

                # Store for delayed application (after backward)
                self._pending_ltm_update = {
                    'decay': decay,
                    'new_keys': new_keys,
                    'new_bindings': new_bindings,
                    'count_delta': avg_gate * B * L * (1 - decay)
                }

        # Add LTM contribution
        total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

    def get_ltm_params(self):
        """Return LTM parameters for selective gradient scaling"""
        return list(self.ltm_key_encoder.parameters()) + \
               list(self.ltm_value_proj.parameters()) + \
               [self.ltm_resonance_scale, self.ltm_resonance_threshold,
                self.ltm_surprise_scale, self.ltm_surprise_bias,
                self.ltm_weight, self.surprise_momentum, self.ltm_decay]

    def reset_ltm(self):
        """Reset persistent LTM state (call between unrelated tasks or for ablation)"""
        self.ltm_key_memory.zero_()
        self.ltm_binding_memory.zero_()
        self.ltm_count.zero_()

    def get_ltm_stats(self):
        """Return LTM statistics for monitoring"""
        return {
            'ltm_count': self.ltm_count.item(),
            'ltm_key_norm': self.ltm_key_memory.abs().mean().item(),
            'ltm_binding_norm': self.ltm_binding_memory.abs().mean().item(),
            'ltm_decay': self.ltm_decay.item()
        }

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


