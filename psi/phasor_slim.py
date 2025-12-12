"""
Phasor Slim: O(P) per step instead of O(P*D).

Key insight: D is for value richness, not binding capacity.
Store scalar/low-dim values, reconstruct full D via learned projection.

Memory 2 changes:
- OLD: phase[P] ⊗ value[D] → [P, D], then cumsum → [B, L, P, D]
- NEW: phase[P] * scalar_value → [P], then cumsum → [B, L, P]
       Retrieved [P] → project to [D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlimPhasorBlock(nn.Module):
    """
    Phasor block with slim associative memory.

    Memory 1: Positional (unchanged)
    Memory 2: Slim - stores scalar values bound to phases, projects back to D
    """
    def __init__(self, dim, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim

        # Fixed random positional phases for Memory 1
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Memory 1: Positional memory (unchanged)
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.offset_predictor = nn.Linear(dim * 2, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))

        # Memory 2: Slim associative memory
        # Key encoder: content → phase angles [P]
        self.key_encoder = nn.Linear(dim, n_phases)

        # Value encoder: content → scalar value (or small vector)
        # We'll use a small value_dim for binding, then project back
        self.value_encoder = nn.Linear(dim, value_dim)

        # Storage key from content + context
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Project retrieved [P * value_dim] back to [D]
        # Retrieved is [B, L, value_dim] after unbinding
        self.kv_out = nn.Linear(self.value_dim, dim)

        # Output MLP
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        # ========== Memory 1: Positional (unchanged) ==========
        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * weighted_v1, dim=1)
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * weighted_v1, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)

        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * math.pi

        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin

        pos_ret = (mem1_cos * query_cos + mem1_sin * query_sin) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        # ========== Memory 2: Slim Associative ==========
        # Query phase (for retrieval) - [B, L, P]
        query_phase = torch.tanh(self.key_encoder(x)) * math.pi

        # Values to store - small! [B, L, value_dim]
        values = self.value_encoder(x)
        store_gate = self.store_gate(x)  # [B, L, 1]

        # Context for storage key
        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        # Storage phase [B, L, P]
        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi

        # Bind: phase[P] ⊗ value[value_dim] → [P, value_dim]
        # This is still an outer product but MUCH smaller: P * value_dim instead of P * D
        store_cos = torch.cos(storage_phase)  # [B, L, P]
        store_sin = torch.sin(storage_phase)  # [B, L, P]

        # Outer product: [B, L, P, 1] * [B, L, 1, value_dim] → [B, L, P, value_dim]
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        # Accumulate
        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)  # [B, L, P, value_dim]
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        # Normalize
        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)
        kv_mem_cos = kv_mem_cos / torch.sqrt(gate_cumsum).unsqueeze(-1)
        kv_mem_sin = kv_mem_sin / torch.sqrt(gate_cumsum).unsqueeze(-1)

        # Retrieve with query phase
        query_cos = torch.cos(query_phase)  # [B, L, P]
        query_sin = torch.sin(query_phase)

        # [B, L, P, 1] * [B, L, P, value_dim] → sum over P → [B, L, value_dim]
        kv_retrieved = (
            query_cos.unsqueeze(-1) * kv_mem_cos +
            query_sin.unsqueeze(-1) * kv_mem_sin
        ).sum(dim=2) / math.sqrt(self.n_phases)  # [B, L, value_dim]

        # Project back to full dimension
        kv_out = self.kv_out(kv_retrieved)  # [B, L, D]

        # Trajectory
        trajectory = x * torch.cos(query_phase).mean(dim=-1, keepdim=True)

        # Combine and output
        combined = torch.cat([pos_out, kv_out, trajectory], dim=-1)
        return x + self.to_out(combined)


class EvolvingPhasorBlock(nn.Module):
    """
    Phasor block with EVOLVING phases (like PHI but with dual memory).

    Key innovation: Instead of fixed random phases, phases evolve via cumsum
    of learned omega (frequency). This gives content-dependent phase dynamics
    while maintaining O(n) complexity.

    The phase evolution creates a "trajectory" through phase space that
    content can ride on - similar to how PHI works but with separate
    storage and retrieval phases.
    """
    def __init__(self, dim, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim

        # Phase evolution (from PHI) - learned omega creates evolving phases
        self.to_omega = nn.Linear(dim, dim)  # Frequency field
        self.omega_scale = nn.Parameter(torch.ones(dim) * 0.01)  # Integration scale

        # Phase initialization (content-dependent starting point)
        self.phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Memory 1: Uses evolved phases for positional memory
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))

        # Query offset for Memory 1 retrieval
        self.query_offset = nn.Linear(dim, dim)

        # Memory 2: Associative memory with separate phase encoding
        self.key_encoder = nn.Linear(dim, n_phases)
        self.value_encoder = nn.Linear(dim, value_dim)

        # Storage uses context-aware phase
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.kv_out = nn.Linear(value_dim, dim)

        # Output combines: evolved trajectory, positional retrieval, associative retrieval
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # ========== Phase Evolution (from PHI) ==========
        # Compute frequency field from content
        omega = self.to_omega(x)  # [B, L, D]
        omega_scaled = omega * self.omega_scale.abs()

        # Initialize phase from content
        phi_init = self.phase_init(x)  # [B, L, D]

        # Phase evolves via cumsum of omega (key innovation from PHI)
        phi = phi_init + torch.cumsum(omega_scaled, dim=1)  # [B, L, D]

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # ========== Memory 1: Positional with Evolved Phases ==========
        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        # Store with evolved phase
        mem1_cos = torch.cumsum(weighted_v1 * cos_phi, dim=1)
        mem1_sin = torch.cumsum(weighted_v1 * sin_phi, dim=1)

        # Normalize by accumulated magnitude
        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        # Query with offset
        query_offset = self.query_offset(x)
        phi_query = phi + query_offset
        cos_q = torch.cos(phi_query)
        sin_q = torch.sin(phi_query)

        pos_ret = (mem1_cos * cos_q + mem1_sin * sin_q) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        # ========== Memory 2: Associative (separate phase space) ==========
        query_phase = torch.tanh(self.key_encoder(x)) * math.pi
        values = self.value_encoder(x)
        store_gate = self.store_gate(x)

        # Context for storage key
        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi

        store_cos = torch.cos(storage_phase)
        store_sin = torch.sin(storage_phase)

        # Outer product binding
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)
        kv_mem_cos = kv_mem_cos / torch.sqrt(gate_cumsum).unsqueeze(-1)
        kv_mem_sin = kv_mem_sin / torch.sqrt(gate_cumsum).unsqueeze(-1)

        q_cos = torch.cos(query_phase)
        q_sin = torch.sin(query_phase)

        kv_retrieved = (
            q_cos.unsqueeze(-1) * kv_mem_cos +
            q_sin.unsqueeze(-1) * kv_mem_sin
        ).sum(dim=2) / math.sqrt(self.n_phases)

        kv_out = self.kv_out(kv_retrieved)

        # ========== Trajectory (content on evolving phase) ==========
        trajectory_real = x * cos_phi
        trajectory_imag = x * sin_phi

        # Combine all signals
        combined = torch.cat([pos_out, kv_out, trajectory_real, trajectory_imag], dim=-1)
        return x + self.to_out(combined)


class MultiHeadPhasorBlock(nn.Module):
    """
    Multi-head phasor block - splits phase space into independent heads.
    Each head has its own subset of phases for diversity.
    """
    def __init__(self, dim, n_phases=128, value_dim=8, n_heads=4, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim
        self.n_heads = n_heads
        self.head_phases = n_phases // n_heads
        self.head_dim = dim // n_heads

        # Fixed random positional phases for Memory 1
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Memory 1: Positional memory (unchanged from original)
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.offset_predictor = nn.Linear(dim * 2, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))

        # Memory 2: Multi-head slim associative memory
        # Each head has its own key/value encoders
        self.key_encoder = nn.Linear(dim, n_phases)  # Split into heads later
        self.value_encoder = nn.Linear(dim, value_dim * n_heads)

        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, n_heads),
            nn.Sigmoid()
        )

        # Output projection per head
        self.kv_out = nn.Linear(value_dim * n_heads, dim)

        # Output MLP
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        P_h = self.head_phases
        V = self.value_dim

        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        # ========== Memory 1: Positional (same as original) ==========
        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * weighted_v1, dim=1)
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * weighted_v1, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)

        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * math.pi

        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos_pos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin_pos = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin

        pos_ret = (mem1_cos * query_cos_pos + mem1_sin * query_sin_pos) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        # ========== Memory 2: Multi-Head Slim Associative ==========
        # Query phase [B, L, P] -> [B, L, H, P_h]
        query_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = query_phase.view(B, L, H, P_h)

        # Values [B, L, H*V] -> [B, L, H, V]
        values = self.value_encoder(x).view(B, L, H, V)

        # Store gate [B, L, H]
        store_gate = self.store_gate(x)

        # Context for storage key
        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        # Storage phase [B, L, P] -> [B, L, H, P_h]
        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi
        storage_phase = storage_phase.view(B, L, H, P_h)

        store_cos = torch.cos(storage_phase)  # [B, L, H, P_h]
        store_sin = torch.sin(storage_phase)

        # Bind per head: [B, L, H, P_h, 1] * [B, L, H, 1, V] * [B, L, H, 1, 1]
        gate_expanded = store_gate.unsqueeze(-1).unsqueeze(-1)  # [B, L, H, 1, 1]
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(3) * gate_expanded
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(3) * gate_expanded

        # Accumulate per head
        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)  # [B, L, H, P_h, V]
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        # Normalize per head
        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)  # [B, L, H]
        norm_factor = torch.sqrt(gate_cumsum).unsqueeze(-1).unsqueeze(-1)  # [B, L, H, 1, 1]
        kv_mem_cos = kv_mem_cos / norm_factor
        kv_mem_sin = kv_mem_sin / norm_factor

        # Retrieve per head
        query_cos = torch.cos(query_phase).unsqueeze(-1)  # [B, L, H, P_h, 1]
        query_sin = torch.sin(query_phase).unsqueeze(-1)

        # [B, L, H, P_h, 1] * [B, L, H, P_h, V] -> sum over P_h -> [B, L, H, V]
        kv_retrieved = (
            query_cos * kv_mem_cos + query_sin * kv_mem_sin
        ).sum(dim=3) / math.sqrt(P_h)

        # Flatten heads and project
        kv_retrieved = kv_retrieved.view(B, L, H * V)  # [B, L, H*V]
        kv_out = self.kv_out(kv_retrieved)

        # Trajectory (use mean of query phases)
        trajectory = x * torch.cos(query_phase).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)

        # Combine and output
        combined = torch.cat([pos_out, kv_out, trajectory], dim=-1)
        return x + self.to_out(combined)


class LocalConvPhasorBlock(nn.Module):
    """
    Phasor block with local convolution for n-gram patterns.
    Adds a causal conv layer to capture short-range dependencies.
    """
    def __init__(self, dim, n_phases=128, value_dim=8, kernel_size=4, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim
        self.kernel_size = kernel_size

        # Local convolution (causal)
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=dim  # Depthwise for efficiency
        )
        self.conv_gate = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=dim
        )
        self.conv_proj = nn.Linear(dim, dim)

        # Fixed random positional phases for Memory 1
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Memory 1: Positional memory
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.offset_predictor = nn.Linear(dim * 2, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))

        # Memory 2: Slim associative memory
        self.key_encoder = nn.Linear(dim, n_phases)
        self.value_encoder = nn.Linear(dim, value_dim)
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.kv_out = nn.Linear(value_dim, dim)

        # Output MLP - now takes 4 inputs: pos, kv, conv
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # ========== Local Convolution (causal) ==========
        # [B, L, D] -> [B, D, L] for conv1d
        x_conv = x.transpose(1, 2)
        conv_out = self.local_conv(x_conv)[:, :, :L]  # Causal: trim future
        conv_gate = torch.sigmoid(self.conv_gate(x_conv)[:, :, :L])
        conv_out = conv_out * conv_gate
        conv_out = conv_out.transpose(1, 2)  # [B, L, D]
        conv_out = self.conv_proj(conv_out)

        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        # ========== Memory 1: Positional ==========
        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * weighted_v1, dim=1)
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * weighted_v1, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)

        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * math.pi

        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos_pos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin_pos = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin

        pos_ret = (mem1_cos * query_cos_pos + mem1_sin * query_sin_pos) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        # ========== Memory 2: Associative ==========
        query_phase = torch.tanh(self.key_encoder(x)) * math.pi
        values = self.value_encoder(x)
        store_gate = self.store_gate(x)

        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi

        store_cos = torch.cos(storage_phase)
        store_sin = torch.sin(storage_phase)

        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)
        kv_mem_cos = kv_mem_cos / torch.sqrt(gate_cumsum).unsqueeze(-1)
        kv_mem_sin = kv_mem_sin / torch.sqrt(gate_cumsum).unsqueeze(-1)

        query_cos = torch.cos(query_phase)
        query_sin = torch.sin(query_phase)

        kv_retrieved = (
            query_cos.unsqueeze(-1) * kv_mem_cos +
            query_sin.unsqueeze(-1) * kv_mem_sin
        ).sum(dim=2) / math.sqrt(self.n_phases)

        kv_out = self.kv_out(kv_retrieved)

        # Combine all four signals
        combined = torch.cat([pos_out, kv_out, conv_out], dim=-1)
        return x + self.to_out(combined)


class SlimPhasorModel(nn.Module):
    """Slim Phasor model - O(n) in sequence, O(P * value_dim) per position."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            SlimPhasorBlock(dim, n_phases, value_dim, max_seq_len)
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


class MultiHeadPhasorModel(nn.Module):
    """Multi-head Phasor model with independent phase heads."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, n_heads=4, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            MultiHeadPhasorBlock(dim, n_phases, value_dim, n_heads, max_seq_len)
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


class LocalConvPhasorModel(nn.Module):
    """Phasor model with local convolution for n-gram patterns."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, kernel_size=4, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            LocalConvPhasorBlock(dim, n_phases, value_dim, kernel_size, max_seq_len)
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


class EvolvingPhasorModel(nn.Module):
    """
    Phasor model with evolving phases (PHI-style dynamics + dual memory).

    Key innovation: Phases evolve via cumsum of learned omega, giving
    content-dependent phase trajectories while maintaining O(n) complexity.
    """
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            EvolvingPhasorBlock(dim, n_phases, value_dim, max_seq_len)
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


class EvolvingLocalConvBlock(nn.Module):
    """
    Best of both worlds: Evolving phases (PHI-style) + Local conv for n-grams.

    This combines:
    1. PHI-style phase evolution (content-dependent dynamics)
    2. Local convolution (captures n-gram patterns)
    3. Associative memory (content-based retrieval)
    """
    def __init__(self, dim, n_phases=128, value_dim=8, kernel_size=4, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim
        self.kernel_size = kernel_size

        # Local convolution (causal) for n-gram patterns
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=dim
        )
        self.conv_gate = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=dim
        )
        self.conv_proj = nn.Linear(dim, dim)

        # Phase evolution (from PHI) - learned omega creates evolving phases
        self.to_omega = nn.Linear(dim, dim)
        self.omega_scale = nn.Parameter(torch.ones(dim) * 0.01)
        self.phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Memory 1: Uses evolved phases
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))
        self.query_offset = nn.Linear(dim, dim)

        # Memory 2: Associative
        self.key_encoder = nn.Linear(dim, n_phases)
        self.value_encoder = nn.Linear(dim, value_dim)
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.kv_out = nn.Linear(value_dim, dim)

        # Output: conv + pos_mem + kv_mem + trajectory_real + trajectory_imag
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # ========== Local Convolution ==========
        x_conv = x.transpose(1, 2)
        conv_out = self.local_conv(x_conv)[:, :, :L]
        conv_gate = torch.sigmoid(self.conv_gate(x_conv)[:, :, :L])
        conv_out = conv_out * conv_gate
        conv_out = conv_out.transpose(1, 2)
        conv_out = self.conv_proj(conv_out)

        # ========== Phase Evolution ==========
        omega = self.to_omega(x)
        omega_scaled = omega * self.omega_scale.abs()
        phi_init = self.phase_init(x)
        phi = phi_init + torch.cumsum(omega_scaled, dim=1)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # ========== Memory 1: Positional with Evolved Phases ==========
        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        mem1_cos = torch.cumsum(weighted_v1 * cos_phi, dim=1)
        mem1_sin = torch.cumsum(weighted_v1 * sin_phi, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        query_off = self.query_offset(x)
        phi_query = phi + query_off
        cos_q = torch.cos(phi_query)
        sin_q = torch.sin(phi_query)

        pos_ret = (mem1_cos * cos_q + mem1_sin * sin_q) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        # ========== Memory 2: Associative ==========
        query_phase = torch.tanh(self.key_encoder(x)) * math.pi
        values = self.value_encoder(x)
        store_gate = self.store_gate(x)

        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi

        store_cos = torch.cos(storage_phase)
        store_sin = torch.sin(storage_phase)

        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)
        kv_mem_cos = kv_mem_cos / torch.sqrt(gate_cumsum).unsqueeze(-1)
        kv_mem_sin = kv_mem_sin / torch.sqrt(gate_cumsum).unsqueeze(-1)

        q_cos = torch.cos(query_phase)
        q_sin = torch.sin(query_phase)

        kv_retrieved = (
            q_cos.unsqueeze(-1) * kv_mem_cos +
            q_sin.unsqueeze(-1) * kv_mem_sin
        ).sum(dim=2) / math.sqrt(self.n_phases)

        kv_out = self.kv_out(kv_retrieved)

        # ========== Trajectory ==========
        trajectory_real = x * cos_phi
        trajectory_imag = x * sin_phi

        # Combine all signals
        combined = torch.cat([conv_out, pos_out, kv_out, trajectory_real, trajectory_imag], dim=-1)
        return x + self.to_out(combined)


class EvolvingLocalConvModel(nn.Module):
    """Full model combining evolving phases with local convolution."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, kernel_size=4, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            EvolvingLocalConvBlock(dim, n_phases, value_dim, kernel_size, max_seq_len)
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


if __name__ == "__main__":
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("SLIM PHASOR - Speed Test")
    print("=" * 60)

    # Compare memory sizes
    print("\n--- Memory Size Comparison ---")
    B, L, P, D = 32, 256, 128, 128
    value_dim = 8

    original_mem = B * L * P * D * 4  # float32
    slim_mem = B * L * P * value_dim * 4

    print(f"Original [B,L,P,D]: {B}x{L}x{P}x{D} = {original_mem/1e6:.1f} MB")
    print(f"Slim [B,L,P,v]:     {B}x{L}x{P}x{value_dim} = {slim_mem/1e6:.1f} MB")
    print(f"Memory reduction:   {original_mem/slim_mem:.1f}x")

    # Speed comparison
    print("\n--- Speed Comparison ---")

    for seq_len in [128, 256, 512]:
        print(f"\nseq_len = {seq_len}:")
        batch_size = 32

        # Slim Phasor
        slim_model = SlimPhasorModel(
            vocab_size=256, dim=128, n_layers=4, n_phases=128, max_seq_len=seq_len+128
        ).to(device)
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = slim_model(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = slim_model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        slim_time = (time.time() - start) / 10
        slim_tps = (batch_size * seq_len) / slim_time
        print(f"  Slim Phasor: {slim_time*1000:.1f}ms, {slim_tps/1000:.1f}K tok/sec")

        del slim_model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Parameter count
    print("\n--- Parameter Count ---")
    slim_model = SlimPhasorModel(vocab_size=256, dim=128, n_layers=4, n_phases=128)
    slim_params = sum(p.numel() for p in slim_model.parameters())
    print(f"Slim Phasor: {slim_params:,} parameters")
