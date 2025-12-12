"""
Phasor Memory Model - Evolving Phase Architecture

Key innovation: Instead of fixed random phases, phases EVOLVE via cumsum
of learned omega (frequency field). This gives content-dependent phase
dynamics while maintaining O(n) complexity.

Two-memory architecture:
- Memory 1: Positional memory with evolving phases
- Memory 2: Content-based associative memory (slim outer product)

The phase evolution creates a "trajectory" through phase space that
content rides on - combining PHI's dynamics with dual-memory structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhasorBlock(nn.Module):
    """
    Phasor block with EVOLVING phases (PHI-style dynamics + dual memory).

    Key innovation: Phases evolve via cumsum of learned omega, giving
    content-dependent phase dynamics while maintaining O(n) complexity.
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


class PhasorModel(nn.Module):
    """
    Phasor model with evolving phases. O(n) complexity.

    Key features:
    - PHI-style phase evolution via cumsum of learned omega
    - Dual memory: positional (evolved phases) + associative (slim outer product)
    - Content-dependent phase trajectories
    """
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            PhasorBlock(dim, n_phases, value_dim, max_seq_len)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Quick test
    x = torch.randint(0, 64, (2, 128), device=device)

    model = PhasorModel(vocab_size=64, dim=64, n_layers=4, n_phases=64, value_dim=8).to(device)
    out = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhasorModel: {x.shape} -> {out.shape}")
    print(f"  Parameters: {n_params:,}")

    # Timing test
    import time
    x = torch.randint(0, 64, (4, 128), device=device)
    torch.cuda.synchronize() if device == 'cuda' else None

    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize() if device == 'cuda' else None

    start = time.time()
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = (time.time() - start) / 50
    print(f"  Avg forward time (128 tokens): {elapsed*1000:.2f}ms")
