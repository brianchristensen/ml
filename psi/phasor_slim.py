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
    def __init__(self, dim, n_phases=128, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

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
        self.value_dim = 8  # Small! Just enough to encode token identity
        self.value_encoder = nn.Linear(dim, self.value_dim)

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


class SlimPhasorModel(nn.Module):
    """Slim Phasor model - O(n) in sequence, O(P * value_dim) per position."""
    def __init__(self, vocab_size, dim=128, n_layers=4, n_phases=128, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            SlimPhasorBlock(dim, n_phases, max_seq_len)
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
