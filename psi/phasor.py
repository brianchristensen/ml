"""
Phasor Memory Model - Optimized

Two-memory architecture with O(n) complexity:
- Memory 1: Position → Content (flat [B,L,P] with learned offset)
- Memory 2: Content → Value (full [B,L,P,D] for associative recall)

Key components:
1. Learned offset for Memory 1 (positional retrieval)
2. Context accumulation via cumsum for Memory 2 storage key (associative recall)
3. Full [B,L,P,D] binding for Memory 2 (D-dimensional values essential)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhasorBlock(nn.Module):
    """
    Phasor memory block with full-dimension phases (like PHI):
    - Memory 1: Positional memory [B,L,D]
    - Memory 2: Content-based KV memory [B,L,D,D] (full dimensions)

    Using dim as n_phases removes the bottleneck that was hurting LM performance.
    """
    def __init__(self, dim, n_phases=None, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        # Use full dimension as phases (like PHI) - ignore n_phases parameter
        self.n_phases = dim

        # Fixed random positional phases - now [max_seq_len, dim]
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Memory 1: Positional memory with learned offset (full dim)
        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.offset_predictor = nn.Linear(dim * 2, dim)

        # Memory 2: Content-based KV memory (full dimensions)
        self.key_encoder = nn.Linear(dim, dim)
        self.value_encoder = nn.Linear(dim, dim)
        # Storage key needs 2-layer MLP for associative recall
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Output - takes trajectory + retrieved (like PHI)
        # Input: [pos_out, kv_retrieved, trajectory_real, trajectory_imag] = dim * 4
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        P = self.n_phases

        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        # ========== Memory 1: Positional with learned offset (flat) ==========
        v1 = self.mem1_value(x)  # [B, L, P]

        # Build memory
        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * v1, dim=1)  # [B, L, P]
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * v1, dim=1)

        # Learned offset for retrieval
        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * math.pi  # [B, L, P]

        # Rotated query
        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin

        # Retrieve
        pos_ret = (mem1_cos * query_cos + mem1_sin * query_sin) / math.sqrt(P)
        pos_out = self.mem1_out(pos_ret)  # [B, L, D]

        # ========== Memory 2: Content-based KV (full dimensions) ==========
        # Query key (for retrieval) - based on current content
        key_phases = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, P]
        key_cos = torch.cos(key_phases)
        key_sin = torch.sin(key_phases)

        values = self.value_encoder(x)  # [B, L, D]
        store_gate = self.store_gate(x)  # [B, L, 1]

        # Context accumulation (essential for associative recall)
        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        # Storage key from current + context
        storage_phases = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi
        store_cos = torch.cos(storage_phases)  # [B, L, P]
        store_sin = torch.sin(storage_phases)

        # Bind values with context-aware storage key (gated)
        # [B, L, P, 1] * [B, L, 1, D] * [B, L, 1, 1] -> [B, L, P, D]
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)  # [B, L, P, D]
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)

        # Query with current content's key phase
        # [B, L, P, 1] * [B, L, P, D] -> sum over P -> [B, L, D]
        query_cos_kv = key_cos.unsqueeze(-1)  # [B, L, P, 1]
        query_sin_kv = key_sin.unsqueeze(-1)

        kv_retrieved = (kv_mem_cos * query_cos_kv + kv_mem_sin * query_sin_kv).sum(dim=2)
        kv_retrieved = kv_retrieved / (torch.sqrt(gate_cumsum) * math.sqrt(P))

        # Compute trajectory (content modulated by phase) like PHI
        # Use the key phases as the trajectory - this gives content-dependent oscillation
        trajectory_real = x * key_cos.mean(dim=-1, keepdim=True)  # [B, L, D]
        trajectory_imag = x * key_sin.mean(dim=-1, keepdim=True)  # [B, L, D]

        # Concatenate all signals: retrieved memory + trajectory
        combined = torch.cat([pos_out, kv_retrieved, trajectory_real, trajectory_imag], dim=-1)
        return x + self.to_out(combined)


class PhasorModel(nn.Module):
    """Phasor model for sequence modeling. O(n) complexity."""
    def __init__(self, vocab_size, dim=64, n_layers=4, n_phases=32, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            PhasorBlock(dim, n_phases, max_seq_len)
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

    model = PhasorModel(vocab_size=64, dim=64, n_layers=4, n_phases=32).to(device)
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
