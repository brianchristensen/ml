"""
Multi-Head Phasor Memory Model

Key features:
1. Multiple heads - each learns its own offset pattern
2. Soft windows - each head retrieves from a Gaussian-weighted window
3. NO torch.roll - all offsets are learned via phase rotation
4. O(n) complexity - cumsum + element-wise ops only

Two-memory architecture:
- Memory 1: Position → Content (multi-head with learned offsets)
- Memory 2: Content → Value (content-based associative retrieval with learned offset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhasorBlock(nn.Module):
    """
    Multi-head phasor memory block with soft windowed retrieval.

    Each head can specialize in different offset patterns.
    All offsets are LEARNED - no hardcoded torch.roll.
    """
    def __init__(self, dim, n_heads=4, n_phases=32, max_seq_len=16384, window_size=5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_phases = n_phases
        self.window_size = window_size

        # Fixed random positional phases
        pos_phases = torch.randn(max_seq_len, n_phases) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Content encoder for Memory 1
        self.mem1_content_encoder = nn.Linear(dim, dim)

        # Vectorized multi-head offset predictor for Memory 1
        # All heads predict in parallel: offset phases + window width per head
        self.head_offset_predictor = nn.Sequential(
            nn.Linear(dim + n_phases, dim),  # Content + position
            nn.GELU(),
            nn.Linear(dim, n_heads * (n_phases + 1))  # All heads: offset phases + window width
        )

        # Precompute window deltas
        half_window = window_size // 2
        window_deltas = torch.arange(-half_window, half_window + 1, dtype=torch.float32)
        self.register_buffer('window_deltas', window_deltas)

        # Content phase encoder for Memory 2
        self.content_phase_encoder = nn.Linear(dim, n_phases)

        # Value encoder
        self.value_encoder = nn.Linear(dim, dim)

        # Memory 2: Context-aware storage key predictor
        # Takes BOTH current content AND accumulated context to predict storage key
        # This lets a value token generate a key matching its preceding key token
        self.kv_context_encoder = nn.Linear(dim, dim)  # For context accumulation
        self.kv_storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),  # current + context
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )

        # Store gate
        self.store_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads

        # Position phases
        pos_phases = self.pos_phases[:L]  # [L, n_phases]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        # ========== Memory 1: Position → Content (Multi-head) ==========
        content = self.mem1_content_encoder(x)  # [B, L, D]

        # Build positional memory
        mem1_bound_cos = pos_cos.unsqueeze(0).unsqueeze(-1) * content.unsqueeze(2)
        mem1_bound_sin = pos_sin.unsqueeze(0).unsqueeze(-1) * content.unsqueeze(2)
        mem1_cos = torch.cumsum(mem1_bound_cos, dim=1)
        mem1_sin = torch.cumsum(mem1_bound_sin, dim=1)

        # Offset input: content + position
        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        # All heads predict in parallel [B, L, H * (n_phases + 1)]
        all_offset_out = self.head_offset_predictor(offset_input)
        all_offset_out = all_offset_out.view(B, L, H, self.n_phases + 1)

        center_offsets = torch.tanh(all_offset_out[..., :self.n_phases]) * math.pi  # [B, L, H, n_phases]
        window_widths = torch.sigmoid(all_offset_out[..., -1:]) * 2 + 0.1  # [B, L, H, 1]

        # Vectorized soft window: expand for all window positions
        W = len(self.window_deltas)  # window_size
        deltas = self.window_deltas.view(1, 1, 1, W)  # [1, 1, 1, W]

        # Gaussian weights for all window positions [B, L, H, W]
        weights = torch.exp(-0.5 * (deltas ** 2) / (window_widths ** 2 + 1e-6))

        # Phase offsets for all window positions [B, L, H, W, n_phases]
        delta_phases = deltas.unsqueeze(-1) * 0.1  # [1, 1, 1, W, 1]
        offset_phases = center_offsets.unsqueeze(3) + delta_phases  # [B, L, H, W, n_phases]

        offset_cos = torch.cos(offset_phases)
        offset_sin = torch.sin(offset_phases)

        # Expand pos_phases for broadcasting [1, L, 1, 1, n_phases]
        pos_cos_exp = pos_cos.view(1, L, 1, 1, self.n_phases)
        pos_sin_exp = pos_sin.view(1, L, 1, 1, self.n_phases)

        # Apply rotation [B, L, H, W, n_phases]
        query_cos = pos_cos_exp * offset_cos - pos_sin_exp * offset_sin
        query_sin = pos_sin_exp * offset_cos + pos_cos_exp * offset_sin

        # Retrieve from memory: mem1 is [B, L, n_phases, D]
        # We need to compute for each (batch, position, head, window_pos)
        # Reshape for efficient einsum
        query_cos_flat = query_cos.view(B, L, H * W, self.n_phases)  # [B, L, H*W, n_phases]
        query_sin_flat = query_sin.view(B, L, H * W, self.n_phases)

        # Retrieval: sum over n_phases dimension
        retrieved = (
            torch.einsum('blpd,blhp->blhd', mem1_cos, query_cos_flat) +
            torch.einsum('blpd,blhp->blhd', mem1_sin, query_sin_flat)
        ) / math.sqrt(self.n_phases)  # [B, L, H*W, D]

        # Reshape back to [B, L, H, W, D]
        retrieved = retrieved.view(B, L, H, W, D)

        # Apply Gaussian weights and sum over window [B, L, H, D]
        weights_exp = weights.unsqueeze(-1)  # [B, L, H, W, 1]
        weighted_sum = (retrieved * weights_exp).sum(dim=3)
        weight_total = weights.sum(dim=3, keepdim=True)  # [B, L, H, 1]
        head_outputs = weighted_sum / (weight_total + 1e-6)

        # Average over heads [B, L, D]
        multi_head_out = head_outputs.mean(dim=2)

        # ========== Memory 2: Content → Value ==========
        # Key insight: use accumulated context (via cumsum) to inform storage key
        # This lets the model learn to store values under keys from prior context

        # Query key (for retrieval) - based on current content
        key_phases = torch.tanh(self.content_phase_encoder(x)) * math.pi  # [B, L, n_phases]
        key_cos = torch.cos(key_phases)
        key_sin = torch.sin(key_phases)

        values = self.value_encoder(x)  # [B, L, D]
        store_gate = self.store_gate(x)  # [B, L, 1]

        # Build context via cumsum (approximates "previous tokens")
        context_encoded = self.kv_context_encoder(x)  # [B, L, D]
        context_accum = torch.cumsum(context_encoded, dim=1)  # [B, L, D]
        # Normalize by position to get average context
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context_accum / positions

        # Storage key uses BOTH current content AND accumulated context
        # This lets value tokens "see" their preceding key tokens
        storage_input = torch.cat([x, context_avg], dim=-1)  # [B, L, dim*2]
        storage_phases = torch.tanh(self.kv_storage_key(storage_input)) * math.pi
        store_cos = torch.cos(storage_phases)
        store_sin = torch.sin(storage_phases)

        # Bind values with context-aware storage key (gated)
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)

        # Query with current content's key phase
        query_cos = key_cos.unsqueeze(-1)
        query_sin = key_sin.unsqueeze(-1)

        kv_retrieved = (kv_mem_cos * query_cos + kv_mem_sin * query_sin).sum(dim=2)
        kv_retrieved = kv_retrieved / (torch.sqrt(gate_cumsum) * math.sqrt(self.n_phases))

        # Combine multi-head positional retrieval with KV retrieval
        combined = multi_head_out + kv_retrieved

        return x + self.to_out(combined)


class PhasorModel(nn.Module):
    """
    Multi-head phasor model for sequence modeling.
    O(n) complexity via cumsum accumulation.
    """
    def __init__(self, vocab_size, dim=64, n_layers=4, n_heads=4, n_phases=32, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            PhasorBlock(dim, n_heads, n_phases, max_seq_len)
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

    model = PhasorModel(vocab_size=64, dim=64, n_layers=4, n_heads=4, n_phases=32).to(device)
    out = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhasorModel: {x.shape} -> {out.shape}")
    print(f"  Parameters: {n_params:,}")
