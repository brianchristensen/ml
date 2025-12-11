"""
Phasor Memory Model - Two-Pass Architecture

Two-pass retrieval within a single layer:
1. Pass 1: Query by content → retrieve position phase where content matched
2. Pass 2: Use retrieved position + learned offset → retrieve value

This mimics transformer's induction head mechanism in O(n) time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhasorBlock(nn.Module):
    """
    Two-pass phasor memory block with LEARNED positional offset.

    Pass 1 (Position → Content):
        - Store content under position phases
        - Query with current position + learned offset → retrieve content from relative position

    Pass 2 (Content → Value):
        - Store values under content phases
        - Query with retrieved content → get associated value

    The key innovation: learned phase offset allows model to learn WHERE to look
    without hardcoded torch.roll. Uses rotation: query(θ + Δ) = rotation_matrix @ query(θ)
    """
    def __init__(self, dim, n_phases=32, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

        # Fixed random positional phases
        pos_phases = torch.randn(max_seq_len, n_phases) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        # Content phase encoder
        self.content_phase_encoder = nn.Linear(dim, n_phases)

        # Value encoder for positional memory
        self.value_encoder = nn.Linear(dim, dim)

        # Memory 1: Position -> Content
        self.mem1_content_encoder = nn.Linear(dim, dim)

        # LEARNED OFFSET: predicts phase offset based on current token
        # This replaces the hardcoded torch.roll
        self.offset_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)  # Output: offset phases per phase dimension
        )

        # Memory 2: Content -> Value (standard associative memory)

        # Gates
        self.content_store_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.pos_store_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # Position phases (fixed random)
        pos_phases = self.pos_phases[:L]  # [L, n_phases]

        # ========== Memory 1: Position -> Content (with LEARNED offset) ==========
        # Store content[t] under pos_phase[t]
        # Query with pos_phase[t] + learned_offset to retrieve content from relative position
        # The offset is learned from the current token - model learns "when I see X, look back by Δ"

        content_for_mem1 = self.mem1_content_encoder(x)  # [B, L, D]

        # Store under current position (no shift!)
        pos_cos = torch.cos(pos_phases)  # [L, n_phases]
        pos_sin = torch.sin(pos_phases)

        # Bind content with position phases
        # Shape: [1, L, n_phases] * [B, L, 1, D] -> [B, L, n_phases, D]
        mem1_bound_cos = pos_cos.unsqueeze(0).unsqueeze(-1) * content_for_mem1.unsqueeze(2)
        mem1_bound_sin = pos_sin.unsqueeze(0).unsqueeze(-1) * content_for_mem1.unsqueeze(2)

        mem1_cos = torch.cumsum(mem1_bound_cos, dim=1)
        mem1_sin = torch.cumsum(mem1_bound_sin, dim=1)

        # LEARNED OFFSET: predict phase offset from current token
        # This replaces hardcoded torch.roll
        offset_phases = torch.tanh(self.offset_predictor(x)) * math.pi  # [B, L, n_phases]
        offset_cos = torch.cos(offset_phases)  # [B, L, n_phases]
        offset_sin = torch.sin(offset_phases)

        # Apply rotation to query: query(θ + Δ) = rotation @ query(θ)
        # cos(θ + Δ) = cos(θ)cos(Δ) - sin(θ)sin(Δ)
        # sin(θ + Δ) = sin(θ)cos(Δ) + cos(θ)sin(Δ)
        query1_cos_base = pos_cos.unsqueeze(0)  # [1, L, n_phases]
        query1_sin_base = pos_sin.unsqueeze(0)

        query1_cos = query1_cos_base * offset_cos - query1_sin_base * offset_sin  # [B, L, n_phases]
        query1_sin = query1_sin_base * offset_cos + query1_cos_base * offset_sin

        # Expand for retrieval
        query1_cos = query1_cos.unsqueeze(-1)  # [B, L, n_phases, 1]
        query1_sin = query1_sin.unsqueeze(-1)

        # Retrieved content from Memory 1
        retrieved_content = (mem1_cos * query1_cos + mem1_sin * query1_sin).sum(dim=2)  # [B, L, D]
        retrieved_content = retrieved_content / math.sqrt(self.n_phases)

        # ========== Memory 2: Content -> Value ==========
        # Store values under content phases, query with retrieved_content

        # Content phases for querying (from current input)
        query_phases = torch.tanh(self.content_phase_encoder(x)) * math.pi  # [B, L, n_phases]

        # Store phases from retrieved content (what key was before this value)
        store_phases = torch.tanh(self.content_phase_encoder(retrieved_content)) * math.pi  # [B, L, n_phases]

        # Gates
        key_gate = self.content_store_gate(x)  # [B, L, 1]
        value_gate = self.pos_store_gate(x)  # [B, L, 1]

        values = self.value_encoder(x)  # [B, L, D]

        # Store in Memory 2
        store_cos = torch.cos(store_phases)
        store_sin = torch.sin(store_phases)

        store_gate = value_gate

        bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        mem2_cos = torch.cumsum(bound_cos, dim=1)
        mem2_sin = torch.cumsum(bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)

        # Query Memory 2 with current content
        query_cos = torch.cos(query_phases).unsqueeze(-1)
        query_sin = torch.sin(query_phases).unsqueeze(-1)

        retrieved = (mem2_cos * query_cos + mem2_sin * query_sin).sum(dim=2)
        retrieved = retrieved / (torch.sqrt(gate_cumsum) * math.sqrt(self.n_phases))

        retrieved = retrieved * key_gate

        return x + self.to_out(retrieved)


class PhasorModel(nn.Module):
    """
    Full phasor memory model for sequence modeling.
    O(n) complexity via cumsum accumulation.
    """
    def __init__(self, vocab_size, dim=64, n_layers=4, n_phases=32, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            PhasorBlock(dim, n_phases, max_seq_len) for _ in range(n_layers)
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
