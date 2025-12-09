"""
Hypersphere: Single Memory, Two Addressing Modes

    memory = cumsum(value ⊗ combined_key)              # SINGLE memory

    Where combined_key = f(position, content)

    Retrieval can use:
    1. Positional query: query_by_position(pos)
    2. Content query: query_by_content(x)
    3. Both: query_by_both(pos, content)

This is analogous to:
- RAM with both address lines AND content-addressable cache
- Hash table with both index and content-based lookup
- Database with both primary key and content index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')


# =============================================================================
# Hypersphere: Single Memory, Multiple Access Modes
# =============================================================================

class Hypersphere(nn.Module):
    """
    Single memory with both positional and content-based addressing.

    Key insight: The "key" for storage combines position AND content.
    Retrieval can query by either or both.

    Storage: memory += value ⊗ composite_key(position, content)

    Retrieval options:
    1. By position: query with pos_component
    2. By content: query with content_component
    3. By both: query with full composite key
    """
    def __init__(self, dim, max_seq_len=2048, key_dim=16):
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim

        # === Storage ===
        self.to_value = nn.Linear(dim, dim)

        # Positional component of key (fixed, like original phasor)
        pos_key = torch.randn(max_seq_len, key_dim)
        pos_key = F.normalize(pos_key, dim=-1)  # Unit vectors
        self.register_buffer('pos_key', pos_key)

        # Content component of key (learned projection)
        self.to_content_key = nn.Linear(dim, key_dim)

        # How to combine pos and content keys
        self.key_combiner = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.Tanh()
        )

        # === Retrieval ===
        # Position-based query
        self.pos_query_weight = nn.Parameter(torch.ones(1) * 0.5)

        # Content-based query
        self.to_content_query = nn.Linear(dim, key_dim)
        self.content_query_weight = nn.Parameter(torch.ones(1) * 0.5)

        # Output
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # === STORAGE: Combine position + content into composite key ===
        value = self.to_value(x)  # [B, L, D]

        # Positional key component
        pos_k = self.pos_key[:L].unsqueeze(0).expand(B, -1, -1)  # [B, L, key_dim]

        # Content key component
        content_k = self.to_content_key(x)  # [B, L, key_dim]
        content_k = F.normalize(content_k, dim=-1)

        # Composite key: combines both
        composite_key = self.key_combiner(torch.cat([pos_k, content_k], dim=-1))
        composite_key = F.normalize(composite_key, dim=-1)  # [B, L, key_dim]

        # Bind value with composite key (hypersphere-style outer product)
        # memory[d] = cumsum(value * key[d]) for each key dimension
        memory = []
        for k in range(self.key_dim):
            key_slice = composite_key[:, :, k:k+1]  # [B, L, 1]
            weighted_value = value * key_slice  # [B, L, D]
            mem_k = torch.cumsum(weighted_value, dim=1)  # [B, L, D]
            memory.append(mem_k)
        # memory is list of [B, L, D], one per key dimension

        # === RETRIEVAL: Can query by position, content, or both ===

        # Positional query component
        pos_q = self.pos_key[:L].unsqueeze(0).expand(B, -1, -1)

        # Content query component
        content_q = self.to_content_query(x)
        content_q = F.normalize(content_q, dim=-1)

        # Composite query (same combination as storage)
        composite_query = self.key_combiner(torch.cat([pos_q, content_q], dim=-1))
        composite_query = F.normalize(composite_query, dim=-1)

        # Retrieve by querying with composite key
        retrieved = torch.zeros_like(value)
        for k in range(self.key_dim):
            query_slice = composite_query[:, :, k:k+1]
            retrieved = retrieved + memory[k] * query_slice

        retrieved = retrieved / math.sqrt(self.key_dim)

        return x + self.to_out(retrieved)

class HypersphereModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, output_dim=None,
                 key_dim=16, variant='composite'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            Hypersphere(hidden_dim, key_dim=key_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.output_proj(h)
    