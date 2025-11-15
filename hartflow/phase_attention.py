"""
Low-Rank Linear Attention - Fast & Content-Based

ALIGNMENT WITH OBJECTIVES:
1. ✓ O(n·k·d) where k << d: Much faster than O(n²·d)
2. ✓ NO sequential loops: All parallel
3. ✓ Content-based selectivity: φ(Q)·ψ(K) matching
4. ✓ Fast: Small k (32-64) makes KV memory tiny
5. ✓ Parallelizable: cumsum is GPU-parallel

KEY TRICK:
Project K and V to low dimension k before outer product.
Memory is [n, k, k] not [n, d, d] - MUCH smaller!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LowRankLinearAttention(nn.Module):
    """
    Linear attention with low-rank keys/values for speed.

    Memory size: O(k²) instead of O(d²) where k << d
    """

    def __init__(self, dim: int, key_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim  # Small dimension for KV memory

        # Project to low-rank space for keys
        self.W_q = nn.Linear(dim, key_dim, bias=False)
        self.W_k = nn.Linear(dim, key_dim, bias=False)
        self.W_v = nn.Linear(dim, key_dim, bias=False)

        # Project back to full dimension
        self.W_o = nn.Linear(key_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            output: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape
        k = self.key_dim

        # Project to low-rank space
        Q = self.W_q(x)  # [batch, seq_len, k]
        K = self.W_k(x)  # [batch, seq_len, k]
        V = self.W_v(x)  # [batch, seq_len, k]

        # Feature map for positivity
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Outer product in LOW-RANK space: [batch, seq_len, k, k]
        KV = K.unsqueeze(-1) * V.unsqueeze(-2)

        # Cumsum for causal attention
        KV_sum = torch.cumsum(KV, dim=1)  # [batch, seq_len, k, k]
        K_sum = torch.cumsum(K, dim=1)    # [batch, seq_len, k]

        # Retrieve: Q @ KV_sum
        output = torch.matmul(Q.unsqueeze(-2), KV_sum).squeeze(-2)

        # Normalize
        normalizer = torch.matmul(Q.unsqueeze(-2), K_sum.unsqueeze(-1)).squeeze(-1)
        output = output / (normalizer + 1e-6)

        # Project back to full dimension
        output = self.W_o(output)

        return output


class PhaseAttentionLM(nn.Module):
    """
    Language Model using Low-Rank Linear Attention.

    Fast and content-based!
    """

    def __init__(self, vocab_size: int = 256, dim: int = 256, hidden_dim: int = 512,
                 max_len: int = 1024, n_heads: int = 2, device: str = 'cuda',
                 context_window: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        # Low-rank linear attention layers
        # Use key_dim=64 for good speed/quality tradeoff
        self.layers = nn.ModuleList([
            LowRankLinearAttention(dim, key_dim=64)
            for _ in range(n_heads)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(n_heads)
        ])

        # Output
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, input_indices: torch.Tensor,
                target_indices: Optional[torch.Tensor] = None,
                idx2token: Optional[dict] = None) -> torch.Tensor:
        batch, seq_len = input_indices.shape

        # Embed
        positions = torch.arange(seq_len, device=self.device)
        x = self.token_embed(input_indices) + self.pos_embed(positions)

        # Process through layers
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(x)
            x = norm(x)

        # Output
        x = self.output_norm(x)
        logits = self.output(x)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias
MultiHeadPhaseAttentionLM = PhaseAttentionLM
