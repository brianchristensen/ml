"""
Fast Batched Phase Attention - Semantic Phase Encoding

Key insight: Phase encodes SEMANTIC similarity (learned from data),
position information is added through a separate channel (additive encoding).

This allows phase coherence to measure semantic relationships rather than
just temporal synchronization, enabling language modeling and other tasks
requiring content-based attention.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class FastHierarchicalEncoder:
    """Batched hierarchical phase encoding."""

    def __init__(self, dim=512, device='cuda'):
        self.dim = dim
        self.device = device
        self.fast_period = 10
        self.medium_period = 100
        self.slow_period = 50
        self.vector_cache = {}

    def get_phases_batched(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get hierarchical phases for batch of positions.
        positions: [batch, seq_len] or [seq_len]
        Returns: [batch, seq_len, dim] or [seq_len, dim] complex
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        batch_size, seq_len = positions.shape

        # Decompose positions
        fast_pos = positions % self.fast_period
        medium_pos = (positions // self.fast_period) % self.medium_period
        slow_pos = (positions // (self.fast_period * self.medium_period)) % self.slow_period

        # Compute phases
        theta_fast = 2.0 * np.pi * fast_pos.float() / self.fast_period
        theta_medium = 2.0 * np.pi * medium_pos.float() / self.medium_period
        theta_slow = 2.0 * np.pi * slow_pos.float() / self.slow_period

        # Create phase tensor
        phase = torch.zeros(batch_size, seq_len, self.dim, dtype=torch.complex64, device=self.device)

        d1 = self.dim // 3
        d2 = 2 * self.dim // 3

        # Broadcast phases to dimensions
        phase[:, :, :d1] = torch.exp(1j * theta_fast.unsqueeze(-1))
        phase[:, :, d1:d2] = torch.exp(1j * theta_medium.unsqueeze(-1))
        phase[:, :, d2:] = torch.exp(1j * theta_slow.unsqueeze(-1))

        return phase


class FastPhaseAttention(nn.Module):
    """Fast batched phase coherence attention."""

    def __init__(self, dim=512, top_k=32):
        super().__init__()
        self.dim = dim
        self.top_k = top_k

        # Learnable weights (initialize small to prevent overflow)
        self.w_fast = nn.Parameter(torch.tensor([0.33]))
        self.w_medium = nn.Parameter(torch.tensor([0.33]))
        self.w_slow = nn.Parameter(torch.tensor([0.33]))
        self.content_scale = nn.Parameter(torch.tensor([1.0]))

        # Fixed temperature (not learnable to avoid NaN gradients with masking)
        self.register_buffer('temperature', torch.tensor([1.0]))

    def compute_coherence_batched(self, queries: torch.Tensor,
                                  keys: torch.Tensor) -> torch.Tensor:
        """
        queries: [batch, n_queries, dim] complex
        keys: [batch, n_keys, dim] complex
        Returns: [batch, n_queries, n_keys] scores
        """
        batch_size = queries.shape[0]
        n_queries = queries.shape[1]
        n_keys = keys.shape[1]

        d1 = self.dim // 3
        d2 = 2 * self.dim // 3

        # Extract phases [batch, n, 1] with safe handling
        # Add small epsilon to avoid NaN from zero complex numbers
        eps = 1e-8

        q_fast = torch.angle(queries[:, :, :d1] + eps).mean(dim=-1, keepdim=True)
        q_medium = torch.angle(queries[:, :, d1:d2] + eps).mean(dim=-1, keepdim=True)
        q_slow = torch.angle(queries[:, :, d2:] + eps).mean(dim=-1, keepdim=True)

        k_fast = torch.angle(keys[:, :, :d1] + eps).mean(dim=-1, keepdim=True)
        k_medium = torch.angle(keys[:, :, d1:d2] + eps).mean(dim=-1, keepdim=True)
        k_slow = torch.angle(keys[:, :, d2:] + eps).mean(dim=-1, keepdim=True)

        # Phase coherence [batch, n_queries, n_keys]
        coh_fast = torch.cos(q_fast - k_fast.transpose(1, 2))
        coh_medium = torch.cos(q_medium - k_medium.transpose(1, 2))
        coh_slow = torch.cos(q_slow - k_slow.transpose(1, 2))

        phase_score = (
            self.w_fast * coh_fast +
            self.w_medium * coh_medium +
            self.w_slow * coh_slow
        )

        # Content similarity
        q_mag = torch.abs(queries)  # [batch, n_queries, dim]
        k_mag = torch.abs(keys)     # [batch, n_keys, dim]

        q_norm = q_mag / (q_mag.norm(dim=-1, keepdim=True) + 1e-8)
        k_norm = k_mag / (k_mag.norm(dim=-1, keepdim=True) + 1e-8)

        content_score = torch.bmm(q_norm, k_norm.transpose(1, 2))

        return phase_score + self.content_scale * content_score

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        queries: [batch, n_queries, dim] complex
        keys: [batch, n_keys, dim] complex
        values: [batch, n_keys, value_dim] real
        key_mask: [batch, n_keys] bool (True = valid)

        Returns: [batch, n_queries, value_dim]
        """
        # Compute scores [batch, n_queries, n_keys]
        scores = self.compute_coherence_batched(queries, keys)

        # Clamp scores for stability
        scores = torch.clamp(scores, -10.0, 10.0)

        # Apply mask
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.unsqueeze(1), float('-inf'))

        # Top-k sparsity (optional - skip for speed during training)
        # if self.top_k < scores.shape[-1]:
        #     top_k_vals, top_k_idx = torch.topk(scores, self.top_k, dim=-1)
        #     sparse_scores = torch.full_like(scores, float('-inf'))
        #     sparse_scores.scatter_(-1, top_k_idx, top_k_vals)
        #     scores = sparse_scores

        # Attention with temperature
        attn_weights = torch.softmax(scores / self.temperature, dim=-1)

        # Check for NaN
        if torch.isnan(attn_weights).any():
            attn_weights = torch.ones_like(attn_weights) / attn_weights.shape[-1]

        output = torch.bmm(attn_weights, values)

        return output


class FastPhaseAttentionModel(nn.Module):
    """Fast batched phase attention model for benchmarking."""

    def __init__(self, vocab_size=30, dim=512, hidden_dim=256,
                 top_k=32, max_len=1000, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Phase encoder (kept for potential future experiments with hierarchical binding)
        # Currently not used in forward pass - using additive positional encoding instead
        self.encoder = FastHierarchicalEncoder(dim=dim, device=device)

        # Token embeddings (LEARNABLE!) - initialize small
        emb_real = torch.randn(vocab_size, dim) * 0.01
        emb_imag = torch.randn(vocab_size, dim) * 0.01

        # Make them learnable parameters (not frozen buffers!)
        self.token_embeddings_real = nn.Parameter(emb_real)
        self.token_embeddings_imag = nn.Parameter(emb_imag)

        # Learnable positional encoding (additive, like transformers)
        # This is real-valued and added to the real component only
        self.pos_encoding = nn.Parameter(torch.randn(max_len, dim) * 0.01)

        # Attention
        self.attention = FastPhaseAttention(dim=dim, top_k=top_k)

        # Value projection
        self.value_proj = nn.Linear(dim * 2, hidden_dim)

        # Output
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def get_embeddings(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Get complex embeddings for tokens."""
        real = self.token_embeddings_real[token_indices]
        imag = self.token_embeddings_imag[token_indices]
        return torch.complex(real, imag)

    def forward(self, input_indices: torch.Tensor,
                target_indices: Optional[torch.Tensor] = None,
                idx2token: Optional[Dict] = None) -> torch.Tensor:
        """
        input_indices: [batch, input_len]
        target_indices: [batch, output_len]

        Returns: [batch, output_len, vocab_size]
        """
        batch_size, input_len = input_indices.shape

        if target_indices is not None:
            output_len = target_indices.shape[1]
        else:
            output_len = input_len  # Default: same length

        # Get token embeddings [batch, input_len, dim]
        # These have SEMANTIC phase (learned from data)
        token_emb = self.get_embeddings(input_indices)

        # Add positional encoding to REAL component only (not phase!)
        # Phase remains semantic, position info added as additive bias
        positions = torch.arange(input_len, device=self.device)
        pos_enc = self.pos_encoding[positions]  # [input_len, dim]

        # Additive positional encoding on real component
        memory_real = token_emb.real + pos_enc  # [batch, input_len, dim]
        memory_imag = token_emb.imag  # Keep semantic phase intact

        memory_complex = torch.complex(memory_real, memory_imag)

        # Normalize to prevent overflow
        mag = torch.abs(memory_complex)
        memory_complex = memory_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        # Project to values [batch, input_len, hidden_dim]
        memory_real = torch.cat([memory_complex.real, memory_complex.imag], dim=-1)

        # Clamp to prevent extreme values
        memory_real = torch.clamp(memory_real, -10.0, 10.0)

        memory_values = self.value_proj(memory_real)

        # Create mask (valid for non-padding)
        memory_mask = (input_indices != 0)  # [batch, input_len]

        # Get query embeddings (semantic phase)
        if target_indices is not None:
            # Teacher forcing: use target tokens
            query_emb = self.get_embeddings(target_indices)
            # Query positions: same as input positions (predicting next token)
            query_positions = torch.arange(target_indices.shape[1], device=self.device)
        else:
            # Use a fixed query embedding (small, normalized)
            real = torch.ones(batch_size, output_len, self.dim, device=self.device) * 0.1
            imag = torch.zeros(batch_size, output_len, self.dim, device=self.device)
            query_emb = torch.complex(real, imag)
            # Query positions: same length as output
            query_positions = torch.arange(output_len, device=self.device)

        # Add positional encoding to REAL component only
        pos_enc = self.pos_encoding[query_positions]  # [query_len, dim]

        query_real = query_emb.real + pos_enc  # [batch, query_len, dim]
        query_imag = query_emb.imag  # Keep semantic phase intact

        queries_complex = torch.complex(query_real, query_imag)

        # Normalize queries
        mag = torch.abs(queries_complex)
        queries_complex = queries_complex / (mag.mean(dim=-1, keepdim=True) + 1e-8)

        # Attention [batch, output_len, hidden_dim]
        attended = self.attention(
            queries_complex,
            memory_complex,
            memory_values,
            memory_mask
        )

        # Output [batch, output_len, vocab_size]
        logits = self.output_head(attended)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
