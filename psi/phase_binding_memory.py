"""
Phase Binding Memory - O(n) Content-Addressable Memory

Core mechanism:
- Keys encoded as complex phasors: amp * exp(i * phase)
- Values modulated by key phasors and summed (superposition)
- Query demodulates via conjugate multiplication
- Matched phases reinforce, mismatched phases cancel
- RoPE-style positional encoding via phase rotation

Complexity: O(n) for n items - no attention matrix needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_freqs(n_oscillators: int, max_len: int, base: float = 10000.0):
    """
    Precompute RoPE frequency table for positional phase offsets.

    Returns angles theta[pos, k] = pos * freq[k] where freq[k] decays geometrically.
    """
    # Frequencies: 1/base^(2k/d) for k in [0, n_oscillators)
    freqs = 1.0 / (base ** (torch.arange(0, n_oscillators, dtype=torch.float32) / n_oscillators))
    # Position indices
    positions = torch.arange(max_len, dtype=torch.float32)
    # Outer product: [max_len, n_oscillators]
    angles = torch.outer(positions, freqs)
    return angles


class PhaseEncoder(nn.Module):
    """Encode content to complex phasor (amplitude * e^(i*phase))."""

    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.to_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, n_oscillators)
        )
        self.to_amp = nn.Sequential(
            nn.Linear(dim, n_oscillators),
            nn.Softplus()
        )

    def forward(self, x, pos_angles=None):
        """
        x: [..., D] -> complex phasor [..., K]
        pos_angles: [..., K] optional positional phase offsets (RoPE)
        """
        phase = torch.tanh(self.to_phase(x)) * math.pi
        if pos_angles is not None:
            phase = phase + pos_angles  # Add positional phase offset
        amp = self.to_amp(x) + 0.1
        return amp * torch.exp(1j * phase)


class PhaseBindingMemory(nn.Module):
    """
    Core O(n) phase binding memory.

    Write: modulate values by key phasors, sum into memory
    Read: demodulate memory with query phasor, matched phases reinforce
    """

    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = PhaseEncoder(dim, n_oscillators)
        self.query_encoder = PhaseEncoder(dim, n_oscillators)
        self.to_value = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, keys, values, queries):
        """
        Batched retrieval - fully vectorized.

        Args:
            keys: [B, N, D] - N key embeddings per batch
            values: [B, N, D] - N value embeddings per batch
            queries: [B, Q, D] - Q query embeddings per batch

        Returns:
            retrieved: [B, Q, D] - retrieved values for each query
        """
        B, N, D = keys.shape
        Q = queries.shape[1]
        K = self.n_oscillators

        # Encode keys to phasors [B, N, K]
        key_phasors = self.key_encoder(keys)

        # Project values [B, N, D]
        V = self.to_value(values)
        V_complex = V.to(torch.complex64)

        # Write: modulate and sum
        # key_phasors: [B, N, K] -> [B, N, K, 1]
        # V_complex: [B, N, D] -> [B, N, 1, D]
        modulated = key_phasors.unsqueeze(-1) * V_complex.unsqueeze(-2)  # [B, N, K, D]
        memory = modulated.sum(dim=1)  # [B, K, D]

        # Encode queries [B, Q, K]
        query_phasors = self.query_encoder(queries)

        # Read: demodulate for each query
        # memory: [B, K, D] -> [B, 1, K, D]
        # query_phasors.conj(): [B, Q, K] -> [B, Q, K, 1]
        demodulated = memory.unsqueeze(1) * query_phasors.conj().unsqueeze(-1)  # [B, Q, K, D]

        # Sum over oscillators, take real part
        retrieved = demodulated.sum(dim=2).real  # [B, Q, D]
        retrieved = retrieved / math.sqrt(N * K)

        return self.out(retrieved)


class PhaseBindingBlock(nn.Module):
    """
    Causal phase binding for sequence modeling.
    Uses cumsum for O(n) causal attention.
    Supports RoPE-style positional encoding via phase rotation.
    """

    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = PhaseEncoder(dim, n_oscillators)
        self.query_encoder = PhaseEncoder(dim, n_oscillators)
        self.to_value = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x, pos_angles=None):
        """
        x: [B, L, D] -> [B, L, D] with causal context
        pos_angles: [L, K] optional RoPE positional phase offsets
        """
        B, L, D = x.shape
        K = self.n_oscillators

        # Pass positional angles to encoders for RoPE
        key_phasors = self.key_encoder(x, pos_angles)  # [B, L, K]
        query_phasors = self.query_encoder(x, pos_angles)  # [B, L, K]
        V = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Modulate and cumsum for causal memory
        modulated = key_phasors.unsqueeze(-1) * V.unsqueeze(-2)  # [B, L, K, D]
        memory = torch.cumsum(modulated, dim=1)  # [B, L, K, D]

        # Demodulate at each position
        demodulated = memory * query_phasors.conj().unsqueeze(-1)  # [B, L, K, D]
        retrieved = demodulated.sum(dim=2).real  # [B, L, D]

        # Position-dependent normalization
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm

        return x + self.out(retrieved)


class PhaseBindingLanguageModel(nn.Module):
    """Language model using stacked PhaseBindingBlocks with RoPE positional encoding."""

    def __init__(self, vocab_size, dim=256, num_layers=8, n_oscillators=32,
                 max_len=2048, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.n_oscillators = n_oscillators

        self.embed = nn.Embedding(vocab_size, dim)
        # Keep additive pos_embed for compatibility, but RoPE in phasors is the key
        self.pos_embed = nn.Parameter(torch.randn(max_len, dim) * 0.02)

        # Precompute RoPE angles for phase binding [max_len, n_oscillators]
        rope_angles = precompute_freqs(n_oscillators, max_len)
        self.register_buffer('rope_angles', rope_angles)

        self.layers = nn.ModuleList([
            PhaseBindingBlock(dim, n_oscillators) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, x, target_indices=None):
        """x: [B, L] token indices -> [B, L, vocab_size] logits"""
        B, L = x.shape
        h = self.embed(x) + self.pos_embed[:L]

        # Get RoPE angles for current sequence length [L, K]
        pos_angles = self.rope_angles[:L]

        for norm, layer in zip(self.norms, self.layers):
            h = layer(norm(h), pos_angles)

        return self.head(self.norm_out(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        """Return the device of the model's parameters."""
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids, max_length, temperature=1.0, top_k=50):
        """Autoregressive generation."""
        self.eval()
        generated = input_ids.clone()

        while generated.shape[1] < max_length:
            context = generated[:, -self.max_len:]
            logits = self(context)[:, -1, :] / temperature

            if top_k > 0:
                top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = F.softmax(top_logits, dim=-1)
                next_token = top_idx.gather(-1, torch.multinomial(probs, 1))
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


class PhaseBindingRecall(nn.Module):
    """
    Associative recall model - fully vectorized.

    Input format: [k1, v1, k2, v2, ..., kN, vN, QUERY, q1, QUERY, q2, ...]
    Assumes fixed structure: n_pairs KV pairs followed by n_queries queries.
    """

    def __init__(self, vocab_size, dim=64, n_oscillators=32, query_token=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.query_token = query_token if query_token is not None else vocab_size

        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.memory = PhaseBindingMemory(dim, n_oscillators)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x, target_indices=None):
        """
        Fully vectorized forward pass.

        Args:
            x: [B, L] token indices or [B, L, V] one-hot

        Returns:
            logits: [B, L, vocab_size]
        """
        # Handle input format
        if x.dim() == 3:
            indices = x.argmax(dim=-1)
        else:
            indices = x

        B, L = indices.shape
        device = indices.device

        h = self.embed(indices)  # [B, L, D]

        # Find first QUERY position (all batches assumed same structure)
        query_mask = (indices == self.query_token)
        first_query_pos = query_mask.int().argmax(dim=1)  # [B]

        # For vectorization, assume all batches have same n_pairs
        # (standard for benchmarks with fixed structure)
        kv_end = first_query_pos[0].item()
        n_pairs = kv_end // 2

        # Extract keys and values (even/odd positions before QUERY)
        # keys: positions 0, 2, 4, ... -> [B, n_pairs, D]
        # values: positions 1, 3, 5, ... -> [B, n_pairs, D]
        key_indices = torch.arange(0, kv_end, 2, device=device)
        val_indices = torch.arange(1, kv_end, 2, device=device)

        keys = h[:, key_indices, :]    # [B, n_pairs, D]
        values = h[:, val_indices, :]  # [B, n_pairs, D]

        # Find all query positions (positions right after QUERY tokens)
        # Query structure: QUERY, q1, QUERY, q2, ...
        # We need positions kv_end+1, kv_end+3, kv_end+5, ...
        query_content_positions = []
        pos = kv_end + 1  # First query content is right after first QUERY
        while pos < L:
            query_content_positions.append(pos)
            pos += 2  # Skip QUERY token and move to next query content

        if not query_content_positions:
            # No queries, return zeros
            return torch.zeros(B, L, self.vocab_size, device=device)

        query_pos = torch.tensor(query_content_positions, device=device)
        n_queries = len(query_content_positions)

        # Extract query embeddings [B, n_queries, D]
        queries = h[:, query_pos, :]

        # Batch retrieval [B, n_queries, D]
        retrieved = self.memory(keys, values, queries)

        # Build output logits
        out = torch.zeros(B, L, self.vocab_size, device=device)
        logits = self.output(retrieved)  # [B, n_queries, vocab_size]

        # Scatter logits to correct positions
        for i, pos in enumerate(query_content_positions):
            out[:, pos, :] = logits[:, i, :]

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards compatibility alias
PhaseSpaceIntegrator = PhaseBindingLanguageModel
