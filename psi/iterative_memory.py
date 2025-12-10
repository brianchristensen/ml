"""
Iterative Memory with Successive Interference Cancellation

Key insight from CDMA/signal processing:
- Single-pass retrieval has SNR ~ 1/n (noise grows with stored items)
- Iterative cancellation: estimate strongest signal, subtract it, repeat
- Each iteration improves SNR for remaining signals

For associative memory:
1. Initial retrieval from cumsum memory (noisy)
2. Clean up via learned projection to nearest "clean" pattern
3. Subtract the cleaned signal from memory
4. Repeat for remaining signals

This is like unrolling an iterative algorithm (LISTA-style) into learnable layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PhasorMemoryBank(nn.Module):
    """
    Single phasor memory bank with cumsum accumulation.
    Returns raw (noisy) retrieval - cleanup happens externally.
    """
    def __init__(self, dim, n_phases=32):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

        # Content-based phase encoding (shared for bind/unbind)
        self.phase_encoder = nn.Linear(dim, n_phases)
        self.to_value = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, L, D]
        Returns: memory state [B, L, D] and phases [B, L, n_phases]
        """
        B, L, D = x.shape

        # Get content-based phases
        phases = torch.tanh(self.phase_encoder(x)) * math.pi  # [B, L, n_phases]
        phasors = torch.exp(1j * phases)  # [B, L, n_phases]

        # Get values
        values = self.to_value(x)  # [B, L, D]

        # Bind: multiply value by phasor (broadcast over D)
        # phasors: [B, L, n_phases] -> [B, L, n_phases, 1]
        # values: [B, L, D] -> [B, L, 1, D]
        bound = phasors.unsqueeze(-1) * values.unsqueeze(2).to(torch.complex64)  # [B, L, n_phases, D]

        # Cumsum over sequence
        memory = torch.cumsum(bound, dim=1)  # [B, L, n_phases, D]

        return memory, phases, phasors

    def retrieve(self, memory, phases):
        """
        Retrieve from memory using query phases.
        memory: [B, L, n_phases, D]
        phases: [B, L, n_phases]
        Returns: [B, L, D]
        """
        query_phasors = torch.exp(-1j * phases)  # Conjugate for unbinding

        # Unbind: multiply memory by conjugate phasor
        retrieved = memory * query_phasors.unsqueeze(-1)  # [B, L, n_phases, D]

        # Sum over phases and take real part
        retrieved = retrieved.sum(dim=2).real  # [B, L, D]

        return retrieved


class CleanupMemory(nn.Module):
    """
    Learned cleanup that projects noisy retrieval toward clean patterns.

    Inspired by:
    - HRR cleanup memory (auto-associative projection)
    - LISTA learned thresholding (soft projection)
    - Hopfield energy minimization (attractor dynamics)
    """
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        hidden = dim * hidden_mult

        # Learned projection to clean up noisy retrieval
        self.cleanup = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, noisy, residual_weight=0.5):
        """
        Clean up noisy retrieval.
        Returns cleaned signal and residual (noise estimate).
        """
        cleaned = self.cleanup(noisy)
        residual = noisy - cleaned * residual_weight
        return cleaned, residual


class IterativeCancellationBlock(nn.Module):
    """
    Iterative interference cancellation for associative memory.

    Algorithm:
    1. Store all key-value pairs in phasor memory via cumsum
    2. For each query, retrieve initial (noisy) estimate
    3. Clean up the estimate via learned projection
    4. Subtract cleaned estimate's contribution from memory (cancellation)
    5. Re-retrieve with improved SNR
    6. Repeat for n_iterations

    This is analogous to:
    - SIC in CDMA: cancel strongest user, decode next
    - LISTA: learned iterative sparse recovery
    - Hopfield: iterative energy minimization
    """
    def __init__(self, dim, n_phases=32, n_iterations=3):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.n_iterations = n_iterations

        # Phasor memory for storage
        self.memory_bank = PhasorMemoryBank(dim, n_phases)

        # Cleanup module for each iteration (could share weights)
        self.cleanups = nn.ModuleList([
            CleanupMemory(dim) for _ in range(n_iterations)
        ])

        # Learned cancellation strength per iteration
        self.cancel_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(n_iterations)
        ])

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # Build memory
        memory, phases, phasors = self.memory_bank(x)  # [B, L, n_phases, D]

        # Initial retrieval
        retrieved = self.memory_bank.retrieve(memory, phases)  # [B, L, D]

        # Normalize by position (cumsum grows with position)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1)
        retrieved = retrieved / (norm * math.sqrt(self.n_phases))

        # Iterative refinement with cancellation
        accumulated = torch.zeros_like(retrieved)

        for i in range(self.n_iterations):
            # Clean up current estimate
            cleaned, residual = self.cleanups[i](retrieved)

            # Accumulate cleaned estimates
            accumulated = accumulated + cleaned

            # Cancel the cleaned signal's contribution from memory
            # Re-bind the cleaned signal and subtract from memory
            cancel_weight = torch.sigmoid(self.cancel_weights[i])

            # Compute what the cleaned signal contributed to memory
            cleaned_bound = phasors.unsqueeze(-1) * cleaned.unsqueeze(2).to(torch.complex64)
            cleaned_contrib = torch.cumsum(cleaned_bound, dim=1) * cancel_weight

            # Subtract from memory
            memory = memory - cleaned_contrib

            # Re-retrieve from cleaned memory
            retrieved = self.memory_bank.retrieve(memory, phases)
            retrieved = retrieved / (norm * math.sqrt(self.n_phases))

        # Final cleanup of remaining signal
        final_cleaned, _ = self.cleanups[-1](retrieved) if self.n_iterations > 0 else (retrieved, None)
        accumulated = accumulated + final_cleaned

        return x + self.to_out(accumulated)


class IterativeMemoryModel(nn.Module):
    """
    Full model with iterative cancellation memory blocks.
    """
    def __init__(self, vocab_size, dim=64, n_layers=2, n_phases=32, n_iterations=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            IterativeCancellationBlock(dim, n_phases, n_iterations)
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


# Alternative: Simpler iterative refinement without explicit cancellation
class IterativeRefinementBlock(nn.Module):
    """
    Simpler approach: multiple retrieval passes with learned refinement.

    Each iteration:
    1. Retrieve from memory
    2. Refine the retrieval via learned projection
    3. Use refined output to update query (like attention refinement)
    """
    def __init__(self, dim, n_phases=32, n_iterations=3):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.n_iterations = n_iterations

        # Phase encoder
        self.phase_encoder = nn.Linear(dim, n_phases)
        self.to_value = nn.Linear(dim, dim)

        # Refinement networks per iteration
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            for _ in range(n_iterations)
        ])

        # Query update gate
        self.query_gates = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(n_iterations)
        ])

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # Build memory once
        phases = torch.tanh(self.phase_encoder(x)) * math.pi
        phasors = torch.exp(1j * phases)
        values = self.to_value(x).to(torch.complex64)

        # Bind and cumsum
        bound = phasors.unsqueeze(-1) * values.unsqueeze(2)  # [B, L, n_phases, D]
        memory = torch.cumsum(bound, dim=1)

        # Normalization
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1) * math.sqrt(self.n_phases)

        # Start with original query
        query = x
        accumulated = torch.zeros_like(x)

        for i in range(self.n_iterations):
            # Get query phases
            q_phases = torch.tanh(self.phase_encoder(query)) * math.pi
            q_phasors = torch.exp(-1j * q_phases)

            # Retrieve
            retrieved = (memory * q_phasors.unsqueeze(-1)).sum(dim=2).real / norm

            # Refine
            refined = self.refiners[i](retrieved)
            accumulated = accumulated + refined

            # Update query for next iteration (gated update)
            gate_input = torch.cat([query, refined], dim=-1)
            query = query + torch.tanh(self.query_gates[i](gate_input))

        return x + self.to_out(accumulated)


class IterativeRefinementModel(nn.Module):
    """Full model with iterative refinement blocks."""
    def __init__(self, vocab_size, dim=64, n_layers=2, n_phases=32, n_iterations=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            IterativeRefinementBlock(dim, n_phases, n_iterations)
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


# ============================================================================
# Enhanced: Multi-Bank Iterative Refinement with Diverse Phase Codes
# ============================================================================

class MultiBankIterativeBlock(nn.Module):
    """
    Multi-bank iterative refinement inspired by CDMA spreading codes.

    Key ideas:
    1. Multiple independent phase banks = different "spreading codes"
    2. Each bank stores the same content but with different addressing
    3. Cross-bank consensus improves retrieval accuracy
    4. Iterative refinement with bank-aware cleanup

    This is like having K parallel CDMA channels - interference averages
    out when we combine retrievals from banks with orthogonal codes.
    """
    def __init__(self, dim, n_phases=16, n_banks=4, n_iterations=3):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.n_banks = n_banks
        self.n_iterations = n_iterations

        # Independent phase encoder per bank (different "spreading codes")
        self.phase_encoders = nn.ModuleList([
            nn.Linear(dim, n_phases) for _ in range(n_banks)
        ])

        # Value projection (shared across banks)
        self.to_value = nn.Linear(dim, dim)

        # Bank-wise refinement
        self.bank_refiners = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim * n_banks),
                nn.Linear(dim * n_banks, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(n_iterations)
        ])

        # Query update (informed by all banks)
        self.query_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Tanh()
            ) for _ in range(n_iterations)
        ])

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        # Initialize phase encoders to be diverse
        for i, enc in enumerate(self.phase_encoders):
            nn.init.orthogonal_(enc.weight)
            # Add bank-specific bias to encourage diversity
            enc.bias.data = torch.randn_like(enc.bias.data) * 0.5

    def forward(self, x):
        B, L, D = x.shape

        # Shared value
        values = self.to_value(x).to(torch.complex64)  # [B, L, D]

        # Build memory for each bank
        memories = []
        all_phasors = []
        for enc in self.phase_encoders:
            phases = torch.tanh(enc(x)) * math.pi  # [B, L, n_phases]
            phasors = torch.exp(1j * phases)  # [B, L, n_phases]

            # Bind: value * phasor (broadcast over dim)
            bound = phasors.unsqueeze(-1) * values.unsqueeze(2)  # [B, L, n_phases, D]
            memory = torch.cumsum(bound, dim=1)  # [B, L, n_phases, D]

            memories.append(memory)
            all_phasors.append(phasors)

        # Normalization
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1) * math.sqrt(self.n_phases)

        # Iterative refinement with cross-bank consensus
        query = x
        accumulated = torch.zeros_like(x)

        for i in range(self.n_iterations):
            # Retrieve from each bank with current query
            bank_retrievals = []
            for j, enc in enumerate(self.phase_encoders):
                q_phases = torch.tanh(enc(query)) * math.pi
                q_phasors = torch.exp(-1j * q_phases)  # Conjugate for unbinding

                # Unbind and sum over phases
                retrieved = (memories[j] * q_phasors.unsqueeze(-1)).sum(dim=2).real / norm
                bank_retrievals.append(retrieved)

            # Concatenate bank retrievals for cross-bank refinement
            combined = torch.cat(bank_retrievals, dim=-1)  # [B, L, D*n_banks]

            # Refine with cross-bank awareness
            refined = self.bank_refiners[i](combined)  # [B, L, D]
            accumulated = accumulated + refined

            # Update query
            gate_input = torch.cat([query, refined], dim=-1)
            query = query + self.query_updates[i](gate_input)

        return x + self.to_out(accumulated)


class MultiBankIterativeModel(nn.Module):
    """Full model with multi-bank iterative refinement."""
    def __init__(self, vocab_size, dim=64, n_layers=2, n_phases=16, n_banks=4, n_iterations=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            MultiBankIterativeBlock(dim, n_phases, n_banks, n_iterations)
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


# ============================================================================
# Enhanced: Attention-Guided Iterative Refinement
# ============================================================================

class AttentionGuidedIterativeBlock(nn.Module):
    """
    Iterative refinement with lightweight attention guidance.

    Key idea: Use cheap O(n*k) attention to guide the iterative cleanup,
    where k << n is a small number of "memory heads".

    This combines:
    1. Phasor memory for O(n) storage
    2. Sparse attention for retrieval guidance
    3. Iterative refinement for noise cleanup
    """
    def __init__(self, dim, n_phases=32, n_iterations=3, n_memory_heads=8):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.n_iterations = n_iterations
        self.n_memory_heads = n_memory_heads

        # Phasor memory
        self.phase_encoder = nn.Linear(dim, n_phases)
        self.to_value = nn.Linear(dim, dim)

        # Memory heads for attention guidance
        self.memory_keys = nn.Parameter(torch.randn(n_memory_heads, dim) * 0.02)
        self.memory_query = nn.Linear(dim, n_memory_heads)

        # Refinement with attention guidance
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim + n_memory_heads),  # Include attention scores
                nn.Linear(dim + n_memory_heads, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(n_iterations)
        ])

        # Query update
        self.query_gates = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(n_iterations)
        ])

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # Build phasor memory
        phases = torch.tanh(self.phase_encoder(x)) * math.pi
        phasors = torch.exp(1j * phases)
        values = self.to_value(x).to(torch.complex64)

        bound = phasors.unsqueeze(-1) * values.unsqueeze(2)
        memory = torch.cumsum(bound, dim=1)

        # Normalization
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype)
        norm = torch.sqrt(positions).view(1, L, 1) * math.sqrt(self.n_phases)

        # Iterative refinement
        query = x
        accumulated = torch.zeros_like(x)

        for i in range(self.n_iterations):
            # Get phasor retrieval
            q_phases = torch.tanh(self.phase_encoder(query)) * math.pi
            q_phasors = torch.exp(-1j * q_phases)
            retrieved = (memory * q_phasors.unsqueeze(-1)).sum(dim=2).real / norm

            # Get attention guidance (what memory heads are activated?)
            attn_scores = F.softmax(self.memory_query(query), dim=-1)  # [B, L, n_heads]

            # Combine retrieval with attention guidance
            combined = torch.cat([retrieved, attn_scores], dim=-1)

            # Refine
            refined = self.refiners[i](combined)
            accumulated = accumulated + refined

            # Update query
            gate_input = torch.cat([query, refined], dim=-1)
            query = query + torch.tanh(self.query_gates[i](gate_input))

        return x + self.to_out(accumulated)


class AttentionGuidedIterativeModel(nn.Module):
    """Full model with attention-guided iterative refinement."""
    def __init__(self, vocab_size, dim=64, n_layers=2, n_phases=32, n_iterations=3, n_memory_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            AttentionGuidedIterativeBlock(dim, n_phases, n_iterations, n_memory_heads)
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
    # Quick test
    model = IterativeMemoryModel(vocab_size=64, dim=64, n_layers=2, n_iterations=3).to(device)
    x = torch.randint(0, 64, (2, 32), device=device)
    out = model(x)
    print(f"IterativeMemoryModel: {x.shape} -> {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model2 = IterativeRefinementModel(vocab_size=64, dim=64, n_layers=2, n_iterations=3).to(device)
    out2 = model2(x)
    print(f"IterativeRefinementModel: {x.shape} -> {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    model3 = MultiBankIterativeModel(vocab_size=64, dim=64, n_layers=2, n_phases=16, n_banks=4, n_iterations=3).to(device)
    out3 = model3(x)
    print(f"MultiBankIterativeModel: {x.shape} -> {out3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")

    model4 = AttentionGuidedIterativeModel(vocab_size=64, dim=64, n_layers=2, n_iterations=3).to(device)
    out4 = model4(x)
    print(f"AttentionGuidedIterativeModel: {x.shape} -> {out4.shape}")
    print(f"Parameters: {sum(p.numel() for p in model4.parameters()):,}")
