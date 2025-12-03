import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PureOrthoPhasor(nn.Module):
    """
    Orthogonal phasor with random fixed phases.

    Key insight: Random phases are approximately orthogonal in high dimensions.
    This enables O(n) associative retrieval via interference.
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        # Random phases - FIXED, not learned
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)

        value = self.to_value(x)

        # Bind: multiply value by complex exponential
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)

        # Cumsum = O(n) memory accumulation
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind: same phase retrieves original value
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)

        return x + self.to_out(retrieved)


class OptimalPhasorModel(nn.Module):
    """
    Minimal architecture: 2 phasor layers, no FFN.

    Why 2 layers: "Layer 1 retrieves, Layer 2 decodes"
    The two retrieval operations seem essential.
    """
    def __init__(self, vocab_size, dim=256, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.phasor1 = PureOrthoPhasor(dim, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor2 = PureOrthoPhasor(dim, max_seq_len)
        self.norm3 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = h + self.phasor1(self.norm1(h))
        h = h + self.phasor2(self.norm2(h))
        return self.head(self.norm3(h))
