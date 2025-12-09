"""
Debug: Why does PSI fail at extrapolation?

The comprehensive benchmark shows PSI degrades 57x when going from seq_len=50 to 800,
while LSTM actually IMPROVES (0.28x). This is unexpected - PSI's cumsum should
theoretically handle any length.

Let's trace what happens at different sequence lengths.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import from psi.py
import sys
sys.path.insert(0, '.')
from psi import PSI, PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def analyze_psi_internals():
    """Trace PSI internals at different sequence lengths."""
    print("=" * 70)
    print("PSI EXTRAPOLATION DEBUG")
    print("=" * 70)

    dim = 64
    psi = PSI(dim).to(device)
    psi.eval()

    # Test at different sequence lengths
    seq_lens = [50, 100, 200, 400, 800]

    print("\n1. PHASE ACCUMULATION ANALYSIS")
    print("-" * 50)

    for seq_len in seq_lens:
        x = torch.randn(1, seq_len, dim, device=device)

        with torch.no_grad():
            # Compute omega
            omega = psi.to_omega(x)

            # Position scaling
            position = torch.arange(1, seq_len + 1, device=device, dtype=x.dtype).view(1, -1, 1)
            pos_scale = 1.0 / torch.sqrt(position)

            scale = torch.exp(psi.log_scale)
            omega_scaled = omega * scale * pos_scale

            # Phase accumulation
            phi = torch.cumsum(omega_scaled, dim=1)

            # Stats
            phi_final = phi[:, -1, :].abs().mean().item()
            phi_max = phi.abs().max().item()
            omega_mean = omega.abs().mean().item()
            pos_scale_final = pos_scale[:, -1, 0].item()

            print(f"\nSeq len {seq_len}:")
            print(f"  omega mean |value|: {omega_mean:.4f}")
            print(f"  pos_scale at end: {pos_scale_final:.4f}")
            print(f"  phi final mean: {phi_final:.4f}")
            print(f"  phi max: {phi_max:.4f}")

    print("\n\n2. MEMORY NORMALIZATION ISSUE")
    print("-" * 50)
    print("The memory is normalized by position: memory / position")
    print("At long sequences, this makes early contributions vanishingly small.")

    for seq_len in seq_lens:
        x = torch.randn(1, seq_len, dim, device=device)

        with torch.no_grad():
            omega = psi.to_omega(x)
            position = torch.arange(1, seq_len + 1, device=device, dtype=x.dtype).view(1, -1, 1)
            pos_scale = 1.0 / torch.sqrt(position)
            scale = torch.exp(psi.log_scale)
            omega_scaled = omega * scale * pos_scale
            phi = torch.cumsum(omega_scaled, dim=1)

            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            memory_real = torch.cumsum(x * cos_phi, dim=1)
            memory_imag = torch.cumsum(x * sin_phi, dim=1)

            # This is the problem:
            memory_real_normalized = memory_real / position
            memory_imag_normalized = memory_imag / position

            # At the final position
            final_mem_real = memory_real_normalized[:, -1, :].abs().mean().item()
            final_mem_imag = memory_imag_normalized[:, -1, :].abs().mean().item()

            # What if we used running average instead?
            # memory_avg = memory / position gives more weight to recent
            # But content magnitude stays ~1, so memory/position -> 0 as position -> inf

            print(f"\nSeq len {seq_len}:")
            print(f"  Final normalized memory magnitude: {final_mem_real:.4f}")
            print(f"  Theoretical decay: 1/{seq_len} = {1/seq_len:.4f}")

    print("\n\n3. THE CORE ISSUE")
    print("-" * 50)
    print("""
    The PSI architecture has a fundamental extrapolation problem:

    memory_normalized = cumsum(x * cos(phi)) / position

    As position -> infinity, memory_normalized -> 0

    This is like computing a running average, which converges to 0 if inputs
    are zero-mean (which they typically are after LayerNorm).

    During training at length 50, the model learns to work with memory ~ 1/50 scale.
    At length 800, memory becomes ~ 1/800 scale = 16x smaller!

    The output projection was trained to expect signals at 1/50 scale,
    so at length 800 it sees signals that are 16x too small.
    """)

    print("\n\n4. POTENTIAL FIXES")
    print("-" * 50)
    print("""
    Option A: Normalize by sqrt(position) instead of position
              This gives O(1/sqrt(n)) decay instead of O(1/n)

    Option B: Use exponential moving average instead of cumsum
              mem[t] = alpha * mem[t-1] + (1-alpha) * x[t]
              This has bounded magnitude regardless of length

    Option C: Chunk-based normalization
              Normalize within fixed-size windows

    Option D: Remove position normalization entirely
              Let the model learn to handle growing memory
              (but this might cause instability)

    Option E: Learn a position-dependent scaling factor
              Instead of 1/position, use learned_scale(position)
    """)

    return


def test_fix_sqrt_normalization():
    """Test if sqrt normalization helps."""
    print("\n\n5. TESTING SQRT NORMALIZATION FIX")
    print("=" * 70)

    dim = 64

    class PSI_SqrtNorm(nn.Module):
        """PSI with sqrt position normalization."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.to_omega = nn.Linear(dim, dim)
            self.log_scale = nn.Parameter(torch.ones(dim) * np.log(0.1))
            self.to_out = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )

        def forward(self, x):
            batch_size, seq_len, dim = x.shape

            omega = self.to_omega(x)
            position = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            pos_scale = 1.0 / torch.sqrt(position)

            scale = torch.exp(self.log_scale)
            omega_scaled = omega * scale * pos_scale
            phi = torch.cumsum(omega_scaled, dim=1)

            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            memory_real = torch.cumsum(x * cos_phi, dim=1)
            memory_imag = torch.cumsum(x * sin_phi, dim=1)

            # KEY CHANGE: sqrt normalization
            memory_real_normalized = memory_real / torch.sqrt(position)
            memory_imag_normalized = memory_imag / torch.sqrt(position)

            retrieved_real = memory_real_normalized * cos_phi + memory_imag_normalized * sin_phi
            retrieved_imag = memory_imag_normalized * cos_phi - memory_real_normalized * sin_phi

            content_modulated_real = x * cos_phi
            content_modulated_imag = x * sin_phi

            context = torch.cat([
                content_modulated_real,
                content_modulated_imag,
                retrieved_real,
                retrieved_imag
            ], dim=-1)

            return x + self.to_out(context)

    psi_orig = PSI(dim).to(device).eval()
    psi_sqrt = PSI_SqrtNorm(dim).to(device).eval()

    print("\nComparing output magnitudes at different lengths:")
    print(f"{'Seq Len':<10} {'Original':<15} {'Sqrt Norm':<15} {'Ratio':<10}")
    print("-" * 50)

    baseline_orig = None
    baseline_sqrt = None

    for seq_len in [50, 100, 200, 400, 800]:
        x = torch.randn(1, seq_len, dim, device=device)

        with torch.no_grad():
            out_orig = psi_orig(x)
            out_sqrt = psi_sqrt(x)

            mag_orig = out_orig.abs().mean().item()
            mag_sqrt = out_sqrt.abs().mean().item()

            if baseline_orig is None:
                baseline_orig = mag_orig
                baseline_sqrt = mag_sqrt

            ratio_orig = mag_orig / baseline_orig
            ratio_sqrt = mag_sqrt / baseline_sqrt

            print(f"{seq_len:<10} {mag_orig:<15.4f} {mag_sqrt:<15.4f} {ratio_sqrt/ratio_orig:<10.2f}")

    print("\nIf sqrt norm helps, the ratio column should be closer to 1.0")


if __name__ == "__main__":
    analyze_psi_internals()
    test_fix_sqrt_normalization()
