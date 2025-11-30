"""
Comprehensive Ablation Study for PSI

Tests which components of PSI are essential vs incidental.

Components to ablate:
1. Cumsum integration (the core claim)
2. Phase separation (complex/sinusoidal representation)
3. Learned magnitude weighting
4. Learned phase init (phi_init)
5. Gate mechanism
6. Integration scale (per-dimension learned rates)
7. Query offset (for retrieval)
8. Magnitude normalization (sqrt normalization)
9. Content modulation (x * trajectory)
10. Memory mechanism (phase-bound accumulation)

We test on a simple but representative task: 3-body gravitational dynamics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.integrate import solve_ivp
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# PSI Variants for Ablation
# ============================================================================

class PSI_Full(nn.Module):
    """Full PSI - the baseline."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoCumsum(nn.Module):
    """Ablation: Replace cumsum with direct learned transformation (no integration)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        # ABLATION: No cumsum - just use gated_omega directly as phase offset
        phi = phi_init + gated_omega  # Instead of cumsum

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        # ABLATION: No cumsum in memory either - just current
        memory_real = weighted_content * torch.cos(phi)
        memory_imag = weighted_content * torch.sin(phi)

        # No normalization needed without accumulation
        memory_real_normalized = memory_real
        memory_imag_normalized = memory_imag

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoPhase(nn.Module):
    """Ablation: Remove phase/complex representation - just use real-valued cumsum."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 2),  # Only 2x now (no complex)
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled

        # ABLATION: No phase - just cumsum the values directly
        integrated = torch.cumsum(gated_omega, dim=1)

        weighted_content = magnitude * x
        memory = torch.cumsum(weighted_content, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_normalized = memory / sqrt_magnitude

        # No phase modulation - just concatenate
        context = torch.cat([
            integrated,
            memory_normalized,
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoMagnitude(nn.Module):
    """Ablation: Remove learned magnitude - all content weighted equally."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        # ABLATION: Fixed magnitude = 1.0
        magnitude = torch.ones_like(x)
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoGate(nn.Module):
    """Ablation: Remove gate mechanism."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        # ABLATION: No gate - just use omega directly

        omega_scaled = omega * self.integration_scale.abs()
        # No gate applied
        phi = phi_init + torch.cumsum(omega_scaled, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoPhaseInit(nn.Module):
    """Ablation: Remove learned phase init - start from zero."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        # ABLATION: No phase init - start from zero
        phi_init = torch.zeros_like(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoQueryOffset(nn.Module):
    """Ablation: Remove query offset - query at same phase as storage."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        # ABLATION: No query offset - use phi directly
        phi_query = phi

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoMagnitudeNorm(nn.Module):
    """Ablation: Remove sqrt magnitude normalization."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        # ABLATION: No magnitude normalization - use raw accumulated memory
        # Just use sequence position for normalization to prevent explosion
        seq_len = x.shape[1]
        position = torch.arange(1, seq_len + 1, device=x.device).float().view(1, -1, 1)
        memory_real_normalized = memory_real / position
        memory_imag_normalized = memory_imag / position

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoContentModulation(nn.Module):
    """Ablation: Remove content modulation by trajectory."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 2),  # Only 2x now (no content modulation)
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        # ABLATION: No content modulation - just use retrieved memory
        context = torch.cat([
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_NoMemory(nn.Module):
    """Ablation: Remove memory accumulation entirely - only use current state + trajectory."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 2),  # Only 2x (trajectory real/imag)
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        # ABLATION: No memory - just content modulation
        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
        ], dim=-1)

        return x + self.to_out(context)


class PSI_RandomPhase(nn.Module):
    """Ablation: Use random (HRR-style) phases instead of learned phases."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Random fixed phases (HRR style)
        self.register_buffer('random_phases', torch.randn(dim) * 2 * np.pi)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0

        # ABLATION: Random fixed phases per dimension, offset by position
        position = torch.arange(seq_len, device=x.device).float().view(1, -1, 1)
        phi = self.random_phases.view(1, 1, -1) * position * 0.01  # Scale down

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


class PSI_FixedIntegrationScale(nn.Module):
    """Ablation: Fixed integration scale instead of learned per-dimension."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.to_gate = nn.Linear(dim, dim)
        # ABLATION: Fixed scale, not learned
        self.register_buffer('integration_scale', torch.ones(dim) * 0.001)
        self.to_magnitude = nn.Linear(dim, dim)
        self.to_query_offset = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        return x + self.to_out(context)


# Simple MLP baseline for comparison
class MLP_Baseline(nn.Module):
    """Simple MLP baseline - no recurrence, no phase."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.net(x)


# ============================================================================
# Block Wrapper
# ============================================================================

class AblationBlock(nn.Module):
    """Generic block wrapper for any PSI variant."""

    def __init__(self, dim, psi_class):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.integration = psi_class(dim)

    def forward(self, x):
        return x + self.integration(self.norm(x))


# ============================================================================
# Full Model
# ============================================================================

class AblationModel(nn.Module):
    """Full model for ablation testing."""

    def __init__(self, state_dim, dim=256, num_layers=8, psi_class=PSI_Full):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.blocks = nn.ModuleList([
            AblationBlock(dim, psi_class) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

    def forward(self, states):
        x = self.state_embedding(states)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Data Generation (3-body)
# ============================================================================

def nbody_dynamics(t, state, n_bodies=3):
    state = state.reshape(n_bodies, 5)
    deriv = np.zeros_like(state)

    for i in range(n_bodies):
        m_i, x_i, y_i, vx_i, vy_i = state[i]
        deriv[i, 0] = 0
        deriv[i, 1] = vx_i
        deriv[i, 2] = vy_i

        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if i != j:
                m_j, x_j, y_j, _, _ = state[j]
                dx = x_j - x_i
                dy = y_j - y_i
                r = np.sqrt(dx**2 + dy**2)
                ax += m_j * dx / (r**3 + 1e-8)
                ay += m_j * dy / (r**3 + 1e-8)

        deriv[i, 3] = ax
        deriv[i, 4] = ay

    return deriv.flatten()


def generate_data(n_trajectories=500, timesteps=50, n_bodies=3):
    """Generate n-body trajectories."""
    print(f"Generating {n_trajectories} {n_bodies}-body trajectories...")

    trajectories = []

    for _ in range(n_trajectories):
        # Random initial config
        state0 = np.zeros((n_bodies, 5))
        state0[:, 0] = 1.0  # mass

        for i in range(n_bodies):
            r = np.random.rand() * 0.5 + 0.75
            theta = 2 * np.pi * i / n_bodies + np.random.randn() * 0.3
            state0[i, 1] = r * np.cos(theta)
            state0[i, 2] = r * np.sin(theta)
            state0[i, 3] = -state0[i, 2] / (r**1.5)
            state0[i, 4] = state0[i, 1] / (r**1.5)

        # Integrate
        t_eval = np.linspace(0, 20, timesteps)
        sol = solve_ivp(
            lambda t, y: nbody_dynamics(t, y, n_bodies),
            [0, 20],
            state0.flatten(),
            t_eval=t_eval,
            rtol=1e-9
        )

        if sol.success:
            traj = sol.y.T.reshape(timesteps, n_bodies, 5)
            # Extract positions and velocities (ignore mass)
            features = []
            for i in range(n_bodies):
                features.append(traj[:, i, 1:])  # x, y, vx, vy
            trajectories.append(np.concatenate(features, axis=-1))

    trajectories = np.array(trajectories, dtype=np.float32)

    # Normalize
    mean = trajectories.mean(axis=(0, 1))
    std = trajectories.std(axis=(0, 1))
    trajectories = (trajectories - mean) / (std + 1e-8)

    return trajectories, mean, std


class DynamicsDataset(Dataset):
    def __init__(self, trajectories, context_len=20):
        self.trajectories = trajectories
        self.context_len = context_len
        self.seqs_per_traj = trajectories.shape[1] - context_len - 1

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.seqs_per_traj
        pos = idx % self.seqs_per_traj

        context = self.trajectories[traj_idx, pos:pos + self.context_len]
        target = self.trajectories[traj_idx, pos + self.context_len]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """Train a model and return best validation loss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)

            pred = model(context)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(device)
                target = batch['target'].to(device)
                pred = model(context)
                loss = criterion(pred, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_val_loss


def evaluate_rollout(model, trajectories, context_len, mean, std, n_samples=10, rollout_steps=20):
    """Evaluate autoregressive rollout MSE."""
    model.eval()

    mses = []

    with torch.no_grad():
        for i in range(min(n_samples, len(trajectories))):
            traj = trajectories[i]

            # Get initial context
            context = torch.tensor(traj[:context_len], dtype=torch.float32).unsqueeze(0).to(device)

            predictions = []
            for _ in range(rollout_steps):
                pred = model(context)
                predictions.append(pred.cpu().numpy()[0])

                # Shift context
                context = torch.cat([context[:, 1:, :], pred.unsqueeze(1)], dim=1)

            predictions = np.array(predictions)
            ground_truth = traj[context_len:context_len + rollout_steps]

            mse = np.mean((predictions - ground_truth) ** 2)
            mses.append(mse)

    return np.mean(mses), np.std(mses)


# ============================================================================
# Main Ablation Study
# ============================================================================

ABLATION_CONFIGS = {
    'Full PSI': PSI_Full,
    'No Cumsum': PSI_NoCumsum,
    'No Phase': PSI_NoPhase,
    'No Magnitude': PSI_NoMagnitude,
    'No Gate': PSI_NoGate,
    'No Phase Init': PSI_NoPhaseInit,
    'No Query Offset': PSI_NoQueryOffset,
    'No Magnitude Norm': PSI_NoMagnitudeNorm,
    'No Content Modulation': PSI_NoContentModulation,
    'No Memory': PSI_NoMemory,
    'Random Phase (HRR)': PSI_RandomPhase,
    'Fixed Integration Scale': PSI_FixedIntegrationScale,
    'MLP Baseline': MLP_Baseline,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajectories', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_bodies', type=int, default=3)
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    args = parser.parse_args()

    print("=" * 80)
    print("PSI Comprehensive Ablation Study")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Task: {args.n_bodies}-body gravitational dynamics")
    print(f"Trajectories: {args.n_trajectories}")
    print(f"Context length: {args.context_len}")
    print(f"Model dim: {args.dim}, layers: {args.num_layers}")
    print(f"Seeds: {args.seeds}")
    print()

    state_dim = args.n_bodies * 4  # x, y, vx, vy per body

    results = {}

    for name, psi_class in ABLATION_CONFIGS.items():
        print("=" * 80)
        print(f"Testing: {name}")
        print("=" * 80)

        seed_results = []

        for seed in range(args.seeds):
            print(f"\n  Seed {seed + 1}/{args.seeds}")

            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate data
            trajectories, mean, std = generate_data(
                args.n_trajectories, args.timesteps, args.n_bodies
            )

            # Split
            n_train = int(len(trajectories) * 0.8)
            train_trajs = trajectories[:n_train]
            val_trajs = trajectories[n_train:]

            train_dataset = DynamicsDataset(train_trajs, args.context_len)
            val_dataset = DynamicsDataset(val_trajs, args.context_len)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

            # Create model
            model = AblationModel(
                state_dim, args.dim, args.num_layers, psi_class
            ).to(device)

            if seed == 0:
                print(f"  Parameters: {model.count_parameters():,}")

            # Train
            start_time = time.time()
            best_val_loss = train_model(model, train_loader, val_loader, args.epochs, args.lr)
            train_time = time.time() - start_time

            # Evaluate rollout
            rollout_mse, rollout_std = evaluate_rollout(
                model, val_trajs, args.context_len, mean, std
            )

            seed_results.append({
                'val_loss': best_val_loss,
                'rollout_mse': rollout_mse,
                'rollout_std': rollout_std,
                'train_time': train_time
            })

            print(f"  Val Loss: {best_val_loss:.6f}, Rollout MSE: {rollout_mse:.6f} ± {rollout_std:.6f}")

        # Aggregate results across seeds
        avg_val_loss = np.mean([r['val_loss'] for r in seed_results])
        std_val_loss = np.std([r['val_loss'] for r in seed_results])
        avg_rollout_mse = np.mean([r['rollout_mse'] for r in seed_results])
        std_rollout_mse = np.std([r['rollout_mse'] for r in seed_results])
        avg_time = np.mean([r['train_time'] for r in seed_results])

        results[name] = {
            'val_loss_mean': avg_val_loss,
            'val_loss_std': std_val_loss,
            'rollout_mse_mean': avg_rollout_mse,
            'rollout_mse_std': std_rollout_mse,
            'train_time': avg_time,
            'params': model.count_parameters()
        }

        print(f"\n  AGGREGATE: Val Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        print(f"             Rollout MSE: {avg_rollout_mse:.6f} ± {std_rollout_mse:.6f}")

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print()

    # Sort by rollout MSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rollout_mse_mean'])

    print(f"{'Variant':<30} {'Val Loss':<20} {'Rollout MSE':<20} {'Params':<12}")
    print("-" * 82)

    baseline_mse = results['Full PSI']['rollout_mse_mean']

    for name, r in sorted_results:
        val_str = f"{r['val_loss_mean']:.6f} ± {r['val_loss_std']:.6f}"
        rollout_str = f"{r['rollout_mse_mean']:.6f} ± {r['rollout_mse_std']:.6f}"

        # Calculate relative degradation
        if name != 'Full PSI':
            degradation = (r['rollout_mse_mean'] - baseline_mse) / baseline_mse * 100
            rollout_str += f" ({degradation:+.1f}%)"

        print(f"{name:<30} {val_str:<20} {rollout_str:<35} {r['params']:>10,}")

    print()

    # Save results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to ablation_results.json")

    # Component importance ranking
    print("\n" + "=" * 80)
    print("COMPONENT IMPORTANCE RANKING (by degradation when removed)")
    print("=" * 80)
    print()

    importance = []
    for name, r in results.items():
        if name != 'Full PSI' and name != 'MLP Baseline':
            degradation = (r['rollout_mse_mean'] - baseline_mse) / baseline_mse * 100
            importance.append((name, degradation))

    importance.sort(key=lambda x: x[1], reverse=True)

    print("Most to least important (higher = more important when removed):")
    for i, (name, deg) in enumerate(importance, 1):
        component = name.replace('No ', '').replace('Fixed ', '')
        print(f"  {i:2}. {component:<25} {deg:+.1f}% degradation")


if __name__ == "__main__":
    main()
