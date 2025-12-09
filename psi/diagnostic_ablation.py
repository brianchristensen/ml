"""
Diagnostic Ablation: Identify the Root Cause of Catastrophic Forgetting

This ablation:
1. Tracks LTM surprise/write-gate values during training
2. Freezes block internals selectively
3. Analyzes the actual key phase distributions for Lorenz vs Chen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

from clifford_memory import ContinuousDynamicsModel, OrthogonalBivectorBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# =============================================================================
# Data Generation (same as projection_ablation.py)
# =============================================================================

def generate_lorenz(batch_size, seq_len, dt=0.01):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x = np.random.uniform(-15, 15, batch_size)
    y = np.random.uniform(-20, 20, batch_size)
    z = np.random.uniform(10, 40, batch_size)
    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        trajectories[:, t, 0] = x
        trajectories[:, t, 1] = y
        trajectories[:, t, 2] = z
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
    return trajectories


def generate_chen(batch_size, seq_len, dt=0.002):
    a, b, c = 35.0, 3.0, 28.0
    x = np.random.uniform(-10, 10, batch_size)
    y = np.random.uniform(-10, 10, batch_size)
    z = np.random.uniform(10, 30, batch_size)
    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        trajectories[:, t, 0] = x
        trajectories[:, t, 1] = y
        trajectories[:, t, 2] = z
        dx = a * (y - x)
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
    return trajectories


def normalize(trajs):
    mean = trajs.mean(axis=(0, 1), keepdims=True)
    std = trajs.std(axis=(0, 1), keepdims=True) + 1e-8
    return (trajs - mean) / std


def create_sequences(trajectories, context_len=20):
    n_traj, traj_len, dim = trajectories.shape
    n_seqs = traj_len - context_len - 1
    contexts = []
    targets = []
    for i in range(n_traj):
        for t in range(n_seqs):
            contexts.append(trajectories[i, t:t+context_len])
            targets.append(trajectories[i, t+context_len])
    return np.array(contexts), np.array(targets)


# =============================================================================
# Instrumented Block for Tracking
# =============================================================================

class InstrumentedBlock(OrthogonalBivectorBlock):
    """OrthogonalBivectorBlock with hooks to track internal values"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking_enabled = False
        self.tracked_values = defaultdict(list)

    def enable_tracking(self):
        self.tracking_enabled = True
        self.tracked_values = defaultdict(list)

    def disable_tracking(self):
        self.tracking_enabled = False

    def get_tracked(self):
        return {k: np.array(v) for k, v in self.tracked_values.items()}

    def forward(self, x):
        B, L, D = x.shape

        # Apply pending LTM updates
        if self._pending_ltm_update is not None:
            update = self._pending_ltm_update
            self.ltm_key_memory.mul_(update['decay'])
            self.ltm_binding_memory.mul_(update['decay'])
            self.ltm_key_memory.add_(update['new_keys'])
            self.ltm_binding_memory.add_(update['new_bindings'])
            self.ltm_count.add_(update['count_delta'])
            self._pending_ltm_update = None

        # Content-based phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        # TRACK: Key phases
        if self.tracking_enabled:
            with torch.no_grad():
                self.tracked_values['key_phase_mean'].append(key_phase.mean().item())
                self.tracked_values['key_phase_std'].append(key_phase.std().item())
                # Track phase distribution (histogram bins)
                phase_flat = key_phase.cpu().numpy().flatten()
                hist, _ = np.histogram(phase_flat, bins=20, range=(-math.pi, math.pi))
                self.tracked_values['key_phase_hist'].append(hist / len(phase_flat))

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        # Process content sets
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        weights = F.softmax(self.set_weights, dim=0)

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        if self.n_sets > 1:
            joint_key = torch.prod(key_phasor, dim=2)
            joint_query = torch.prod(query_phasor, dim=2)

            V_real = self.to_value(x)

            # Working memory surprise computation
            key_memory = torch.zeros_like(joint_key)
            key_memory[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)

            resonance = key_memory * joint_query.conj()
            resonance_magnitude = resonance.abs().mean(dim=-1, keepdim=True)

            positions = torch.arange(L, device=x.device, dtype=torch.float32).clamp(min=1.0)
            normalized_resonance = resonance_magnitude / positions.view(1, -1, 1).sqrt()

            scale = self.resonance_scale.clamp(min=1.0, max=20.0)
            threshold = self.resonance_threshold.clamp(min=0.1, max=0.9)
            surprise = 0.5 * (1.0 - torch.tanh(scale * (normalized_resonance - threshold)))

            write_gate = torch.sigmoid(self.surprise_scale * (surprise - 0.5) + self.surprise_bias)

            # TRACK: Working memory surprise and gate
            if self.tracking_enabled:
                with torch.no_grad():
                    self.tracked_values['wm_surprise_mean'].append(surprise.mean().item())
                    self.tracked_values['wm_surprise_std'].append(surprise.std().item())
                    self.tracked_values['wm_write_gate_mean'].append(write_gate.mean().item())
                    self.tracked_values['wm_resonance_mean'].append(normalized_resonance.mean().item())

            V_real_gated = V_real * write_gate
            V_gated = V_real_gated.to(torch.complex64)

            bound = joint_key.unsqueeze(-1) * V_gated.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * joint_query.conj().unsqueeze(-1)
            cross_bank_retrieved = retrieved.sum(dim=2).real

            bound_all = key_phasor.unsqueeze(-1) * V_gated.unsqueeze(2).unsqueeze(3)
            B_, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B_, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B_, L_seq, n_s, pp, D_dim)

            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real

            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

            cross_weight = 1.0 / (self.n_sets + 1)
            total_retrieved = cross_weight * (total_retrieved + cross_bank_retrieved)
        else:
            bound_all = key_phasor.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)
            B_, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B_, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B_, L_seq, n_s, pp, D_dim)

            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real

            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

        # Positional planes
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
        pos_phase = pos * self.pos_freqs * 2 * math.pi
        pos_phasor = torch.exp(1j * pos_phase)

        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)

        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)
        pos_retention = torch.sigmoid(self.pos_retention_logit)
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real

        gate = self.content_pos_gate(x)
        pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos
        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)

        # LTM with sequence-level context
        cumsum = torch.cumsum(x, dim=1)
        positions_for_mean = torch.arange(1, L + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
        running_mean = cumsum / positions_for_mean

        cumsum_sq = torch.cumsum(x ** 2, dim=1)
        running_var = (cumsum_sq / positions_for_mean) - (running_mean ** 2)
        running_std = (running_var.clamp(min=1e-8)).sqrt()

        ltm_key_input = torch.cat([x, running_mean, running_std], dim=-1)

        # TRACK: LTM key input statistics
        if self.tracking_enabled:
            with torch.no_grad():
                self.tracked_values['ltm_input_x_mean'].append(x.mean().item())
                self.tracked_values['ltm_input_x_std'].append(x.std().item())
                self.tracked_values['ltm_input_running_mean_mean'].append(running_mean.mean().item())
                self.tracked_values['ltm_input_running_std_mean'].append(running_std.mean().item())

        ltm_key_phase = torch.tanh(self.ltm_key_encoder(ltm_key_input)) * math.pi
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = self.ltm_value_proj(x)

        # TRACK: LTM key phases
        if self.tracking_enabled:
            with torch.no_grad():
                self.tracked_values['ltm_key_phase_mean'].append(ltm_key_phase.mean().item())
                self.tracked_values['ltm_key_phase_std'].append(ltm_key_phase.std().item())
                # Histogram
                ltm_phase_flat = ltm_key_phase.cpu().numpy().flatten()
                hist, _ = np.histogram(ltm_phase_flat, bins=20, range=(-math.pi, math.pi))
                self.tracked_values['ltm_key_phase_hist'].append(hist / len(ltm_phase_flat))

        # LTM surprise computation
        key_resonance = self.ltm_key_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()
        total_resonance = key_resonance.sum(dim=-1)
        resonance_magnitude = total_resonance.abs()

        count_norm = (self.ltm_count.detach().clamp(min=1.0) * math.sqrt(self.ltm_planes))
        normalized_resonance = resonance_magnitude / count_norm

        ltm_scale = self.ltm_resonance_scale.clamp(min=1.0, max=20.0)
        ltm_threshold = self.ltm_resonance_threshold.clamp(min=0.1, max=0.9)

        ltm_surprise = 0.5 * (1.0 - torch.tanh(ltm_scale * (normalized_resonance - ltm_threshold)))
        ltm_surprise = ltm_surprise.unsqueeze(-1)

        ltm_write_gate = torch.sigmoid(self.ltm_surprise_scale * (ltm_surprise - 0.5) + self.ltm_surprise_bias)

        # TRACK: LTM surprise and gate
        if self.tracking_enabled:
            with torch.no_grad():
                self.tracked_values['ltm_surprise_mean'].append(ltm_surprise.mean().item())
                self.tracked_values['ltm_surprise_std'].append(ltm_surprise.std().item())
                self.tracked_values['ltm_write_gate_mean'].append(ltm_write_gate.mean().item())
                self.tracked_values['ltm_resonance_mean'].append(normalized_resonance.mean().item())
                self.tracked_values['ltm_count'].append(self.ltm_count.item())
                self.tracked_values['ltm_key_memory_norm'].append(self.ltm_key_memory.abs().mean().item())

        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)

        # LTM retrieval
        persistent_retrieved = self.ltm_binding_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj().unsqueeze(-1)
        persistent_retrieved = persistent_retrieved.sum(dim=2).real
        persistent_norm = (self.ltm_count.detach().clamp(min=1.0) * self.ltm_planes).sqrt()
        ltm_retrieved = persistent_retrieved / persistent_norm

        # Pending LTM update
        if self.training:
            with torch.no_grad():
                decay = self.ltm_decay.detach().clamp(min=0.9, max=0.9999)
                new_keys = (ltm_write_gate.detach() * ltm_key_phasor.detach()).mean(dim=(0, 1))
                new_bindings = gated_binding.detach().mean(dim=(0, 1))
                avg_gate = ltm_write_gate.detach().mean().item()

                self._pending_ltm_update = {
                    'decay': decay,
                    'new_keys': new_keys,
                    'new_bindings': new_bindings,
                    'count_delta': avg_gate * B * L * (1 - decay)
                }

        total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)


class InstrumentedModel(nn.Module):
    """ContinuousDynamicsModel with instrumented blocks"""

    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            InstrumentedBlock(hidden_dim, n_orthogonal_sets, planes_per_set,
                             pos_planes=pos_planes)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.output_proj(h)

    def enable_tracking(self):
        for block in self.blocks:
            block.enable_tracking()

    def disable_tracking(self):
        for block in self.blocks:
            block.disable_tracking()

    def get_tracked(self):
        """Return tracked values from all blocks"""
        all_tracked = {}
        for i, block in enumerate(self.blocks):
            for k, v in block.get_tracked().items():
                all_tracked[f'block{i}_{k}'] = v
        return all_tracked

    def reset_ltm(self):
        for block in self.blocks:
            block.reset_ltm()


# =============================================================================
# Training with Tracking
# =============================================================================

def train_epoch_tracked(model, contexts, targets, optimizer, batch_size=64):
    model.train()
    n_samples = len(contexts)
    indices = np.random.permutation(n_samples)
    total_loss = 0
    n_batches = 0
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        ctx = torch.tensor(contexts[batch_idx], device=device)
        tgt = torch.tensor(targets[batch_idx], device=device)
        optimizer.zero_grad()
        pred = model(ctx)[:, -1, :]
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, contexts, targets, batch_size=256):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i+batch_size], device=device)
            tgt = torch.tensor(targets[i:i+batch_size], device=device)
            pred = model(ctx)[:, -1, :]
            loss = F.mse_loss(pred, tgt)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


# =============================================================================
# Selective Freezing Strategies
# =============================================================================

def freeze_key_query_encoders(model):
    """Freeze only key and query encoders in blocks"""
    frozen = 0
    for block in model.blocks:
        for param in block.key_encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
        for param in block.query_encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def freeze_value_output(model):
    """Freeze value and output projections in blocks"""
    frozen = 0
    for block in model.blocks:
        for param in block.to_value.parameters():
            param.requires_grad = False
            frozen += param.numel()
        for param in block.to_out.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def freeze_ltm_components(model):
    """Freeze LTM-specific components"""
    frozen = 0
    for block in model.blocks:
        for param in block.ltm_key_encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
        for param in block.ltm_value_proj.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def freeze_content_pos_gate(model):
    """Freeze content-position gating"""
    frozen = 0
    for block in model.blocks:
        for param in block.content_pos_gate.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


FREEZE_STRATEGIES = {
    'none': lambda m: 0,
    'key_query': freeze_key_query_encoders,
    'value_output': freeze_value_output,
    'ltm': freeze_ltm_components,
    'content_pos_gate': freeze_content_pos_gate,
    'key_query+value_output': lambda m: freeze_key_query_encoders(m) + freeze_value_output(m),
    'all_block_internals': lambda m: (freeze_key_query_encoders(m) +
                                       freeze_value_output(m) +
                                       freeze_ltm_components(m) +
                                       freeze_content_pos_gate(m)),
}


# =============================================================================
# Phase Distribution Analysis
# =============================================================================

def analyze_phase_distributions(model, lorenz_data, chen_data, n_samples=500):
    """Analyze how key phases differ between Lorenz and Chen"""
    model.eval()

    # Sample from each dataset
    lorenz_ctx = torch.tensor(lorenz_data[:n_samples], device=device)
    chen_ctx = torch.tensor(chen_data[:n_samples], device=device)

    lorenz_phases = []
    chen_phases = []
    lorenz_ltm_phases = []
    chen_ltm_phases = []

    with torch.no_grad():
        # Get hidden representations
        lorenz_h = model.input_proj(lorenz_ctx)
        chen_h = model.input_proj(chen_ctx)

        for i, (norm, block) in enumerate(zip(model.norms, model.blocks)):
            lorenz_normed = norm(lorenz_h)
            chen_normed = norm(chen_h)

            # Key phases
            lorenz_key_phase = torch.tanh(block.key_encoder(lorenz_normed)) * math.pi
            chen_key_phase = torch.tanh(block.key_encoder(chen_normed)) * math.pi

            lorenz_phases.append(lorenz_key_phase.cpu().numpy())
            chen_phases.append(chen_key_phase.cpu().numpy())

            # LTM key phases with sequence context
            B, L, D = lorenz_normed.shape
            positions_for_mean = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, -1, 1)

            # Lorenz LTM input
            lorenz_cumsum = torch.cumsum(lorenz_normed, dim=1)
            lorenz_running_mean = lorenz_cumsum / positions_for_mean
            lorenz_cumsum_sq = torch.cumsum(lorenz_normed ** 2, dim=1)
            lorenz_running_var = (lorenz_cumsum_sq / positions_for_mean) - (lorenz_running_mean ** 2)
            lorenz_running_std = (lorenz_running_var.clamp(min=1e-8)).sqrt()
            lorenz_ltm_input = torch.cat([lorenz_normed, lorenz_running_mean, lorenz_running_std], dim=-1)
            lorenz_ltm_phase = torch.tanh(block.ltm_key_encoder(lorenz_ltm_input)) * math.pi

            # Chen LTM input
            chen_cumsum = torch.cumsum(chen_normed, dim=1)
            chen_running_mean = chen_cumsum / positions_for_mean
            chen_cumsum_sq = torch.cumsum(chen_normed ** 2, dim=1)
            chen_running_var = (chen_cumsum_sq / positions_for_mean) - (chen_running_mean ** 2)
            chen_running_std = (chen_running_var.clamp(min=1e-8)).sqrt()
            chen_ltm_input = torch.cat([chen_normed, chen_running_mean, chen_running_std], dim=-1)
            chen_ltm_phase = torch.tanh(block.ltm_key_encoder(chen_ltm_input)) * math.pi

            lorenz_ltm_phases.append(lorenz_ltm_phase.cpu().numpy())
            chen_ltm_phases.append(chen_ltm_phase.cpu().numpy())

            # Pass through block for next layer
            lorenz_h = block(lorenz_normed)
            chen_h = block(chen_normed)

    return {
        'key_phases': {'lorenz': lorenz_phases, 'chen': chen_phases},
        'ltm_phases': {'lorenz': lorenz_ltm_phases, 'chen': chen_ltm_phases}
    }


def compute_phase_overlap(lorenz_phases, chen_phases, n_bins=50):
    """Compute histogram overlap between Lorenz and Chen phase distributions"""
    overlaps = []

    for layer_idx in range(len(lorenz_phases)):
        lorenz_flat = lorenz_phases[layer_idx].flatten()
        chen_flat = chen_phases[layer_idx].flatten()

        # Compute histograms
        bins = np.linspace(-math.pi, math.pi, n_bins + 1)
        lorenz_hist, _ = np.histogram(lorenz_flat, bins=bins, density=True)
        chen_hist, _ = np.histogram(chen_flat, bins=bins, density=True)

        # Normalize to probability
        lorenz_hist = lorenz_hist / lorenz_hist.sum()
        chen_hist = chen_hist / chen_hist.sum()

        # Overlap = sum of min at each bin (intersection of distributions)
        overlap = np.minimum(lorenz_hist, chen_hist).sum()
        overlaps.append(overlap)

    return overlaps


# =============================================================================
# Main Diagnostic
# =============================================================================

def run_diagnostic():
    print("=" * 70)
    print("DIAGNOSTIC ABLATION: Root Cause Analysis")
    print("=" * 70)
    print()

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    context_len = 20
    n_epochs = 10

    # Generate data
    print("Generating datasets...")
    lorenz_train = normalize(generate_lorenz(500, 200))
    lorenz_test = normalize(generate_lorenz(100, 200))
    chen_train = normalize(generate_chen(500, 200))
    chen_test = normalize(generate_chen(100, 200))

    lorenz_ctx_train, lorenz_tgt_train = create_sequences(lorenz_train, context_len)
    lorenz_ctx_test, lorenz_tgt_test = create_sequences(lorenz_test, context_len)
    chen_ctx_train, chen_tgt_train = create_sequences(chen_train, context_len)
    chen_ctx_test, chen_tgt_test = create_sequences(chen_test, context_len)

    print(f"  Lorenz: {len(lorenz_ctx_train)} train, {len(lorenz_ctx_test)} test")
    print(f"  Chen: {len(chen_ctx_train)} train, {len(chen_ctx_test)} test")

    # =========================================================================
    # Part 1: Track LTM metrics during training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Tracking LTM Metrics During Training")
    print("=" * 70)

    model = InstrumentedModel(
        input_dim=3, hidden_dim=128, n_layers=4,
        n_orthogonal_sets=4, planes_per_set=16
    ).to(device)
    model.reset_ltm()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Train on Lorenz with tracking
    print("\nPhase 1: Training on Lorenz...")
    model.enable_tracking()

    lorenz_tracked = defaultdict(list)
    for epoch in range(n_epochs):
        loss = train_epoch_tracked(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)
        if (epoch + 1) % 2 == 0:
            val_loss = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Epoch {epoch+1}: train={loss:.6f}, val={val_loss:.6f}")

    # Collect Lorenz tracking data
    lorenz_tracking = model.get_tracked()

    lorenz_after_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    print(f"\n  Lorenz after Task A: MSE={lorenz_after_A:.6f}")

    # Analyze phase distributions BEFORE Chen training
    print("\n  Analyzing phase distributions before Chen training...")
    phase_before = analyze_phase_distributions(model, lorenz_ctx_test, chen_ctx_test)
    key_overlap_before = compute_phase_overlap(
        phase_before['key_phases']['lorenz'],
        phase_before['key_phases']['chen']
    )
    ltm_overlap_before = compute_phase_overlap(
        phase_before['ltm_phases']['lorenz'],
        phase_before['ltm_phases']['chen']
    )

    print(f"  Key phase overlap (Lorenz vs Chen): {[f'{o:.3f}' for o in key_overlap_before]}")
    print(f"  LTM phase overlap (Lorenz vs Chen): {[f'{o:.3f}' for o in ltm_overlap_before]}")

    # Reset tracking for Chen phase
    for block in model.blocks:
        block.tracked_values = defaultdict(list)

    # Train on Chen with tracking
    print("\nPhase 2: Training on Chen...")
    for epoch in range(n_epochs):
        loss = train_epoch_tracked(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)
        if (epoch + 1) % 2 == 0:
            chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
            lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

    chen_tracking = model.get_tracked()
    model.disable_tracking()

    lorenz_after_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    chen_final = evaluate(model, chen_ctx_test, chen_tgt_test)
    forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100

    print(f"\n  Final Results:")
    print(f"    Lorenz after A: {lorenz_after_A:.6f}")
    print(f"    Lorenz after B: {lorenz_after_B:.6f}")
    print(f"    Forgetting: {forgetting:+.1f}%")
    print(f"    Chen final: {chen_final:.6f}")

    # Analyze phase distributions AFTER Chen training
    print("\n  Analyzing phase distributions after Chen training...")
    phase_after = analyze_phase_distributions(model, lorenz_ctx_test, chen_ctx_test)
    key_overlap_after = compute_phase_overlap(
        phase_after['key_phases']['lorenz'],
        phase_after['key_phases']['chen']
    )
    ltm_overlap_after = compute_phase_overlap(
        phase_after['ltm_phases']['lorenz'],
        phase_after['ltm_phases']['chen']
    )

    print(f"  Key phase overlap (Lorenz vs Chen): {[f'{o:.3f}' for o in key_overlap_after]}")
    print(f"  LTM phase overlap (Lorenz vs Chen): {[f'{o:.3f}' for o in ltm_overlap_after]}")

    # Print LTM metrics summary
    print("\n  LTM Metrics Summary (Block 0):")
    print(f"    During Lorenz training:")
    if 'block0_ltm_surprise_mean' in lorenz_tracking:
        print(f"      LTM surprise: {np.mean(lorenz_tracking['block0_ltm_surprise_mean']):.4f} (mean)")
        print(f"      LTM write gate: {np.mean(lorenz_tracking['block0_ltm_write_gate_mean']):.4f} (mean)")
        print(f"      LTM count (end): {lorenz_tracking['block0_ltm_count'][-1]:.1f}")

    print(f"    During Chen training:")
    if 'block0_ltm_surprise_mean' in chen_tracking:
        print(f"      LTM surprise: {np.mean(chen_tracking['block0_ltm_surprise_mean']):.4f} (mean)")
        print(f"      LTM write gate: {np.mean(chen_tracking['block0_ltm_write_gate_mean']):.4f} (mean)")
        print(f"      LTM count (end): {chen_tracking['block0_ltm_count'][-1]:.1f}")

    # =========================================================================
    # Part 2: Selective Freezing Ablation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Selective Freezing Ablation")
    print("=" * 70)

    results = {}

    for strategy_name, freeze_fn in FREEZE_STRATEGIES.items():
        print(f"\n--- Strategy: {strategy_name} ---")

        # Fresh model
        model = InstrumentedModel(
            input_dim=3, hidden_dim=128, n_layers=4,
            n_orthogonal_sets=4, planes_per_set=16
        ).to(device)
        model.reset_ltm()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Phase 1: Train on Lorenz
        for epoch in range(n_epochs):
            train_epoch_tracked(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)

        lorenz_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)

        # Apply freezing strategy
        frozen_params = freeze_fn(model)
        print(f"  Frozen {frozen_params:,} parameters")

        # New optimizer with only trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(trainable) > 0:
            optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)

        # Phase 2: Train on Chen
        for epoch in range(n_epochs):
            if len(trainable) > 0:
                train_epoch_tracked(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)

        lorenz_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        chen_F = evaluate(model, chen_ctx_test, chen_tgt_test)
        forget = (lorenz_B - lorenz_A) / lorenz_A * 100

        results[strategy_name] = {
            'lorenz_A': lorenz_A,
            'lorenz_B': lorenz_B,
            'chen': chen_F,
            'forgetting': forget,
            'frozen': frozen_params
        }

        print(f"  Lorenz A: {lorenz_A:.6f}, B: {lorenz_B:.6f}")
        print(f"  Forgetting: {forget:+.1f}%, Chen: {chen_F:.6f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SELECTIVE FREEZING SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Strategy':<25} {'Frozen':>10} {'Forget %':>12} {'Chen':>12}")
    print("-" * 60)

    for name, r in sorted(results.items(), key=lambda x: x[1]['forgetting']):
        print(f"{name:<25} {r['frozen']:>10,} {r['forgetting']:>+11.1f}% {r['chen']:>12.6f}")

    # =========================================================================
    # Part 3: Analysis Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    # Check if phase overlap increased after Chen training
    print("\n1. PHASE DISTRIBUTION ANALYSIS:")
    key_overlap_change = np.mean(key_overlap_after) - np.mean(key_overlap_before)
    ltm_overlap_change = np.mean(ltm_overlap_after) - np.mean(ltm_overlap_before)

    print(f"   Key phase overlap change: {key_overlap_change:+.3f}")
    print(f"   LTM phase overlap change: {ltm_overlap_change:+.3f}")

    if np.mean(key_overlap_before) > 0.7:
        print("   WARNING: High initial overlap suggests keys don't discriminate tasks well")

    if key_overlap_change > 0.1:
        print("   WARNING: Key phases became MORE similar after Chen training (interference)")

    # Identify best freezing strategy
    print("\n2. FREEZING ANALYSIS:")
    best_strategy = min(results.items(), key=lambda x: x[1]['forgetting'])
    worst_strategy = max(results.items(), key=lambda x: x[1]['forgetting'])

    print(f"   Best strategy: {best_strategy[0]} (forgetting: {best_strategy[1]['forgetting']:+.1f}%)")
    print(f"   Worst strategy: {worst_strategy[0]} (forgetting: {worst_strategy[1]['forgetting']:+.1f}%)")

    if best_strategy[1]['forgetting'] < results['none']['forgetting'] * 0.5:
        print(f"   FINDING: Freezing {best_strategy[0]} significantly reduces forgetting!")
    else:
        print("   FINDING: No freezing strategy substantially reduces forgetting")
        print("            -> Forgetting may be intrinsic to memory mechanism")

    # Check LTM surprise behavior
    print("\n3. LTM SURPRISE BEHAVIOR:")
    if 'block0_ltm_surprise_mean' in lorenz_tracking and 'block0_ltm_surprise_mean' in chen_tracking:
        lorenz_surprise = np.mean(lorenz_tracking['block0_ltm_surprise_mean'])
        chen_surprise = np.mean(chen_tracking['block0_ltm_surprise_mean'])

        print(f"   Lorenz surprise: {lorenz_surprise:.4f}")
        print(f"   Chen surprise: {chen_surprise:.4f}")

        if chen_surprise < 0.3:
            print("   WARNING: Low Chen surprise suggests LTM sees Chen as 'familiar'")
            print("            -> LTM keys may not discriminate between systems")
        elif chen_surprise > 0.7:
            print("   GOOD: High Chen surprise means LTM recognizes it as novel")

    return results, lorenz_tracking, chen_tracking


if __name__ == "__main__":
    results, lorenz_tracking, chen_tracking = run_diagnostic()
