"""
LTM Diagnostic Test

Diagnose WHY the persistent LTM isn't preventing forgetting during continual learning.

Key questions:
1. Is the LTM actually accumulating information? (ltm_count, memory norms)
2. Is surprise being computed correctly? (novel vs familiar patterns)
3. Is the write gate opening/closing appropriately?
4. Is retrieval from LTM actually contributing to output?
5. Are the LTM keys encoding useful information?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from clifford_memory import ContinuousDynamicsModel, OrthogonalBivectorBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def generate_lorenz(batch_size, seq_len, dt=0.01):
    """Lorenz attractor"""
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
        x, y, z = x + dx*dt, y + dy*dt, z + dz*dt

    mean = trajectories.mean(axis=(0, 1), keepdims=True)
    std = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8
    return (trajectories - mean) / std


def generate_chen(batch_size, seq_len, dt=0.002):
    """Chen attractor - different dynamics"""
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
        x, y, z = x + dx*dt, y + dy*dt, z + dz*dt

    mean = trajectories.mean(axis=(0, 1), keepdims=True)
    std = trajectories.std(axis=(0, 1), keepdims=True) + 1e-8
    return (trajectories - mean) / std


class LTMProbe:
    """Hooks to probe LTM internals during forward pass"""

    def __init__(self, block: OrthogonalBivectorBlock):
        self.block = block
        self.logs = []

    def capture(self, x, hidden, label=""):
        """Capture LTM state during a forward pass

        Args:
            x: raw input (for reference)
            hidden: hidden representation from model (after input_proj)
            label: description for logging
        """
        B, L, D = hidden.shape

        # Get persistent memory state
        ltm_count = self.block.ltm_count.item()
        ltm_key_norm = self.block.ltm_key_memory.abs().mean().item()
        ltm_binding_norm = self.block.ltm_binding_memory.abs().mean().item()

        # Compute what surprise would be for this input
        with torch.no_grad():
            # Compute sequence context (same as in forward pass)
            cumsum = torch.cumsum(hidden, dim=1)  # [B, L, D]
            positions_for_mean = torch.arange(1, L + 1, device=hidden.device, dtype=torch.float32).view(1, -1, 1)
            running_mean = cumsum / positions_for_mean  # [B, L, D]
            cumsum_sq = torch.cumsum(hidden ** 2, dim=1)
            running_var = (cumsum_sq / positions_for_mean) - (running_mean ** 2)
            running_std = (running_var.clamp(min=1e-8)).sqrt()

            # Concatenate for LTM key encoder
            ltm_key_input = torch.cat([hidden, running_mean, running_std], dim=-1)  # [B, L, D*3]

            ltm_key_phase = torch.tanh(self.block.ltm_key_encoder(ltm_key_input)) * 3.14159
            ltm_key_phasor = torch.exp(1j * ltm_key_phase)

            # Query persistent key memory
            key_resonance = self.block.ltm_key_memory.unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()
            total_resonance = key_resonance.sum(dim=-1)
            resonance_magnitude = total_resonance.abs()

            # Normalize
            import math
            count_norm = max(1.0, ltm_count) * math.sqrt(self.block.ltm_planes)
            normalized_resonance = resonance_magnitude / count_norm

            # Compute surprise using LTM-specific params
            ltm_scale = self.block.ltm_resonance_scale.clamp(min=1.0, max=20.0).item()
            ltm_threshold = self.block.ltm_resonance_threshold.clamp(min=0.1, max=0.9).item()

            # Transform
            surprise = 0.5 * (1.0 - torch.tanh(ltm_scale * (normalized_resonance - ltm_threshold)))

            # Write gate
            ltm_surprise_scale = self.block.ltm_surprise_scale.item()
            ltm_surprise_bias = self.block.ltm_surprise_bias.item()
            write_gate = torch.sigmoid(ltm_surprise_scale * (surprise - 0.5) + ltm_surprise_bias)

            # LTM retrieval contribution
            ltm_value = self.block.ltm_value_proj(hidden)
            ltm_value_complex = ltm_value.to(torch.complex64)

            persistent_retrieved = self.block.ltm_binding_memory.unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj().unsqueeze(-1)
            persistent_retrieved = persistent_retrieved.sum(dim=2).real
            persistent_norm = max(1.0, ltm_count) * self.block.ltm_planes
            ltm_retrieved = persistent_retrieved / math.sqrt(persistent_norm)

            ltm_weight = torch.sigmoid(self.block.ltm_weight).item()
            ltm_contribution = ltm_weight * ltm_retrieved

        self.logs.append({
            'label': label,
            'ltm_count': ltm_count,
            'ltm_key_norm': ltm_key_norm,
            'ltm_binding_norm': ltm_binding_norm,
            'resonance_mean': normalized_resonance.mean().item(),
            'resonance_std': normalized_resonance.std().item(),
            'surprise_mean': surprise.mean().item(),
            'surprise_std': surprise.std().item(),
            'write_gate_mean': write_gate.mean().item(),
            'write_gate_std': write_gate.std().item(),
            'ltm_retrieval_norm': ltm_retrieved.norm().item() / (B * L),
            'ltm_contribution_norm': ltm_contribution.norm().item() / (B * L),
            'ltm_weight': ltm_weight,
            'ltm_scale': ltm_scale,
            'ltm_threshold': ltm_threshold,
            'ltm_surprise_scale': ltm_surprise_scale,
            'ltm_surprise_bias': ltm_surprise_bias,
        })

        return self.logs[-1]

    def print_log(self, log):
        print(f"\n  [{log['label']}]")
        print(f"    LTM State: count={log['ltm_count']:.1f}, key_norm={log['ltm_key_norm']:.4f}, binding_norm={log['ltm_binding_norm']:.4f}")
        print(f"    Resonance: mean={log['resonance_mean']:.4f}, std={log['resonance_std']:.4f}")
        print(f"    Surprise:  mean={log['surprise_mean']:.4f}, std={log['surprise_std']:.4f}")
        print(f"    Write Gate: mean={log['write_gate_mean']:.4f}, std={log['write_gate_std']:.4f}")
        print(f"    LTM Retrieval: norm={log['ltm_retrieval_norm']:.4f}, contribution={log['ltm_contribution_norm']:.4f}")
        print(f"    Params: weight={log['ltm_weight']:.3f}, scale={log['ltm_scale']:.2f}, thresh={log['ltm_threshold']:.3f}")


def train_epoch(model, data, optimizer, context_len=20, batch_size=64):
    """Train one epoch, return loss"""
    model.train()
    n_traj, traj_len, dim = data.shape

    # Create sequences
    contexts, targets = [], []
    for i in range(n_traj):
        for t in range(traj_len - context_len - 1):
            contexts.append(data[i, t:t+context_len])
            targets.append(data[i, t+context_len])

    contexts = np.array(contexts)
    targets = np.array(targets)

    # Shuffle
    idx = np.random.permutation(len(contexts))
    contexts, targets = contexts[idx], targets[idx]

    total_loss = 0
    n_batches = 0

    for i in range(0, len(contexts), batch_size):
        ctx = torch.tensor(contexts[i:i+batch_size], device=device)
        tgt = torch.tensor(targets[i:i+batch_size], device=device)

        optimizer.zero_grad()
        pred = model(ctx)[:, -1, :]
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, data, context_len=20, batch_size=256):
    """Evaluate model"""
    model.eval()
    n_traj, traj_len, dim = data.shape

    contexts, targets = [], []
    for i in range(n_traj):
        for t in range(traj_len - context_len - 1):
            contexts.append(data[i, t:t+context_len])
            targets.append(data[i, t+context_len])

    contexts = np.array(contexts)
    targets = np.array(targets)

    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i+batch_size], device=device)
            tgt = torch.tensor(targets[i:i+batch_size], device=device)
            pred = model(ctx)[:, -1, :]
            total_loss += F.mse_loss(pred, tgt).item()
            n_batches += 1

    return total_loss / n_batches


def run_diagnostic():
    print("=" * 70)
    print("LTM DIAGNOSTIC: Why isn't LTM preventing forgetting?")
    print("=" * 70)

    # Generate data (smaller for faster diagnosis)
    print("\nGenerating data...")
    lorenz_train = generate_lorenz(100, 100)
    lorenz_test = generate_lorenz(30, 100)
    chen_train = generate_chen(100, 100)
    chen_test = generate_chen(30, 100)

    # Create model
    model = ContinuousDynamicsModel(
        input_dim=3,
        hidden_dim=128,
        n_layers=4,
        n_orthogonal_sets=4,
        planes_per_set=16
    ).to(device)

    # Get first block for probing
    block = model.blocks[0]
    probe = LTMProbe(block)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Sample inputs for probing
    lorenz_sample = torch.tensor(lorenz_train[:16, :20, :], device=device)
    chen_sample = torch.tensor(chen_train[:16, :20, :], device=device)

    # Compute hidden representations (needed for LTM probing)
    with torch.no_grad():
        lorenz_hidden = model.input_proj(lorenz_sample)
        chen_hidden = model.input_proj(chen_sample)

    print("\n" + "=" * 70)
    print("PHASE 1: Initial state (before any training)")
    print("=" * 70)

    log = probe.capture(lorenz_sample, lorenz_hidden, "Initial - Lorenz input")
    probe.print_log(log)
    log = probe.capture(chen_sample, chen_hidden, "Initial - Chen input")
    probe.print_log(log)

    print("\n" + "=" * 70)
    print("PHASE 2: Training on Lorenz (3 epochs)")
    print("=" * 70)

    for epoch in range(3):
        loss = train_epoch(model, lorenz_train, optimizer)
        if (epoch + 1) % 1 == 0:
            val_loss = evaluate(model, lorenz_test)
            print(f"  Epoch {epoch+1}: train={loss:.6f}, val={val_loss:.6f}")

    # Recompute hidden representations after training
    with torch.no_grad():
        lorenz_hidden = model.input_proj(lorenz_sample)
        chen_hidden = model.input_proj(chen_sample)

    print("\nLTM state after Lorenz training:")
    log = probe.capture(lorenz_sample, lorenz_hidden, "After Lorenz - Lorenz input")
    probe.print_log(log)
    log = probe.capture(chen_sample, chen_hidden, "After Lorenz - Chen input")
    probe.print_log(log)

    lorenz_after_A = evaluate(model, lorenz_test)
    print(f"\n  Lorenz test MSE: {lorenz_after_A:.6f}")

    print("\n" + "=" * 70)
    print("PHASE 3: Training on Chen (3 epochs) - potential forgetting")
    print("=" * 70)

    for epoch in range(3):
        loss = train_epoch(model, chen_train, optimizer)
        if (epoch + 1) % 1 == 0:
            chen_val = evaluate(model, chen_test)
            lorenz_val = evaluate(model, lorenz_test)
            print(f"  Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

    # Recompute hidden representations after Chen training
    with torch.no_grad():
        lorenz_hidden = model.input_proj(lorenz_sample)
        chen_hidden = model.input_proj(chen_sample)

    print("\nLTM state after Chen training:")
    log = probe.capture(lorenz_sample, lorenz_hidden, "After Chen - Lorenz input")
    probe.print_log(log)
    log = probe.capture(chen_sample, chen_hidden, "After Chen - Chen input")
    probe.print_log(log)

    lorenz_after_B = evaluate(model, lorenz_test)
    chen_final = evaluate(model, chen_test)

    forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    print(f"\nPerformance:")
    print(f"  Lorenz after Phase A: {lorenz_after_A:.6f}")
    print(f"  Lorenz after Phase B: {lorenz_after_B:.6f}")
    print(f"  Forgetting: {forgetting:+.1f}%")
    print(f"  Chen final: {chen_final:.6f}")

    print("\nLTM Analysis:")
    final_log = probe.logs[-2]  # Lorenz input after Chen training

    issues = []

    # Check 1: Is LTM accumulating?
    if final_log['ltm_count'] < 10:
        issues.append("LOW LTM COUNT: Memory not accumulating enough items")

    # Check 2: Is resonance distinguishing patterns?
    if final_log['resonance_std'] < 0.01:
        issues.append("LOW RESONANCE VARIANCE: LTM not distinguishing patterns")

    # Check 3: Is surprise varying?
    if final_log['surprise_std'] < 0.01:
        issues.append("LOW SURPRISE VARIANCE: All patterns treated same novelty")

    # Check 4: Is write gate actually gating?
    if final_log['write_gate_std'] < 0.01:
        issues.append("CONSTANT WRITE GATE: Not selectively writing")

    # Check 5: Is LTM retrieval contributing?
    if final_log['ltm_contribution_norm'] < 0.001:
        issues.append("NEGLIGIBLE LTM CONTRIBUTION: Retrieved values not affecting output")

    # Check 6: Is LTM weight reasonable?
    if final_log['ltm_weight'] < 0.1:
        issues.append("LOW LTM WEIGHT: Model not using LTM output")

    # Check 7: Is surprise scale reasonable?
    if final_log['ltm_scale'] < 2.0:
        issues.append("LOW SURPRISE SCALE: Can't distinguish novel/familiar")

    # Check 8: Are we writing too much? (always high gate)
    if final_log['write_gate_mean'] > 0.8:
        issues.append("ALWAYS WRITING: Write gate too open, overwriting everything")

    # Check 9: Are we writing too little?
    if final_log['write_gate_mean'] < 0.2:
        issues.append("NEVER WRITING: Write gate too closed, not storing")

    if issues:
        print("\nPOTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n  No obvious issues detected in LTM mechanics")
        print("  Forgetting may be due to:")
        print("    - Network weights being overwritten (not LTM's job)")
        print("    - LTM capacity too small for task")
        print("    - Key encodings not capturing task-relevant features")

    # Additional analysis: Compare resonance for same vs different data
    print("\n" + "=" * 70)
    print("CROSS-TASK RESONANCE ANALYSIS")
    print("=" * 70)

    with torch.no_grad():
        # Compute sequence context for both tasks
        def compute_ltm_key_input(hidden):
            B, L, D = hidden.shape
            cumsum = torch.cumsum(hidden, dim=1)
            positions_for_mean = torch.arange(1, L + 1, device=hidden.device, dtype=torch.float32).view(1, -1, 1)
            running_mean = cumsum / positions_for_mean
            cumsum_sq = torch.cumsum(hidden ** 2, dim=1)
            running_var = (cumsum_sq / positions_for_mean) - (running_mean ** 2)
            running_std = (running_var.clamp(min=1e-8)).sqrt()
            return torch.cat([hidden, running_mean, running_std], dim=-1)

        # Get resonance for Lorenz on LTM trained with both
        # Use hidden representations with sequence context
        lorenz_ltm_input = compute_ltm_key_input(lorenz_hidden)
        chen_ltm_input = compute_ltm_key_input(chen_hidden)

        ltm_key_phase_lorenz = torch.tanh(block.ltm_key_encoder(lorenz_ltm_input)) * 3.14159
        ltm_key_phase_chen = torch.tanh(block.ltm_key_encoder(chen_ltm_input)) * 3.14159

        # Check if keys are similar across tasks
        # Use phase difference (cosine similarity of phases), not phasor product
        # Mean phase across batch and sequence for each dim
        mean_phase_lorenz = ltm_key_phase_lorenz.mean(dim=(0, 1))  # [ltm_planes]
        mean_phase_chen = ltm_key_phase_chen.mean(dim=(0, 1))      # [ltm_planes]

        # Phase difference (how similar are the mean keys?)
        phase_diff = (mean_phase_lorenz - mean_phase_chen).abs()
        mean_phase_diff = phase_diff.mean().item()

        # Also compute cosine similarity of the phases themselves
        cos_sim = torch.cos(mean_phase_lorenz - mean_phase_chen).mean().item()

        print(f"\n  Mean phase difference: {mean_phase_diff:.4f} radians")
        print(f"  Cosine similarity of mean phases: {cos_sim:.4f}")
        print(f"  (cos_sim=1.0 means identical phases, -1.0 means opposite)")

        # Also check variance within each task
        lorenz_phase_std = ltm_key_phase_lorenz.std(dim=(0, 1)).mean().item()
        chen_phase_std = ltm_key_phase_chen.std(dim=(0, 1)).mean().item()
        print(f"\n  Lorenz phase std (within-task variance): {lorenz_phase_std:.4f}")
        print(f"  Chen phase std (within-task variance): {chen_phase_std:.4f}")

        if cos_sim > 0.9:
            print("\n  WARNING: Keys too similar - LTM can't distinguish tasks!")
        elif cos_sim < 0.5:
            print("\n  Keys are distinct - LTM should be able to separate tasks")
        else:
            print("\n  Keys have moderate overlap")

    return probe.logs


if __name__ == "__main__":
    logs = run_diagnostic()
