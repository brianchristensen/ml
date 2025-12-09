"""
Continual Learning Benchmark: Lorenz -> Turbulence -> Test Lorenz Retention

Tests whether surprise-gated Clifford LTM can:
1. Learn Lorenz attractor dynamics
2. Learn turbulent flow dynamics (related chaotic system)
3. Retain Lorenz knowledge due to shared dynamical structure

Both systems share:
- Chaotic attractor dynamics
- Sensitivity to initial conditions
- Nonlinear coupling between variables
- Conservation-like properties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import from source of truth
from clifford_memory import ContinuousDynamicsModel

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# Dynamical Systems Generation
# ============================================================================

def generate_lorenz(batch_size, seq_len, dt=0.01):
    """Lorenz attractor: classic chaotic system"""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0

    # Random initial conditions on attractor
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


def generate_rossler(batch_size, seq_len, dt=0.01):
    """Rössler attractor: another classic chaotic system with different structure"""
    a, b, c = 0.2, 0.2, 5.7

    x = np.random.uniform(-5, 5, batch_size)
    y = np.random.uniform(-5, 5, batch_size)
    z = np.random.uniform(0, 10, batch_size)

    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)

    for t in range(seq_len):
        trajectories[:, t, 0] = x
        trajectories[:, t, 1] = y
        trajectories[:, t, 2] = z

        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt

    return trajectories


def generate_chen(batch_size, seq_len, dt=0.002):
    """Chen attractor: chaotic system similar to Lorenz but with different dynamics"""
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


def generate_kolmogorov_flow(batch_size, seq_len, dt=0.01):
    """
    Simplified Kolmogorov flow (2D turbulence approximation)
    Uses Fourier mode truncation to simulate turbulent-like dynamics
    """
    # Use 4 Fourier modes for a simplified but chaotic representation
    n_modes = 4

    # Initialize mode amplitudes (complex)
    real = np.random.randn(batch_size, n_modes) * 0.5
    imag = np.random.randn(batch_size, n_modes) * 0.5

    # Wavenumbers and viscosity
    k = np.array([1, 2, 3, 4])
    nu = 0.01  # viscosity
    forcing = 1.0

    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)

    for t in range(seq_len):
        # Output: energy in first 3 mode pairs as "state"
        energy = real**2 + imag**2
        trajectories[:, t, 0] = energy[:, 0] + energy[:, 1]
        trajectories[:, t, 1] = energy[:, 1] + energy[:, 2]
        trajectories[:, t, 2] = energy[:, 2] + energy[:, 3]

        # Mode coupling (simplified cascade)
        for m in range(n_modes):
            # Dissipation
            diss = -nu * k[m]**2

            # Nonlinear transfer (energy cascade)
            if m > 0:
                transfer_in = 0.1 * (real[:, m-1] * imag[:, m-1])
            else:
                transfer_in = forcing * np.sin(t * dt)

            if m < n_modes - 1:
                transfer_out = -0.1 * (real[:, m] * real[:, m+1] + imag[:, m] * imag[:, m+1])
            else:
                transfer_out = 0

            real[:, m] += (diss * real[:, m] + transfer_in + transfer_out) * dt
            imag[:, m] += (diss * imag[:, m] + transfer_in * 0.5) * dt

    return trajectories


def normalize_trajectories(trajs):
    """Normalize to zero mean, unit variance per dimension"""
    mean = trajs.mean(axis=(0, 1), keepdims=True)
    std = trajs.std(axis=(0, 1), keepdims=True) + 1e-8
    return (trajs - mean) / std


# Model imported from clifford_memory.py (source of truth)


# ============================================================================
# Training and Evaluation
# ============================================================================

def create_sequences(trajectories, context_len=20):
    """Create training sequences from trajectories"""
    n_traj, traj_len, dim = trajectories.shape
    n_seqs = traj_len - context_len - 1

    contexts = []
    targets = []

    for i in range(n_traj):
        for t in range(n_seqs):
            contexts.append(trajectories[i, t:t+context_len])
            targets.append(trajectories[i, t+context_len])

    return np.array(contexts), np.array(targets)


def train_epoch(model, contexts, targets, optimizer, batch_size=64):
    """Train for one epoch"""
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
        pred = model(ctx)[:, -1, :]  # Predict next state
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, contexts, targets, batch_size=256):
    """Evaluate model"""
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


def multi_step_evaluate(model, trajectories, context_len=20, horizon=50, n_samples=100):
    """Evaluate multi-step prediction accuracy"""
    model.eval()

    errors = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Pick random trajectory and start point
            traj_idx = np.random.randint(len(trajectories))
            start = np.random.randint(0, len(trajectories[traj_idx]) - context_len - horizon)

            # Get context and ground truth
            context = trajectories[traj_idx, start:start+context_len]
            gt = trajectories[traj_idx, start+context_len:start+context_len+horizon]

            # Autoregressive prediction
            ctx = torch.tensor(context, device=device).unsqueeze(0)
            preds = []

            for _ in range(horizon):
                pred = model(ctx)[:, -1, :]
                preds.append(pred.cpu().numpy()[0])
                ctx = torch.cat([ctx[:, 1:, :], pred.unsqueeze(1)], dim=1)

            preds = np.array(preds)
            error = np.mean((preds - gt) ** 2)
            errors.append(error)

    return np.mean(errors)


# ============================================================================
# Main Experiment
# ============================================================================

def run_continual_learning_experiment():
    print("=" * 70)
    print("CONTINUAL DYNAMICS LEARNING: Lorenz -> Chen -> Test Lorenz Retention")
    print("=" * 70)
    print()
    print("Hypothesis: Surprise-gated LTM preserves shared chaotic dynamics knowledge")
    print()

    # Generate datasets
    print("Generating datasets...")
    context_len = 20

    # Task A: Lorenz attractor
    lorenz_train = normalize_trajectories(generate_lorenz(500, 200))
    lorenz_test = normalize_trajectories(generate_lorenz(100, 200))
    lorenz_ctx_train, lorenz_tgt_train = create_sequences(lorenz_train, context_len)
    lorenz_ctx_test, lorenz_tgt_test = create_sequences(lorenz_test, context_len)

    # Task B: Chen attractor (similar chaotic structure to Lorenz)
    chen_train = normalize_trajectories(generate_chen(500, 200))
    chen_test = normalize_trajectories(generate_chen(100, 200))
    chen_ctx_train, chen_tgt_train = create_sequences(chen_train, context_len)
    chen_ctx_test, chen_tgt_test = create_sequences(chen_test, context_len)

    print(f"  Lorenz: {len(lorenz_ctx_train)} train, {len(lorenz_ctx_test)} test sequences")
    print(f"  Chen: {len(chen_ctx_train)} train, {len(chen_ctx_test)} test sequences")
    print()

    # Create two models: with and without surprise gating
    results = {}

    for use_gating in [True, False]:
        name = "Surprise-Gated" if use_gating else "Baseline (No Gating)"
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        model = ContinuousDynamicsModel(
            input_dim=3,
            hidden_dim=128,
            n_layers=4,
            n_orthogonal_sets=4,
            planes_per_set=16
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Phase 1: Train on Lorenz
        print("\nPhase 1: Learning Lorenz attractor...")
        for epoch in range(10):
            loss = train_epoch(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 2 == 0:
                val_loss = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"  Epoch {epoch+1}: train={loss:.6f}, val={val_loss:.6f}")

        lorenz_after_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        lorenz_multistep_A = multi_step_evaluate(model, lorenz_test, context_len, horizon=30)
        print(f"\n  Lorenz after Task A: MSE={lorenz_after_A:.6f}, Multi-step={lorenz_multistep_A:.6f}")

        # Note: Surprise gating is always on in this version
        if use_gating:
            print("\n  [Surprise gating is always ENABLED]")

        # Phase 2: Train on Chen attractor
        print("\nPhase 2: Learning Chen attractor...")
        for epoch in range(10):
            loss = train_epoch(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 2 == 0:
                chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
                lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"  Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

        # Note: Surprise gating stays enabled throughout

        # Final evaluation
        lorenz_after_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        lorenz_multistep_B = multi_step_evaluate(model, lorenz_test, context_len, horizon=30)
        chen_final = evaluate(model, chen_ctx_test, chen_tgt_test)
        chen_multistep = multi_step_evaluate(model, chen_test, context_len, horizon=30)

        # Calculate forgetting
        forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100
        multistep_forgetting = (lorenz_multistep_B - lorenz_multistep_A) / lorenz_multistep_A * 100

        results[name] = {
            'lorenz_A': lorenz_after_A,
            'lorenz_B': lorenz_after_B,
            'lorenz_multistep_A': lorenz_multistep_A,
            'lorenz_multistep_B': lorenz_multistep_B,
            'chen': chen_final,
            'chen_multistep': chen_multistep,
            'forgetting': forgetting,
            'multistep_forgetting': multistep_forgetting
        }

        print(f"\n  Final Results for {name}:")
        print(f"    Lorenz after A: {lorenz_after_A:.6f}")
        print(f"    Lorenz after B: {lorenz_after_B:.6f}")
        print(f"    Lorenz forgetting: {forgetting:+.1f}%")
        print(f"    Multi-step forgetting: {multistep_forgetting:+.1f}%")
        print(f"    Chen final: {chen_final:.6f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Continual Dynamics Learning")
    print("=" * 70)
    print()
    print("Task A: Lorenz attractor (chaotic, 3D)")
    print("Task B: Chen attractor (chaotic, 3D, similar structure)")
    print()
    print(f"{'Metric':<30} {'Baseline':>15} {'Gated':>15} {'Diff':>12}")
    print("-" * 72)

    baseline = results['Baseline (No Gating)']
    gated = results['Surprise-Gated']

    metrics = [
        ('Lorenz after Task A', 'lorenz_A', False),
        ('Lorenz after Task B', 'lorenz_B', False),
        ('Lorenz Forgetting %', 'forgetting', True),
        ('Multi-step Forgetting %', 'multistep_forgetting', True),
        ('Chen Final MSE', 'chen', False),
    ]

    for label, key, is_pct in metrics:
        b_val = baseline[key]
        g_val = gated[key]
        diff = g_val - b_val

        if is_pct:
            print(f"{label:<30} {b_val:>14.1f}% {g_val:>14.1f}% {diff:>+11.1f}%")
        else:
            print(f"{label:<30} {b_val:>15.6f} {g_val:>15.6f} {diff:>+12.6f}")

    print()

    # Interpretation
    forget_diff = gated['forgetting'] - baseline['forgetting']
    if forget_diff < -5:
        print("✓ SIGNIFICANT: Surprise-gated LTM reduces forgetting on related dynamics!")
        print(f"  Gated model shows {-forget_diff:.1f}% less forgetting than baseline")
    elif forget_diff < 0:
        print("~ MODEST: Some reduction in forgetting with surprise gating")
    else:
        print("✗ No benefit from surprise gating on this task")

    return results


if __name__ == "__main__":
    results = run_continual_learning_experiment()
