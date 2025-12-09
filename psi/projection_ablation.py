"""
Ablation: Are input/output projections the cause of forgetting?

Test hypothesis: LTM stores knowledge correctly but corrupted projections
prevent proper retrieval.

Approach:
1. Train on Lorenz
2. FREEZE input_proj and output_proj
3. Train on Chen (only LTM and block internals can adapt)
4. Test Lorenz retention

If forgetting is eliminated, projections were the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from clifford_memory import ContinuousDynamicsModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


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


def train_epoch(model, contexts, targets, optimizer, batch_size=64):
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


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def run_ablation():
    print("=" * 70)
    print("ABLATION: Do Input/Output Projections Cause Forgetting?")
    print("=" * 70)
    print()

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    context_len = 20

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
    print()

    results = {}

    for freeze_projections in [False, True]:
        name = "Frozen Projections" if freeze_projections else "Baseline (All Trainable)"
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        # Create fresh model
        model = ContinuousDynamicsModel(
            input_dim=3,
            hidden_dim=128,
            n_layers=4,
            n_orthogonal_sets=4,
            planes_per_set=16
        ).to(device)

        # Reset LTM
        for block in model.blocks:
            block.reset_ltm()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Phase 1: Train on Lorenz
        print("\nPhase 1: Learning Lorenz...")
        for epoch in range(10):
            loss = train_epoch(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 5 == 0:
                val_loss = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"  Epoch {epoch+1}: train={loss:.6f}, val={val_loss:.6f}")

        lorenz_after_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        print(f"\n  Lorenz after Task A: MSE={lorenz_after_A:.6f}")

        # Optionally freeze projections
        if freeze_projections:
            print("\n  >>> FREEZING input_proj and output_proj <<<")
            for param in model.input_proj.parameters():
                param.requires_grad = False
            for param in model.output_proj.parameters():
                param.requires_grad = False

            # Also freeze norms since they're part of the pathway
            for norm in model.norms:
                for param in norm.parameters():
                    param.requires_grad = False

            print(f"  Trainable params: {count_params(model):,} (was ~{count_params(model, False):,})")

            # Create new optimizer with only trainable params
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-3, weight_decay=0.01
            )

        # Phase 2: Train on Chen
        print("\nPhase 2: Learning Chen...")
        for epoch in range(10):
            loss = train_epoch(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 5 == 0:
                chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
                lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"  Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

        lorenz_after_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        chen_final = evaluate(model, chen_ctx_test, chen_tgt_test)

        forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100

        results[name] = {
            'lorenz_A': lorenz_after_A,
            'lorenz_B': lorenz_after_B,
            'chen': chen_final,
            'forgetting': forgetting
        }

        print(f"\n  Final Results:")
        print(f"    Lorenz after A: {lorenz_after_A:.6f}")
        print(f"    Lorenz after B: {lorenz_after_B:.6f}")
        print(f"    Forgetting: {forgetting:+.1f}%")
        print(f"    Chen final: {chen_final:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Projection Ablation")
    print("=" * 70)
    print()

    baseline = results['Baseline (All Trainable)']
    frozen = results['Frozen Projections']

    print(f"{'Metric':<25} {'Baseline':>15} {'Frozen':>15} {'Diff':>12}")
    print("-" * 67)

    for label, key in [('Lorenz after Task A', 'lorenz_A'),
                       ('Lorenz after Task B', 'lorenz_B'),
                       ('Forgetting %', 'forgetting'),
                       ('Chen Final', 'chen')]:
        b_val = baseline[key]
        f_val = frozen[key]
        diff = f_val - b_val

        if 'Forgetting' in label:
            print(f"{label:<25} {b_val:>14.1f}% {f_val:>14.1f}% {diff:>+11.1f}%")
        else:
            print(f"{label:<25} {b_val:>15.6f} {f_val:>15.6f} {diff:>+12.6f}")

    print()

    # Interpretation
    baseline_forget = baseline['forgetting']
    frozen_forget = frozen['forgetting']

    if frozen_forget < baseline_forget * 0.5:
        print("CONFIRMED: Frozen projections significantly reduce forgetting!")
        print(f"  Forgetting reduced from {baseline_forget:.1f}% to {frozen_forget:.1f}%")
        print()
        print("IMPLICATION: The projections are the bottleneck, not LTM.")
        print("  Options:")
        print("    1. Use separate projections per task (modular)")
        print("    2. Make projections task-agnostic (shared structure)")
        print("    3. Protect projections with EWC-style regularization")
        print("    4. Let LTM bypass projections entirely")
    elif frozen_forget < baseline_forget * 0.9:
        print("PARTIAL: Frozen projections help somewhat")
    else:
        print("NOT CONFIRMED: Freezing projections doesn't prevent forgetting")
        print("  The forgetting must come from inside the blocks")

    # Also check if frozen model can still learn Chen
    if frozen['chen'] > baseline['chen'] * 1.5:
        print()
        print("WARNING: Frozen model struggles to learn Chen")
        print("  This suggests projections ARE needed for learning new tasks")
    else:
        print()
        print("GOOD: Frozen model can still learn Chen reasonably well")

    return results


if __name__ == "__main__":
    results = run_ablation()
