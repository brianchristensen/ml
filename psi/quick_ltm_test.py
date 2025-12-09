"""
Quick LTM Architecture Test (~30-60 seconds)

Exercises:
- Cross-bank binding (joint_key = product across banks)
- Surprise-gated writing (Titans-style prediction error)
- Positional phase planes

Uses small data, few epochs, just checks the components work and train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from clifford_memory import ContinuousDynamicsModel

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# Quick synthetic data
# ============================================================================

def generate_lorenz_quick(batch_size, seq_len, dt=0.01):
    """Small Lorenz batch"""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x = np.random.uniform(-15, 15, batch_size)
    y = np.random.uniform(-20, 20, batch_size)
    z = np.random.uniform(10, 40, batch_size)

    trajs = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        trajs[:, t, 0] = x
        trajs[:, t, 1] = y
        trajs[:, t, 2] = z
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dx*dt, y + dy*dt, z + dz*dt

    # Normalize
    mean = trajs.mean(axis=(0,1), keepdims=True)
    std = trajs.std(axis=(0,1), keepdims=True) + 1e-8
    return (trajs - mean) / std


def make_batches(trajs, context_len=10, batch_size=64):
    """Create context->target batches"""
    n_traj, traj_len, dim = trajs.shape
    contexts, targets = [], []

    for i in range(n_traj):
        for t in range(traj_len - context_len - 1):
            contexts.append(trajs[i, t:t+context_len])
            targets.append(trajs[i, t+context_len])

    contexts = np.array(contexts)
    targets = np.array(targets)

    # Shuffle and batch
    idx = np.random.permutation(len(contexts))
    contexts, targets = contexts[idx], targets[idx]

    batches = []
    for i in range(0, len(contexts), batch_size):
        ctx = torch.tensor(contexts[i:i+batch_size], device=device)
        tgt = torch.tensor(targets[i:i+batch_size], device=device)
        batches.append((ctx, tgt))
    return batches


# ============================================================================
# Quick test
# ============================================================================

def run_quick_test():
    print("=" * 60)
    print("QUICK LTM ARCHITECTURE TEST")
    print("=" * 60)
    print()
    print("Components being tested:")
    print("  - Cross-bank binding (joint_key product)")
    print("  - Surprise-gated writing (Titans-style)")
    print("  - Positional phase planes")
    print()

    start = time.time()

    # Small data: 50 trajectories, length 50
    print("Generating data...")
    train_data = generate_lorenz_quick(50, 50)
    val_data = generate_lorenz_quick(20, 50)

    context_len = 10
    train_batches = make_batches(train_data, context_len, batch_size=64)
    val_batches = make_batches(val_data, context_len, batch_size=64)

    print(f"  Train: {len(train_batches)} batches")
    print(f"  Val: {len(val_batches)} batches")
    print()

    # Create model
    model = ContinuousDynamicsModel(
        input_dim=3,
        hidden_dim=64,        # Smaller for speed
        n_layers=2,           # Fewer layers
        n_orthogonal_sets=4,
        planes_per_set=8      # Fewer planes
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Train 5 epochs - just enough to see learning
    print("Training (5 epochs)...")
    for epoch in range(5):
        model.train()
        train_loss = 0
        for ctx, tgt in train_batches:
            optimizer.zero_grad()
            pred = model(ctx)[:, -1, :]
            loss = F.mse_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ctx, tgt in val_batches:
                pred = model(ctx)[:, -1, :]
                val_loss += F.mse_loss(pred, tgt).item()

        train_loss /= len(train_batches)
        val_loss /= len(val_batches)
        print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

    elapsed = time.time() - start
    print()
    print(f"Completed in {elapsed:.1f}s")

    # Check: did it learn?
    if val_loss < 0.5:
        print("[OK] Model is learning (val_loss < 0.5)")
    else:
        print("[X] Model may not be learning properly")

    # Multi-step rollout test (5 steps)
    print()
    print("Multi-step rollout (5 steps)...")
    model.eval()

    test_traj = val_data[0]
    context = torch.tensor(test_traj[:context_len], device=device).unsqueeze(0)
    gt = test_traj[context_len:context_len+5]

    preds = []
    with torch.no_grad():
        for _ in range(5):
            pred = model(context)[:, -1, :]
            preds.append(pred.cpu().numpy()[0])
            context = torch.cat([context[:, 1:, :], pred.unsqueeze(1)], dim=1)

    preds = np.array(preds)
    errors = [np.mean((preds[i] - gt[i])**2) for i in range(5)]

    print(f"  Step 1: {errors[0]:.6f}")
    print(f"  Step 3: {errors[2]:.6f}")
    print(f"  Step 5: {errors[4]:.6f}")

    if errors[4] < 2.0:
        print("[OK] Multi-step prediction working")
    else:
        print("[~] Multi-step errors growing (expected for chaotic system)")

    print()
    print("=" * 60)
    print(f"TOTAL TIME: {elapsed:.1f}s")
    print("=" * 60)

    return val_loss, errors


if __name__ == "__main__":
    run_quick_test()
