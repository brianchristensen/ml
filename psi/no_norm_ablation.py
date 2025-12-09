"""
Ablation: Remove normalization to preserve task identity

Hypothesis: Without global normalization, Lorenz and Chen have different
natural scales that will create distinct phase distributions in LTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict

from clifford_memory import ContinuousDynamicsModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def generate_lorenz(batch_size, seq_len, dt=0.01):
    """Lorenz system - natural scale ~[-20, 45]"""
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
    """Chen system - natural scale ~[-25, 35]"""
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


def light_normalize(trajs, scale=100.0):
    """
    Light normalization: just divide by a constant scale factor.
    Preserves relative differences between systems while keeping values in reasonable range.
    """
    return trajs / scale


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


def analyze_data_statistics(lorenz, chen, name_l="Lorenz", name_c="Chen"):
    """Print statistics to verify task identity is preserved"""
    print(f"\n  Data Statistics:")
    print(f"    {name_l}:")
    print(f"      x: [{lorenz[:,:,0].min():.2f}, {lorenz[:,:,0].max():.2f}], mean={lorenz[:,:,0].mean():.2f}, std={lorenz[:,:,0].std():.2f}")
    print(f"      y: [{lorenz[:,:,1].min():.2f}, {lorenz[:,:,1].max():.2f}], mean={lorenz[:,:,1].mean():.2f}, std={lorenz[:,:,1].std():.2f}")
    print(f"      z: [{lorenz[:,:,2].min():.2f}, {lorenz[:,:,2].max():.2f}], mean={lorenz[:,:,2].mean():.2f}, std={lorenz[:,:,2].std():.2f}")
    print(f"    {name_c}:")
    print(f"      x: [{chen[:,:,0].min():.2f}, {chen[:,:,0].max():.2f}], mean={chen[:,:,0].mean():.2f}, std={chen[:,:,0].std():.2f}")
    print(f"      y: [{chen[:,:,1].min():.2f}, {chen[:,:,1].max():.2f}], mean={chen[:,:,1].mean():.2f}, std={chen[:,:,1].std():.2f}")
    print(f"      z: [{chen[:,:,2].min():.2f}, {chen[:,:,2].max():.2f}], mean={chen[:,:,2].mean():.2f}, std={chen[:,:,2].std():.2f}")


def compute_phase_overlap(model, lorenz_ctx, chen_ctx, device):
    """Compute phase overlap between Lorenz and Chen keys"""
    model.eval()
    n_samples = min(500, len(lorenz_ctx), len(chen_ctx))

    lorenz_data = torch.tensor(lorenz_ctx[:n_samples], device=device)
    chen_data = torch.tensor(chen_ctx[:n_samples], device=device)

    overlaps_key = []
    overlaps_ltm = []

    with torch.no_grad():
        lorenz_h = model.input_proj(lorenz_data)
        chen_h = model.input_proj(chen_data)

        for i, (norm, block) in enumerate(zip(model.norms, model.blocks)):
            lorenz_normed = norm(lorenz_h)
            chen_normed = norm(chen_h)

            # Key phases
            lorenz_key_phase = torch.tanh(block.key_encoder(lorenz_normed)) * math.pi
            chen_key_phase = torch.tanh(block.key_encoder(chen_normed)) * math.pi

            # Compute histogram overlap
            lorenz_flat = lorenz_key_phase.cpu().numpy().flatten()
            chen_flat = chen_key_phase.cpu().numpy().flatten()

            bins = np.linspace(-math.pi, math.pi, 51)
            lorenz_hist, _ = np.histogram(lorenz_flat, bins=bins, density=True)
            chen_hist, _ = np.histogram(chen_flat, bins=bins, density=True)
            lorenz_hist = lorenz_hist / lorenz_hist.sum()
            chen_hist = chen_hist / chen_hist.sum()
            overlap = np.minimum(lorenz_hist, chen_hist).sum()
            overlaps_key.append(overlap)

            # LTM key phases
            B, L, D = lorenz_normed.shape
            positions = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, -1, 1)

            # Lorenz LTM
            l_cumsum = torch.cumsum(lorenz_normed, dim=1)
            l_mean = l_cumsum / positions
            l_var = (torch.cumsum(lorenz_normed**2, dim=1) / positions) - l_mean**2
            l_std = l_var.clamp(min=1e-8).sqrt()
            l_ltm_input = torch.cat([lorenz_normed, l_mean, l_std], dim=-1)
            l_ltm_phase = torch.tanh(block.ltm_key_encoder(l_ltm_input)) * math.pi

            # Chen LTM
            c_cumsum = torch.cumsum(chen_normed, dim=1)
            c_mean = c_cumsum / positions
            c_var = (torch.cumsum(chen_normed**2, dim=1) / positions) - c_mean**2
            c_std = c_var.clamp(min=1e-8).sqrt()
            c_ltm_input = torch.cat([chen_normed, c_mean, c_std], dim=-1)
            c_ltm_phase = torch.tanh(block.ltm_key_encoder(c_ltm_input)) * math.pi

            l_flat = l_ltm_phase.cpu().numpy().flatten()
            c_flat = c_ltm_phase.cpu().numpy().flatten()
            l_hist, _ = np.histogram(l_flat, bins=bins, density=True)
            c_hist, _ = np.histogram(c_flat, bins=bins, density=True)
            l_hist = l_hist / l_hist.sum()
            c_hist = c_hist / c_hist.sum()
            ltm_overlap = np.minimum(l_hist, c_hist).sum()
            overlaps_ltm.append(ltm_overlap)

            # Pass through block
            lorenz_h = block(lorenz_normed)
            chen_h = block(chen_normed)

    return overlaps_key, overlaps_ltm


def run_experiment():
    print("=" * 70)
    print("ABLATION: No Normalization (Preserve Task Identity)")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    context_len = 20
    n_epochs = 5  # Reduced for faster iteration

    # Generate RAW data (no normalization) - reduced size for speed
    print("\nGenerating datasets (NO normalization)...")
    lorenz_train_raw = generate_lorenz(100, 100)
    lorenz_test_raw = generate_lorenz(30, 100)
    chen_train_raw = generate_chen(100, 100)
    chen_test_raw = generate_chen(30, 100)

    # Show raw statistics
    analyze_data_statistics(lorenz_train_raw, chen_train_raw)

    # Light normalization (just scale, preserve differences)
    print("\n  Applying light normalization (divide by 100)...")
    lorenz_train = light_normalize(lorenz_train_raw)
    lorenz_test = light_normalize(lorenz_test_raw)
    chen_train = light_normalize(chen_train_raw)
    chen_test = light_normalize(chen_test_raw)

    # Show normalized statistics
    analyze_data_statistics(lorenz_train, chen_train, "Lorenz (scaled)", "Chen (scaled)")

    lorenz_ctx_train, lorenz_tgt_train = create_sequences(lorenz_train, context_len)
    lorenz_ctx_test, lorenz_tgt_test = create_sequences(lorenz_test, context_len)
    chen_ctx_train, chen_tgt_train = create_sequences(chen_train, context_len)
    chen_ctx_test, chen_tgt_test = create_sequences(chen_test, context_len)

    print(f"\n  Lorenz: {len(lorenz_ctx_train)} train, {len(lorenz_ctx_test)} test")
    print(f"  Chen: {len(chen_ctx_train)} train, {len(chen_ctx_test)} test")

    # Create model
    model = ContinuousDynamicsModel(
        input_dim=3,
        hidden_dim=128,
        n_layers=4,
        n_orthogonal_sets=4,
        planes_per_set=16
    ).to(device)

    for block in model.blocks:
        block.reset_ltm()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Phase 1: Train on Lorenz
    print("\n" + "=" * 60)
    print("Phase 1: Learning Lorenz...")
    print("=" * 60)

    for epoch in range(n_epochs):
        loss = train_epoch(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)
        if (epoch + 1) % 2 == 0:
            val_loss = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Epoch {epoch+1}: train={loss:.6f}, val={val_loss:.6f}")

    lorenz_after_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    print(f"\n  Lorenz after Task A: MSE={lorenz_after_A:.6f}")

    # Analyze phase overlap BEFORE Chen
    print("\n  Analyzing phase distributions...")
    key_overlap_before, ltm_overlap_before = compute_phase_overlap(
        model, lorenz_ctx_test, chen_ctx_test, device
    )
    print(f"  Key phase overlap:  {[f'{o:.3f}' for o in key_overlap_before]}")
    print(f"  LTM phase overlap:  {[f'{o:.3f}' for o in ltm_overlap_before]}")

    # Get LTM stats
    ltm_stats_before = model.blocks[0].get_ltm_stats()
    print(f"  LTM count: {ltm_stats_before['ltm_count']:.1f}")

    # Phase 2: Train on Chen
    print("\n" + "=" * 60)
    print("Phase 2: Learning Chen...")
    print("=" * 60)

    for epoch in range(n_epochs):
        loss = train_epoch(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)
        if (epoch + 1) % 2 == 0:
            chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
            lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

    lorenz_after_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    chen_final = evaluate(model, chen_ctx_test, chen_tgt_test)

    # Analyze phase overlap AFTER Chen
    print("\n  Analyzing phase distributions after Chen...")
    key_overlap_after, ltm_overlap_after = compute_phase_overlap(
        model, lorenz_ctx_test, chen_ctx_test, device
    )
    print(f"  Key phase overlap:  {[f'{o:.3f}' for o in key_overlap_after]}")
    print(f"  LTM phase overlap:  {[f'{o:.3f}' for o in ltm_overlap_after]}")

    ltm_stats_after = model.blocks[0].get_ltm_stats()
    print(f"  LTM count: {ltm_stats_after['ltm_count']:.1f}")

    # Results
    forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100

    print("\n" + "=" * 70)
    print("RESULTS: No Normalization")
    print("=" * 70)
    print(f"  Lorenz after A: {lorenz_after_A:.6f}")
    print(f"  Lorenz after B: {lorenz_after_B:.6f}")
    print(f"  Forgetting: {forgetting:+.1f}%")
    print(f"  Chen final: {chen_final:.6f}")

    print("\n  Phase Overlap Comparison:")
    print(f"    Key overlap (before -> after): {np.mean(key_overlap_before):.3f} -> {np.mean(key_overlap_after):.3f}")
    print(f"    LTM overlap (before -> after): {np.mean(ltm_overlap_before):.3f} -> {np.mean(ltm_overlap_after):.3f}")

    # Compare to normalized baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO NORMALIZED BASELINE")
    print("=" * 70)
    print("  Metric                    Normalized    No-Norm")
    print("  " + "-" * 50)
    print(f"  Key overlap (before):       0.888         {np.mean(key_overlap_before):.3f}")
    print(f"  LTM overlap (before):       0.928         {np.mean(ltm_overlap_before):.3f}")
    print(f"  Forgetting:                +10781%        {forgetting:+.0f}%")

    if forgetting < 5000:
        print("\n  IMPROVEMENT: Removing normalization reduced forgetting!")
        if np.mean(ltm_overlap_before) < 0.8:
            print("  REASON: Lower phase overlap allows better task discrimination")
    else:
        print("\n  NO IMPROVEMENT: Forgetting still severe")
        print("  IMPLICATION: Need additional mechanism for task separation")

    return {
        'lorenz_A': lorenz_after_A,
        'lorenz_B': lorenz_after_B,
        'chen': chen_final,
        'forgetting': forgetting,
        'key_overlap_before': key_overlap_before,
        'ltm_overlap_before': ltm_overlap_before,
        'key_overlap_after': key_overlap_after,
        'ltm_overlap_after': ltm_overlap_after,
    }


if __name__ == "__main__":
    results = run_experiment()
