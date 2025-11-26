"""
Springs N-Body Experiment: Direct Comparison with NRI Baseline

NRI Benchmark (Kipf et al., ICML 2018):
- Springs (5 particles): MSE @ 20 steps = 2.13e-5
- Charged (5 particles): MSE @ 20 steps = 7.06e-3

This experiment replicates the NRI springs setup for direct comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Springs Data Generation (NRI-style)
# ============================================================================

def spring_dynamics(state, edges, spring_const=0.1, damping=0.0):
    """
    Compute derivatives for spring-connected particle system.

    Matches NRI setup:
    - Particles in 2D box with elastic wall collisions
    - Springs between particle pairs follow Hooke's law
    - F = -k * (r - rest_length) * direction

    State: [n_particles, 4] - [x, y, vx, vy] per particle
    Edges: [n_particles, n_particles] - 1 if spring exists, 0 otherwise
    """
    n_particles = state.shape[0]
    positions = state[:, :2]
    velocities = state[:, 2:]

    # Compute accelerations from springs
    accelerations = np.zeros((n_particles, 2))

    rest_length = 1.0  # NRI uses rest length of 1

    for i in range(n_particles):
        for j in range(n_particles):
            if i != j and edges[i, j] == 1:
                # Vector from i to j
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff) + 1e-8
                direction = diff / dist

                # Hooke's law: F = -k * (dist - rest_length)
                # Positive when stretched (dist > rest), pulls toward j
                force_magnitude = spring_const * (dist - rest_length)
                accelerations[i] += force_magnitude * direction

    # Add damping
    accelerations -= damping * velocities

    return np.concatenate([velocities, accelerations], axis=1)


def simulate_springs(n_particles, n_steps, dt=0.001, spring_const=0.1,
                     edge_prob=0.5, box_size=5.0):
    """
    Simulate spring-connected particles in a box.

    Returns:
        trajectory: [n_steps, n_particles, 4] - states over time
        edges: [n_particles, n_particles] - adjacency matrix
    """
    # Random initial positions in box
    positions = np.random.uniform(-box_size/2, box_size/2, (n_particles, 2))

    # Random initial velocities (small)
    velocities = np.random.randn(n_particles, 2) * 0.5

    # Random edge structure (symmetric)
    edges = np.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            if np.random.rand() < edge_prob:
                edges[i, j] = 1
                edges[j, i] = 1

    state = np.concatenate([positions, velocities], axis=1)
    trajectory = [state.copy()]

    # Simulate with simple Euler integration (like NRI)
    for _ in range(n_steps - 1):
        deriv = spring_dynamics(state, edges, spring_const)
        state = state + dt * deriv

        # Box boundary collisions (elastic)
        for i in range(n_particles):
            for d in range(2):
                if state[i, d] < -box_size/2:
                    state[i, d] = -box_size/2
                    state[i, d + 2] *= -1  # Reverse velocity
                elif state[i, d] > box_size/2:
                    state[i, d] = box_size/2
                    state[i, d + 2] *= -1

        trajectory.append(state.copy())

    return np.array(trajectory), edges


def generate_springs_data(n_particles=5, n_trajectories=1000, n_steps=10000,
                          sample_freq=100, spring_const=0.1, edge_prob=0.5):
    """
    Generate springs dataset matching NRI format.

    NRI uses:
    - 5 particles
    - 5000 timesteps, sampled every 100 (= 50 frames)
    - Spring constant 0.1
    - Edge probability 0.5

    Returns:
        trajectories: [n_trajectories, n_frames, n_particles, 4]
        edges: [n_trajectories, n_particles, n_particles]
    """
    print(f"Generating {n_trajectories} spring trajectories...")
    print(f"  Particles: {n_particles}")
    print(f"  Raw steps: {n_steps}, Sample freq: {sample_freq}")
    print(f"  Final frames: {n_steps // sample_freq}")
    print(f"  Edge probability: {edge_prob}")

    n_frames = n_steps // sample_freq
    trajectories = np.zeros((n_trajectories, n_frames, n_particles, 4))
    all_edges = np.zeros((n_trajectories, n_particles, n_particles))

    for i in range(n_trajectories):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_trajectories}")

        traj, edges = simulate_springs(
            n_particles, n_steps,
            spring_const=spring_const,
            edge_prob=edge_prob
        )

        # Subsample
        trajectories[i] = traj[::sample_freq]
        all_edges[i] = edges

    # Statistics
    print(f"\nData statistics:")
    print(f"  Position range: [{trajectories[:,:,:,:2].min():.2f}, {trajectories[:,:,:,:2].max():.2f}]")
    print(f"  Velocity range: [{trajectories[:,:,:,2:].min():.2f}, {trajectories[:,:,:,2:].max():.2f}]")
    print(f"  Avg edges per trajectory: {all_edges.sum(axis=(1,2)).mean() / 2:.1f}")

    return trajectories, all_edges


# ============================================================================
# Charged Particles Data Generation (NRI-style)
# ============================================================================

def charged_dynamics(state, charges, interaction_strength=1.0, damping=0.0):
    """
    Compute derivatives for charged particle system.

    Coulomb's law: F = k * q1 * q2 / r^2
    - Same sign charges repel
    - Opposite sign charges attract

    State: [n_particles, 4] - [x, y, vx, vy] per particle
    Charges: [n_particles] - +1 or -1
    """
    n_particles = state.shape[0]
    positions = state[:, :2]
    velocities = state[:, 2:]

    accelerations = np.zeros((n_particles, 2))

    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff) + 1e-8
                direction = diff / dist

                # Coulomb: F = k * q1 * q2 / r^2
                # Positive force = attraction (toward j)
                # charges[i] * charges[j] < 0 means opposite signs = attract
                force_magnitude = -interaction_strength * charges[i] * charges[j] / (dist ** 2)
                accelerations[i] += force_magnitude * direction

    accelerations -= damping * velocities

    return np.concatenate([velocities, accelerations], axis=1)


def simulate_charged(n_particles, n_steps, dt=0.001, interaction_strength=1.0, box_size=5.0):
    """
    Simulate charged particles in a box.
    """
    positions = np.random.uniform(-box_size/2, box_size/2, (n_particles, 2))
    velocities = np.random.randn(n_particles, 2) * 0.5

    # Random charges: +1 or -1 with equal probability
    charges = np.random.choice([-1, 1], size=n_particles)

    state = np.concatenate([positions, velocities], axis=1)
    trajectory = [state.copy()]

    for _ in range(n_steps - 1):
        deriv = charged_dynamics(state, charges, interaction_strength)
        state = state + dt * deriv

        # Box collisions
        for i in range(n_particles):
            for d in range(2):
                if state[i, d] < -box_size/2:
                    state[i, d] = -box_size/2
                    state[i, d + 2] *= -1
                elif state[i, d] > box_size/2:
                    state[i, d] = box_size/2
                    state[i, d + 2] *= -1

        trajectory.append(state.copy())

    return np.array(trajectory), charges


def generate_charged_data(n_particles=5, n_trajectories=1000, n_steps=10000,
                          sample_freq=100, interaction_strength=1.0):
    """
    Generate charged particles dataset matching NRI format.
    """
    print(f"Generating {n_trajectories} charged particle trajectories...")
    print(f"  Particles: {n_particles}")
    print(f"  Raw steps: {n_steps}, Sample freq: {sample_freq}")
    print(f"  Final frames: {n_steps // sample_freq}")

    n_frames = n_steps // sample_freq
    trajectories = np.zeros((n_trajectories, n_frames, n_particles, 4))
    all_charges = np.zeros((n_trajectories, n_particles))

    for i in range(n_trajectories):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_trajectories}")

        traj, charges = simulate_charged(
            n_particles, n_steps,
            interaction_strength=interaction_strength
        )

        trajectories[i] = traj[::sample_freq]
        all_charges[i] = charges

    print(f"\nData statistics:")
    print(f"  Position range: [{trajectories[:,:,:,:2].min():.2f}, {trajectories[:,:,:,:2].max():.2f}]")
    print(f"  Velocity range: [{trajectories[:,:,:,2:].min():.2f}, {trajectories[:,:,:,2:].max():.2f}]")

    return trajectories, all_charges


# ============================================================================
# Dataset
# ============================================================================

class SpringsDataset(Dataset):
    """Dataset for springs/charged particle trajectories."""

    def __init__(self, trajectories, context_len=49, pred_len=1, split='train',
                 train_frac=0.8, normalize=True):
        """
        Args:
            trajectories: [n_traj, n_frames, n_particles, 4]
            context_len: Number of frames to condition on (NRI uses 49)
            pred_len: Number of frames to predict (for multi-step)
            split: 'train' or 'val'
        """
        self.n_particles = trajectories.shape[2]
        self.context_len = context_len
        self.pred_len = pred_len

        # Flatten particles into feature dimension: [n_traj, n_frames, n_particles * 4]
        self.trajectories = trajectories.reshape(
            trajectories.shape[0], trajectories.shape[1], -1
        ).astype(np.float32)

        # Normalize
        self.normalize = normalize
        if normalize:
            self.mean = self.trajectories.mean(axis=(0, 1))
            self.std = self.trajectories.std(axis=(0, 1))
            self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)
        else:
            self.mean = np.zeros(self.trajectories.shape[-1])
            self.std = np.ones(self.trajectories.shape[-1])

        # Split
        n_train = int(len(self.trajectories) * train_frac)
        if split == 'train':
            self.trajectories = self.trajectories[:n_train]
        else:
            self.trajectories = self.trajectories[n_train:]

        # Each trajectory gives multiple sequences
        self.n_frames = self.trajectories.shape[1]
        self.seqs_per_traj = self.n_frames - context_len - pred_len + 1

        print(f"{split.upper()}: {len(self.trajectories)} trajectories, "
              f"{len(self)} sequences, {self.n_particles} particles")

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.seqs_per_traj
        pos = idx % self.seqs_per_traj

        traj = self.trajectories[traj_idx]
        context = traj[pos:pos + self.context_len]
        target = traj[pos + self.context_len:pos + self.context_len + self.pred_len]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target.squeeze(0) if self.pred_len == 1 else target,
                                   dtype=torch.float32)
        }


# ============================================================================
# Model
# ============================================================================

class PSIDynamicsPredictor(nn.Module):
    """PSI model for particle dynamics prediction."""

    def __init__(self, state_dim, dim=256, num_layers=6, max_len=100, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        self.blocks = nn.ModuleList([PSIBlock(dim=dim) for _ in range(num_layers)])

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, state_dim)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, states):
        batch_size, seq_len, _ = states.shape
        x = self.state_embedding(states)
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        for block in self.blocks:
            x = block(x)

        final_state = x[:, -1, :]
        prediction = self.output_head(final_state)
        return prediction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        context = batch['context'].to(device)
        target = batch['target'].to(device)

        prediction = model(context)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            prediction = model(context)
            loss = criterion(prediction, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ============================================================================
# Multi-step Rollout Evaluation (NRI-style)
# ============================================================================

def evaluate_rollout(model, trajectories, mean, std, context_len, pred_steps, device):
    """
    Evaluate model with multi-step rollout.

    NRI evaluation:
    - Feed encoder first 49 timesteps
    - Predict next 20 timesteps autoregressively
    - Report MSE at steps 1, 10, 20

    Returns MSE at various prediction horizons in NORMALIZED scale (like NRI).
    """
    model.eval()

    n_particles = trajectories.shape[2]
    n_frames = trajectories.shape[1]

    # Check if trajectories are long enough
    required_frames = context_len + pred_steps
    if n_frames < required_frames:
        print(f"Warning: Trajectories have {n_frames} frames, need {required_frames}")
        print(f"  Adjusting pred_steps from {pred_steps} to {n_frames - context_len}")
        pred_steps = n_frames - context_len

    # Flatten: [n_traj, n_frames, n_particles * 4]
    traj_flat = trajectories.reshape(trajectories.shape[0], trajectories.shape[1], -1)

    # Normalize
    traj_norm = (traj_flat - mean) / (std + 1e-8)

    all_mse_norm = {1: [], 10: [], 20: []}
    all_mse_orig = {1: [], 10: [], 20: []}

    with torch.no_grad():
        for traj_idx in range(len(trajectories)):
            traj = traj_norm[traj_idx]
            traj_orig = traj_flat[traj_idx]

            # Context: first context_len frames
            context = torch.tensor(traj[:context_len], dtype=torch.float32).unsqueeze(0).to(device)

            predictions_norm = []

            # Autoregressive rollout
            for step in range(pred_steps):
                pred = model(context)  # [1, state_dim]
                predictions_norm.append(pred.cpu().numpy()[0])

                # Shift context
                pred_expanded = pred.unsqueeze(1)
                context = torch.cat([context[:, 1:, :], pred_expanded], dim=1)

            predictions_norm = np.array(predictions_norm)  # [pred_steps, state_dim]

            # Ground truth normalized
            gt_norm = traj[context_len:context_len + pred_steps]

            # Also compute original scale for reference
            predictions_orig = predictions_norm * (std + 1e-8) + mean
            gt_orig = traj_orig[context_len:context_len + pred_steps]

            # MSE at specific horizons (positions only, not velocities)
            pos_indices = []
            for p in range(n_particles):
                pos_indices.extend([p * 4, p * 4 + 1])  # x, y for each particle

            for horizon in [1, 10, 20]:
                if horizon <= pred_steps and horizon <= len(gt_norm):
                    # Normalized MSE (for NRI comparison)
                    pred_pos_norm = predictions_norm[horizon - 1, pos_indices]
                    gt_pos_norm = gt_norm[horizon - 1, pos_indices]
                    mse_norm = np.mean((pred_pos_norm - gt_pos_norm) ** 2)
                    all_mse_norm[horizon].append(mse_norm)

                    # Original scale MSE (for interpretability)
                    pred_pos_orig = predictions_orig[horizon - 1, pos_indices]
                    gt_pos_orig = gt_orig[horizon - 1, pos_indices]
                    mse_orig = np.mean((pred_pos_orig - gt_pos_orig) ** 2)
                    all_mse_orig[horizon].append(mse_orig)

    # Average MSE across trajectories
    results_norm = {}
    results_orig = {}
    for horizon in [1, 10, 20]:
        if all_mse_norm[horizon]:
            results_norm[horizon] = np.mean(all_mse_norm[horizon])
            results_orig[horizon] = np.mean(all_mse_orig[horizon])

    return results_norm, results_orig


# ============================================================================
# Visualization
# ============================================================================

def visualize_trajectories(gt_traj, pred_traj, n_particles, title, filename):
    """Visualize ground truth vs predicted trajectories."""
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    for p in range(n_particles):
        x = gt_traj[:, p * 4]
        y = gt_traj[:, p * 4 + 1]
        axes[0].plot(x, y, color=colors[p % len(colors)], alpha=0.7, linewidth=2, label=f'P{p+1}')
        axes[0].scatter(x[0], y[0], c=colors[p % len(colors)], s=100, marker='o', zorder=5)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # PSI predictions
    for p in range(n_particles):
        x = pred_traj[:, p * 4]
        y = pred_traj[:, p * 4 + 1]
        axes[1].plot(x, y, color=colors[p % len(colors)], alpha=0.7, linewidth=2, label=f'P{p+1}')
        axes[1].scatter(x[0], y[0], c=colors[p % len(colors)], s=100, marker='o', zorder=5)
    axes[1].set_title('PSI Predictions')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    # Overlay
    for p in range(n_particles):
        gt_x, gt_y = gt_traj[:, p * 4], gt_traj[:, p * 4 + 1]
        pred_x, pred_y = pred_traj[:, p * 4], pred_traj[:, p * 4 + 1]
        axes[2].plot(gt_x, gt_y, color=colors[p % len(colors)], alpha=0.4, linewidth=3)
        axes[2].plot(pred_x, pred_y, color=colors[p % len(colors)], alpha=0.8, linewidth=2, linestyle='--')
    axes[2].set_title('Overlay (solid=GT, dashed=PSI)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Springs/Charged Particles Experiment')
    parser.add_argument('--system', type=str, default='springs', choices=['springs', 'charged'],
                        help='Which system to simulate')
    parser.add_argument('--n_particles', type=int, default=5, help='Number of particles')
    parser.add_argument('--n_trajectories', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('--context_len', type=int, default=49, help='Context length (NRI uses 49)')
    parser.add_argument('--pred_steps', type=int, default=20, help='Prediction steps for evaluation')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of PSI layers')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on existing model')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PSI {args.system.upper()} Experiment - NRI Benchmark Comparison")
    print("=" * 80)
    print()
    print("NRI Baselines (Kipf et al., ICML 2018):")
    print("  Springs (5 particles):  1-step=3.12e-8, 10-step=3.29e-6, 20-step=2.13e-5")
    print("  Charged (5 particles):  1-step=1.05e-3, 10-step=3.21e-3, 20-step=7.06e-3")
    print()

    # Generate data
    if args.system == 'springs':
        trajectories, edges = generate_springs_data(
            n_particles=args.n_particles,
            n_trajectories=args.n_trajectories
        )
    else:
        trajectories, charges = generate_charged_data(
            n_particles=args.n_particles,
            n_trajectories=args.n_trajectories
        )

    # Create datasets
    state_dim = args.n_particles * 4
    train_dataset = SpringsDataset(trajectories, args.context_len, split='train')
    val_dataset = SpringsDataset(trajectories, args.context_len, split='val')

    # Store normalization params for evaluation
    mean, std = train_dataset.mean, train_dataset.std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = PSIDynamicsPredictor(
        state_dim=state_dim,
        dim=args.dim,
        num_layers=args.num_layers,
        max_len=args.context_len + 10
    ).to(device)

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print()

    if args.eval_only:
        # Load existing model
        print("Loading existing model for evaluation...")
        checkpoint = torch.load(f'{args.system}_best.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        mean = checkpoint['mean']
        std = checkpoint['std']
        best_val_loss = checkpoint['val_loss']
        total_train_time = 0
        print(f"Loaded model with val loss: {best_val_loss:.6f}")
    else:
        # Training
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        train_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) - "
                      f"Train: {train_loss:.6f} - Val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'mean': mean,
                    'std': std,
                }, f'{args.system}_best.pt')

        total_train_time = time.time() - train_start
        print(f"\nTraining complete in {total_train_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")

        # Load best model
        checkpoint = torch.load(f'{args.system}_best.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate with multi-step rollout
    print("\n" + "=" * 80)
    print("Multi-step Rollout Evaluation (NRI-style)")
    print("=" * 80)

    # Use validation trajectories
    val_trajectories = trajectories[int(len(trajectories) * 0.8):]

    # Time the inference
    inference_start = time.time()
    mse_results_norm, mse_results_orig = evaluate_rollout(
        model, val_trajectories, mean, std,
        args.context_len, args.pred_steps, device
    )
    inference_time = time.time() - inference_start

    print(f"\nPSI Results ({args.system}, {args.n_particles} particles):")
    print("-" * 50)
    print("NORMALIZED MSE (for NRI comparison):")
    for horizon, mse in sorted(mse_results_norm.items()):
        print(f"  {horizon:2d}-step MSE: {mse:.6e}")
    print("\nOriginal scale MSE (for interpretability):")
    for horizon, mse in sorted(mse_results_orig.items()):
        print(f"  {horizon:2d}-step MSE: {mse:.6e}")

    print(f"\nInference time for {len(val_trajectories)} trajectories: {inference_time:.2f}s")
    print(f"Per-trajectory: {inference_time/len(val_trajectories)*1000:.2f}ms")

    # NRI comparison
    print("\n" + "=" * 80)
    print("Comparison with NRI Baseline (NORMALIZED MSE)")
    print("=" * 80)

    if args.system == 'springs':
        nri_baselines = {1: 3.12e-8, 10: 3.29e-6, 20: 2.13e-5}
    else:
        nri_baselines = {1: 1.05e-3, 10: 3.21e-3, 20: 7.06e-3}

    print(f"\n{'Horizon':<10} {'NRI':<15} {'PSI':<15} {'Ratio (PSI/NRI)':<15}")
    print("-" * 55)
    for horizon in [1, 10, 20]:
        if horizon in mse_results_norm:
            nri = nri_baselines[horizon]
            psi = mse_results_norm[horizon]
            ratio = psi / nri
            print(f"{horizon:<10} {nri:<15.2e} {psi:<15.2e} {ratio:<15.1f}x")

    # Visualize some trajectories
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    model.eval()
    n_viz = min(3, len(val_trajectories))

    for i in range(n_viz):
        traj = val_trajectories[i]
        traj_flat = traj.reshape(traj.shape[0], -1)
        traj_norm = (traj_flat - mean) / (std + 1e-8)

        # Get context and predict
        context = torch.tensor(traj_norm[:args.context_len], dtype=torch.float32).unsqueeze(0).to(device)

        predictions = []
        with torch.no_grad():
            for _ in range(args.pred_steps):
                pred = model(context)
                predictions.append(pred.cpu().numpy()[0])
                context = torch.cat([context[:, 1:, :], pred.unsqueeze(1)], dim=1)

        predictions = np.array(predictions)
        predictions_orig = predictions * (std + 1e-8) + mean

        gt = traj_flat[args.context_len:args.context_len + args.pred_steps]

        visualize_trajectories(
            gt, predictions_orig, args.n_particles,
            f'{args.system.title()} Sample {i+1}',
            f'{args.system}_comparison_{i}.png'
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"System: {args.system}")
    print(f"Particles: {args.n_particles}")
    print(f"Trajectories: {args.n_trajectories}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training time: {total_train_time:.1f}s")
    print(f"Best val loss (normalized): {best_val_loss:.6f}")
    print(f"\nNormalized MSE Results (NRI comparison):")
    for horizon, mse in sorted(mse_results_norm.items()):
        nri = nri_baselines[horizon]
        ratio = mse / nri
        status = "✓" if ratio < 100 else "✗"
        print(f"  {horizon:2d}-step: {mse:.2e} (NRI: {nri:.2e}, {ratio:.1f}x) {status}")


if __name__ == "__main__":
    main()
