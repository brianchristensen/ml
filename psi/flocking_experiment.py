"""
Flocking (Boids) Emergent Behavior: Generate, Train, Evaluate

Test PSI on complex adaptive systems with emergent properties.
Can it learn the implicit governing equations for collective behavior
that emerges from simple local rules (separation, alignment, cohesion)?

Key questions:
1. Can PSI learn the local interaction rules?
2. Can it predict emergent flock formations?
3. Does it capture phase transitions (disorder -> order)?
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

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Boids Flocking Simulation
# ============================================================================

class BoidsSimulation:
    """
    Classic Boids algorithm with three rules:
    1. Separation: Steer to avoid crowding local flockmates
    2. Alignment: Steer towards average heading of local flockmates
    3. Cohesion: Steer towards average position of local flockmates

    Plus boundary avoidance to keep flock in view.
    """

    def __init__(
        self,
        n_boids=20,
        width=100.0,
        height=100.0,
        max_speed=2.0,
        max_force=0.1,
        separation_radius=5.0,
        alignment_radius=15.0,
        cohesion_radius=25.0,
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        boundary_margin=10.0,
        boundary_force=0.5
    ):
        self.n_boids = n_boids
        self.width = width
        self.height = height
        self.max_speed = max_speed
        self.max_force = max_force

        # Perception radii
        self.separation_radius = separation_radius
        self.alignment_radius = alignment_radius
        self.cohesion_radius = cohesion_radius

        # Rule weights
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight

        # Boundary handling
        self.boundary_margin = boundary_margin
        self.boundary_force = boundary_force

        # State: [x, y, vx, vy] per boid
        self.positions = None
        self.velocities = None

    def initialize(self, cluster_start=False):
        """Initialize boid positions and velocities."""
        if cluster_start:
            # Start clustered in center - good for observing flock formation
            center = np.array([self.width / 2, self.height / 2])
            self.positions = center + np.random.randn(self.n_boids, 2) * 10
        else:
            # Random positions
            self.positions = np.random.rand(self.n_boids, 2) * np.array([self.width, self.height])

        # Random initial velocities
        angles = np.random.rand(self.n_boids) * 2 * np.pi
        speeds = np.random.rand(self.n_boids) * self.max_speed * 0.5 + self.max_speed * 0.5
        self.velocities = np.stack([
            np.cos(angles) * speeds,
            np.sin(angles) * speeds
        ], axis=1)

    def get_state(self):
        """Return current state as [n_boids, 4] array."""
        return np.concatenate([self.positions, self.velocities], axis=1)

    def set_state(self, state):
        """Set state from [n_boids, 4] array."""
        self.positions = state[:, :2].copy()
        self.velocities = state[:, 2:].copy()

    def _limit_magnitude(self, vectors, max_mag):
        """Limit vector magnitudes."""
        mags = np.linalg.norm(vectors, axis=1, keepdims=True)
        mags = np.maximum(mags, 1e-8)
        scale = np.minimum(1.0, max_mag / mags)
        return vectors * scale

    def _separation(self, boid_idx):
        """Separation: steer away from nearby boids."""
        steer = np.zeros(2)
        count = 0

        for j in range(self.n_boids):
            if j != boid_idx:
                diff = self.positions[boid_idx] - self.positions[j]
                dist = np.linalg.norm(diff)

                if 0 < dist < self.separation_radius:
                    # Weight by inverse distance (closer = stronger repulsion)
                    steer += diff / (dist * dist)
                    count += 1

        if count > 0:
            steer /= count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed
                steer -= self.velocities[boid_idx]
                steer = self._limit_magnitude(steer.reshape(1, -1), self.max_force)[0]

        return steer

    def _alignment(self, boid_idx):
        """Alignment: steer towards average heading of neighbors."""
        avg_velocity = np.zeros(2)
        count = 0

        for j in range(self.n_boids):
            if j != boid_idx:
                dist = np.linalg.norm(self.positions[boid_idx] - self.positions[j])

                if dist < self.alignment_radius:
                    avg_velocity += self.velocities[j]
                    count += 1

        if count > 0:
            avg_velocity /= count
            if np.linalg.norm(avg_velocity) > 0:
                avg_velocity = avg_velocity / np.linalg.norm(avg_velocity) * self.max_speed
                steer = avg_velocity - self.velocities[boid_idx]
                steer = self._limit_magnitude(steer.reshape(1, -1), self.max_force)[0]
                return steer

        return np.zeros(2)

    def _cohesion(self, boid_idx):
        """Cohesion: steer towards center of mass of neighbors."""
        center = np.zeros(2)
        count = 0

        for j in range(self.n_boids):
            if j != boid_idx:
                dist = np.linalg.norm(self.positions[boid_idx] - self.positions[j])

                if dist < self.cohesion_radius:
                    center += self.positions[j]
                    count += 1

        if count > 0:
            center /= count
            desired = center - self.positions[boid_idx]
            if np.linalg.norm(desired) > 0:
                desired = desired / np.linalg.norm(desired) * self.max_speed
                steer = desired - self.velocities[boid_idx]
                steer = self._limit_magnitude(steer.reshape(1, -1), self.max_force)[0]
                return steer

        return np.zeros(2)

    def _boundary_avoidance(self, boid_idx):
        """Steer away from boundaries."""
        steer = np.zeros(2)
        pos = self.positions[boid_idx]

        # Left boundary
        if pos[0] < self.boundary_margin:
            steer[0] = self.boundary_force
        # Right boundary
        elif pos[0] > self.width - self.boundary_margin:
            steer[0] = -self.boundary_force

        # Bottom boundary
        if pos[1] < self.boundary_margin:
            steer[1] = self.boundary_force
        # Top boundary
        elif pos[1] > self.height - self.boundary_margin:
            steer[1] = -self.boundary_force

        return steer

    def step(self, dt=1.0):
        """Advance simulation by one timestep."""
        accelerations = np.zeros((self.n_boids, 2))

        for i in range(self.n_boids):
            # Apply three rules
            sep = self._separation(i) * self.separation_weight
            ali = self._alignment(i) * self.alignment_weight
            coh = self._cohesion(i) * self.cohesion_weight
            bnd = self._boundary_avoidance(i)

            accelerations[i] = sep + ali + coh + bnd

        # Update velocities
        self.velocities += accelerations * dt
        self.velocities = self._limit_magnitude(self.velocities, self.max_speed)

        # Update positions
        self.positions += self.velocities * dt

        # Wrap around boundaries (toroidal) or clamp
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height)

    def compute_metrics(self):
        """Compute flock metrics for analysis."""
        # Average velocity (alignment measure)
        avg_vel = np.mean(self.velocities, axis=0)
        alignment = np.linalg.norm(avg_vel) / self.max_speed

        # Average distance to centroid (cohesion measure)
        centroid = np.mean(self.positions, axis=0)
        distances = np.linalg.norm(self.positions - centroid, axis=1)
        cohesion = np.mean(distances)

        # Polarization (how aligned are all velocities)
        vel_norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        vel_norms = np.maximum(vel_norms, 1e-8)
        unit_vels = self.velocities / vel_norms
        polarization = np.linalg.norm(np.mean(unit_vels, axis=0))

        return {
            'alignment': alignment,
            'cohesion': cohesion,
            'polarization': polarization
        }


def generate_flocking_data(
    n_boids=20,
    trials=500,
    timesteps=100,
    dt=1.0,
    cluster_start=True,
    vary_params=True,
    fixed_separation=None,
    fixed_alignment=None,
    fixed_cohesion=None
):
    """
    Generate flocking dynamics dataset.

    Returns:
        x: [trials, timesteps, n_boids, 4] - positions and velocities
        metrics: [trials, timesteps, 3] - alignment, cohesion, polarization
    """
    print(f"Generating {trials} flocking simulations with {n_boids} boids...")
    print(f"  Timesteps: {timesteps}")
    print(f"  Cluster start: {cluster_start}")
    print(f"  Varying parameters: {vary_params}")

    if fixed_separation is not None or fixed_alignment is not None or fixed_cohesion is not None:
        print(f"  FIXED PARAMS (OOD test):")
        print(f"    Separation: {fixed_separation}")
        print(f"    Alignment: {fixed_alignment}")
        print(f"    Cohesion: {fixed_cohesion}")

    x = np.zeros((trials, timesteps, n_boids, 4))
    metrics = np.zeros((trials, timesteps, 3))

    for trial in range(trials):
        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1}/{trials}")

        # Use fixed params if provided (OOD test), otherwise vary or use defaults
        if fixed_separation is not None:
            sep_weight = fixed_separation
        elif vary_params:
            sep_weight = np.random.uniform(1.0, 2.0)
        else:
            sep_weight = 1.5

        if fixed_alignment is not None:
            ali_weight = fixed_alignment
        elif vary_params:
            ali_weight = np.random.uniform(0.8, 1.2)
        else:
            ali_weight = 1.0

        if fixed_cohesion is not None:
            coh_weight = fixed_cohesion
        elif vary_params:
            coh_weight = np.random.uniform(0.8, 1.2)
        else:
            coh_weight = 1.0

        sim = BoidsSimulation(
            n_boids=n_boids,
            separation_weight=sep_weight,
            alignment_weight=ali_weight,
            cohesion_weight=coh_weight
        )
        sim.initialize(cluster_start=cluster_start)

        for t in range(timesteps):
            x[trial, t] = sim.get_state()
            m = sim.compute_metrics()
            metrics[trial, t] = [m['alignment'], m['cohesion'], m['polarization']]
            sim.step(dt)

    # Report statistics
    final_polarization = metrics[:, -1, 2]
    print(f"\nFlock statistics (final timestep):")
    print(f"  Polarization: {np.mean(final_polarization):.3f} ± {np.std(final_polarization):.3f}")
    print(f"  (1.0 = perfect alignment, 0.0 = random directions)")

    return x, metrics


# ============================================================================
# Dataset
# ============================================================================

class FlockingDataset(Dataset):
    """Flocking dynamics dataset."""

    def __init__(self, trajectories, metrics, context_len=20, split='train', train_frac=0.8):
        self.n_boids = trajectories.shape[2]

        # Flatten boid states: [trials, timesteps, n_boids * 4]
        self.trajectories = trajectories.reshape(
            trajectories.shape[0], trajectories.shape[1], -1
        ).astype(np.float32)

        self.metrics = metrics.astype(np.float32)

        # Normalize positions and velocities
        self.mean = self.trajectories.mean(axis=(0, 1))
        self.std = self.trajectories.std(axis=(0, 1))
        self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        # Split
        num_train = int(len(self.trajectories) * train_frac)
        if split == 'train':
            self.trajectories = self.trajectories[:num_train]
            self.metrics = self.metrics[:num_train]
        else:
            self.trajectories = self.trajectories[num_train:]
            self.metrics = self.metrics[num_train:]

        self.context_len = context_len
        self.seqs_per_traj = self.trajectories.shape[1] - context_len - 1

        print(f"{split.upper()}: {len(self.trajectories)} trajectories, {len(self)} sequences")

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.seqs_per_traj
        pos = idx % self.seqs_per_traj

        trajectory = self.trajectories[traj_idx]
        context = trajectory[pos:pos + self.context_len]
        target = trajectory[pos + self.context_len]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


# ============================================================================
# Model
# ============================================================================

class PSIFlockingPredictor(nn.Module):
    """PSI model for flocking dynamics."""

    def __init__(self, state_dim, dim=256, num_layers=8, max_len=100, device='cuda'):
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
            nn.Dropout(0.1),
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
# Visualization
# ============================================================================

def predict_trajectory(model, initial_states, num_steps, mean, std, device):
    """Rollout PSI predictions autoregressively."""
    model.eval()
    predictions = []

    context = (initial_states - mean) / (std + 1e-8)
    context = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_steps):
            next_state = model(context)
            next_state_denorm = next_state.cpu().numpy()[0] * (std + 1e-8) + mean
            predictions.append(next_state_denorm)

            next_state_expanded = next_state.unsqueeze(1)
            context = torch.cat([context[:, 1:, :], next_state_expanded], dim=1)

    return np.array(predictions)


def visualize_flocking(ground_truth, psi_predictions, n_boids, sample_idx=0, save_dir='.'):
    """
    Visualize flocking trajectories and emergent behavior.

    Creates:
    1. Trajectory comparison (GT vs PSI)
    2. Snapshot comparison at different times
    3. Velocity field visualization
    """
    colors = plt.cm.tab20(np.linspace(0, 1, n_boids))

    # Reshape predictions: [timesteps, n_boids, 4]
    gt = ground_truth.reshape(-1, n_boids, 4)
    pred = psi_predictions.reshape(-1, n_boids, 4)

    num_steps = min(len(gt), len(pred))
    gt = gt[:num_steps]
    pred = pred[:num_steps]

    # ===== Figure 1: Trajectory Comparison =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth trajectories
    for i in range(n_boids):
        axes[0].plot(gt[:, i, 0], gt[:, i, 1], color=colors[i], alpha=0.6, linewidth=1)
        axes[0].scatter(gt[0, i, 0], gt[0, i, 1], color=colors[i], s=50, marker='o', zorder=5)
        axes[0].scatter(gt[-1, i, 0], gt[-1, i, 1], color=colors[i], s=50, marker='s', zorder=5)
    axes[0].set_title('Ground Truth Trajectories')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # PSI predictions
    for i in range(n_boids):
        axes[1].plot(pred[:, i, 0], pred[:, i, 1], color=colors[i], alpha=0.6, linewidth=1)
        axes[1].scatter(pred[0, i, 0], pred[0, i, 1], color=colors[i], s=50, marker='o', zorder=5)
        axes[1].scatter(pred[-1, i, 0], pred[-1, i, 1], color=colors[i], s=50, marker='s', zorder=5)
    axes[1].set_title('PSI Predictions (No Boids Rules!)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    # Overlay
    for i in range(n_boids):
        axes[2].plot(gt[:, i, 0], gt[:, i, 1], color=colors[i], alpha=0.3, linewidth=2)
        axes[2].plot(pred[:, i, 0], pred[:, i, 1], color=colors[i], alpha=0.8, linewidth=1, linestyle='--')
    axes[2].set_title('Overlay (solid=GT, dashed=PSI)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')

    plt.tight_layout()
    filename = f'{save_dir}/flocking_trajectories_sample_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

    # ===== Figure 2: Snapshots with Velocity Vectors =====
    snapshot_times = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    fig, axes = plt.subplots(2, len(snapshot_times), figsize=(20, 8))

    for col, t in enumerate(snapshot_times):
        # Ground truth
        ax = axes[0, col]
        ax.quiver(
            gt[t, :, 0], gt[t, :, 1],
            gt[t, :, 2], gt[t, :, 3],
            color=[colors[i] for i in range(n_boids)],
            scale=50, width=0.008
        )
        ax.scatter(gt[t, :, 0], gt[t, :, 1], c=colors, s=30, zorder=5)
        ax.set_title(f'GT t={t}')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # PSI prediction
        ax = axes[1, col]
        ax.quiver(
            pred[t, :, 0], pred[t, :, 1],
            pred[t, :, 2], pred[t, :, 3],
            color=[colors[i] for i in range(n_boids)],
            scale=50, width=0.008
        )
        ax.scatter(pred[t, :, 0], pred[t, :, 1], c=colors, s=30, zorder=5)
        ax.set_title(f'PSI t={t}')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Ground Truth')
    axes[1, 0].set_ylabel('PSI Prediction')

    plt.suptitle(f'Flocking Snapshots - Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    filename = f'{save_dir}/flocking_snapshots_sample_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def visualize_emergent_metrics(ground_truth, psi_predictions, n_boids, sample_idx=0, save_dir='.'):
    """
    Visualize emergent behavior metrics over time.
    Compare alignment, cohesion, polarization between GT and PSI.
    """
    gt = ground_truth.reshape(-1, n_boids, 4)
    pred = psi_predictions.reshape(-1, n_boids, 4)

    num_steps = min(len(gt), len(pred))

    def compute_metrics_series(states):
        """Compute metrics for a trajectory."""
        alignment = []
        cohesion = []
        polarization = []

        for t in range(len(states)):
            positions = states[t, :, :2]
            velocities = states[t, :, 2:]

            # Alignment: magnitude of average velocity
            avg_vel = np.mean(velocities, axis=0)
            alignment.append(np.linalg.norm(avg_vel))

            # Cohesion: average distance to centroid
            centroid = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            cohesion.append(np.mean(distances))

            # Polarization: alignment of unit velocities
            vel_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
            vel_norms = np.maximum(vel_norms, 1e-8)
            unit_vels = velocities / vel_norms
            polarization.append(np.linalg.norm(np.mean(unit_vels, axis=0)))

        return np.array(alignment), np.array(cohesion), np.array(polarization)

    gt_align, gt_coh, gt_pol = compute_metrics_series(gt[:num_steps])
    pred_align, pred_coh, pred_pol = compute_metrics_series(pred[:num_steps])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Alignment
    axes[0].plot(gt_align, 'b-', label='Ground Truth', linewidth=2)
    axes[0].plot(pred_align, 'r--', label='PSI Prediction', linewidth=2)
    axes[0].set_title('Alignment (Avg Velocity Magnitude)')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Alignment')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cohesion
    axes[1].plot(gt_coh, 'b-', label='Ground Truth', linewidth=2)
    axes[1].plot(pred_coh, 'r--', label='PSI Prediction', linewidth=2)
    axes[1].set_title('Cohesion (Avg Distance to Centroid)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Polarization
    axes[2].plot(gt_pol, 'b-', label='Ground Truth', linewidth=2)
    axes[2].plot(pred_pol, 'r--', label='PSI Prediction', linewidth=2)
    axes[2].set_title('Polarization (Velocity Alignment)')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Polarization')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Emergent Behavior Metrics - Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    filename = f'{save_dir}/flocking_metrics_sample_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

    # Return final metrics for summary
    return {
        'gt_polarization': gt_pol[-1],
        'pred_polarization': pred_pol[-1],
        'gt_cohesion': gt_coh[-1],
        'pred_cohesion': pred_coh[-1]
    }


def visualize_error_rollout(all_errors, save_dir='.'):
    """Plot prediction error over rollout horizon."""
    errors = np.array(all_errors)
    mean_error = errors.mean(axis=0)
    std_error = errors.std(axis=0)

    plt.figure(figsize=(10, 5))
    timesteps = np.arange(len(mean_error))
    plt.plot(timesteps, mean_error, 'b-', linewidth=2, label='Mean MSE')
    plt.fill_between(timesteps, mean_error - std_error, mean_error + std_error,
                     alpha=0.3, color='blue', label='±1 Std')
    plt.xlabel('Rollout Step')
    plt.ylabel('MSE')
    plt.title('Prediction Error vs Rollout Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'{save_dir}/flocking_error_rollout.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_boids', type=int, default=20, help='Number of boids')
    parser.add_argument('--trials', type=int, default=500, help='Number of trajectories')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per trajectory')
    parser.add_argument('--context_len', type=int, default=20, help='Context length')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--rollout_steps', type=int, default=50, help='Steps to predict')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, load best model and evaluate')
    parser.add_argument('--generalization_test', action='store_true', help='Test on out-of-distribution parameters')
    parser.add_argument('--test_separation', type=float, default=None, help='OOD separation weight')
    parser.add_argument('--test_alignment', type=float, default=None, help='OOD alignment weight')
    parser.add_argument('--test_cohesion', type=float, default=None, help='OOD cohesion weight')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PSI Flocking (Boids) Emergent Behavior Experiment")
    print("=" * 80)
    print(f"Testing: Can PSI learn the implicit governing equations")
    print(f"         for emergent collective behavior?")
    print("=" * 80)
    print()

    state_dim = args.n_boids * 4  # Each boid: x, y, vx, vy

    if args.eval_only or args.generalization_test:
        # Load checkpoint to get normalization stats and generate minimal eval data
        mode = "Generalization test" if args.generalization_test else "Eval-only"
        print(f"{mode} mode: Loading saved model...")
        checkpoint = torch.load('flocking_best.pt', weights_only=False)
        best_val_loss = checkpoint['val_loss']
        print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss {best_val_loss:.6f}")

        # Generate eval dataset - with OOD params if generalization test
        print("\nGenerating evaluation trajectories...")

        if args.generalization_test:
            print("\n*** GENERALIZATION TEST ***")
            print("Training distribution:")
            print("  Separation: [1.0, 2.0]")
            print("  Alignment:  [0.8, 1.2]")
            print("  Cohesion:   [0.8, 1.2]")
            print(f"\nTest distribution (OOD):")
            print(f"  Separation: {args.test_separation or 'varied'}")
            print(f"  Alignment:  {args.test_alignment or 'varied'}")
            print(f"  Cohesion:   {args.test_cohesion or 'varied'}")
            print()

        x, metrics = generate_flocking_data(
            n_boids=args.n_boids,
            trials=100,  # Small set for eval
            timesteps=args.timesteps,
            cluster_start=True,
            vary_params=not args.generalization_test,  # Don't vary if testing specific OOD params
            fixed_separation=args.test_separation,
            fixed_alignment=args.test_alignment,
            fixed_cohesion=args.test_cohesion
        )

        # Create dataset just to get normalization stats
        eval_dataset = FlockingDataset(x, metrics, args.context_len, split='train', train_frac=1.0)

        # Create and load model
        model = PSIFlockingPredictor(
            state_dim, args.dim, args.num_layers, args.context_len, device
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model parameters: {model.count_parameters():,}\n")

        # Use all generated trajectories for eval
        val_trajectories = x
        train_dataset = eval_dataset  # For normalization stats

    else:
        # Full training mode
        # Generate data
        x, metrics = generate_flocking_data(
            n_boids=args.n_boids,
            trials=args.trials,
            timesteps=args.timesteps,
            cluster_start=True,
            vary_params=True
        )

        # Create datasets
        train_dataset = FlockingDataset(x, metrics, args.context_len, split='train')
        val_dataset = FlockingDataset(x, metrics, args.context_len, split='val')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Create model
        model = PSIFlockingPredictor(
            state_dim, args.dim, args.num_layers, args.context_len, device
        ).to(device)
        print(f"\nModel parameters: {model.count_parameters():,}\n")

        # Train
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{args.epochs} ({time.time()-start:.1f}s) - "
                  f"Train: {train_loss:.6f} - Val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, 'flocking_best.pt')
                print(f"  Saved best model")

        # Load best model
        checkpoint = torch.load('flocking_best.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        val_trajectories = x[int(len(x) * 0.8):]

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating Emergent Behavior Prediction...")
    print("=" * 80)

    all_errors = []
    all_emergent_metrics = []

    for i in range(min(5, len(val_trajectories))):
        gt_traj = val_trajectories[i]  # [timesteps, n_boids, 4]

        # Get initial context
        initial = gt_traj[:args.context_len].reshape(args.context_len, -1)

        # Predict
        psi_pred = predict_trajectory(
            model, initial, args.rollout_steps,
            train_dataset.mean, train_dataset.std, device
        )

        # Ground truth for comparison
        gt_future = gt_traj[args.context_len:args.context_len + args.rollout_steps]
        gt_flat = gt_future.reshape(-1, args.n_boids * 4)

        # Visualize
        visualize_flocking(gt_flat, psi_pred, args.n_boids, i)
        emergent = visualize_emergent_metrics(gt_flat, psi_pred, args.n_boids, i)
        all_emergent_metrics.append(emergent)

        # Compute per-step errors
        errors = []
        for t in range(min(len(gt_flat), len(psi_pred))):
            mse = np.mean((gt_flat[t] - psi_pred[t])**2)
            errors.append(mse)
        all_errors.append(errors)

        print(f"Sample {i}:")
        print(f"  Final MSE: {errors[-1]:.6f}")
        print(f"  GT Polarization: {emergent['gt_polarization']:.3f}")
        print(f"  PSI Polarization: {emergent['pred_polarization']:.3f}")

    # Plot error rollout
    visualize_error_rollout(all_errors)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")

    avg_gt_pol = np.mean([m['gt_polarization'] for m in all_emergent_metrics])
    avg_pred_pol = np.mean([m['pred_polarization'] for m in all_emergent_metrics])
    print(f"Average GT Polarization: {avg_gt_pol:.3f}")
    print(f"Average PSI Polarization: {avg_pred_pol:.3f}")
    print(f"Polarization Preservation: {avg_pred_pol/avg_gt_pol*100:.1f}%")


if __name__ == "__main__":
    main()
