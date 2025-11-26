"""
nuScenes Multi-Sensor Fusion Experiment for PSI

Tests PSI on real-world sensor fusion for object tracking:
- Fuses radar, lidar, and camera detections
- Predicts future object trajectories
- No hand-engineered sensor models - learns from data

Download nuScenes mini from:
https://www.kaggle.com/datasets/aadimator/nuscenes-mini

Extract to: data/nuscenes/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# nuScenes Data Loading
# ============================================================================

def check_nuscenes_available(dataroot):
    """Check if nuScenes data is available."""
    path = Path(dataroot)
    required = ['v1.0-mini', 'samples']

    if not path.exists():
        return False, f"Directory {dataroot} does not exist"

    for req in required:
        if not (path / req).exists():
            return False, f"Missing {req} directory in {dataroot}"

    return True, "nuScenes mini dataset found"


class NuScenesTrackingDataset(Dataset):
    """
    Dataset for trajectory forecasting using nuScenes tracked object data.

    NOTE: This is trajectory forecasting, NOT sensor fusion. The objects are
    already detected and tracked by nuScenes. We're predicting future motion
    given past trajectory.

    For each tracked instance, we gather:
    - Position (x, y, z) in global coordinates
    - Velocity (vx, vy) computed from position differences
    - Bounding box size (width, length, height)
    - Heading angle
    """

    # Class-level cache for extracted tracks (avoid re-extracting for each split)
    _tracks_cache = None
    _nusc_cache = None

    def __init__(self, dataroot, context_len=10, pred_len=10, min_track_len=25,
                 track_indices=None):
        """
        Args:
            dataroot: Path to nuScenes data
            context_len: Number of past timesteps to use as context
            pred_len: Number of future timesteps to predict
            min_track_len: Minimum track length to include
            track_indices: If provided, only use these track indices (for train/val/test split)
        """
        self.context_len = context_len
        self.pred_len = pred_len
        self.min_track_len = min_track_len

        # Load nuScenes and extract tracks (cached at class level)
        if NuScenesTrackingDataset._tracks_cache is None:
            from nuscenes.nuscenes import NuScenes
            print(f"Loading nuScenes from {dataroot}...")
            NuScenesTrackingDataset._nusc_cache = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
            NuScenesTrackingDataset._tracks_cache = self._extract_all_tracks(
                NuScenesTrackingDataset._nusc_cache, min_track_len
            )
            print(f"Extracted {len(NuScenesTrackingDataset._tracks_cache)} tracks total")

        self.nusc = NuScenesTrackingDataset._nusc_cache
        all_tracks = NuScenesTrackingDataset._tracks_cache

        # Filter to specified track indices if provided (for proper train/val/test split)
        if track_indices is not None:
            self.tracks = [all_tracks[i] for i in track_indices if i < len(all_tracks)]
        else:
            self.tracks = all_tracks

        # Create samples (context, target pairs)
        self.samples = self._create_samples()

        print(f"Created {len(self.samples)} samples from {len(self.tracks)} tracks")

    @staticmethod
    def _extract_all_tracks(nusc, min_track_len):
        """Extract all object tracks from the dataset (static method for caching)."""
        tracks = []

        for instance in tqdm(nusc.instance, desc="Extracting tracks"):
            track = NuScenesTrackingDataset._get_instance_track_static(nusc, instance)

            if len(track) >= min_track_len:
                # Compute velocities from position differences
                track = NuScenesTrackingDataset._compute_velocities_static(track)
                tracks.append(track)

        return tracks

    @staticmethod
    def _get_instance_track_static(nusc, instance):
        """Get the full trajectory of a tracked instance (static method)."""
        from pyquaternion import Quaternion

        track = []
        ann_token = instance['first_annotation_token']

        while ann_token != '':
            ann = nusc.get('sample_annotation', ann_token)
            sample = nusc.get('sample', ann['sample_token'])

            # Get ego pose at this sample
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

            # Extract state
            state = NuScenesTrackingDataset._extract_state_static(nusc, ann, sample, ego_pose)
            track.append(state)

            ann_token = ann['next']

        return np.array(track) if track else np.array([]).reshape(0, 13)

    @staticmethod
    def _extract_state_static(nusc, annotation, sample, ego_pose):
        """Extract state vector (static method)."""
        from pyquaternion import Quaternion

        # Position
        pos = np.array(annotation['translation'])

        # Velocity - placeholder, will compute from positions
        vel = np.zeros(2)

        # Size
        size = np.array(annotation['size'])

        # Heading
        q = Quaternion(annotation['rotation'])
        yaw = q.yaw_pitch_roll[0]

        # Sensor flags (approximate based on distance)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_pos = np.array(ego_pose_data['translation'][:2])
        obj_pos = pos[:2]
        dist = np.linalg.norm(obj_pos - ego_pos)

        lidar_flag = 1.0 if dist < 70 else 0.0
        radar_front = 1.0 if dist < 200 else 0.0
        radar_side = 1.0 if dist < 100 else 0.0
        camera = 1.0 if dist < 50 else 0.0

        state = np.concatenate([
            pos,
            vel,
            size,
            [yaw],
            [lidar_flag, radar_front, radar_side, camera]
        ])

        return state.astype(np.float32)

    @staticmethod
    def _compute_velocities_static(track):
        """Compute velocities from position differences (static method)."""
        dt = 0.5

        for i in range(len(track)):
            if np.allclose(track[i, 3:5], 0):
                if i > 0:
                    dx = track[i, 0] - track[i-1, 0]
                    dy = track[i, 1] - track[i-1, 1]
                    track[i, 3] = dx / dt
                    track[i, 4] = dy / dt
                elif i < len(track) - 1:
                    dx = track[i+1, 0] - track[i, 0]
                    dy = track[i+1, 1] - track[i, 1]
                    track[i, 3] = dx / dt
                    track[i, 4] = dy / dt

        return track

    @classmethod
    def get_num_tracks(cls, dataroot, min_track_len=25):
        """Get number of tracks without creating samples (for splitting)."""
        if cls._tracks_cache is None:
            from nuscenes.nuscenes import NuScenes
            print(f"Loading nuScenes from {dataroot}...")
            cls._nusc_cache = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
            cls._tracks_cache = cls._extract_all_tracks(cls._nusc_cache, min_track_len)
            print(f"Extracted {len(cls._tracks_cache)} tracks total")
        return len(cls._tracks_cache)

    @classmethod
    def clear_cache(cls):
        """Clear the class-level cache."""
        cls._tracks_cache = None
        cls._nusc_cache = None

    def _create_samples(self):
        """Create (context, target) pairs from tracks."""
        samples = []

        for track in self.tracks:
            track_len = len(track)

            # Slide window over track
            for start in range(track_len - self.context_len - self.pred_len + 1):
                context = track[start:start + self.context_len].copy()
                target = track[start + self.context_len:start + self.context_len + self.pred_len].copy()

                # Normalize to be relative to last context position
                # This is crucial - global coords are huge (UTM), we need relative motion
                ref_pos = context[-1, :3].copy()  # Last context position (x, y, z)
                ref_yaw = context[-1, 8]  # Last context heading

                # Transform positions to be relative to reference point
                context[:, :3] -= ref_pos
                target[:, :3] -= ref_pos

                # Optionally rotate to align with heading (ego-centric frame)
                # For now, just translate - rotation can help but adds complexity

                samples.append((context, target))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


class SyntheticMultiSensorDataset(Dataset):
    """
    Synthetic dataset mimicking multi-sensor tracking for testing without nuScenes.

    Generates trajectories with:
    - Realistic vehicle dynamics (constant velocity + noise)
    - Simulated sensor dropouts
    - Varying observation quality
    """

    def __init__(self, n_tracks=1000, track_len=50, context_len=10, pred_len=10):
        self.context_len = context_len
        self.pred_len = pred_len

        print("Generating synthetic multi-sensor tracking data...")
        self.tracks = self._generate_tracks(n_tracks, track_len)
        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} samples from {n_tracks} synthetic tracks")

    def _generate_tracks(self, n_tracks, track_len):
        """Generate synthetic vehicle tracks."""
        tracks = []

        for _ in range(n_tracks):
            # Initial state
            x, y = np.random.uniform(-50, 50, 2)
            z = 0.0  # Ground level
            vx = np.random.uniform(-15, 15)  # m/s
            vy = np.random.uniform(-15, 15)
            yaw = np.arctan2(vy, vx)

            # Vehicle size (typical car)
            width = np.random.uniform(1.8, 2.2)
            length = np.random.uniform(4.0, 5.0)
            height = np.random.uniform(1.4, 1.8)

            track = []
            dt = 0.5  # nuScenes keyframe rate is 2Hz

            for t in range(track_len):
                # Sensor flags based on distance from ego (at origin)
                dist = np.sqrt(x**2 + y**2)
                lidar = 1.0 if dist < 70 and np.random.random() > 0.05 else 0.0
                radar_f = 1.0 if dist < 200 and np.random.random() > 0.1 else 0.0
                radar_s = 1.0 if dist < 100 and np.random.random() > 0.15 else 0.0
                camera = 1.0 if dist < 50 and np.random.random() > 0.1 else 0.0

                state = np.array([
                    x, y, z,
                    vx, vy,
                    width, length, height,
                    yaw,
                    lidar, radar_f, radar_s, camera
                ], dtype=np.float32)

                track.append(state)

                # Update with constant velocity + noise
                x += vx * dt + np.random.normal(0, 0.1)
                y += vy * dt + np.random.normal(0, 0.1)

                # Small velocity changes (acceleration)
                vx += np.random.normal(0, 0.5)
                vy += np.random.normal(0, 0.5)

                # Clip velocities
                vx = np.clip(vx, -20, 20)
                vy = np.clip(vy, -20, 20)

                yaw = np.arctan2(vy, vx)

            tracks.append(np.array(track))

        return tracks

    def _create_samples(self):
        """Create (context, target) pairs from tracks."""
        samples = []

        for track in self.tracks:
            track_len = len(track)

            for start in range(track_len - self.context_len - self.pred_len + 1):
                context = track[start:start + self.context_len]
                target = track[start + self.context_len:start + self.context_len + self.pred_len]
                samples.append((context, target))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# ============================================================================
# PSI Trajectory Predictor
# ============================================================================

class PSISensorFusion(nn.Module):
    """
    PSI-based multi-sensor fusion for trajectory prediction.

    Architecture:
    1. Input projection: sensor state -> latent space
    2. PSI blocks: learn temporal dynamics with phase integration
    3. Output projection: predict future states

    The key insight: PSI's cumsum-based integration should naturally
    learn to integrate noisy sensor observations into coherent trajectories.
    """

    def __init__(self, state_dim=13, dim=256, num_layers=6, pred_len=10):
        super().__init__()

        self.state_dim = state_dim
        self.pred_len = pred_len

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # PSI blocks for temporal dynamics
        self.psi_blocks = nn.ModuleList([
            PSIBlock(dim) for _ in range(num_layers)
        ])

        # Prediction head - output future trajectory
        self.pred_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, pred_len * state_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, context_len, state_dim] - past observations

        Returns:
            pred: [batch, pred_len, state_dim] - predicted future states
        """
        batch_size = x.shape[0]

        # Project to latent space
        h = self.input_proj(x)  # [batch, context_len, dim]

        # Apply PSI blocks
        for block in self.psi_blocks:
            h = block(h)

        # Use last hidden state for prediction
        h_last = h[:, -1, :]  # [batch, dim]

        # Predict future trajectory
        pred = self.pred_head(h_last)  # [batch, pred_len * state_dim]
        pred = pred.view(batch_size, self.pred_len, self.state_dim)

        return pred


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Train the sensor fusion model."""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Loss weights: position matters most, then velocity
    # State: [x,y,z, vx,vy, w,l,h, yaw, sensor_flags...]
    loss_weights = torch.tensor([
        5.0, 5.0, 1.0,    # position (x, y, z)
        2.0, 2.0,          # velocity
        0.5, 0.5, 0.5,     # size (shouldn't change)
        1.0,               # yaw
        0.1, 0.1, 0.1, 0.1 # sensor flags (not predicting these)
    ], device=device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for context, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(context)

            # Weighted MSE loss
            diff = (pred - target) ** 2
            weighted_diff = diff * loss_weights.view(1, 1, -1)
            loss = weighted_diff.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_pos_error = 0.0

        with torch.no_grad():
            for context, target in val_loader:
                context = context.to(device)
                target = target.to(device)

                pred = model(context)

                # Weighted loss
                diff = (pred - target) ** 2
                weighted_diff = diff * loss_weights.view(1, 1, -1)
                loss = weighted_diff.mean()
                val_loss += loss.item()

                # Position error (ADE - Average Displacement Error)
                pos_error = torch.sqrt(((pred[:, :, :2] - target[:, :, :2]) ** 2).sum(dim=-1)).mean()
                val_pos_error += pos_error.item()

        val_loss /= len(val_loader)
        val_pos_error /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val ADE={val_pos_error:.2f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_ade': val_pos_error
            }, 'sensor_fusion_best.pt')

    return best_val_loss


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(model, test_loader, device):
    """
    Compute standard trajectory prediction metrics.

    Metrics:
    - ADE (Average Displacement Error): Mean L2 distance over all timesteps
    - FDE (Final Displacement Error): L2 distance at final timestep
    - Miss Rate: % of predictions > threshold from ground truth
    """
    model.eval()

    all_ade = []
    all_fde = []
    miss_threshold = 2.0  # meters
    misses = 0
    total = 0

    with torch.no_grad():
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)

            pred = model(context)

            # Position only (x, y)
            pred_pos = pred[:, :, :2]
            target_pos = target[:, :, :2]

            # ADE: average over all timesteps
            displacement = torch.sqrt(((pred_pos - target_pos) ** 2).sum(dim=-1))
            ade = displacement.mean(dim=1)  # [batch]
            all_ade.extend(ade.cpu().numpy())

            # FDE: final timestep only
            fde = displacement[:, -1]  # [batch]
            all_fde.extend(fde.cpu().numpy())

            # Miss rate
            misses += (fde > miss_threshold).sum().item()
            total += fde.shape[0]

    metrics = {
        'ADE': np.mean(all_ade),
        'FDE': np.mean(all_fde),
        'MissRate': misses / total * 100
    }

    return metrics


# ============================================================================
# Visualization
# ============================================================================

def visualize_predictions(model, dataset, device, n_samples=5):
    """Visualize predicted vs ground truth trajectories."""

    model.eval()

    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        context, target = dataset[idx]
        context = context.unsqueeze(0).to(device)
        target = target.numpy()

        with torch.no_grad():
            pred = model(context)[0].cpu().numpy()

        # Plot context trajectory
        ctx = dataset.samples[idx][0]
        ax.plot(ctx[:, 0], ctx[:, 1], 'b.-', label='Context', alpha=0.6)
        ax.scatter(ctx[-1, 0], ctx[-1, 1], c='blue', s=100, marker='o', zorder=5)

        # Plot ground truth future
        ax.plot(target[:, 0], target[:, 1], 'g.-', label='Ground Truth', linewidth=2)
        ax.scatter(target[-1, 0], target[-1, 1], c='green', s=100, marker='*', zorder=5)

        # Plot prediction
        ax.plot(pred[:, 0], pred[:, 1], 'r--', label='PSI Prediction', linewidth=2)
        ax.scatter(pred[-1, 0], pred[-1, 1], c='red', s=100, marker='^', zorder=5)

        # Compute error for this sample
        ade = np.sqrt(((pred[:, :2] - target[:, :2])**2).sum(axis=1)).mean()
        fde = np.sqrt(((pred[-1, :2] - target[-1, :2])**2).sum())

        ax.set_title(f'ADE={ade:.2f}m, FDE={fde:.2f}m')
        ax.legend(fontsize=8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensor_fusion_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved sensor_fusion_predictions.png")
    plt.close()


def visualize_error_over_horizon(model, test_loader, device, pred_len):
    """Plot prediction error vs prediction horizon."""

    model.eval()

    errors_by_step = [[] for _ in range(pred_len)]

    with torch.no_grad():
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)

            pred = model(context)

            # Displacement at each timestep
            displacement = torch.sqrt(((pred[:, :, :2] - target[:, :, :2]) ** 2).sum(dim=-1))

            for t in range(pred_len):
                errors_by_step[t].extend(displacement[:, t].cpu().numpy())

    mean_errors = [np.mean(errs) for errs in errors_by_step]
    std_errors = [np.std(errs) for errs in errors_by_step]

    plt.figure(figsize=(10, 5))
    timesteps = np.arange(1, pred_len + 1) * 0.5  # Convert to seconds (2Hz)

    plt.plot(timesteps, mean_errors, 'b-', linewidth=2, label='Mean Error')
    plt.fill_between(timesteps,
                     np.array(mean_errors) - np.array(std_errors),
                     np.array(mean_errors) + np.array(std_errors),
                     alpha=0.3)

    plt.xlabel('Prediction Horizon (seconds)')
    plt.ylabel('Displacement Error (m)')
    plt.title('PSI Sensor Fusion: Error vs Prediction Horizon')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('sensor_fusion_error_horizon.png', dpi=150, bbox_inches='tight')
    print("Saved sensor_fusion_error_horizon.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Sensor Fusion Experiment with PSI')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of nuScenes')
    parser.add_argument('--n_tracks', type=int, default=2000,
                        help='Number of synthetic tracks to generate')
    parser.add_argument('--context_len', type=int, default=10,
                        help='Context length (past timesteps)')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='Prediction length (future timesteps)')
    parser.add_argument('--dim', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of PSI layers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()

    print("=" * 80)
    print("PSI Trajectory Forecasting Experiment (nuScenes)")
    print("=" * 80)
    print("NOTE: This is trajectory forecasting on already-tracked objects,")
    print("      NOT raw sensor fusion. Objects are pre-detected by nuScenes.")
    print()

    # Load data
    if args.synthetic:
        print("Using synthetic multi-sensor tracking data")
        dataset = SyntheticMultiSensorDataset(
            n_tracks=args.n_tracks,
            track_len=50,
            context_len=args.context_len,
            pred_len=args.pred_len
        )
    else:
        # Check for nuScenes
        available, msg = check_nuscenes_available(args.dataroot)

        if not available:
            print(f"nuScenes not found: {msg}")
            print()
            print("To download nuScenes mini:")
            print("1. Go to: https://www.kaggle.com/datasets/aadimator/nuscenes-mini")
            print("2. Download and extract to: data/nuscenes/")
            print()
            print("Or run with --synthetic to use synthetic data")
            print()
            print("Falling back to synthetic data...")
            print()

            dataset = SyntheticMultiSensorDataset(
                n_tracks=args.n_tracks,
                track_len=50,
                context_len=args.context_len,
                pred_len=args.pred_len
            )
        else:
            print(f"Loading nuScenes: {msg}")

            # Get number of tracks to split at TRACK level (not sample level)
            # This prevents data leakage from overlapping windows of the same track
            n_tracks = NuScenesTrackingDataset.get_num_tracks(args.dataroot)

            # Split tracks (not samples!) into train/val/test
            np.random.seed(42)
            track_indices = np.random.permutation(n_tracks)

            n_train_tracks = int(0.7 * n_tracks)
            n_val_tracks = int(0.15 * n_tracks)

            train_track_idx = track_indices[:n_train_tracks].tolist()
            val_track_idx = track_indices[n_train_tracks:n_train_tracks + n_val_tracks].tolist()
            test_track_idx = track_indices[n_train_tracks + n_val_tracks:].tolist()

            print(f"\nTrack-level split: Train={len(train_track_idx)}, Val={len(val_track_idx)}, Test={len(test_track_idx)} tracks")
            print("(No data leakage: train/val/test contain different vehicles)")

            # Create separate datasets for each split (uses cached track extraction)
            train_dataset = NuScenesTrackingDataset(
                dataroot=args.dataroot,
                context_len=args.context_len,
                pred_len=args.pred_len,
                track_indices=train_track_idx
            )
            val_dataset = NuScenesTrackingDataset(
                dataroot=args.dataroot,
                context_len=args.context_len,
                pred_len=args.pred_len,
                track_indices=val_track_idx
            )
            test_dataset = NuScenesTrackingDataset(
                dataroot=args.dataroot,
                context_len=args.context_len,
                pred_len=args.pred_len,
                track_indices=test_track_idx
            )

            # For visualization later
            dataset = test_dataset

    if args.synthetic:
        # For synthetic data, sample-level split is fine (tracks are independent)
        n_samples = len(dataset)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        n_test = n_samples - n_train - n_val

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

    print(f"Sample counts: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    state_dim = 13  # x,y,z, vx,vy, w,l,h, yaw, 4 sensor flags
    model = PSISensorFusion(
        state_dim=state_dim,
        dim=args.dim,
        num_layers=args.num_layers,
        pred_len=args.pred_len
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"State dimension: {state_dim}")
    print(f"Context: {args.context_len} steps ({args.context_len * 0.5:.1f}s)")
    print(f"Prediction: {args.pred_len} steps ({args.pred_len * 0.5:.1f}s)")
    print()

    # Train
    print("=" * 80)
    print("Training")
    print("=" * 80)

    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Load best model
    checkpoint = torch.load('sensor_fusion_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    metrics = compute_metrics(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  ADE (Average Displacement Error): {metrics['ADE']:.3f} m")
    print(f"  FDE (Final Displacement Error): {metrics['FDE']:.3f} m")
    print(f"  Miss Rate (>{2.0}m): {metrics['MissRate']:.1f}%")

    # Context for these numbers
    print("\nContext (typical values from literature):")
    print("  - Social-LSTM on ETH/UCY: ADE ~0.7m, FDE ~1.5m")
    print("  - Trajectron++ on nuScenes: ADE ~1.5m, FDE ~3.5m (but 6s horizon)")
    print(f"  - Our horizon: {args.pred_len * 0.5:.1f}s")

    # Visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    visualize_predictions(model, dataset, device, n_samples=5)
    visualize_error_over_horizon(model, test_loader, device, args.pred_len)

    print("\nDone!")


if __name__ == "__main__":
    main()
