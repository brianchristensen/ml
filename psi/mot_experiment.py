"""
Multi-Object Tracking with PSI - Objects as N-Bodies

Key insight: Tracked objects are like gravitational bodies!
- Each object has state: [x, y, z, vx, vy, vz, w, l, h]
- Objects move according to dynamics (physics + driver intent)
- PSI learns to predict how objects evolve over time

This is the same approach as n-body: learn dynamics from sequences of states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Data Loading - Extract object trajectories from nuScenes
# ============================================================================

def load_nuscenes_trajectories(dataroot: str = 'data/nuscenes', version: str = 'v1.0-mini'):
    """
    Load object trajectories from nuScenes annotations.

    Each trajectory is a sequence of object states over time.
    State: [x, y, z, vx, vy, vz, w, l, h] (position, velocity, size)

    Returns:
        trajectories: dict mapping instance_token -> list of (timestamp, state)
        scene_trajectories: dict mapping scene_token -> list of trajectories in that scene
    """
    print("Loading nuScenes annotations...")
    version_path = Path(dataroot) / version

    with open(version_path / 'scene.json') as f:
        scenes = {s['token']: s for s in json.load(f)}

    with open(version_path / 'sample.json') as f:
        samples = {s['token']: s for s in json.load(f)}

    with open(version_path / 'sample_annotation.json') as f:
        annotations = json.load(f)

    with open(version_path / 'instance.json') as f:
        instances = {i['token']: i for i in json.load(f)}

    with open(version_path / 'category.json') as f:
        categories = {c['token']: c for c in json.load(f)}

    # Group annotations by instance to build trajectories
    instance_annotations = defaultdict(list)
    for ann in annotations:
        instance_annotations[ann['instance_token']].append(ann)

    # Build trajectories for each instance
    trajectories = {}
    instance_to_scene = {}

    # Track which categories we want (vehicles, pedestrians, cyclists)
    valid_categories = {'vehicle', 'human', 'pedestrian', 'bicycle', 'motorcycle'}

    for inst_token, anns in instance_annotations.items():
        instance = instances[inst_token]
        category = categories[instance['category_token']]

        # Filter by category
        cat_name = category['name'].lower()
        if not any(vc in cat_name for vc in valid_categories):
            continue

        # Sort by timestamp
        ann_with_time = []
        for ann in anns:
            sample = samples[ann['sample_token']]
            ann_with_time.append((sample['timestamp'], ann, sample['scene_token']))
        ann_with_time.sort(key=lambda x: x[0])

        if len(ann_with_time) < 2:
            continue

        # Build trajectory with velocity estimation
        traj = []
        scene_token = ann_with_time[0][2]

        for i, (timestamp, ann, _) in enumerate(ann_with_time):
            x, y, z = ann['translation']
            w, l, h = ann['size']

            # Estimate velocity from finite differences
            if i > 0:
                prev_time, prev_ann, _ = ann_with_time[i-1]
                dt = (timestamp - prev_time) / 1e6  # Convert microseconds to seconds
                if dt > 0:
                    px, py, pz = prev_ann['translation']
                    vx = (x - px) / dt
                    vy = (y - py) / dt
                    vz = (z - pz) / dt
                else:
                    vx, vy, vz = 0, 0, 0
            else:
                vx, vy, vz = 0, 0, 0

            state = np.array([x, y, z, vx, vy, vz, w, l, h], dtype=np.float32)
            traj.append((timestamp, state))

        trajectories[inst_token] = traj
        instance_to_scene[inst_token] = scene_token

    # Group by scene
    scene_trajectories = defaultdict(list)
    for inst_token, traj in trajectories.items():
        scene_token = instance_to_scene[inst_token]
        scene_trajectories[scene_token].append(traj)

    print(f"Loaded {len(trajectories)} object trajectories across {len(scene_trajectories)} scenes")

    # Stats
    traj_lens = [len(t) for t in trajectories.values()]
    print(f"Trajectory lengths: min={min(traj_lens)}, max={max(traj_lens)}, mean={np.mean(traj_lens):.1f}")

    return trajectories, scene_trajectories


def create_multi_object_sequences(scene_trajectories: dict, max_objects: int = 20,
                                   context_len: int = 10, min_seq_len: int = 15):
    """
    Create sequences where each timestep has multiple objects.
    Like n-body but with variable number of objects.

    IMPORTANT: Uses sequence-relative coordinates!
    All positions are relative to the centroid of the first frame.
    This way the model learns motion patterns, not absolute positions.

    Returns list of sequences, each sequence is:
        [timesteps, max_objects, state_dim] with padding for missing objects
    """
    sequences = []
    state_dim = 9  # x, y, z, vx, vy, vz, w, l, h

    for scene_token, trajs in scene_trajectories.items():
        if len(trajs) == 0:
            continue

        # Get all unique timestamps in this scene
        all_timestamps = set()
        for traj in trajs:
            for ts, _ in traj:
                all_timestamps.add(ts)
        timestamps = sorted(all_timestamps)

        if len(timestamps) < min_seq_len:
            continue

        # Build frame-aligned data: for each timestamp, collect all visible objects
        frames = []
        for ts in timestamps:
            frame_objects = []
            for traj in trajs:
                # Find if this object is visible at this timestamp
                for t, state in traj:
                    if t == ts:
                        frame_objects.append(state.copy())
                        break
            frames.append(frame_objects)

        # Create sliding windows
        for start in range(len(frames) - context_len):
            window = frames[start:start + context_len + 1]  # +1 for target

            # Compute reference point: centroid of first frame's objects
            first_frame_objs = window[0]
            if len(first_frame_objs) == 0:
                continue

            ref_x = np.mean([obj[0] for obj in first_frame_objs])
            ref_y = np.mean([obj[1] for obj in first_frame_objs])
            ref_z = np.mean([obj[2] for obj in first_frame_objs])

            # Convert to padded array with relative coordinates
            seq = np.zeros((context_len + 1, max_objects, state_dim), dtype=np.float32)
            mask = np.zeros((context_len + 1, max_objects), dtype=np.float32)

            for t, frame_objs in enumerate(window):
                for i, obj_state in enumerate(frame_objs[:max_objects]):
                    # Make position relative to reference
                    rel_state = obj_state.copy()
                    rel_state[0] -= ref_x  # x
                    rel_state[1] -= ref_y  # y
                    rel_state[2] -= ref_z  # z
                    # Velocities stay the same (already relative)

                    seq[t, i] = rel_state
                    mask[t, i] = 1.0

            # Only include if we have at least some objects
            if mask.sum() > context_len:  # At least 1 object per timestep on average
                sequences.append({
                    'sequence': seq,
                    'mask': mask
                })

    print(f"Created {len(sequences)} multi-object sequences")
    return sequences


def augment_sequences(sequences: list, num_augments: int = 10) -> list:
    """
    Augment sequences by rotation, flipping, and noise.
    This multiplies our dataset size significantly.
    """
    augmented = []

    for seq_data in sequences:
        seq = seq_data['sequence']  # [T, N, F]
        mask = seq_data['mask']  # [T, N]

        # Original
        augmented.append(seq_data)

        for _ in range(num_augments):
            aug_seq = seq.copy()

            # Random rotation around Z axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Rotate positions (x, y) and velocities (vx, vy)
            for t in range(aug_seq.shape[0]):
                for i in range(aug_seq.shape[1]):
                    if mask[t, i] > 0:
                        # Position
                        x, y = aug_seq[t, i, 0], aug_seq[t, i, 1]
                        aug_seq[t, i, 0] = cos_t * x - sin_t * y
                        aug_seq[t, i, 1] = sin_t * x + cos_t * y

                        # Velocity
                        vx, vy = aug_seq[t, i, 3], aug_seq[t, i, 4]
                        aug_seq[t, i, 3] = cos_t * vx - sin_t * vy
                        aug_seq[t, i, 4] = sin_t * vx + cos_t * vy

            # Random flip with 50% probability
            if np.random.rand() > 0.5:
                aug_seq[:, :, 1] *= -1  # Flip Y
                aug_seq[:, :, 4] *= -1  # Flip vy

            # Small position noise
            noise_scale = 0.1  # 10cm noise
            for t in range(aug_seq.shape[0]):
                for i in range(aug_seq.shape[1]):
                    if mask[t, i] > 0:
                        aug_seq[t, i, 0] += np.random.randn() * noise_scale
                        aug_seq[t, i, 1] += np.random.randn() * noise_scale
                        aug_seq[t, i, 2] += np.random.randn() * noise_scale * 0.1

            augmented.append({
                'sequence': aug_seq,
                'mask': mask.copy()
            })

    print(f"Augmented {len(sequences)} -> {len(augmented)} sequences ({num_augments}x + original)")
    return augmented


# ============================================================================
# Dataset
# ============================================================================

class MOTDataset(Dataset):
    """Multi-object tracking dataset - like n-body but with cars/pedestrians."""

    def __init__(self, sequences: list, split: str = 'train', train_frac: float = 0.7,
                 val_frac: float = 0.15):
        # Split sequences
        np.random.seed(42)
        indices = np.random.permutation(len(sequences))

        n_train = int(len(sequences) * train_frac)
        n_val = int(len(sequences) * val_frac)

        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]

        self.sequences = [sequences[i] for i in indices]

        # Compute normalization stats from training data only
        if split == 'train':
            all_states = []
            for seq in self.sequences:
                mask = seq['mask']
                states = seq['sequence']
                for t in range(states.shape[0]):
                    for i in range(states.shape[1]):
                        if mask[t, i] > 0:
                            all_states.append(states[t, i])

            all_states = np.array(all_states)
            self.mean = all_states.mean(axis=0)
            self.std = all_states.std(axis=0)
            # Prevent division by zero for constant features
            self.std = np.maximum(self.std, 1e-6)
        else:
            self.mean = None
            self.std = None

        print(f"{split.upper()}: {len(self.sequences)} sequences")

    def set_normalization(self, mean, std):
        """Set normalization stats (for val/test sets)."""
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        seq = seq_data['sequence'].copy()  # [T+1, N, F]
        mask = seq_data['mask'].copy()  # [T+1, N]

        # Normalize
        if self.mean is not None:
            seq = (seq - self.mean) / self.std

        # Context: all but last timestep
        # Target: last timestep
        context = seq[:-1]  # [T, N, F]
        context_mask = mask[:-1]  # [T, N]
        target = seq[-1]  # [N, F]
        target_mask = mask[-1]  # [N]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'context_mask': torch.tensor(context_mask, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'target_mask': torch.tensor(target_mask, dtype=torch.float32)
        }


# ============================================================================
# Model - Same architecture as n-body
# ============================================================================

class PSIDynamicsPredictor(nn.Module):
    """PSI model for multi-object dynamics - same as n-body."""

    def __init__(self, state_dim: int, max_objects: int = 20, dim: int = 256,
                 num_layers: int = 8, max_len: int = 50):
        super().__init__()
        self.state_dim = state_dim
        self.max_objects = max_objects
        self.dim = dim

        # Flatten all objects into one state vector (like n-body)
        self.input_dim = max_objects * state_dim

        self.state_embedding = nn.Sequential(
            nn.Linear(self.input_dim, dim),
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
            nn.Linear(dim, self.input_dim)
        )

    def _create_sinusoidal_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            context: [batch, T, N, F] - sequence of multi-object states
            context_mask: [batch, T, N] - which objects are valid (optional)

        Returns:
            prediction: [batch, N, F] - predicted next state for all objects
        """
        batch, T, N, F = context.shape

        # Flatten objects into single state vector per timestep
        x = context.view(batch, T, N * F)  # [batch, T, N*F]

        # Embed
        x = self.state_embedding(x)  # [batch, T, dim]

        # Add positional encoding
        x = x + self.pos_encoding[:T].unsqueeze(0)

        # PSI blocks
        for block in self.blocks:
            x = block(x)

        # Predict from last timestep
        x = x[:, -1]  # [batch, dim]
        pred = self.output_head(x)  # [batch, N*F]

        # Reshape to [batch, N, F]
        pred = pred.view(batch, N, F)

        return pred


# ============================================================================
# Training
# ============================================================================

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MSE loss only on valid (non-padded) objects."""
    # pred, target: [batch, N, F]
    # mask: [batch, N]

    mask = mask.unsqueeze(-1)  # [batch, N, 1]

    # Only compute loss on position (x, y, z) - indices 0, 1, 2
    pred_pos = pred[:, :, :3]
    target_pos = target[:, :, :3]

    squared_error = (pred_pos - target_pos) ** 2
    masked_error = squared_error * mask

    # Mean over valid elements
    loss = masked_error.sum() / (mask.sum() * 3 + 1e-8)

    return loss


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        context = batch['context'].to(device)
        context_mask = batch['context_mask'].to(device)
        target = batch['target'].to(device)
        target_mask = batch['target_mask'].to(device)

        pred = model(context, context_mask)
        loss = masked_mse_loss(pred, target, target_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            context_mask = batch['context_mask'].to(device)
            target = batch['target'].to(device)
            target_mask = batch['target_mask'].to(device)

            pred = model(context, context_mask)
            loss = masked_mse_loss(pred, target, target_mask)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def compute_metrics(model, dataloader, mean, std, device):
    """Compute position error metrics."""
    model.eval()

    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            target_mask = batch['target_mask'].to(device)

            pred = model(context)

            # Denormalize
            pred_denorm = pred.cpu().numpy() * std + mean
            target_denorm = target.cpu().numpy() * std + mean
            mask = target_mask.cpu().numpy()

            # Compute position errors
            for b in range(pred_denorm.shape[0]):
                for i in range(pred_denorm.shape[1]):
                    if mask[b, i] > 0:
                        pred_pos = pred_denorm[b, i, :3]
                        true_pos = target_denorm[b, i, :3]
                        error = np.linalg.norm(pred_pos - true_pos)
                        all_errors.append(error)

    all_errors = np.array(all_errors)

    return {
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'p90_error': np.percentile(all_errors, 90),
        'all_errors': all_errors
    }


def visualize_predictions(model, dataloader, mean, std, device, n_samples=3):
    """Visualize predicted vs actual object trajectories."""
    model.eval()

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_idx >= n_samples:
                break

            context = batch['context'].to(device)
            context_mask = batch['context_mask']
            target = batch['target']
            target_mask = batch['target_mask']

            pred = model(context)

            # Take first sample in batch
            b = 0

            ctx = context[b].cpu().numpy() * std + mean  # [T, N, F]
            tgt = target[b].cpu().numpy() * std + mean  # [N, F]
            prd = pred[b].cpu().numpy() * std + mean  # [N, F]
            ctx_m = context_mask[b].cpu().numpy()  # [T, N]
            tgt_m = target_mask[b].cpu().numpy()  # [N]

            T, N, F = ctx.shape

            # Left plot: X-Y trajectories
            ax1 = axes[sample_idx, 0]
            colors = plt.cm.tab10(np.linspace(0, 1, N))

            for i in range(N):
                # Get trajectory for this object
                traj_x, traj_y = [], []
                for t in range(T):
                    if ctx_m[t, i] > 0:
                        traj_x.append(ctx[t, i, 0])
                        traj_y.append(ctx[t, i, 1])

                if len(traj_x) > 1:
                    # Plot context trajectory
                    ax1.plot(traj_x, traj_y, 'o-', color=colors[i], alpha=0.6,
                             markersize=4, label=f'Obj {i}' if i < 5 else None)

                    # Plot target (ground truth next position)
                    if tgt_m[i] > 0:
                        ax1.scatter(tgt[i, 0], tgt[i, 1], c=[colors[i]], s=100,
                                   marker='s', edgecolor='black', zorder=5)

                    # Plot prediction
                    if tgt_m[i] > 0:
                        ax1.scatter(prd[i, 0], prd[i, 1], c=[colors[i]], s=100,
                                   marker='x', linewidths=3, zorder=5)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'Sample {sample_idx+1}: Trajectories (o), Target (□), Pred (×)')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Right plot: Error per object
            ax2 = axes[sample_idx, 1]
            errors = []
            obj_ids = []

            for i in range(N):
                if tgt_m[i] > 0:
                    error = np.linalg.norm(prd[i, :3] - tgt[i, :3])
                    errors.append(error)
                    obj_ids.append(i)

            if errors:
                ax2.bar(range(len(errors)), errors, color=[colors[i] for i in obj_ids])
                ax2.set_xlabel('Object')
                ax2.set_ylabel('Position Error (m)')
                ax2.set_title(f'Sample {sample_idx+1}: Prediction Error (Mean={np.mean(errors):.2f}m)')
                ax2.set_xticks(range(len(errors)))
                ax2.set_xticklabels([f'{i}' for i in obj_ids])
                ax2.grid(True, alpha=0.3)

            sample_idx += 1

    plt.tight_layout()
    plt.savefig('mot_predictions.png', dpi=150)
    print("Saved mot_predictions.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Multi-Object Tracking with PSI - Objects as N-Bodies")
    print("=" * 80)
    print()
    print("Approach: Treat tracked objects like gravitational bodies.")
    print("Each object has state [x, y, z, vx, vy, vz, w, l, h].")
    print("PSI learns to predict how objects evolve - same as n-body!")
    print()

    # Config
    max_objects = 20
    context_len = 10
    state_dim = 9  # x, y, z, vx, vy, vz, w, l, h
    dim = 256
    num_layers = 8
    epochs = 50
    batch_size = 32
    lr = 1e-4

    # Load data
    trajectories, scene_trajectories = load_nuscenes_trajectories()

    # Create sequences
    sequences = create_multi_object_sequences(
        scene_trajectories,
        max_objects=max_objects,
        context_len=context_len
    )

    # Augment training data (10x more sequences through rotation/flip/noise)
    sequences = augment_sequences(sequences, num_augments=10)

    # Create datasets
    train_dataset = MOTDataset(sequences, split='train')
    val_dataset = MOTDataset(sequences, split='val')
    test_dataset = MOTDataset(sequences, split='test')

    # Share normalization stats
    val_dataset.set_normalization(train_dataset.mean, train_dataset.std)
    test_dataset.set_normalization(train_dataset.mean, train_dataset.std)

    print(f"\nNormalization - Mean: {train_dataset.mean}")
    print(f"Normalization - Std: {train_dataset.std}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    # Create model
    print("\n" + "=" * 80)
    print("Training PSI Dynamics Model")
    print("=" * 80)

    model = PSIDynamicsPredictor(
        state_dim=state_dim,
        max_objects=max_objects,
        dim=dim,
        num_layers=num_layers,
        max_len=context_len + 1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Input dim: {max_objects} objects × {state_dim} features = {max_objects * state_dim}")

    # Train
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} ({time.time()-start:.1f}s) - "
              f"Train: {train_loss:.6f} - Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mot_dynamics_model.pt')
            print(f"  Saved best model")

    # Load best model
    model.load_state_dict(torch.load('mot_dynamics_model.pt', map_location=device, weights_only=True))

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    metrics = compute_metrics(model, test_loader, train_dataset.mean, train_dataset.std, device)

    print(f"\nPSI Multi-Object Tracking Results:")
    print(f"  Mean Position Error: {metrics['mean_error']:.4f} m")
    print(f"  Median Position Error: {metrics['median_error']:.4f} m")
    print(f"  90th Percentile Error: {metrics['p90_error']:.4f} m")

    # Baseline 1: Copy last position
    print("\nBaseline 1 (copy last position):")
    baseline_copy_errors = []

    for batch in test_loader:
        context = batch['context']  # [1, T, N, F]
        target = batch['target']  # [1, N, F]
        target_mask = batch['target_mask']  # [1, N]

        last_ctx = context[0, -1]  # [N, F]

        last_ctx_denorm = last_ctx.numpy() * train_dataset.std + train_dataset.mean
        target_denorm = target[0].numpy() * train_dataset.std + train_dataset.mean
        mask = target_mask[0].numpy()

        for i in range(last_ctx.shape[0]):
            if mask[i] > 0:
                error = np.linalg.norm(last_ctx_denorm[i, :3] - target_denorm[i, :3])
                baseline_copy_errors.append(error)

    baseline_copy_errors = np.array(baseline_copy_errors)
    print(f"  Mean Position Error: {np.mean(baseline_copy_errors):.4f} m")
    print(f"  Median Position Error: {np.median(baseline_copy_errors):.4f} m")

    # Baseline 2: Constant velocity (pos + vel * dt)
    # dt is ~0.5 seconds between keyframes in nuScenes
    print("\nBaseline 2 (constant velocity):")
    baseline_cv_errors = []
    dt = 0.5  # nuScenes keyframe interval

    for batch in test_loader:
        context = batch['context']  # [1, T, N, F]
        target = batch['target']  # [1, N, F]
        target_mask = batch['target_mask']  # [1, N]

        last_ctx = context[0, -1]  # [N, F]

        last_ctx_denorm = last_ctx.numpy() * train_dataset.std + train_dataset.mean
        target_denorm = target[0].numpy() * train_dataset.std + train_dataset.mean
        mask = target_mask[0].numpy()

        for i in range(last_ctx.shape[0]):
            if mask[i] > 0:
                # Predict: pos + vel * dt
                # State: [x, y, z, vx, vy, vz, w, l, h]
                pred_x = last_ctx_denorm[i, 0] + last_ctx_denorm[i, 3] * dt
                pred_y = last_ctx_denorm[i, 1] + last_ctx_denorm[i, 4] * dt
                pred_z = last_ctx_denorm[i, 2] + last_ctx_denorm[i, 5] * dt

                true_pos = target_denorm[i, :3]
                pred_pos = np.array([pred_x, pred_y, pred_z])
                error = np.linalg.norm(pred_pos - true_pos)
                baseline_cv_errors.append(error)

    baseline_cv_errors = np.array(baseline_cv_errors)
    print(f"  Mean Position Error: {np.mean(baseline_cv_errors):.4f} m")
    print(f"  Median Position Error: {np.median(baseline_cv_errors):.4f} m")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"PSI Mean Error:              {metrics['mean_error']:.4f} m")
    print(f"Copy Baseline Mean Error:    {np.mean(baseline_copy_errors):.4f} m")
    print(f"Const Vel Baseline Mean Error: {np.mean(baseline_cv_errors):.4f} m")

    improvement_copy = (np.mean(baseline_copy_errors) - metrics['mean_error']) / np.mean(baseline_copy_errors) * 100
    improvement_cv = (np.mean(baseline_cv_errors) - metrics['mean_error']) / np.mean(baseline_cv_errors) * 100
    print(f"\nPSI improvement over copy baseline: {improvement_copy:.1f}%")
    print(f"PSI improvement over const vel baseline: {improvement_cv:.1f}%")

    # Visualize
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    visualize_predictions(model, test_loader, train_dataset.mean, train_dataset.std, device, n_samples=3)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(metrics['all_errors'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(metrics['mean_error'], color='r', linestyle='--', label=f"Mean={metrics['mean_error']:.2f}m")
    plt.axvline(np.mean(baseline_errors), color='g', linestyle='--', label=f"Baseline={np.mean(baseline_errors):.2f}m")
    plt.xlabel('Position Error (m)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mot_training.png', dpi=150)
    print("Saved mot_training.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
