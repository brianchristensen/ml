"""
True Sensor Fusion Experiment for PSI

This tests PSI's ability to fuse noisy, asynchronous, unreliable sensor data
into coherent state estimates - the actual sensor fusion problem.

Scenario: Track a moving object using multiple simulated sensors:
- Sensor A (e.g., Radar): Long range, noisy position, provides velocity
- Sensor B (e.g., Lidar): Medium range, accurate position, no velocity
- Sensor C (e.g., Camera): Short range, 2D only (no depth), high noise

Each sensor has:
- Different noise characteristics
- Different update rates (asynchronous)
- Dropout probability (missed detections)
- False positive rate (ghost detections)
- Different fields of view / range limits

The task: Given a stream of noisy, asynchronous sensor observations,
estimate the true object state (position, velocity).

Baseline: Extended Kalman Filter (the standard solution)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple
import argparse
from tqdm import tqdm

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Sensor Models
# ============================================================================

@dataclass
class SensorConfig:
    """Configuration for a simulated sensor."""
    name: str
    update_rate: float      # Hz (updates per second)
    position_noise: float   # meters (std dev)
    velocity_noise: float   # m/s (std dev), 0 if sensor doesn't measure velocity
    dropout_prob: float     # probability of missed detection
    false_positive_prob: float  # probability of ghost detection per timestep
    max_range: float        # meters
    has_velocity: bool      # whether sensor provides velocity
    has_depth: bool         # whether sensor provides depth (z)


# Realistic sensor configurations
RADAR = SensorConfig(
    name="radar",
    update_rate=10.0,       # 10 Hz
    position_noise=1.5,     # Radar is noisy in position
    velocity_noise=0.3,     # But good at velocity (Doppler)
    dropout_prob=0.1,
    false_positive_prob=0.05,
    max_range=200.0,
    has_velocity=True,
    has_depth=True
)

LIDAR = SensorConfig(
    name="lidar",
    update_rate=10.0,       # 10 Hz
    position_noise=0.1,     # Lidar is very accurate
    velocity_noise=0.0,     # No direct velocity
    dropout_prob=0.05,
    false_positive_prob=0.02,
    max_range=70.0,
    has_velocity=False,
    has_depth=True
)

CAMERA = SensorConfig(
    name="camera",
    update_rate=30.0,       # 30 Hz (higher rate)
    position_noise=2.0,     # Camera depth estimation is poor
    velocity_noise=0.0,     # No velocity
    dropout_prob=0.15,      # Occlusion, lighting issues
    false_positive_prob=0.1,
    max_range=50.0,
    has_velocity=False,
    has_depth=False         # Only 2D (bearing)
)


# ============================================================================
# Ground Truth Trajectory Generation
# ============================================================================

def generate_trajectory(duration: float, dt: float = 0.01) -> np.ndarray:
    """
    Generate a realistic ground truth trajectory.

    Returns array of shape [n_timesteps, 6]: x, y, z, vx, vy, vz
    """
    n_steps = int(duration / dt)

    # Initial state
    x, y, z = np.random.uniform(-20, 20), np.random.uniform(-20, 20), 0.0
    vx, vy, vz = np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0.0

    trajectory = []

    # Motion model: constant velocity with random accelerations
    for t in range(n_steps):
        # Random acceleration (simulates turns, speed changes)
        ax = np.random.normal(0, 0.5)
        ay = np.random.normal(0, 0.5)
        az = 0  # Keep on ground plane

        # Update velocity
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # Clip velocity
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 20:  # Max 20 m/s
            vx *= 20 / speed
            vy *= 20 / speed

        # Update position
        x += vx * dt
        y += vy * dt
        z += vz * dt

        trajectory.append([x, y, z, vx, vy, vz])

    return np.array(trajectory)


# ============================================================================
# Sensor Observation Generation
# ============================================================================

@dataclass
class SensorObservation:
    """A single sensor observation."""
    timestamp: float
    sensor_id: int          # Which sensor (0, 1, 2, ...)
    position: np.ndarray    # Observed position [x, y, z] (may be partial)
    velocity: Optional[np.ndarray]  # Observed velocity [vx, vy, vz] if available
    is_valid: bool          # False if dropout
    is_false_positive: bool # True if ghost detection
    confidence: float       # Sensor confidence (for weighting)


def generate_sensor_observations(
    trajectory: np.ndarray,
    sensors: List[SensorConfig],
    dt: float = 0.01,
    ego_position: np.ndarray = np.array([0, 0, 0])
) -> List[SensorObservation]:
    """
    Generate noisy sensor observations from ground truth trajectory.

    Returns list of observations sorted by timestamp.
    """
    observations = []
    n_steps = len(trajectory)
    duration = n_steps * dt

    for sensor_id, sensor in enumerate(sensors):
        # Determine observation times for this sensor
        sensor_dt = 1.0 / sensor.update_rate
        obs_times = np.arange(0, duration, sensor_dt)

        for t in obs_times:
            step_idx = min(int(t / dt), n_steps - 1)
            true_state = trajectory[step_idx]
            true_pos = true_state[:3]
            true_vel = true_state[3:6]

            # Check range
            dist = np.linalg.norm(true_pos - ego_position)
            if dist > sensor.max_range:
                continue  # Out of range

            # Check dropout
            if np.random.random() < sensor.dropout_prob:
                observations.append(SensorObservation(
                    timestamp=t,
                    sensor_id=sensor_id,
                    position=np.zeros(3),
                    velocity=None,
                    is_valid=False,
                    is_false_positive=False,
                    confidence=0.0
                ))
                continue

            # Generate noisy observation
            noisy_pos = true_pos.copy()
            noisy_pos[:2] += np.random.normal(0, sensor.position_noise, 2)
            if sensor.has_depth:
                noisy_pos[2] += np.random.normal(0, sensor.position_noise * 0.5)
            else:
                noisy_pos[2] = 0  # No depth info

            noisy_vel = None
            if sensor.has_velocity:
                noisy_vel = true_vel.copy()
                noisy_vel += np.random.normal(0, sensor.velocity_noise, 3)

            # Confidence based on range and noise
            confidence = 1.0 - (dist / sensor.max_range) * 0.5
            confidence *= (1.0 / (1.0 + sensor.position_noise))

            observations.append(SensorObservation(
                timestamp=t,
                sensor_id=sensor_id,
                position=noisy_pos,
                velocity=noisy_vel,
                is_valid=True,
                is_false_positive=False,
                confidence=confidence
            ))

        # Add false positives
        n_false_positives = np.random.poisson(sensor.false_positive_prob * duration * sensor.update_rate)
        for _ in range(n_false_positives):
            t = np.random.uniform(0, duration)
            fake_pos = np.random.uniform(-50, 50, 3)
            fake_pos[2] = 0  # Ground plane

            observations.append(SensorObservation(
                timestamp=t,
                sensor_id=sensor_id,
                position=fake_pos,
                velocity=np.random.uniform(-5, 5, 3) if sensor.has_velocity else None,
                is_valid=True,
                is_false_positive=True,
                confidence=np.random.uniform(0.1, 0.5)  # Low confidence
            ))

    # Sort by timestamp
    observations.sort(key=lambda x: x.timestamp)
    return observations


# ============================================================================
# Dataset
# ============================================================================

class SensorFusionDataset(Dataset):
    """
    Dataset for sensor fusion task.

    Each sample is a sequence of sensor observations and the corresponding
    ground truth states at those timestamps.

    IMPORTANT: Split at trajectory level to avoid data leakage.
    """

    def __init__(
        self,
        n_trajectories: int = 1000,
        duration: float = 10.0,
        sensors: List[SensorConfig] = None,
        seq_len: int = 50,
        dt: float = 0.01,
        split: str = 'all',  # 'train', 'val', 'test', or 'all'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        if sensors is None:
            sensors = [RADAR, LIDAR, CAMERA]

        self.sensors = sensors
        self.n_sensors = len(sensors)
        self.seq_len = seq_len
        self.dt = dt
        self.split = split

        # Determine which trajectories belong to this split
        np.random.seed(seed)
        traj_indices = np.random.permutation(n_trajectories)

        n_train = int(train_ratio * n_trajectories)
        n_val = int(val_ratio * n_trajectories)

        if split == 'train':
            my_indices = set(traj_indices[:n_train])
            desc = f"train ({n_train} trajectories)"
        elif split == 'val':
            my_indices = set(traj_indices[n_train:n_train+n_val])
            desc = f"val ({n_val} trajectories)"
        elif split == 'test':
            my_indices = set(traj_indices[n_train+n_val:])
            desc = f"test ({n_trajectories - n_train - n_val} trajectories)"
        else:
            my_indices = set(range(n_trajectories))
            desc = f"all ({n_trajectories} trajectories)"

        print(f"Generating {desc} with {len(sensors)} sensors...")
        self.samples = self._generate_samples(n_trajectories, duration, my_indices)
        print(f"Created {len(self.samples)} samples for {split}")

    def _generate_samples(self, n_trajectories, duration, my_indices):
        samples = []

        for traj_id in tqdm(range(n_trajectories), desc="Generating data"):
            if traj_id not in my_indices:
                continue

            # Generate ground truth
            trajectory = generate_trajectory(duration, self.dt)

            # Generate sensor observations
            observations = generate_sensor_observations(
                trajectory, self.sensors, self.dt
            )

            # Filter to valid observations only for training
            valid_obs = [o for o in observations if o.is_valid and not o.is_false_positive]

            if len(valid_obs) < self.seq_len:
                continue

            # Create sliding window samples
            for start in range(0, len(valid_obs) - self.seq_len, self.seq_len // 2):
                obs_window = valid_obs[start:start + self.seq_len]

                # Build input tensor: [seq_len, input_dim]
                # Input: sensor_id (one-hot), position, velocity (if any), confidence, dt
                input_dim = self.n_sensors + 3 + 3 + 1 + 1  # one-hot + pos + vel + conf + dt
                inputs = np.zeros((self.seq_len, input_dim))
                targets = np.zeros((self.seq_len, 6))  # x, y, z, vx, vy, vz

                prev_time = obs_window[0].timestamp
                for i, obs in enumerate(obs_window):
                    # One-hot sensor ID
                    inputs[i, obs.sensor_id] = 1.0

                    # Position
                    inputs[i, self.n_sensors:self.n_sensors+3] = obs.position

                    # Velocity (0 if not available)
                    if obs.velocity is not None:
                        inputs[i, self.n_sensors+3:self.n_sensors+6] = obs.velocity

                    # Confidence
                    inputs[i, self.n_sensors+6] = obs.confidence

                    # Time delta
                    inputs[i, self.n_sensors+7] = obs.timestamp - prev_time
                    prev_time = obs.timestamp

                    # Ground truth at this timestamp
                    step_idx = min(int(obs.timestamp / self.dt), len(trajectory) - 1)
                    targets[i] = trajectory[step_idx]

                samples.append((inputs.astype(np.float32), targets.astype(np.float32)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, targets = self.samples[idx]
        return torch.tensor(inputs), torch.tensor(targets)


# ============================================================================
# Kalman Filter Baseline
# ============================================================================

class KalmanFilter:
    """
    Extended Kalman Filter for sensor fusion baseline.

    State: [x, y, z, vx, vy, vz]
    """

    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.state = np.zeros(6)
        self.P = np.eye(6) * 10  # Covariance

        # Process noise
        self.Q = np.eye(6) * process_noise
        self.Q[3:, 3:] *= 2  # More noise in velocity

        # Base measurement noise (adjusted per sensor)
        self.R_base = measurement_noise

        self.initialized = False

    def predict(self, dt):
        """Predict step."""
        # State transition: constant velocity
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, measurement, sensor_config: SensorConfig):
        """Update step with sensor measurement."""
        if not self.initialized:
            self.state[:3] = measurement[:3]
            if sensor_config.has_velocity and len(measurement) > 3:
                self.state[3:6] = measurement[3:6]
            self.initialized = True
            return

        # Measurement matrix
        if sensor_config.has_velocity:
            H = np.eye(6)
            z = measurement[:6]
            R = np.eye(6) * sensor_config.position_noise**2
            R[3:, 3:] = sensor_config.velocity_noise**2
        else:
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)
            z = measurement[:3]
            R = np.eye(3) * sensor_config.position_noise**2

        if not sensor_config.has_depth:
            # Zero out z component
            if sensor_config.has_velocity:
                H[2, :] = 0
                R[2, 2] = 1e6  # Large uncertainty
            else:
                H[2, :] = 0
                R[2, 2] = 1e6

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_state(self):
        return self.state.copy()


def evaluate_kalman_baseline(dataset: SensorFusionDataset) -> dict:
    """Evaluate Kalman filter on the dataset."""

    position_errors = []
    velocity_errors = []

    # Use all samples for fair comparison with PSI
    for inputs, targets in tqdm(dataset.samples, desc="Evaluating Kalman"):
        kf = KalmanFilter(process_noise=0.5, measurement_noise=1.0)

        seq_errors_pos = []
        seq_errors_vel = []

        prev_time = 0
        for i in range(len(inputs)):
            # Get sensor ID
            sensor_id = np.argmax(inputs[i, :dataset.n_sensors])
            sensor = dataset.sensors[sensor_id]

            # Get measurement
            pos = inputs[i, dataset.n_sensors:dataset.n_sensors+3]
            vel = inputs[i, dataset.n_sensors+3:dataset.n_sensors+6]
            dt = inputs[i, -1]

            # Predict
            if dt > 0:
                kf.predict(dt)

            # Update
            if sensor.has_velocity:
                measurement = np.concatenate([pos, vel])
            else:
                measurement = pos
            kf.update(measurement, sensor)

            # Compute error
            est_state = kf.get_state()
            true_state = targets[i]

            pos_error = np.linalg.norm(est_state[:3] - true_state[:3])
            vel_error = np.linalg.norm(est_state[3:6] - true_state[3:6])

            seq_errors_pos.append(pos_error)
            seq_errors_vel.append(vel_error)

        position_errors.extend(seq_errors_pos)
        velocity_errors.extend(seq_errors_vel)

    return {
        'position_rmse': np.sqrt(np.mean(np.array(position_errors)**2)),
        'velocity_rmse': np.sqrt(np.mean(np.array(velocity_errors)**2)),
        'position_mae': np.mean(position_errors),
        'velocity_mae': np.mean(velocity_errors)
    }


# ============================================================================
# PSI Sensor Fusion Model
# ============================================================================

class PSISensorFusionModel(nn.Module):
    """
    PSI-based sensor fusion model.

    Takes asynchronous, multi-sensor observations and estimates true state.
    """

    def __init__(self, input_dim, state_dim=6, hidden_dim=256, num_layers=6):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.psi_blocks = nn.ModuleList([
            PSIBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output head - estimate state at each timestep
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] - sensor observations

        Returns:
            state: [batch, seq_len, state_dim] - estimated states
        """
        h = self.input_proj(x)

        for block in self.psi_blocks:
            h = block(h)

        state = self.output_head(h)
        return state


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Train the PSI sensor fusion model."""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Weight position more than velocity
    loss_weights = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0], device=device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            pred = model(inputs)

            # Weighted MSE
            diff = (pred - targets) ** 2
            weighted_diff = diff * loss_weights
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
        val_vel_error = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                pred = model(inputs)

                diff = (pred - targets) ** 2
                weighted_diff = diff * loss_weights
                loss = weighted_diff.mean()
                val_loss += loss.item()

                # Position RMSE
                pos_error = torch.sqrt(((pred[:, :, :3] - targets[:, :, :3])**2).sum(dim=-1)).mean()
                vel_error = torch.sqrt(((pred[:, :, 3:6] - targets[:, :, 3:6])**2).sum(dim=-1)).mean()
                val_pos_error += pos_error.item()
                val_vel_error += vel_error.item()

        val_loss /= len(val_loader)
        val_pos_error /= len(val_loader)
        val_vel_error /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Pos RMSE={val_pos_error:.3f}m, Vel RMSE={val_vel_error:.3f}m/s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'sensor_fusion_model.pt')

    return best_val_loss


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate PSI model."""
    model.eval()

    position_errors = []
    velocity_errors = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred = model(inputs)

            # Errors per timestep
            pos_err = torch.sqrt(((pred[:, :, :3] - targets[:, :, :3])**2).sum(dim=-1))
            vel_err = torch.sqrt(((pred[:, :, 3:6] - targets[:, :, 3:6])**2).sum(dim=-1))

            position_errors.extend(pos_err.cpu().numpy().flatten())
            velocity_errors.extend(vel_err.cpu().numpy().flatten())

    return {
        'position_rmse': np.sqrt(np.mean(np.array(position_errors)**2)),
        'velocity_rmse': np.sqrt(np.mean(np.array(velocity_errors)**2)),
        'position_mae': np.mean(position_errors),
        'velocity_mae': np.mean(velocity_errors)
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_fusion(model, dataset, device, n_samples=3):
    """Visualize sensor fusion results."""

    model.eval()

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for row, idx in enumerate(indices):
        inputs, targets = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inputs)[0].cpu().numpy()

        targets = targets.numpy()
        raw_inputs = dataset.samples[idx][0]

        # Left plot: Position
        ax = axes[row, 0]

        # Plot raw sensor observations
        for s_id in range(dataset.n_sensors):
            mask = raw_inputs[:, s_id] == 1
            sensor_pos = raw_inputs[mask, dataset.n_sensors:dataset.n_sensors+2]
            ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], alpha=0.3, s=20,
                      label=f'{dataset.sensors[s_id].name}')

        # Plot ground truth
        ax.plot(targets[:, 0], targets[:, 1], 'g-', linewidth=2, label='Ground Truth')

        # Plot PSI estimate
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='PSI Estimate')

        pos_rmse = np.sqrt(np.mean((pred[:, :3] - targets[:, :3])**2))
        ax.set_title(f'Position (RMSE={pos_rmse:.3f}m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Right plot: Velocity
        ax = axes[row, 1]
        timesteps = np.arange(len(targets))

        ax.plot(timesteps, targets[:, 3], 'g-', linewidth=2, label='True Vx')
        ax.plot(timesteps, pred[:, 3], 'r--', linewidth=2, label='Est Vx')
        ax.plot(timesteps, targets[:, 4], 'b-', linewidth=2, label='True Vy')
        ax.plot(timesteps, pred[:, 4], 'm--', linewidth=2, label='Est Vy')

        vel_rmse = np.sqrt(np.mean((pred[:, 3:6] - targets[:, 3:6])**2))
        ax.set_title(f'Velocity (RMSE={vel_rmse:.3f}m/s)')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensor_fusion_results.png', dpi=150, bbox_inches='tight')
    print("Saved sensor_fusion_results.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sensor Fusion Experiment with PSI')
    parser.add_argument('--n_trajectories', type=int, default=2000,
                        help='Number of trajectories to generate')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration of each trajectory (seconds)')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Sequence length for training')
    parser.add_argument('--dim', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of PSI layers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--skip_kalman', action='store_true',
                        help='Skip Kalman filter baseline')
    args = parser.parse_args()

    print("=" * 80)
    print("PSI Sensor Fusion Experiment")
    print("=" * 80)
    print()
    print("Task: Fuse noisy observations from multiple sensors to estimate true state")
    print("Sensors:")
    print(f"  - Radar: noisy position (std=1.5m), good velocity, 200m range, 10Hz")
    print(f"  - Lidar: accurate position (std=0.1m), no velocity, 70m range, 10Hz")
    print(f"  - Camera: poor depth (std=2.0m), no velocity, 50m range, 30Hz")
    print()

    # Generate datasets with trajectory-level splits (avoids data leakage)
    sensors = [RADAR, LIDAR, CAMERA]

    train_dataset = SensorFusionDataset(
        n_trajectories=args.n_trajectories,
        duration=args.duration,
        sensors=sensors,
        seq_len=args.seq_len,
        split='train'
    )

    val_dataset = SensorFusionDataset(
        n_trajectories=args.n_trajectories,
        duration=args.duration,
        sensors=sensors,
        seq_len=args.seq_len,
        split='val'
    )

    test_dataset = SensorFusionDataset(
        n_trajectories=args.n_trajectories,
        duration=args.duration,
        sensors=sensors,
        seq_len=args.seq_len,
        split='test'
    )

    print(f"\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Kalman filter baseline (evaluated on test set for fair comparison)
    if not args.skip_kalman:
        print("\n" + "=" * 80)
        print("Kalman Filter Baseline")
        print("=" * 80)

        kalman_metrics = evaluate_kalman_baseline(test_dataset)
        print(f"\nKalman Filter Results:")
        print(f"  Position RMSE: {kalman_metrics['position_rmse']:.3f} m")
        print(f"  Velocity RMSE: {kalman_metrics['velocity_rmse']:.3f} m/s")
        print(f"  Position MAE:  {kalman_metrics['position_mae']:.3f} m")
        print(f"  Velocity MAE:  {kalman_metrics['velocity_mae']:.3f} m/s")

    # PSI model
    print("\n" + "=" * 80)
    print("Training PSI Model")
    print("=" * 80)

    input_dim = len(sensors) + 3 + 3 + 1 + 1  # one-hot + pos + vel + conf + dt
    model = PSISensorFusionModel(
        input_dim=input_dim,
        state_dim=6,
        hidden_dim=args.dim,
        num_layers=args.num_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Load best model
    checkpoint = torch.load('sensor_fusion_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    psi_metrics = evaluate_model(model, test_loader, device)
    print(f"\nPSI Model Results:")
    print(f"  Position RMSE: {psi_metrics['position_rmse']:.3f} m")
    print(f"  Velocity RMSE: {psi_metrics['velocity_rmse']:.3f} m/s")
    print(f"  Position MAE:  {psi_metrics['position_mae']:.3f} m")
    print(f"  Velocity MAE:  {psi_metrics['velocity_mae']:.3f} m/s")

    if not args.skip_kalman:
        print("\n" + "-" * 40)
        print("Comparison:")
        print(f"  Position RMSE: Kalman={kalman_metrics['position_rmse']:.3f}m, PSI={psi_metrics['position_rmse']:.3f}m")
        print(f"  Velocity RMSE: Kalman={kalman_metrics['velocity_rmse']:.3f}m/s, PSI={psi_metrics['velocity_rmse']:.3f}m/s")

        pos_improvement = (kalman_metrics['position_rmse'] - psi_metrics['position_rmse']) / kalman_metrics['position_rmse'] * 100
        vel_improvement = (kalman_metrics['velocity_rmse'] - psi_metrics['velocity_rmse']) / kalman_metrics['velocity_rmse'] * 100
        print(f"\n  Position improvement: {pos_improvement:+.1f}%")
        print(f"  Velocity improvement: {vel_improvement:+.1f}%")

    # Visualize
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    visualize_fusion(model, test_dataset, device, n_samples=3)

    print("\nDone!")


if __name__ == "__main__":
    main()
