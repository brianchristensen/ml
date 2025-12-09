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

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Sensor Dropout Augmentation
# ============================================================================

def apply_sensor_dropout(inputs: torch.Tensor, n_sensors: int, dropout_prob: float = 0.3,
                         velocity_sensor_indices: list = None) -> torch.Tensor:
    """
    Randomly drop entire sensors during training to encourage robustness.

    For synchronized format where each row contains ALL sensors:
    [sensor0_valid, sensor0_pos(3), sensor0_vel(3), sensor0_conf,
     sensor1_valid, sensor1_pos(3), sensor1_vel(3), sensor1_conf, ...]

    For each sample in the batch, EACH sensor is independently dropped with
    probability dropout_prob. This means with dropout_prob=0.5:
    - 50% chance each sensor is dropped
    - Sample might have 0, 1, 2, or all 3 sensors dropped
    - Ensures model sees many single-sensor and two-sensor scenarios

    IMPORTANT: If velocity_sensor_indices is provided, ensures at least ONE
    velocity sensor remains active (tests redundant failover, not total failure).

    Args:
        inputs: [batch, seq_len, n_sensors * 8] tensor of sensor observations
        n_sensors: number of sensors
        dropout_prob: probability of dropping EACH sensor independently (0.0-1.0)
        velocity_sensor_indices: list of sensor indices that provide velocity
                                 (ensures at least one remains)

    Returns:
        Modified inputs with some sensors dropped (entire columns zeroed)
    """
    if dropout_prob <= 0:
        return inputs

    batch_size = inputs.shape[0]
    inputs = inputs.clone()

    for b in range(batch_size):
        # Independently decide to drop each sensor
        sensors_to_drop = []
        for s in range(n_sensors):
            if torch.rand(1).item() < dropout_prob:
                sensors_to_drop.append(s)

        # Don't drop ALL sensors - need at least one
        if len(sensors_to_drop) == n_sensors:
            # Keep one random sensor
            keep_sensor = torch.randint(0, n_sensors, (1,)).item()
            sensors_to_drop.remove(keep_sensor)

        # Ensure at least one velocity sensor remains if specified
        if velocity_sensor_indices is not None:
            velocity_sensors_remaining = [v for v in velocity_sensor_indices if v not in sensors_to_drop]
            if len(velocity_sensors_remaining) == 0:
                # All velocity sensors would be dropped - keep one random velocity sensor
                keep_velocity = velocity_sensor_indices[torch.randint(0, len(velocity_sensor_indices), (1,)).item()]
                if keep_velocity in sensors_to_drop:
                    sensors_to_drop.remove(keep_velocity)

        # Zero out dropped sensors across ALL timesteps
        for sensor_to_drop in sensors_to_drop:
            offset = sensor_to_drop * SENSOR_INPUT_DIM
            # Zero all 8 values for this sensor: valid, pos(3), vel(3), conf
            inputs[b, :, offset:offset + SENSOR_INPUT_DIM] = 0.0

    return inputs


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

# IMU - Inertial Measurement Unit (secondary velocity source)
# Measures acceleration and integrates to velocity - different physics than radar Doppler
# Good at relative motion, but drifts over time (we simulate this with higher noise at longer ranges)
IMU = SensorConfig(
    name="imu",
    update_rate=100.0,      # 100 Hz (very fast, typical for IMUs)
    position_noise=5.0,     # IMU doesn't measure position directly (this is drift)
    velocity_noise=0.5,     # Velocity from integrated acceleration (noisier than radar)
    dropout_prob=0.02,      # IMUs are very reliable
    false_positive_prob=0.0,  # No false positives (always measuring something)
    max_range=float('inf'), # No range limit (measures own motion)
    has_velocity=True,      # Primary purpose is velocity/acceleration
    has_depth=True
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

# Input dimension per sensor: valid(1) + position(3) + velocity(3) + confidence(1) = 8
SENSOR_INPUT_DIM = 8

def compute_input_dim(n_sensors: int) -> int:
    """Compute total input dimension for synchronized sensor format."""
    return n_sensors * SENSOR_INPUT_DIM


class SensorFusionDataset(Dataset):
    """
    Dataset for sensor fusion task with SYNCHRONIZED time slices.

    Each row represents a SINGLE TIMESTEP with ALL sensor readings concatenated:
    [radar_valid, radar_pos(3), radar_vel(3), radar_conf,
     lidar_valid, lidar_pos(3), lidar_vel(3), lidar_conf,
     camera_valid, camera_pos(3), camera_vel(3), camera_conf]

    This allows PSI to learn temporal dynamics of state evolution,
    with sensors providing simultaneous observations at each timestep.

    Sensors that don't have a reading at a given timestep have valid=0
    and all other values zeroed.

    IMPORTANT: Split at trajectory level to avoid data leakage.
    """

    def __init__(
        self,
        n_trajectories: int = 1000,
        duration: float = 10.0,
        sensors: List[SensorConfig] = None,
        seq_len: int = 50,
        sample_dt: float = 0.1,  # Time between samples (synchronized timesteps)
        sim_dt: float = 0.01,    # Simulation timestep for trajectory
        split: str = 'all',      # 'train', 'val', 'test', or 'all'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        if sensors is None:
            sensors = [RADAR, LIDAR, CAMERA]

        self.sensors = sensors
        self.n_sensors = len(sensors)
        self.seq_len = seq_len
        self.sample_dt = sample_dt
        self.sim_dt = sim_dt
        self.split = split

        # Input format: for each sensor: [valid, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, confidence]
        self.input_dim = compute_input_dim(self.n_sensors)

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

        print(f"Generating {desc} with {len(sensors)} sensors (synchronized format)...")
        self.samples = self._generate_samples(n_trajectories, duration, my_indices)
        print(f"Created {len(self.samples)} samples for {split}")

    def _generate_samples(self, n_trajectories, duration, my_indices):
        samples = []

        for traj_id in tqdm(range(n_trajectories), desc="Generating data"):
            if traj_id not in my_indices:
                continue

            # Generate ground truth trajectory
            trajectory = generate_trajectory(duration, self.sim_dt)

            # Generate sensor observations (still asynchronous initially)
            observations = generate_sensor_observations(
                trajectory, self.sensors, self.sim_dt
            )

            # Filter to valid observations only
            valid_obs = [o for o in observations if o.is_valid and not o.is_false_positive]

            # Organize observations by sensor
            obs_by_sensor = {i: [] for i in range(self.n_sensors)}
            for obs in valid_obs:
                obs_by_sensor[obs.sensor_id].append(obs)

            # Sort each sensor's observations by time
            for sensor_id in obs_by_sensor:
                obs_by_sensor[sensor_id].sort(key=lambda x: x.timestamp)

            # Create synchronized time slices
            n_timesteps = int(duration / self.sample_dt)

            if n_timesteps < self.seq_len:
                continue

            # Build synchronized data for entire trajectory
            sync_inputs = np.zeros((n_timesteps, self.input_dim), dtype=np.float32)
            sync_targets = np.zeros((n_timesteps, 6), dtype=np.float32)

            # Track last observation index for each sensor
            sensor_indices = {i: 0 for i in range(self.n_sensors)}

            for t_idx in range(n_timesteps):
                t = t_idx * self.sample_dt

                # Get ground truth at this time
                gt_idx = min(int(t / self.sim_dt), len(trajectory) - 1)
                sync_targets[t_idx] = trajectory[gt_idx]

                # For each sensor, find most recent observation
                for sensor_id in range(self.n_sensors):
                    sensor_obs = obs_by_sensor[sensor_id]
                    idx = sensor_indices[sensor_id]

                    # Advance to most recent observation at or before time t
                    while idx < len(sensor_obs) - 1 and sensor_obs[idx + 1].timestamp <= t:
                        idx += 1
                    sensor_indices[sensor_id] = idx

                    # Check if we have a valid recent observation (within 2x sensor period)
                    sensor_config = self.sensors[sensor_id]
                    max_age = 2.0 / sensor_config.update_rate

                    offset = sensor_id * SENSOR_INPUT_DIM

                    if idx < len(sensor_obs):
                        obs = sensor_obs[idx]
                        age = t - obs.timestamp

                        if age >= 0 and age < max_age:
                            # Valid recent observation
                            sync_inputs[t_idx, offset] = 1.0  # valid flag
                            sync_inputs[t_idx, offset+1:offset+4] = obs.position
                            if obs.velocity is not None:
                                sync_inputs[t_idx, offset+4:offset+7] = obs.velocity
                            sync_inputs[t_idx, offset+7] = obs.confidence
                        # else: leave as zeros (no valid reading)

            # Create sliding window samples
            stride = max(1, self.seq_len // 2)
            for start in range(0, n_timesteps - self.seq_len + 1, stride):
                end = start + self.seq_len
                samples.append((
                    sync_inputs[start:end].copy(),
                    sync_targets[start:end].copy()
                ))

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
    """Evaluate Kalman filter on the dataset (synchronized format)."""

    position_errors = []
    velocity_errors = []

    # Use all samples for fair comparison with PSI
    for inputs, targets in tqdm(dataset.samples, desc="Evaluating Kalman"):
        kf = KalmanFilter(process_noise=0.5, measurement_noise=1.0)

        seq_errors_pos = []
        seq_errors_vel = []

        for i in range(len(inputs)):
            # Predict with fixed dt (synchronized format uses constant sample_dt)
            if i > 0:
                kf.predict(dataset.sample_dt)

            # Process each sensor's observation at this timestep
            for sensor_id in range(dataset.n_sensors):
                offset = sensor_id * SENSOR_INPUT_DIM
                valid = inputs[i, offset]

                if valid > 0.5:  # Sensor has valid reading
                    sensor = dataset.sensors[sensor_id]
                    pos = inputs[i, offset+1:offset+4]
                    vel = inputs[i, offset+4:offset+7]

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

def train_model(model, train_loader, val_loader, epochs, lr, device,
                sensor_dropout_prob=0.0, n_sensors=3, velocity_sensor_indices=None):
    """
    Train the PSI sensor fusion model.

    Args:
        model: PSI model to train
        train_loader: training data loader
        val_loader: validation data loader
        epochs: number of training epochs
        lr: learning rate
        device: device to train on
        sensor_dropout_prob: probability of dropping a sensor during training (0.0-1.0)
        n_sensors: number of sensors for dropout augmentation
        velocity_sensor_indices: indices of sensors that provide velocity (for redundancy protection)
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Weight position more than velocity
    loss_weights = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0], device=device)

    best_val_loss = float('inf')

    if sensor_dropout_prob > 0:
        print(f"\nSensor dropout augmentation enabled: {sensor_dropout_prob*100:.0f}% of batches will have a random sensor dropped")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Apply sensor dropout augmentation during training
            if sensor_dropout_prob > 0:
                inputs = apply_sensor_dropout(inputs, n_sensors, sensor_dropout_prob,
                                              velocity_sensor_indices=velocity_sensor_indices)

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

        # Plot raw sensor observations (new synchronized format)
        for s_id in range(dataset.n_sensors):
            offset = s_id * SENSOR_INPUT_DIM
            # valid flag is at offset, position at offset+1:offset+4
            mask = raw_inputs[:, offset] == 1.0
            if mask.sum() > 0:
                sensor_pos = raw_inputs[mask, offset+1:offset+3]  # x, y only
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
    parser.add_argument('--n_trajectories', type=int, default=10000,
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
    parser.add_argument('--sensor_dropout', type=float, default=0.0,
                        help='Sensor dropout probability during training (0.0-1.0). '
                             'Higher values make model more robust to sensor failure.')
    args = parser.parse_args()

    print("=" * 80)
    print("PSI Sensor Fusion Experiment")
    print("=" * 80)
    print()
    print("Task: Fuse noisy observations from multiple sensors to estimate true state")
    print("Sensors:")
    print(f"  - Radar: noisy position (std=1.5m), good velocity (Doppler), 200m range, 10Hz")
    print(f"  - Lidar: accurate position (std=0.1m), no velocity, 70m range, 10Hz")
    print(f"  - Camera: poor depth (std=2.0m), no velocity, 50m range, 30Hz")
    print(f"  - IMU: poor position (drift), moderate velocity (integrated accel), unlimited range, 100Hz")
    print()
    print("Velocity redundancy: Radar (idx=0) and IMU (idx=3) both provide velocity")
    print("Dropout ensures at least one velocity sensor remains active")
    print()

    # Generate datasets with trajectory-level splits (avoids data leakage)
    # Now with 4 sensors: Radar, Lidar, Camera, IMU
    sensors = [RADAR, LIDAR, CAMERA, IMU]

    # Track which sensors provide velocity for redundancy protection
    velocity_sensor_indices = [i for i, s in enumerate(sensors) if s.has_velocity]
    print(f"Velocity sensors: {[sensors[i].name for i in velocity_sensor_indices]} (indices: {velocity_sensor_indices})")

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

    # New synchronized format: each sensor has 8 values (valid, pos(3), vel(3), conf)
    input_dim = compute_input_dim(len(sensors))
    print(f"\nInput format: synchronized time slices")
    print(f"  Each timestep contains all {len(sensors)} sensors")
    print(f"  Per sensor: valid(1) + pos(3) + vel(3) + conf(1) = 8 values")
    print(f"  Total input dim: {input_dim}")

    model = PSISensorFusionModel(
        input_dim=input_dim,
        state_dim=6,
        hidden_dim=args.dim,
        num_layers=args.num_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    train_model(model, train_loader, val_loader, args.epochs, args.lr, device,
                sensor_dropout_prob=args.sensor_dropout, n_sensors=len(sensors),
                velocity_sensor_indices=velocity_sensor_indices)

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
