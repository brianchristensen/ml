"""
Lorenz Fusion Experiment - State Reconstruction from Noisy Observations

This tests the hypothesis that PSI learns state reconstruction when the
true state is HIDDEN behind noisy observations, vs learning dynamics
when the state is directly given.

Setup:
- True state: (x, y, z) from Lorenz attractor
- Observations: Multiple noisy, partial, redundant sensors
- Task: Reconstruct true state from observations

Sensors:
1. Position sensor A: Measures x with noise
2. Position sensor B: Measures y with noise
3. Position sensor C: Measures z with noise
4. Coupled sensor: Measures x+y with noise (redundant)
5. Velocity proxy: Measures x-x_prev (noisy finite difference)
6. Altitude sensor: Measures z with different noise characteristics

This mirrors the sensor fusion experiment but on Lorenz dynamics.
If our hypothesis is correct, the learned phase space should show
HIGH correlation with true state (like sensor fusion), not the
compressed dynamics representation we saw with direct state input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Optional

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Lorenz System Generation
# ============================================================================

def lorenz_step(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    """One step of Lorenz system using RK4 integration."""
    def derivatives(x, y, z):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    # RK4 integration for better accuracy
    k1 = derivatives(x, y, z)
    k2 = derivatives(x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], z + 0.5*dt*k1[2])
    k3 = derivatives(x + 0.5*dt*k2[0], y + 0.5*dt*k2[1], z + 0.5*dt*k2[2])
    k4 = derivatives(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])

    x_new = x + (dt/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y_new = y + (dt/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    z_new = z + (dt/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    return x_new, y_new, z_new


def generate_lorenz_trajectory(length=500, dt=0.01, warmup=1000):
    """Generate a single Lorenz trajectory with warmup period."""
    # Random initial conditions
    x = np.random.uniform(-15, 15)
    y = np.random.uniform(-20, 20)
    z = np.random.uniform(5, 45)

    # Warmup to get onto attractor
    for _ in range(warmup):
        x, y, z = lorenz_step(x, y, z, dt=dt)

    trajectory = np.zeros((length, 3), dtype=np.float32)
    for i in range(length):
        trajectory[i] = [x, y, z]
        x, y, z = lorenz_step(x, y, z, dt=dt)

    return trajectory


# ============================================================================
# Sensor Models
# ============================================================================

@dataclass
class LorenzSensorConfig:
    """Configuration for a simulated sensor observing Lorenz system."""
    name: str
    observes: str  # 'x', 'y', 'z', 'xy', 'xz', 'yz', 'dx', 'dy', 'dz'
    noise_std: float
    dropout_prob: float
    bias: float = 0.0
    scale: float = 1.0


# Define sensors with different characteristics
SENSORS = [
    LorenzSensorConfig(
        name="x_sensor",
        observes='x',
        noise_std=2.0,
        dropout_prob=0.1,
    ),
    LorenzSensorConfig(
        name="y_sensor",
        observes='y',
        noise_std=2.0,
        dropout_prob=0.1,
    ),
    LorenzSensorConfig(
        name="z_sensor",
        observes='z',
        noise_std=3.0,  # Z has larger range, more noise
        dropout_prob=0.1,
    ),
    LorenzSensorConfig(
        name="xy_coupled",
        observes='xy',  # Measures x + y
        noise_std=2.5,
        dropout_prob=0.15,
    ),
    LorenzSensorConfig(
        name="z_altitude",
        observes='z',
        noise_std=1.5,  # More accurate Z sensor
        dropout_prob=0.2,  # But drops out more
        bias=0.5,
    ),
    LorenzSensorConfig(
        name="dx_velocity",
        observes='dx',  # Finite difference velocity
        noise_std=5.0,  # Velocity is noisy
        dropout_prob=0.1,
    ),
]


def generate_sensor_observation(state: np.ndarray, prev_state: Optional[np.ndarray],
                                sensor: LorenzSensorConfig, dt: float = 0.01) -> Tuple[float, float, float]:
    """
    Generate a single sensor observation.

    Returns: (value, valid, confidence)
    """
    x, y, z = state

    # Check dropout
    if np.random.random() < sensor.dropout_prob:
        return 0.0, 0.0, 0.0  # Invalid reading

    # Get true value based on what sensor observes
    if sensor.observes == 'x':
        true_val = x
    elif sensor.observes == 'y':
        true_val = y
    elif sensor.observes == 'z':
        true_val = z
    elif sensor.observes == 'xy':
        true_val = x + y
    elif sensor.observes == 'xz':
        true_val = x + z
    elif sensor.observes == 'yz':
        true_val = y + z
    elif sensor.observes == 'dx':
        if prev_state is None:
            return 0.0, 0.0, 0.0
        true_val = (x - prev_state[0]) / dt
    elif sensor.observes == 'dy':
        if prev_state is None:
            return 0.0, 0.0, 0.0
        true_val = (y - prev_state[1]) / dt
    elif sensor.observes == 'dz':
        if prev_state is None:
            return 0.0, 0.0, 0.0
        true_val = (z - prev_state[2]) / dt
    else:
        raise ValueError(f"Unknown observation type: {sensor.observes}")

    # Apply sensor characteristics
    observed = true_val * sensor.scale + sensor.bias
    observed += np.random.normal(0, sensor.noise_std)

    # Confidence based on noise level
    confidence = 1.0 / (1.0 + sensor.noise_std)

    return observed, 1.0, confidence


# ============================================================================
# Dataset
# ============================================================================

# Input format per sensor: [valid, value, confidence] = 3 values
SENSOR_INPUT_DIM = 3

def compute_input_dim(n_sensors: int) -> int:
    return n_sensors * SENSOR_INPUT_DIM


class LorenzFusionDataset(Dataset):
    """
    Dataset for Lorenz state reconstruction from noisy observations.

    Each sample:
    - Input: Sequence of sensor observations [seq_len, n_sensors * 3]
    - Target: True state at final timestep [3]
    """

    def __init__(self, n_trajectories: int, trajectory_length: int = 500,
                 seq_len: int = 50, sensors: List[LorenzSensorConfig] = SENSORS,
                 split: str = 'train', seed: int = None):

        if seed is None:
            seed = {'train': 42, 'val': 43, 'test': 44}[split]
        np.random.seed(seed)

        self.seq_len = seq_len
        self.sensors = sensors
        self.n_sensors = len(sensors)
        self.input_dim = compute_input_dim(self.n_sensors)

        # Generate trajectories
        print(f"Generating {n_trajectories} Lorenz trajectories for {split}...")
        self.trajectories = []
        for _ in tqdm(range(n_trajectories), desc=f"Generating {split}"):
            traj = generate_lorenz_trajectory(trajectory_length)
            self.trajectories.append(traj)

        # Compute normalization stats from trajectories
        all_states = np.concatenate(self.trajectories, axis=0)
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0)

        # Pre-generate all sequences and observations
        print(f"Generating sensor observations...")
        self.sequences = []
        self.targets = []

        for traj in tqdm(self.trajectories, desc="Processing"):
            # Normalize trajectory
            traj_norm = (traj - self.state_mean) / (self.state_std + 1e-8)

            # Generate sequences
            for start in range(0, len(traj) - seq_len, 5):  # Step by 5
                # Generate observations for this sequence
                obs_sequence = np.zeros((seq_len, self.input_dim), dtype=np.float32)

                for t in range(seq_len):
                    state = traj_norm[start + t]
                    prev_state = traj_norm[start + t - 1] if t > 0 else None

                    for s_idx, sensor in enumerate(sensors):
                        value, valid, conf = generate_sensor_observation(
                            state, prev_state, sensor
                        )
                        offset = s_idx * SENSOR_INPUT_DIM
                        obs_sequence[t, offset] = valid
                        obs_sequence[t, offset + 1] = value
                        obs_sequence[t, offset + 2] = conf

                # Target is true state at final timestep
                target = traj_norm[start + seq_len - 1]

                self.sequences.append(obs_sequence)
                self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

        print(f"  Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# ============================================================================
# Model
# ============================================================================

class PSILorenzFusionModel(nn.Module):
    """
    PSI model for Lorenz state reconstruction from noisy observations.
    """

    def __init__(self, input_dim: int, state_dim: int = 3,
                 hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        # PSI blocks
        self.psi_blocks = nn.ModuleList([
            PSIBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] sensor observations
        Returns:
            state: [batch, state_dim] reconstructed state
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        h = self.input_proj(x)

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]

        # Apply PSI blocks
        for block in self.psi_blocks:
            h = block(h)

        # Output from last timestep
        h_final = h[:, -1, :]
        state = self.output_proj(h_final)

        return state


# ============================================================================
# Training
# ============================================================================

def apply_sensor_dropout(inputs: torch.Tensor, n_sensors: int,
                         dropout_prob: float = 0.3) -> torch.Tensor:
    """Apply random sensor dropout during training."""
    if dropout_prob <= 0:
        return inputs

    batch_size = inputs.shape[0]
    inputs = inputs.clone()

    for b in range(batch_size):
        for s in range(n_sensors):
            if torch.rand(1).item() < dropout_prob:
                offset = s * SENSOR_INPUT_DIM
                inputs[b, :, offset:offset + SENSOR_INPUT_DIM] = 0.0

    return inputs


def train_model(model, train_loader, val_loader, epochs, lr, device,
                sensor_dropout_prob=0.0, n_sensors=6):
    """Train the model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if sensor_dropout_prob > 0:
                inputs = apply_sensor_dropout(inputs, n_sensors, sensor_dropout_prob)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = nn.functional.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred = model(inputs)
                val_loss += nn.functional.mse_loss(pred, targets).item()
        val_loss /= len(val_loader)

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, 'lorenz_fusion_model.pt')
            print(f"  → Saved best model")

    return best_val_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            pred = model(inputs)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Per-dimension metrics
    mse = np.mean((preds - targets) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Correlation per dimension
    correlations = []
    for d in range(3):
        corr = np.corrcoef(preds[:, d], targets[:, d])[0, 1]
        correlations.append(corr)

    return {
        'mse': mse,
        'rmse': rmse,
        'total_rmse': np.sqrt(np.mean((preds - targets) ** 2)),
        'correlations': correlations,
        'preds': preds,
        'targets': targets
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_reconstruction(model, test_dataset, device, n_samples=3):
    """Visualize reconstruction quality."""
    model.eval()

    fig = plt.figure(figsize=(15, 5 * n_samples))

    indices = np.random.choice(len(test_dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            inputs, target = test_dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            pred = model(inputs).cpu().numpy()[0]
            target = target.numpy()

            # 3D plot
            ax = fig.add_subplot(n_samples, 2, 2*i + 1, projection='3d')
            ax.scatter([target[0]], [target[1]], [target[2]],
                      c='blue', s=100, label='True', marker='o')
            ax.scatter([pred[0]], [pred[1]], [pred[2]],
                      c='red', s=100, label='Predicted', marker='x')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {i+1}: 3D State')
            ax.legend()

            # Bar comparison
            ax2 = fig.add_subplot(n_samples, 2, 2*i + 2)
            x_pos = np.arange(3)
            width = 0.35
            ax2.bar(x_pos - width/2, target, width, label='True', color='blue', alpha=0.7)
            ax2.bar(x_pos + width/2, pred, width, label='Predicted', color='red', alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(['X', 'Y', 'Z'])
            ax2.set_ylabel('Value (normalized)')
            ax2.set_title(f'Sample {i+1}: State Comparison')
            ax2.legend()

    plt.tight_layout()
    plt.savefig('lorenz_fusion_reconstruction.png', dpi=150)
    print("Saved lorenz_fusion_reconstruction.png")
    plt.close()


def visualize_attractor_reconstruction(metrics, save_path='lorenz_fusion_attractor.png'):
    """Visualize reconstructed vs true attractor."""
    preds = metrics['preds']
    targets = metrics['targets']

    fig = plt.figure(figsize=(15, 5))

    # True attractor
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(targets[:, 0], targets[:, 1], targets[:, 2],
               c=np.arange(len(targets)), cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True States')

    # Reconstructed attractor
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(preds[:, 0], preds[:, 1], preds[:, 2],
               c=np.arange(len(preds)), cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Reconstructed States')

    # Error distribution
    ax3 = fig.add_subplot(1, 3, 3)
    errors = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
    ax3.hist(errors, bins=50, density=True, alpha=0.7)
    ax3.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
    ax3.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.3f}')
    ax3.set_xlabel('Reconstruction Error')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Lorenz Fusion Experiment')
    parser.add_argument('--n_trajectories', type=int, default=2000)
    parser.add_argument('--trajectory_length', type=int, default=500)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sensor_dropout', type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 80)
    print("LORENZ FUSION EXPERIMENT")
    print("State Reconstruction from Noisy Observations")
    print("=" * 80)
    print()
    print("Hypothesis: When state is HIDDEN behind noisy sensors,")
    print("PSI will learn a representation that correlates with TRUE STATE")
    print("(like sensor fusion), not compressed dynamics (like direct Lorenz).")
    print()
    print("Sensors:")
    for sensor in SENSORS:
        print(f"  - {sensor.name}: observes {sensor.observes}, noise={sensor.noise_std}, dropout={sensor.dropout_prob}")
    print()

    # Generate datasets
    train_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
        seq_len=args.seq_len,
        split='train'
    )

    val_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories // 5,
        trajectory_length=args.trajectory_length,
        seq_len=args.seq_len,
        split='val'
    )

    test_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories // 5,
        trajectory_length=args.trajectory_length,
        seq_len=args.seq_len,
        split='test'
    )

    print(f"\nDatasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    input_dim = compute_input_dim(len(SENSORS))
    print(f"\nModel: input_dim={input_dim}, hidden={args.hidden_dim}, layers={args.num_layers}")

    model = PSILorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    train_model(
        model, train_loader, val_loader,
        args.epochs, args.lr, device,
        sensor_dropout_prob=args.sensor_dropout,
        n_sensors=len(SENSORS)
    )

    # Load best model
    checkpoint = torch.load('lorenz_fusion_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    metrics = evaluate_model(model, test_loader, device)

    print(f"\nReconstruction Results:")
    print(f"  Total RMSE: {metrics['total_rmse']:.4f}")
    print(f"  Per-dimension RMSE: X={metrics['rmse'][0]:.4f}, Y={metrics['rmse'][1]:.4f}, Z={metrics['rmse'][2]:.4f}")
    print(f"  Per-dimension Correlation: X={metrics['correlations'][0]:.4f}, Y={metrics['correlations'][1]:.4f}, Z={metrics['correlations'][2]:.4f}")

    # Visualizations
    print("\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)

    visualize_reconstruction(model, test_dataset, device)
    visualize_attractor_reconstruction(metrics)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("If hypothesis is correct, running lorenz_phase_space_viz.py")
    print("(adapted for this model) should show HIGH state correlations")
    print("similar to sensor fusion (~0.9+), not compressed dynamics (~0.7).")
    print()
    print("Next: Run phase space visualization on this model!")
    print()


if __name__ == "__main__":
    main()
