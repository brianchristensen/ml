"""
DynaBench PDE Dynamics Experiment: Generate, Train, Evaluate

Tests PSI on benchmark PDEs without closed-form solutions:
- Advection
- Burgers' equation
- Kuramoto-Sivashinsky (chaotic)
- Reaction-Diffusion
- Wave equation
- Diffusion (heat equation)

Uses fast finite difference methods for data generation.
Reference: https://github.com/badulion/dynabench
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
import h5py
import glob

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EQUATIONS = ['advection', 'burgers', 'kuramotosivashinsky', 'reactiondiffusion', 'wave', 'gasdynamics', 'diffusion']

# DynaBench uses no underscores in equation names
EQUATION_INFO = {
    'advection': {'components': 1, 'description': 'Linear advection (1st order)'},
    'burgers': {'components': 1, 'description': "Burgers' equation (nonlinear)"},
    'kuramotosivashinsky': {'components': 1, 'description': 'Kuramoto-Sivashinsky (chaotic, no closed-form)'},
    'reactiondiffusion': {'components': 2, 'description': 'Reaction-diffusion system'},
    'wave': {'components': 2, 'description': 'Wave equation (2nd order time)'},
    'gasdynamics': {'components': 4, 'description': 'Compressible gas dynamics (Euler equations)'},
    'diffusion': {'components': 1, 'description': 'Heat/Diffusion equation'},
}


# ============================================================================
# Data Generation - Fast Finite Difference Methods
# ============================================================================

def laplacian_2d(u, dx):
    """2D Laplacian using finite differences with periodic boundaries."""
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    ) / (dx ** 2)


def gradient_x(u, dx):
    """x-derivative using central differences with periodic boundaries."""
    return (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)


def gradient_y(u, dx):
    """y-derivative using central differences with periodic boundaries."""
    return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)


def biharmonic_2d(u, dx):
    """Biharmonic operator (Laplacian of Laplacian) for KS equation."""
    return laplacian_2d(laplacian_2d(u, dx), dx)


def generate_fast_pde_data(equation='advection', n_samples=500, grid_size=32, n_timesteps=50):
    """
    Generate PDE simulation data using fast finite difference methods.
    Much faster than py-pde due to no JIT compilation overhead.
    """
    print(f"Generating {n_samples} {equation} simulations...")
    print(f"  Grid: {grid_size}x{grid_size}, {n_timesteps} timesteps")

    dx = 1.0 / grid_size
    x = np.linspace(0, 1, grid_size, endpoint=False)
    y = np.linspace(0, 1, grid_size, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    trajectories = []

    for i in range(n_samples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Generated {i + 1}/{n_samples} simulations...")

        np.random.seed(i)

        if equation == 'advection':
            # Linear advection: du/dt + c * (du/dx + du/dy) = 0
            c = 0.5 + 0.3 * np.random.randn()
            dt = 0.4 * dx / max(abs(c), 0.1)
            total_time = 2.0
            n_steps = int(total_time / dt) + 1

            # Initial condition: random modes
            u = np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
            u += 0.5 * np.sin(4 * np.pi * xx + np.random.rand() * np.pi)
            u += 0.3 * np.random.randn(grid_size, grid_size)

            traj = [u.copy()]
            for _ in range(n_steps):
                if c > 0:
                    du_dx = (u - np.roll(u, 1, axis=1)) / dx
                    du_dy = (u - np.roll(u, 1, axis=0)) / dx
                else:
                    du_dx = (np.roll(u, -1, axis=1) - u) / dx
                    du_dy = (np.roll(u, -1, axis=0) - u) / dx
                u = u - c * dt * (du_dx + du_dy)
                traj.append(u.copy())

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 1)

        elif equation == 'burgers':
            # Burgers equation: du/dt + u * du/dx = nu * laplacian(u)
            nu = 0.01 + 0.005 * np.random.randn()
            dt = 0.2 * dx ** 2 / max(nu, 0.001)
            total_time = 1.0
            n_steps = int(total_time / dt) + 1

            u = np.sin(2 * np.pi * xx) + 0.5 * np.random.randn(grid_size, grid_size)

            traj = [u.copy()]
            for _ in range(n_steps):
                lap = laplacian_2d(u, dx)
                du_dx = np.where(u > 0,
                                (u - np.roll(u, 1, axis=1)) / dx,
                                (np.roll(u, -1, axis=1) - u) / dx)
                u = u + dt * (-u * du_dx + nu * lap)
                traj.append(u.copy())

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 1)

        elif equation == 'reaction_diffusion':
            # Gray-Scott reaction-diffusion
            Du, Dv = 0.16, 0.08
            f = 0.04 + 0.01 * np.random.randn()
            k = 0.06 + 0.01 * np.random.randn()
            dt = 0.2 * dx ** 2 / max(Du, Dv)
            total_time = 5.0
            n_steps = int(total_time / dt) + 1

            u = np.ones((grid_size, grid_size))
            v = np.zeros((grid_size, grid_size))
            cx, cy = grid_size // 2, grid_size // 2
            r = grid_size // 8
            mask = ((xx - x[cx]) ** 2 + (yy - y[cy]) ** 2) < (r * dx) ** 2
            u[mask] = 0.5
            v[mask] = 0.25
            u += 0.05 * np.random.randn(grid_size, grid_size)
            v += 0.05 * np.random.randn(grid_size, grid_size)

            traj = [np.stack([u, v], axis=-1)]
            for _ in range(n_steps):
                uvv = u * v * v
                u_new = u + dt * (Du * laplacian_2d(u, dx) - uvv + f * (1 - u))
                v_new = v + dt * (Dv * laplacian_2d(v, dx) + uvv - (f + k) * v)
                u, v = u_new, v_new
                traj.append(np.stack([u, v], axis=-1))

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 2)

        elif equation == 'wave':
            # Wave equation: d2u/dt2 = c^2 * laplacian(u)
            c = 1.0 + 0.2 * np.random.randn()
            dt = 0.4 * dx / max(abs(c), 0.1)
            total_time = 2.0
            n_steps = int(total_time / dt) + 1

            u = np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
            u += 0.3 * np.random.randn(grid_size, grid_size)
            v = np.zeros((grid_size, grid_size))

            traj = [np.stack([u, v], axis=-1)]
            for _ in range(n_steps):
                lap = laplacian_2d(u, dx)
                v_new = v + dt * c ** 2 * lap
                u_new = u + dt * v_new
                u, v = u_new, v_new
                traj.append(np.stack([u, v], axis=-1))

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 2)

        elif equation == 'kuramoto_sivashinsky':
            # KS equation (chaotic): du/dt = -laplacian(u) - biharmonic(u) - 0.5 * |grad(u)|^2
            dt = 0.001 * dx ** 4
            total_time = 0.1
            n_steps = int(total_time / dt) + 1

            u = 0.1 * np.random.randn(grid_size, grid_size)
            u += 0.1 * np.sin(4 * np.pi * xx)

            traj = [u.copy()]
            for _ in range(n_steps):
                lap = laplacian_2d(u, dx)
                bih = biharmonic_2d(u, dx)
                gx, gy = gradient_x(u, dx), gradient_y(u, dx)
                grad_sq = gx ** 2 + gy ** 2
                u = u + dt * (-lap - bih - 0.5 * grad_sq)
                traj.append(u.copy())

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 1)

        else:  # diffusion
            # Heat equation: du/dt = D * laplacian(u)
            D = 0.1 + 0.05 * np.random.randn()
            dt = 0.2 * dx ** 2 / max(D, 0.01)
            total_time = 2.0
            n_steps = int(total_time / dt) + 1

            u = np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
            u += 0.5 * np.random.randn(grid_size, grid_size)

            traj = [u.copy()]
            for _ in range(n_steps):
                u = u + dt * D * laplacian_2d(u, dx)
                traj.append(u.copy())

            indices = np.linspace(0, len(traj) - 1, n_timesteps, dtype=int)
            traj = np.array([traj[j] for j in indices])
            traj = traj.reshape(n_timesteps, -1, 1)

        trajectories.append(traj)

    trajectories = np.array(trajectories)

    # Create grid points
    points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    points = np.tile(points[np.newaxis, :, :], (len(trajectories), 1, 1))

    n_features = trajectories.shape[-1]
    print(f"  Generated {len(trajectories)} simulations")
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  Features: {n_features}")

    return trajectories, points


# ============================================================================
# DynaBench Real Data Loading
# ============================================================================

def load_dynabench_data(equation='kuramotosivashinsky', data_dir='data', resolution='low',
                        max_samples=None, n_timesteps=50):
    """
    Load real DynaBench data from H5 files.

    Args:
        equation: PDE name (must match folder name exactly, e.g., 'kuramotosivashinsky', 'gasdynamics')
        data_dir: Root data directory
        resolution: 'low' (15x15), 'medium' (31x31), or 'high' (63x63)
        max_samples: Maximum samples to load (None for all)
        n_timesteps: Number of timesteps to subsample to (original is 201)

    Returns:
        trajectories: (n_samples, n_timesteps, n_points, n_features)
        points: (n_samples, n_points, 2)
    """
    base_path = Path(data_dir) / equation / 'grid' / resolution

    if not base_path.exists():
        print(f"DynaBench data not found at {base_path}")
        return None, None

    # Find all H5 files
    h5_files = sorted(base_path.glob(f'{equation}_*.h5'))

    if not h5_files:
        print(f"No H5 files found in {base_path}")
        return None, None

    print(f"Loading DynaBench {equation} data from {base_path}")
    print(f"  Found {len(h5_files)} H5 files")

    all_data = []
    all_points = []

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # data shape: (n_samples, n_timesteps, n_components, grid_x, grid_y)
            data = f['data'][:]
            # points shape: (n_samples, grid_x, grid_y, 2)
            points = f['points'][:]

        all_data.append(data)
        all_points.append(points)

        print(f"  Loaded {h5_file.name}: {data.shape[0]} samples")

        if max_samples and sum(d.shape[0] for d in all_data) >= max_samples:
            break

    # Concatenate all data
    data = np.concatenate(all_data, axis=0)
    points = np.concatenate(all_points, axis=0)

    if max_samples:
        data = data[:max_samples]
        points = points[:max_samples]

    # Original data shape: (n_samples, n_timesteps=201, n_components, grid_x, grid_y)
    n_samples, orig_timesteps, n_components, grid_x, grid_y = data.shape

    # Subsample timesteps
    if n_timesteps < orig_timesteps:
        indices = np.linspace(0, orig_timesteps - 1, n_timesteps, dtype=int)
        data = data[:, indices]

    # Reshape to expected format: (n_samples, n_timesteps, n_points, n_features)
    # From: (n_samples, n_timesteps, n_components, grid_x, grid_y)
    # To: (n_samples, n_timesteps, grid_x*grid_y, n_components)
    data = data.transpose(0, 1, 3, 4, 2)  # (n_samples, n_timesteps, grid_x, grid_y, n_components)
    data = data.reshape(n_samples, n_timesteps, grid_x * grid_y, n_components)

    # Reshape points: (n_samples, grid_x, grid_y, 2) -> (n_samples, n_points, 2)
    points = points.reshape(n_samples, grid_x * grid_y, 2)

    print(f"\nLoaded {n_samples} samples")
    print(f"  Trajectory shape: {data.shape}")
    print(f"  Points shape: {points.shape}")
    print(f"  Grid: {grid_x}x{grid_y}, Components: {n_components}")

    return data, points


def check_dynabench_available(equation, data_dir='data', resolution='low'):
    """Check if DynaBench data is available for a given equation."""
    base_path = Path(data_dir) / equation / 'grid' / resolution
    if not base_path.exists():
        return False
    h5_files = list(base_path.glob(f'{equation}_*.h5'))
    return len(h5_files) > 0


# ============================================================================
# Dataset
# ============================================================================

class PDEDataset(Dataset):
    """PyTorch Dataset for PDE dynamics."""

    def __init__(self, trajectories, points, lookback=4, rollout=16,
                 split='train', train_frac=0.8):
        self.lookback = lookback
        self.rollout = rollout

        # Split
        num_train = int(len(trajectories) * train_frac)
        if split == 'train':
            self.trajectories = trajectories[:num_train]
            self.points = points[:num_train]
        else:
            self.trajectories = trajectories[num_train:]
            self.points = points[num_train:]

        # Get dimensions
        self.n_points = self.trajectories.shape[2]
        self.n_features = self.trajectories.shape[3]
        self.n_timesteps = self.trajectories.shape[1]

        # Compute normalization stats
        self.mean = self.trajectories.mean(axis=(0, 1, 2))
        self.std = self.trajectories.std(axis=(0, 1, 2)) + 1e-8

        # Valid sequence starts
        self.seqs_per_traj = max(1, self.n_timesteps - lookback - rollout)

        print(f"{split.upper()}: {len(self.trajectories)} trajectories, "
              f"{len(self)} sequences, {self.n_features} features")

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.seqs_per_traj
        pos = idx % self.seqs_per_traj

        traj = self.trajectories[traj_idx]
        pts = self.points[traj_idx]

        # Normalize
        traj_norm = (traj - self.mean) / self.std

        # Context and target
        context = traj_norm[pos:pos + self.lookback]  # (lookback, n_points, n_features)
        target = traj_norm[pos + self.lookback]  # (n_points, n_features)

        # Flatten spatial dimension
        context_flat = context.reshape(self.lookback, -1)
        target_flat = target.reshape(-1)

        return {
            'context': torch.tensor(context_flat, dtype=torch.float32),
            'target': torch.tensor(target_flat, dtype=torch.float32),
            'points': torch.tensor(pts, dtype=torch.float32)
        }


# ============================================================================
# Model
# ============================================================================

class PSIDynamicsPredictor(nn.Module):
    """PSI model for PDE dynamics prediction."""

    def __init__(self, state_dim, dim=256, num_layers=8, max_len=64, device='cuda'):
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
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, state_dim)
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

def predict_trajectory(model, initial_context, num_steps, device):
    """Rollout PSI predictions autoregressively."""
    model.eval()
    predictions = []

    context = torch.tensor(initial_context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_steps):
            next_state = model(context)
            predictions.append(next_state.cpu().numpy()[0])

            next_state_expanded = next_state.unsqueeze(1)
            context = torch.cat([context[:, 1:, :], next_state_expanded], dim=1)

    return np.array(predictions)


def visualize_pde(ground_truth, predictions, points, equation, n_features, grid_size, sample_idx=0):
    """Visualize PDE evolution: ground truth vs predictions."""

    n_steps = min(len(ground_truth), len(predictions))

    # Select time steps to visualize
    vis_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    vis_steps = sorted(set([max(0, min(s, n_steps - 1)) for s in vis_steps]))

    fig, axes = plt.subplots(2, len(vis_steps), figsize=(4 * len(vis_steps), 8))

    if len(vis_steps) == 1:
        axes = axes.reshape(2, 1)

    # Plot first feature component
    feature_idx = 0

    for i, step in enumerate(vis_steps):
        gt = ground_truth[step].reshape(grid_size, grid_size, n_features)[:, :, feature_idx]
        pred = predictions[step].reshape(grid_size, grid_size, n_features)[:, :, feature_idx]

        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        # Ground truth
        im1 = axes[0, i].imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, i].set_title(f'GT t={step}')
        if i == 0:
            axes[0, i].set_ylabel('Ground Truth')
        plt.colorbar(im1, ax=axes[0, i])

        # Predictions
        im2 = axes[1, i].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f'PSI t={step}')
        if i == 0:
            axes[1, i].set_ylabel('PSI Prediction')
        plt.colorbar(im2, ax=axes[1, i])

    plt.suptitle(f'{equation.replace("_", " ").title()} - Sample {sample_idx}')
    plt.tight_layout()

    filename = f'{equation}_comparison_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def plot_error_over_time(errors, equation):
    """Plot prediction error vs rollout step."""
    plt.figure(figsize=(10, 5))
    plt.plot(errors, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Rollout Step')
    plt.ylabel('MSE')
    plt.title(f'{equation.replace("_", " ").title()} - Error vs Rollout Step')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    filename = f'{equation}_error_rollout.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DynaBench PDE Experiment with PSI')
    parser.add_argument('--equation', type=str, default='advection',
                        choices=EQUATIONS, help='PDE to test')
    parser.add_argument('--n_samples', type=int, default=None, help='Max samples to load (None=all for real data, 500 for synthetic)')
    parser.add_argument('--grid_size', type=int, default=32, help='Spatial grid size (for synthetic data)')
    parser.add_argument('--n_timesteps', type=int, default=50, help='Time steps per simulation')
    parser.add_argument('--lookback', type=int, default=4, help='Number of past frames')
    parser.add_argument('--rollout', type=int, default=16, help='Number of future frames to predict')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of PSI layers')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for DynaBench data')
    parser.add_argument('--resolution', type=str, default='low', choices=['low', 'medium', 'high'],
                        help='DynaBench grid resolution (low=15x15, medium=31x31, high=63x63)')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Force use of synthetic data even if real data is available')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PSI DynaBench Experiment: {args.equation}")
    print("=" * 80)
    print()
    print(f"Equation: {EQUATION_INFO[args.equation]['description']}")

    # Check for real DynaBench data
    use_real_data = False
    if not args.use_synthetic and check_dynabench_available(args.equation, args.data_dir, args.resolution):
        print(f"Found real DynaBench data for {args.equation}!")
        use_real_data = True
    else:
        print(f"Using synthetic data generation for {args.equation}")
        print(f"  (To use real data, download DynaBench datasets to {args.data_dir}/)")

    print()

    # Load or generate data
    if use_real_data:
        trajectories, points = load_dynabench_data(
            equation=args.equation,
            data_dir=args.data_dir,
            resolution=args.resolution,
            max_samples=args.n_samples,  # None = load all
            n_timesteps=args.n_timesteps
        )
        # Get grid size from loaded data
        grid_size = int(np.sqrt(trajectories.shape[2]))
    else:
        n_samples = args.n_samples if args.n_samples else 500  # Default 500 for synthetic
        print(f"Samples: {n_samples}")
        print(f"Grid: {args.grid_size}x{args.grid_size}, {args.n_timesteps} timesteps")
        trajectories, points = generate_fast_pde_data(
            equation=args.equation,
            n_samples=n_samples,
            grid_size=args.grid_size,
            n_timesteps=args.n_timesteps
        )
        grid_size = args.grid_size

    print(f"Lookback: {args.lookback}, Rollout: {args.rollout}")
    print()

    if trajectories is None:
        print("Failed to load/generate data!")
        return

    # Create datasets
    train_dataset = PDEDataset(
        trajectories, points,
        lookback=args.lookback,
        rollout=args.rollout,
        split='train'
    )

    val_dataset = PDEDataset(
        trajectories, points,
        lookback=args.lookback,
        rollout=args.rollout,
        split='val'
    )

    # Copy normalization stats
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    state_dim = train_dataset.n_points * train_dataset.n_features
    n_features = train_dataset.n_features

    model = PSIDynamicsPredictor(
        state_dim=state_dim,
        dim=args.dim,
        num_layers=args.num_layers,
        max_len=args.lookback + args.rollout,
        device=device
    ).to(device)

    print()
    print(f"State dimension: {state_dim} ({grid_size}x{grid_size}x{n_features})")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print()

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Training...")
    print("-" * 60)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({elapsed:5.1f}s) | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'equation': args.equation,
                'args': vars(args),
            }, f'{args.equation}_best.pt')
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

    # Evaluation
    print()
    print("=" * 80)
    print("Evaluation")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load(f'{args.equation}_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get validation samples
    val_trajectories = trajectories[int(len(trajectories) * 0.8):]
    val_points = points[int(len(points) * 0.8):]

    all_errors = []

    for sample_idx in range(min(3, len(val_trajectories))):
        traj = val_trajectories[sample_idx]
        pts = val_points[sample_idx]

        # Normalize
        traj_norm = (traj - train_dataset.mean) / train_dataset.std

        # Get context
        context = traj_norm[:args.lookback].reshape(args.lookback, -1)

        # Predict rollout steps
        predictions = predict_trajectory(model, context, args.rollout, device)

        # Ground truth
        gt = traj_norm[args.lookback:args.lookback + args.rollout].reshape(args.rollout, -1)

        # Compute errors
        errors = np.mean((predictions - gt) ** 2, axis=1)
        all_errors.append(errors)

        print(f"\nSample {sample_idx}:")
        print(f"  Step 1 MSE:  {errors[0]:.6f}")
        print(f"  Step 8 MSE:  {errors[min(7, len(errors)-1)]:.6f}")
        print(f"  Step {len(errors)} MSE: {errors[-1]:.6f}")

        # Denormalize for visualization
        # std/mean are (n_features,), need to tile to (n_points * n_features,)
        std_tiled = np.tile(train_dataset.std, train_dataset.n_points)
        mean_tiled = np.tile(train_dataset.mean, train_dataset.n_points)
        predictions_denorm = predictions * std_tiled + mean_tiled
        gt_denorm = traj[args.lookback:args.lookback + args.rollout].reshape(args.rollout, -1)

        visualize_pde(
            gt_denorm, predictions_denorm, pts,
            args.equation, train_dataset.n_features, grid_size, sample_idx
        )

    # Plot average error
    avg_errors = np.mean(all_errors, axis=0)
    plot_error_over_time(avg_errors, args.equation)

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Equation: {args.equation} ({EQUATION_INFO[args.equation]['description']})")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Average 1-step MSE: {avg_errors[0]:.6f}")
    print(f"Average {len(avg_errors)}-step MSE: {avg_errors[-1]:.6f}")
    print()
    print("Generated artifacts:")
    print(f"  - {args.equation}_best.pt (model checkpoint)")
    print(f"  - {args.equation}_comparison_*.png (visualizations)")
    print(f"  - {args.equation}_error_rollout.png (error plot)")


if __name__ == "__main__":
    main()
