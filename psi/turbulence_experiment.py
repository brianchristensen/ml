"""
2D Navier-Stokes Turbulence Experiment for PSI

Generates 2D incompressible Navier-Stokes data using pseudo-spectral methods.
This is "real" turbulence - the same equations used in CFD research.

Equation (vorticity form):
    ∂ω/∂t + (u·∇)ω = ν∇²ω + f

where:
    ω = ∇×u is vorticity (scalar in 2D)
    ν is kinematic viscosity (controls Reynolds number)
    f is optional forcing (for sustained turbulence)

References:
- Kolmogorov flow: sinusoidal forcing, classic turbulence benchmark
- Decaying turbulence: random IC, no forcing, energy decays over time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import hashlib
import os

from psi import PSIBlock


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Pseudo-Spectral Navier-Stokes Solver
# ============================================================================

class SpectralNavierStokes2D:
    """
    Pseudo-spectral solver for 2D incompressible Navier-Stokes.

    Solves in vorticity-streamfunction formulation:
        ∂ω/∂t = -u·∇ω + ν∇²ω + f
        ∇²ψ = -ω
        u = ∂ψ/∂y, v = -∂ψ/∂x

    Uses 2/3 dealiasing rule and RK4 time integration.
    """

    def __init__(self, N=64, L=2*np.pi, nu=1e-3, dt=0.01, forcing='kolmogorov', forcing_k=4):
        """
        Args:
            N: Grid resolution (N x N)
            L: Domain size (L x L), default 2π for periodic
            nu: Kinematic viscosity (smaller = higher Reynolds number)
            dt: Time step
            forcing: 'kolmogorov', 'random', or None (decaying)
            forcing_k: Forcing wavenumber for Kolmogorov flow
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt
        self.forcing_type = forcing
        self.forcing_k = forcing_k

        # Grid
        self.dx = L / N
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # Wavenumbers
        self.kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.ky = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1  # Avoid division by zero

        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias = np.ones((N, N))
        self.dealias[np.abs(self.KX) > kmax * 2 * np.pi / L] = 0
        self.dealias[np.abs(self.KY) > kmax * 2 * np.pi / L] = 0

        # Precompute forcing
        self._setup_forcing()

    def _setup_forcing(self):
        """Setup forcing term."""
        if self.forcing_type == 'kolmogorov':
            # Kolmogorov flow: f = sin(k*y)
            self.forcing = np.sin(self.forcing_k * self.yy)
        elif self.forcing_type == 'random':
            # Random forcing at low wavenumbers
            f_hat = np.zeros((self.N, self.N), dtype=complex)
            k_force = 4
            mask = (np.abs(self.KX) <= k_force * 2 * np.pi / self.L) & \
                   (np.abs(self.KY) <= k_force * 2 * np.pi / self.L) & \
                   (self.K2 > 0)
            f_hat[mask] = np.random.randn(mask.sum()) + 1j * np.random.randn(mask.sum())
            self.forcing = np.real(np.fft.ifft2(f_hat))
            self.forcing *= 0.1 / (np.std(self.forcing) + 1e-8)
        else:
            self.forcing = np.zeros((self.N, self.N))

    def random_initial_condition(self, energy_spectrum='k2', seed=None):
        """
        Generate random initial vorticity field.

        Args:
            energy_spectrum: 'k2' (peaks at k~2), 'k4', or 'flat'
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        # Random phases
        omega_hat = np.zeros((self.N, self.N), dtype=complex)
        phases = np.random.uniform(0, 2*np.pi, (self.N, self.N))

        # Energy spectrum
        k_mag = np.sqrt(self.K2)
        if energy_spectrum == 'k2':
            # E(k) peaks around k=2
            amplitude = k_mag * np.exp(-k_mag / 2)
        elif energy_spectrum == 'k4':
            # E(k) peaks around k=4
            amplitude = k_mag * np.exp(-k_mag / 4)
        else:
            # Flat spectrum up to some kmax
            amplitude = np.ones_like(k_mag)
            amplitude[k_mag > 10] = 0

        omega_hat = amplitude * np.exp(1j * phases)
        omega_hat[0, 0] = 0  # Zero mean
        omega_hat *= self.dealias

        omega = np.real(np.fft.ifft2(omega_hat))

        # Normalize to reasonable amplitude
        omega *= 5.0 / (np.std(omega) + 1e-8)

        return omega

    def vorticity_to_velocity(self, omega):
        """Compute velocity from vorticity via streamfunction."""
        omega_hat = np.fft.fft2(omega)

        # Solve ∇²ψ = -ω for streamfunction
        psi_hat = -omega_hat / self.K2
        psi_hat[0, 0] = 0

        # u = ∂ψ/∂y, v = -∂ψ/∂x
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat

        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))

        return u, v

    def rhs(self, omega):
        """Compute right-hand side: -u·∇ω + ν∇²ω + f"""
        omega_hat = np.fft.fft2(omega)

        # Velocity from vorticity
        u, v = self.vorticity_to_velocity(omega)

        # Gradients of vorticity
        domega_dx = np.real(np.fft.ifft2(1j * self.KX * omega_hat))
        domega_dy = np.real(np.fft.ifft2(1j * self.KY * omega_hat))

        # Advection term (with dealiasing)
        advection = u * domega_dx + v * domega_dy
        advection_hat = np.fft.fft2(advection) * self.dealias
        advection = np.real(np.fft.ifft2(advection_hat))

        # Diffusion term
        diffusion = np.real(np.fft.ifft2(-self.nu * self.K2 * omega_hat))

        return -advection + diffusion + self.forcing

    def step_rk4(self, omega):
        """RK4 time integration."""
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * self.dt * k1)
        k3 = self.rhs(omega + 0.5 * self.dt * k2)
        k4 = self.rhs(omega + self.dt * k3)

        return omega + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def simulate(self, omega0, n_steps, save_every=1):
        """
        Run simulation from initial condition.

        Args:
            omega0: Initial vorticity field
            n_steps: Number of time steps
            save_every: Save every N steps

        Returns:
            trajectory: Array of shape (n_saved, N, N)
        """
        omega = omega0.copy()
        trajectory = [omega.copy()]

        for step in range(n_steps):
            omega = self.step_rk4(omega)

            if (step + 1) % save_every == 0:
                trajectory.append(omega.copy())

        return np.array(trajectory)


# ============================================================================
# Data Generation
# ============================================================================

def generate_turbulence_data(n_samples=1000, grid_size=64, n_timesteps=50,
                              nu=1e-3, forcing='kolmogorov',
                              dt=0.01, sim_steps_per_frame=10,
                              cache_dir='data/turbulence', use_cache=True):
    """
    Generate 2D Navier-Stokes turbulence dataset.

    Args:
        n_samples: Number of trajectories
        grid_size: Spatial resolution (N x N)
        n_timesteps: Number of frames per trajectory
        nu: Viscosity (1e-3 = moderate turbulence, 1e-4 = high Re)
        forcing: 'kolmogorov', 'random', or None
        dt: Solver time step
        sim_steps_per_frame: Simulation steps between saved frames
        cache_dir: Directory to cache generated data
        use_cache: Whether to use cached data if available

    Returns:
        trajectories: (n_samples, n_timesteps, n_points, 1)
        points: (n_samples, n_points, 2)
    """
    # Create cache key from parameters
    cache_key = f"ns2d_n{n_samples}_g{grid_size}_t{n_timesteps}_nu{nu}_f{forcing}_dt{dt}_spf{sim_steps_per_frame}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_file = Path(cache_dir) / f"turbulence_{cache_hash}.npz"

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading cached turbulence data from {cache_file}")
        data = np.load(cache_file)
        return data['trajectories'], data['points']

    print(f"Generating {n_samples} 2D Navier-Stokes turbulence simulations...")
    print(f"  Grid: {grid_size}x{grid_size}, {n_timesteps} frames")
    print(f"  Viscosity: {nu} (Re ~ {1/nu:.0f})")
    print(f"  Forcing: {forcing}")
    print(f"  Solver: dt={dt}, {sim_steps_per_frame} steps/frame")

    solver = SpectralNavierStokes2D(
        N=grid_size,
        nu=nu,
        dt=dt,
        forcing=forcing
    )

    trajectories = []

    start_time = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"  Generated {i + 1}/{n_samples} ({rate:.1f}/s, ETA: {eta:.0f}s)")

        # Random initial condition
        omega0 = solver.random_initial_condition(seed=i)

        # Spinup: let transients die out
        spinup_steps = 100
        omega = omega0
        for _ in range(spinup_steps):
            omega = solver.step_rk4(omega)

        # Generate trajectory
        total_sim_steps = (n_timesteps - 1) * sim_steps_per_frame
        traj = solver.simulate(omega, total_sim_steps, save_every=sim_steps_per_frame)

        # traj shape: (n_timesteps, grid_size, grid_size)
        trajectories.append(traj)

    trajectories = np.array(trajectories)  # (n_samples, n_timesteps, grid_size, grid_size)

    # Reshape to (n_samples, n_timesteps, n_points, n_features)
    n_points = grid_size * grid_size
    trajectories = trajectories.reshape(n_samples, n_timesteps, n_points, 1)

    # Create grid points
    x = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    y = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    points = np.tile(points[np.newaxis, :, :], (n_samples, 1, 1))

    print(f"\nGenerated {n_samples} simulations in {time.time() - start_time:.1f}s")
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  Points shape: {points.shape}")

    # Cache to disk
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, trajectories=trajectories, points=points)
        print(f"  Cached to {cache_file}")

    return trajectories, points


# ============================================================================
# Dataset
# ============================================================================

class TurbulenceDataset(Dataset):
    """PyTorch Dataset for turbulence dynamics."""

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
        context = traj_norm[pos:pos + self.lookback]
        target = traj_norm[pos + self.lookback]

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
    """PSI model for turbulence dynamics prediction."""

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


def visualize_turbulence(ground_truth, predictions, grid_size, sample_idx=0):
    """Visualize vorticity evolution: ground truth vs predictions."""

    n_steps = min(len(ground_truth), len(predictions))

    # Select time steps to visualize
    vis_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    vis_steps = sorted(set([max(0, min(s, n_steps - 1)) for s in vis_steps]))

    fig, axes = plt.subplots(2, len(vis_steps), figsize=(4 * len(vis_steps), 8))

    if len(vis_steps) == 1:
        axes = axes.reshape(2, 1)

    for i, step in enumerate(vis_steps):
        gt = ground_truth[step].reshape(grid_size, grid_size)
        pred = predictions[step].reshape(grid_size, grid_size)

        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        # Ground truth
        im1 = axes[0, i].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, i].set_title(f'GT t={step}')
        if i == 0:
            axes[0, i].set_ylabel('Ground Truth')
        plt.colorbar(im1, ax=axes[0, i])

        # Predictions
        im2 = axes[1, i].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, i].set_title(f'PSI t={step}')
        if i == 0:
            axes[1, i].set_ylabel('PSI Prediction')
        plt.colorbar(im2, ax=axes[1, i])

    plt.suptitle(f'2D Navier-Stokes Turbulence - Sample {sample_idx}')
    plt.tight_layout()

    filename = f'turbulence_comparison_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def plot_error_over_time(errors, title='Turbulence'):
    """Plot prediction error vs rollout step."""
    plt.figure(figsize=(10, 5))
    plt.plot(errors, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Rollout Step')
    plt.ylabel('MSE')
    plt.title(f'{title} - Error vs Rollout Step')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    filename = 'turbulence_error_rollout.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def visualize_sample_field(solver, omega, title="Vorticity Field"):
    """Quick visualization of a single vorticity field."""
    plt.figure(figsize=(8, 6))
    plt.imshow(omega, cmap='RdBu_r', origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi])
    plt.colorbar(label='Vorticity ω')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('turbulence_sample.png', dpi=150)
    print("Saved turbulence_sample.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='2D Navier-Stokes Turbulence Experiment with PSI')
    parser.add_argument('--n_samples', type=int, default=20000, help='Number of trajectories')
    parser.add_argument('--grid_size', type=int, default=32, help='Spatial grid size')
    parser.add_argument('--n_timesteps', type=int, default=50, help='Frames per trajectory')
    parser.add_argument('--nu', type=float, default=1e-3, help='Viscosity (1e-3=moderate, 1e-4=high Re)')
    parser.add_argument('--forcing', type=str, default='kolmogorov',
                        choices=['kolmogorov', 'random', 'none'], help='Forcing type')
    parser.add_argument('--lookback', type=int, default=4, help='Context frames')
    parser.add_argument('--rollout', type=int, default=16, help='Prediction frames')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of PSI layers')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no_cache', action='store_true', help='Disable data caching')
    parser.add_argument('--generate_only', action='store_true', help='Only generate data, no training')
    args = parser.parse_args()

    forcing = None if args.forcing == 'none' else args.forcing

    print("=" * 80)
    print("PSI 2D Navier-Stokes Turbulence Experiment")
    print("=" * 80)
    print()
    print(f"Samples: {args.n_samples}")
    print(f"Grid: {args.grid_size}x{args.grid_size}, {args.n_timesteps} frames")
    print(f"Viscosity: {args.nu} (Re ~ {1/args.nu:.0f})")
    print(f"Forcing: {args.forcing}")
    print()

    # Generate/load data
    trajectories, points = generate_turbulence_data(
        n_samples=args.n_samples,
        grid_size=args.grid_size,
        n_timesteps=args.n_timesteps,
        nu=args.nu,
        forcing=forcing,
        use_cache=not args.no_cache
    )

    if args.generate_only:
        print("\nData generation complete. Exiting (--generate_only).")
        return

    print(f"\nLookback: {args.lookback}, Rollout: {args.rollout}")
    print()

    # Create datasets
    train_dataset = TurbulenceDataset(
        trajectories, points,
        lookback=args.lookback,
        rollout=args.rollout,
        split='train'
    )

    val_dataset = TurbulenceDataset(
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
    grid_size = args.grid_size

    model = PSIDynamicsPredictor(
        state_dim=state_dim,
        dim=args.dim,
        num_layers=args.num_layers,
        max_len=args.lookback + args.rollout,
        device=device
    ).to(device)

    print()
    print(f"State dimension: {state_dim} ({grid_size}x{grid_size})")
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
                'args': vars(args),
            }, 'turbulence_best.pt')
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

    # Evaluation
    print()
    print("=" * 80)
    print("Evaluation")
    print("=" * 80)

    # Load best model
    checkpoint = torch.load('turbulence_best.pt', weights_only=False)
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
        std_tiled = np.tile(train_dataset.std, train_dataset.n_points)
        mean_tiled = np.tile(train_dataset.mean, train_dataset.n_points)
        predictions_denorm = predictions * std_tiled + mean_tiled
        gt_denorm = traj[args.lookback:args.lookback + args.rollout].reshape(args.rollout, -1)

        visualize_turbulence(gt_denorm, predictions_denorm, grid_size, sample_idx)

    # Plot average error
    avg_errors = np.mean(all_errors, axis=0)
    plot_error_over_time(avg_errors)

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"2D Navier-Stokes (Re ~ {1/args.nu:.0f}, forcing: {args.forcing})")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Average 1-step MSE: {avg_errors[0]:.6f}")
    print(f"Average {len(avg_errors)}-step MSE: {avg_errors[-1]:.6f}")
    print()
    print("Generated artifacts:")
    print("  - turbulence_best.pt (model checkpoint)")
    print("  - turbulence_comparison_*.png (visualizations)")
    print("  - turbulence_error_rollout.png (error plot)")


if __name__ == "__main__":
    main()
