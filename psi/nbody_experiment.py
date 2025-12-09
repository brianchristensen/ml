"""
N-Body Gravitational Dynamics: Generate, Train, Evaluate

Generalized to handle 2-body, 3-body, 4-body, etc.
Test where HNN struggled (3-body chaos)!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# N-Body Data Generation
# ============================================================================

def nbody_dynamics(t, state, n_bodies):
    """
    Compute derivatives for N-body gravitational system.

    State: [n_bodies * 5] flattened array of [m, x, y, vx, vy] per body
    Returns: d/dt of state
    """
    state = state.reshape(n_bodies, 5)
    deriv = np.zeros_like(state)

    for i in range(n_bodies):
        m_i, x_i, y_i, vx_i, vy_i = state[i]
        deriv[i, 0] = 0  # dm/dt = 0
        deriv[i, 1] = vx_i  # dx/dt = vx
        deriv[i, 2] = vy_i  # dy/dt = vy

        # Gravitational acceleration from all other bodies
        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if i != j:
                m_j, x_j, y_j, vx_j, vy_j = state[j]
                dx = x_j - x_i
                dy = y_j - y_i
                r = np.sqrt(dx**2 + dy**2)

                # F = G*m1*m2/r^2, with G=1
                ax += m_j * dx / (r**3 + 1e-8)
                ay += m_j * dy / (r**3 + 1e-8)

        deriv[i, 3] = ax
        deriv[i, 4] = ay

    return deriv.flatten()


def random_nbody_config(n_bodies, orbit_noise=5e-2):
    """
    Generate random initial configuration for N-body system.
    Creates approximately circular orbits with perturbations.
    """
    state = np.zeros((n_bodies, 5))
    state[:, 0] = 1.0  # All bodies have mass = 1

    # Distribute bodies in a circle/ring
    for i in range(n_bodies):
        # Random radius and angle
        r = np.random.rand() * 0.5 + 0.75  # [0.75, 1.25]
        theta = 2 * np.pi * i / n_bodies + np.random.randn() * 0.3

        # Position
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        state[i, 1:3] = [x, y]

        # Velocity (perpendicular to radius, scaled for approximate orbit)
        vx = -y / (r**1.5) * (1 + orbit_noise * np.random.randn())
        vy = x / (r**1.5) * (1 + orbit_noise * np.random.randn())
        state[i, 3:5] = [vx, vy]

    return state


def compute_energy(state, n_bodies):
    """Compute total energy for N-body system."""
    state = state.reshape(n_bodies, 5)

    # Kinetic energy
    KE = 0.0
    for i in range(n_bodies):
        m, x, y, vx, vy = state[i]
        KE += 0.5 * m * (vx**2 + vy**2)

    # Potential energy
    PE = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            m_i, x_i, y_i = state[i, 0], state[i, 1], state[i, 2]
            m_j, x_j, y_j = state[j, 0], state[j, 1], state[j, 2]
            r = np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
            PE -= m_i * m_j / (r + 1e-8)

    return KE + PE


def generate_nbody_data(n_bodies=2, trials=1000, timesteps=50, t_span=[0, 20], orbit_noise=5e-2):
    """
    Generate N-body gravitational dynamics dataset.

    Returns:
        x: [trials, timesteps, n_bodies, 5]
        dx: [trials, timesteps, n_bodies, 5]
        h: [trials, timesteps] (energy)
    """
    print(f"Generating {trials} {n_bodies}-body trajectories...")
    print(f"  Timesteps: {timesteps}")
    print(f"  Time span: {t_span}")

    x = np.zeros((trials, timesteps, n_bodies, 5))
    dx = np.zeros_like(x)
    h = np.zeros((trials, timesteps))

    for trial in range(trials):
        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1}/{trials}")

        # Random initial condition
        state0 = random_nbody_config(n_bodies, orbit_noise=orbit_noise)

        # Integrate
        t_eval = np.linspace(t_span[0], t_span[1], timesteps)
        sol = solve_ivp(
            lambda t, y: nbody_dynamics(t, y, n_bodies),
            t_span,
            state0.flatten(),
            t_eval=t_eval,
            rtol=1e-9
        )

        # Store
        for i in range(timesteps):
            x[trial, i] = sol.y[:, i].reshape(n_bodies, 5)
            dx[trial, i] = nbody_dynamics(sol.t[i], sol.y[:, i], n_bodies).reshape(n_bodies, 5)
            h[trial, i] = compute_energy(sol.y[:, i], n_bodies)

    energy_conservation = np.mean(np.abs(np.diff(h, axis=1)))
    print(f"Energy conservation: {energy_conservation:.6f}")

    return x, dx, h


# ============================================================================
# Dataset
# ============================================================================

class NBodyDataset(Dataset):
    """N-body dynamics dataset."""

    def __init__(self, trajectories, energies, context_len=20, split='train', train_frac=0.8):
        # Extract positions and velocities (ignore constant mass)
        n_bodies = trajectories.shape[2]
        state_features = []
        for i in range(n_bodies):
            state_features.append(trajectories[:, :, i, 1:])  # [x, y, vx, vy]

        self.trajectories = np.concatenate(state_features, axis=-1).astype(np.float32)
        self.energies = energies
        self.n_bodies = n_bodies

        # Normalize
        self.mean = self.trajectories.mean(axis=(0, 1))
        self.std = self.trajectories.std(axis=(0, 1))
        self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        # Split
        num_train = int(len(self.trajectories) * train_frac)
        if split == 'train':
            self.trajectories = self.trajectories[:num_train]
            self.energies = self.energies[:num_train]
        else:
            self.trajectories = self.trajectories[num_train:]
            self.energies = self.energies[num_train:]

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

class PSIDynamicsPredictor(nn.Module):
    """PSI model for N-body dynamics."""

    def __init__(self, state_dim, dim=256, num_layers=8, max_len=50, device='cuda'):
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


def visualize_nbody(ground_truth, phi_predictions, n_bodies, sample_idx=0):
    """Visualize N-body trajectories."""
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    for i in range(n_bodies):
        gt_x = ground_truth[:, i, 1]
        gt_y = ground_truth[:, i, 2]
        axes[0].plot(gt_x, gt_y, color=colors[i % len(colors)], alpha=0.6, linewidth=2, label=f'Body {i+1}')
        axes[0].scatter(gt_x[0], gt_y[0], c=colors[i % len(colors)], s=100, marker='o', zorder=5)
    axes[0].set_title('Ground Truth (Scipy Solver)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # PSI predictions
    for i in range(n_bodies):
        pred_x = phi_predictions[:, i*4]
        pred_y = phi_predictions[:, i*4 + 1]
        axes[1].plot(pred_x, pred_y, color=colors[i % len(colors)], alpha=0.6, linewidth=2, label=f'Body {i+1}')
        axes[1].scatter(pred_x[0], pred_y[0], c=colors[i % len(colors)], s=100, marker='o', zorder=5)
    axes[1].set_title('PSI Predictions (No Solver!)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    # Overlay
    for i in range(n_bodies):
        gt_x = ground_truth[:, i, 1]
        gt_y = ground_truth[:, i, 2]
        pred_x = phi_predictions[:, i*4]
        pred_y = phi_predictions[:, i*4 + 1]
        axes[2].plot(gt_x, gt_y, color=colors[i % len(colors)], alpha=0.4, linewidth=3, label=f'GT {i+1}')
        axes[2].plot(pred_x, pred_y, color=colors[i % len(colors)], alpha=0.8, linewidth=2, linestyle='--', label=f'PSI {i+1}')
    axes[2].set_title('Overlay Comparison')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')

    plt.tight_layout()
    filename = f'{n_bodies}body_comparison_sample_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=2, help='Number of bodies')
    parser.add_argument('--trials', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('--timesteps', type=int, default=50, help='Timesteps per trajectory')
    parser.add_argument('--context_len', type=int, default=20, help='Context length')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PSI {args.n_bodies}-Body Gravitational Dynamics Experiment")
    print("=" * 80)
    print()

    # Generate data
    x, dx, h = generate_nbody_data(args.n_bodies, args.trials, args.timesteps)

    # Create datasets
    state_dim = args.n_bodies * 4  # Each body: x, y, vx, vy
    train_dataset = NBodyDataset(x, h, args.context_len, split='train')
    val_dataset = NBodyDataset(x, h, args.context_len, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = PSIDynamicsPredictor(state_dim, args.dim, args.num_layers, args.context_len, device).to(device)
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
            }, f'{args.n_bodies}body_best.pt')
            print(f"  Saved best model")

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating...")
    print("=" * 80)

    val_trajectories = x[int(len(x) * 0.8):]
    for i in range(min(3, len(val_trajectories))):
        gt_traj = val_trajectories[i]
        initial = np.concatenate([gt_traj[:args.context_len, j, 1:] for j in range(args.n_bodies)], axis=-1)

        phi_pred = predict_trajectory(model, initial, 30, train_dataset.mean, train_dataset.std, device)
        gt_viz = gt_traj[args.context_len:args.context_len+30]

        visualize_nbody(gt_viz, phi_pred, args.n_bodies, i)

        # MSE
        gt_pos = np.concatenate([gt_viz[:, j, 1:3] for j in range(args.n_bodies)], axis=-1)
        pred_indices = []
        for j in range(args.n_bodies):
            pred_indices.extend([j*4, j*4+1])
        pred_pos = phi_pred[:, pred_indices]
        mse = np.mean((gt_pos - pred_pos)**2)
        print(f"Sample {i} MSE: {mse:.6f}")

    print(f"\nBest val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
