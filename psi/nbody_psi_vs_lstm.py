"""
PSI vs LSTM Benchmark on 3-Body Problem

Tests whether PSI has any advantage over LSTM on chaotic gravitational dynamics.
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

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# =============================================================================
# Data Generation (from nbody_experiment.py)
# =============================================================================

def nbody_dynamics(t, state, n_bodies):
    """Compute derivatives for N-body gravitational system."""
    state = state.reshape(n_bodies, 5)
    deriv = np.zeros_like(state)

    for i in range(n_bodies):
        m_i, x_i, y_i, vx_i, vy_i = state[i]
        deriv[i, 0] = 0
        deriv[i, 1] = vx_i
        deriv[i, 2] = vy_i

        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if i != j:
                m_j, x_j, y_j, vx_j, vy_j = state[j]
                dx = x_j - x_i
                dy = y_j - y_i
                r = np.sqrt(dx**2 + dy**2)
                ax += m_j * dx / (r**3 + 1e-8)
                ay += m_j * dy / (r**3 + 1e-8)

        deriv[i, 3] = ax
        deriv[i, 4] = ay

    return deriv.flatten()


def random_nbody_config(n_bodies, orbit_noise=5e-2):
    """Generate random initial configuration for N-body system."""
    state = np.zeros((n_bodies, 5))
    state[:, 0] = 1.0

    for i in range(n_bodies):
        r = np.random.rand() * 0.5 + 0.75
        theta = 2 * np.pi * i / n_bodies + np.random.randn() * 0.3
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        state[i, 1:3] = [x, y]

        vx = -y / (r**1.5) * (1 + orbit_noise * np.random.randn())
        vy = x / (r**1.5) * (1 + orbit_noise * np.random.randn())
        state[i, 3:5] = [vx, vy]

    return state


def generate_nbody_data(n_bodies=3, trials=1000, timesteps=50, t_span=[0, 20]):
    """Generate N-body gravitational dynamics dataset."""
    print(f"Generating {trials} {n_bodies}-body trajectories...")

    x = np.zeros((trials, timesteps, n_bodies, 5))

    for trial in range(trials):
        if (trial + 1) % 200 == 0:
            print(f"  {trial + 1}/{trials}")

        state0 = random_nbody_config(n_bodies)
        t_eval = np.linspace(t_span[0], t_span[1], timesteps)

        sol = solve_ivp(
            lambda t, y: nbody_dynamics(t, y, n_bodies),
            t_span,
            state0.flatten(),
            t_eval=t_eval,
            rtol=1e-9
        )

        for i in range(timesteps):
            x[trial, i] = sol.y[:, i].reshape(n_bodies, 5)

    return x


# =============================================================================
# Dataset
# =============================================================================

class NBodyDataset(Dataset):
    def __init__(self, trajectories, context_len=20, split='train', train_frac=0.8):
        n_bodies = trajectories.shape[2]
        state_features = []
        for i in range(n_bodies):
            state_features.append(trajectories[:, :, i, 1:])

        self.trajectories = np.concatenate(state_features, axis=-1).astype(np.float32)
        self.n_bodies = n_bodies

        self.mean = self.trajectories.mean(axis=(0, 1))
        self.std = self.trajectories.std(axis=(0, 1))
        self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        num_train = int(len(self.trajectories) * train_frac)
        if split == 'train':
            self.trajectories = self.trajectories[:num_train]
        else:
            self.trajectories = self.trajectories[num_train:]

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


# =============================================================================
# Models
# =============================================================================

class PSIDynamicsPredictor(nn.Module):
    """PSI model for N-body dynamics."""

    def __init__(self, state_dim, dim=128, num_layers=6, max_len=50):
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


class ManualLSTMCell(nn.Module):
    """Pure PyTorch LSTM cell - no cuDNN optimization."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=-1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMDynamicsPredictor(nn.Module):
    """Manual LSTM model for N-body dynamics - no cuDNN, fair comparison."""

    def __init__(self, state_dim, dim=128, num_layers=6):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.num_layers = num_layers

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Manual LSTM cells instead of nn.LSTM
        self.cells = nn.ModuleList([
            ManualLSTMCell(dim, dim) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

    def forward(self, states):
        batch, seq_len, _ = states.shape
        x = self.state_embedding(states)

        # Initialize hidden states
        h = [torch.zeros(batch, self.dim, device=states.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.dim, device=states.device) for _ in range(self.num_layers)]

        # Sequential processing through time
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]

        final_state = h[-1]
        prediction = self.output_head(final_state)
        return prediction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """Train a model and return best validation loss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            prediction = model(context)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(device)
                target = batch['target'].to(device)
                prediction = model(context)
                val_loss += criterion(prediction, target).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return best_val_loss, history


def predict_trajectory(model, initial_states, num_steps, mean, std):
    """Rollout model predictions autoregressively."""
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


def compute_rollout_error(model, trajectories, context_len, mean, std, n_bodies, num_samples=20):
    """Compute autoregressive rollout error."""
    rollout_steps = 30
    errors = []

    for i in range(min(num_samples, len(trajectories))):
        gt_traj = trajectories[i]
        initial = np.concatenate([gt_traj[:context_len, j, 1:] for j in range(n_bodies)], axis=-1)

        pred = predict_trajectory(model, initial, rollout_steps, mean, std)
        gt_viz = gt_traj[context_len:context_len+rollout_steps]

        # Position error
        gt_pos = np.concatenate([gt_viz[:, j, 1:3] for j in range(n_bodies)], axis=-1)
        pred_indices = []
        for j in range(n_bodies):
            pred_indices.extend([j*4, j*4+1])
        pred_pos = pred[:, pred_indices]

        mse = np.mean((gt_pos - pred_pos)**2)
        errors.append(mse)

    return np.mean(errors), np.std(errors)


# =============================================================================
# Visualization
# =============================================================================

def visualize_comparison(gt_traj, psi_pred, lstm_pred, n_bodies, sample_idx=0):
    """Visualize PSI vs LSTM trajectories."""
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    ax = axes[0]
    for i in range(n_bodies):
        gt_x = gt_traj[:, i, 1]
        gt_y = gt_traj[:, i, 2]
        ax.plot(gt_x, gt_y, color=colors[i], alpha=0.7, linewidth=2, label=f'Body {i+1}')
        ax.scatter(gt_x[0], gt_y[0], c=colors[i], s=100, marker='o', zorder=5)
    ax.set_title('Ground Truth', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # PSI
    ax = axes[1]
    for i in range(n_bodies):
        pred_x = psi_pred[:, i*4]
        pred_y = psi_pred[:, i*4 + 1]
        ax.plot(pred_x, pred_y, color=colors[i], alpha=0.7, linewidth=2, label=f'Body {i+1}')
        ax.scatter(pred_x[0], pred_y[0], c=colors[i], s=100, marker='o', zorder=5)
    ax.set_title('PSI Predictions', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # LSTM
    ax = axes[2]
    for i in range(n_bodies):
        pred_x = lstm_pred[:, i*4]
        pred_y = lstm_pred[:, i*4 + 1]
        ax.plot(pred_x, pred_y, color=colors[i], alpha=0.7, linewidth=2, label=f'Body {i+1}')
        ax.scatter(pred_x[0], pred_y[0], c=colors[i], s=100, marker='o', zorder=5)
    ax.set_title('LSTM Predictions', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(f'nbody_psi_vs_lstm_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=3, help='Number of bodies')
    parser.add_argument('--trials', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('--timesteps', type=int, default=50, help='Timesteps per trajectory')
    parser.add_argument('--context_len', type=int, default=20, help='Context length')
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    print("=" * 80)
    print(f"PSI vs LSTM: {args.n_bodies}-Body Problem Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Generate data
    print("1. GENERATING DATA")
    print("-" * 40)
    x = generate_nbody_data(args.n_bodies, args.trials, args.timesteps)
    print()

    # Create datasets
    state_dim = args.n_bodies * 4
    train_dataset = NBodyDataset(x, args.context_len, split='train')
    val_dataset = NBodyDataset(x, args.context_len, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create models
    print("\n2. CREATING MODELS")
    print("-" * 40)

    psi_model = PSIDynamicsPredictor(state_dim, args.dim, args.num_layers, args.context_len).to(device)
    lstm_model = LSTMDynamicsPredictor(state_dim, args.dim, args.num_layers).to(device)

    print(f"PSI parameters:  {psi_model.count_parameters():,}")
    print(f"LSTM parameters: {lstm_model.count_parameters():,}")

    # Train PSI
    print("\n3. TRAINING PSI")
    print("-" * 40)
    start = time.time()
    psi_val_loss, psi_history = train_model(psi_model, train_loader, val_loader, args.epochs, args.lr)
    psi_time = time.time() - start
    print(f"PSI training time: {psi_time:.1f}s")
    print(f"PSI best val loss: {psi_val_loss:.6f}")

    # Train LSTM
    print("\n4. TRAINING LSTM")
    print("-" * 40)
    start = time.time()
    lstm_val_loss, lstm_history = train_model(lstm_model, train_loader, val_loader, args.epochs, args.lr)
    lstm_time = time.time() - start
    print(f"LSTM training time: {lstm_time:.1f}s")
    print(f"LSTM best val loss: {lstm_val_loss:.6f}")

    # Autoregressive rollout evaluation
    print("\n5. AUTOREGRESSIVE ROLLOUT EVALUATION")
    print("-" * 40)

    val_trajectories = x[int(len(x) * 0.8):]

    psi_rollout_mse, psi_rollout_std = compute_rollout_error(
        psi_model, val_trajectories, args.context_len,
        train_dataset.mean, train_dataset.std, args.n_bodies
    )
    lstm_rollout_mse, lstm_rollout_std = compute_rollout_error(
        lstm_model, val_trajectories, args.context_len,
        train_dataset.mean, train_dataset.std, args.n_bodies
    )

    print(f"PSI 30-step rollout MSE:  {psi_rollout_mse:.6f} (+/- {psi_rollout_std:.6f})")
    print(f"LSTM 30-step rollout MSE: {lstm_rollout_mse:.6f} (+/- {lstm_rollout_std:.6f})")

    # Visualize
    print("\n6. GENERATING VISUALIZATIONS")
    print("-" * 40)

    for i in range(min(3, len(val_trajectories))):
        gt_traj = val_trajectories[i]
        initial = np.concatenate([gt_traj[:args.context_len, j, 1:] for j in range(args.n_bodies)], axis=-1)

        psi_pred = predict_trajectory(psi_model, initial, 30, train_dataset.mean, train_dataset.std)
        lstm_pred = predict_trajectory(lstm_model, initial, 30, train_dataset.mean, train_dataset.std)
        gt_viz = gt_traj[args.context_len:args.context_len+30]

        visualize_comparison(gt_viz, psi_pred, lstm_pred, args.n_bodies, i)
        print(f"  Saved nbody_psi_vs_lstm_sample_{i}.png")

    # Learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(psi_history['val'], label='PSI', linewidth=2)
    plt.plot(lstm_history['val'], label='LSTM', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'{args.n_bodies}-Body Problem: Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nbody_psi_vs_lstm_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved nbody_psi_vs_lstm_learning_curves.png")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} {'PSI':<15} {'LSTM':<15} {'Winner':<10}")
    print("-" * 70)

    print(f"{'Parameters':<30} {psi_model.count_parameters():<15,} {lstm_model.count_parameters():<15,}", end="")
    print(f" {'PSI' if psi_model.count_parameters() < lstm_model.count_parameters() else 'LSTM':<10}")

    print(f"{'Training Time (s)':<30} {psi_time:<15.1f} {lstm_time:<15.1f}", end="")
    print(f" {'PSI' if psi_time < lstm_time else 'LSTM':<10}")

    print(f"{'Val Loss (1-step)':<30} {psi_val_loss:<15.6f} {lstm_val_loss:<15.6f}", end="")
    print(f" {'PSI' if psi_val_loss < lstm_val_loss else 'LSTM':<10}")

    print(f"{'Rollout MSE (30-step)':<30} {psi_rollout_mse:<15.6f} {lstm_rollout_mse:<15.6f}", end="")
    print(f" {'PSI' if psi_rollout_mse < lstm_rollout_mse else 'LSTM':<10}")

    print()
    if psi_rollout_mse < lstm_rollout_mse:
        improvement = (lstm_rollout_mse - psi_rollout_mse) / lstm_rollout_mse * 100
        print(f"PSI wins rollout by {improvement:.1f}%")
    else:
        improvement = (psi_rollout_mse - lstm_rollout_mse) / psi_rollout_mse * 100
        print(f"LSTM wins rollout by {improvement:.1f}%")


if __name__ == "__main__":
    main()
