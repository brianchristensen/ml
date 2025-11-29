"""
Dynamics PSI v2 - Proper comparison with Neural ODE-like approaches.

The key insight: We need to compare apples to apples.

Neural ODE: x(t) = x(0) + integral f(x(s)) ds
- Learns local dynamics f(x)
- Integrates sequentially (slow) or with adjoint method

Our goal: Learn f(x) but integrate via cumsum (parallel).

The challenge: True parallel integration requires knowing x(t) at all t,
but x(t) depends on the integral. This is the "chicken and egg" problem.

Solutions:
1. Teacher forcing: During training, use ground truth trajectory
2. Iterative refinement: Start with guess, refine
3. Sequential during inference, parallel during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, '.')
from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class VectorFieldMLP(nn.Module):
    """Simple MLP that learns dx/dt = f(x)."""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # Tanh for bounded gradients
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        return self.net(x)


class NeuralODEModel(nn.Module):
    """
    Neural ODE: Learn f(x) and integrate sequentially.
    This is the baseline we want to beat in speed while matching accuracy.
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.vector_field = VectorFieldMLP(state_dim, hidden_dim)
        self.dt_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x0, num_steps, dt=0.01):
        """Sequential integration (Euler method)."""
        trajectory = [x0]
        x = x0

        for _ in range(num_steps):
            dx_dt = self.vector_field(x)
            x = x + dx_dt * self.dt_scale * dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)


class TeacherForcedDynamics(nn.Module):
    """
    Teacher-forced training: Use ground truth states to predict derivatives.

    Training: Given trajectory x(0), x(1), ..., x(T), predict dx/dt at each point
    Inference: Sequential rollout using learned f(x)

    This allows parallel training but sequential inference.
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.vector_field = VectorFieldMLP(state_dim, hidden_dim)
        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward_train(self, trajectory, dt=0.01):
        """
        Training mode: Predict dx/dt at all trajectory points in parallel.

        Args:
            trajectory: [batch, seq_len, state_dim] - ground truth trajectory

        Returns:
            dx_dt_pred: [batch, seq_len-1, state_dim] - predicted derivatives
        """
        # Predict dx/dt at each point (parallel)
        dx_dt = self.vector_field(trajectory[:, :-1, :])
        return dx_dt

    def forward(self, x0, num_steps, dt=0.01):
        """Inference: Sequential rollout."""
        trajectory = [x0]
        x = x0

        for _ in range(num_steps):
            dx_dt = self.vector_field(x)
            x = x + dx_dt * self.dt_scale * dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)


class PSIDynamics(nn.Module):
    """
    PSI-based dynamics learning.

    Uses PSI blocks to process the state sequence and predict derivatives.
    The key difference: PSI can "see" context from the sequence when
    predicting derivatives (memory of past states).
    """

    def __init__(self, state_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.state_dim = state_dim

        # Project state to hidden
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # PSI blocks for processing
        self.blocks = nn.ModuleList([PSIBlock(hidden_dim) for _ in range(num_layers)])

        # Output dx/dt
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )

        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward_train(self, trajectory, dt=0.01):
        """Training: Process trajectory through PSI, predict dx/dt."""
        h = self.input_proj(trajectory[:, :-1, :])

        for block in self.blocks:
            h = block(h)

        dx_dt = self.output_proj(h)
        return dx_dt

    def forward(self, x0, num_steps, dt=0.01):
        """Inference: Sequential rollout (PSI reduces to single-step)."""
        trajectory = [x0]
        x = x0

        for _ in range(num_steps):
            # Single step through PSI
            h = self.input_proj(x.unsqueeze(1))
            for block in self.blocks:
                h = block(h)
            dx_dt = self.output_proj(h).squeeze(1)

            x = x + dx_dt * self.dt_scale * dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)


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


class LSTMDynamics(nn.Module):
    """Manual LSTM baseline for dynamics learning - no cuDNN, fair comparison."""

    def __init__(self, state_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ManualLSTMCell(state_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward_train(self, trajectory, dt=0.01):
        """Training: Process trajectory, predict dx/dt."""
        batch, seq_len, _ = trajectory[:, :-1, :].shape
        h = [torch.zeros(batch, self.hidden_dim, device=trajectory.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_dim, device=trajectory.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = trajectory[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            outputs.append(h[-1])

        out = torch.stack(outputs, dim=1)
        dx_dt = self.output_proj(out)
        return dx_dt

    def forward(self, x0, num_steps, dt=0.01):
        """Inference: Sequential rollout."""
        trajectory = [x0]
        x = x0
        h = [torch.zeros(x0.shape[0], self.hidden_dim, device=x0.device) for _ in range(self.num_layers)]
        c = [torch.zeros(x0.shape[0], self.hidden_dim, device=x0.device) for _ in range(self.num_layers)]

        for _ in range(num_steps):
            inp = x
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            dx_dt = self.output_proj(h[-1])
            x = x + dx_dt * self.dt_scale * dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)


# =============================================================================
# Data Generation
# =============================================================================

def lorenz_dynamics(state, t, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_lorenz_data(n_trajectories, n_steps, dt=0.01):
    trajectories = []
    for _ in range(n_trajectories):
        x0 = np.random.randn(3) * 5 + np.array([0, 0, 25])
        t = np.linspace(0, n_steps * dt, n_steps + 1)
        traj = odeint(lorenz_dynamics, x0, t)
        trajectories.append(traj)
    return torch.tensor(np.array(trajectories), dtype=torch.float32)


# =============================================================================
# Training
# =============================================================================

def compute_derivative_loss(model, trajectory, dt=0.01):
    """
    Loss: How well does predicted dx/dt match actual dx/dt from trajectory?
    """
    # Actual derivatives (finite difference)
    actual_dx_dt = (trajectory[:, 1:, :] - trajectory[:, :-1, :]) / dt

    # Predicted derivatives
    pred_dx_dt = model.forward_train(trajectory, dt=dt)

    return F.mse_loss(pred_dx_dt, actual_dx_dt)


def compute_trajectory_loss(model, trajectory, dt=0.01):
    """
    Loss: How well does integrated trajectory match ground truth?
    """
    x0 = trajectory[:, 0, :]
    num_steps = trajectory.shape[1] - 1

    pred_traj = model(x0, num_steps, dt=dt)

    return F.mse_loss(pred_traj, trajectory)


def train_model(model, train_data, val_data, epochs=100, lr=1e-3, dt=0.01,
                use_derivative_loss=True, use_trajectory_loss=True):
    """Train a dynamics model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if use_derivative_loss:
            loss = loss + compute_derivative_loss(model, train_data, dt)
        if use_trajectory_loss:
            # Only use trajectory loss occasionally (expensive)
            if epoch % 5 == 0:
                loss = loss + 0.1 * compute_trajectory_loss(model, train_data, dt)

        if not use_derivative_loss and not use_trajectory_loss:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = compute_trajectory_loss(model, val_data, dt)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train_loss={loss.item():.6f}, val_traj_loss={val_loss.item():.6f}")

    return best_val_loss.item()


def evaluate_rollout(model, test_data, dt=0.01, rollout_steps=None):
    """Evaluate rollout accuracy."""
    model.eval()
    test_data = test_data.to(device)

    if rollout_steps is None:
        rollout_steps = test_data.shape[1] - 1

    x0 = test_data[:, 0, :]
    target = test_data[:, :rollout_steps + 1, :]

    with torch.no_grad():
        pred = model(x0, rollout_steps, dt=dt)

    mse = F.mse_loss(pred, target).item()
    return mse, pred.cpu(), target.cpu()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("DYNAMICS LEARNING COMPARISON")
    print("="*70)
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating Lorenz data...")
    dt = 0.01
    train_steps = 100
    extrap_steps = 200

    train_data = generate_lorenz_data(500, train_steps, dt=dt)
    val_data = generate_lorenz_data(100, train_steps, dt=dt)
    test_data = generate_lorenz_data(100, extrap_steps, dt=dt)

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Models to compare
    models = {
        'NeuralODE': NeuralODEModel(state_dim=3, hidden_dim=64).to(device),
        'TeacherForced': TeacherForcedDynamics(state_dim=3, hidden_dim=64).to(device),
        'PSIDynamics': PSIDynamics(state_dim=3, hidden_dim=64, num_layers=2).to(device),
        'LSTMDynamics': LSTMDynamics(state_dim=3, hidden_dim=64, num_layers=2).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Determine loss type
        if name == 'NeuralODE':
            # NeuralODE uses trajectory loss directly
            best_val = train_model(model, train_data, val_data, epochs=100, lr=1e-3, dt=dt,
                                   use_derivative_loss=False, use_trajectory_loss=True)
        else:
            # Others use derivative loss (parallelizable)
            best_val = train_model(model, train_data, val_data, epochs=100, lr=1e-3, dt=dt,
                                   use_derivative_loss=True, use_trajectory_loss=True)

        # Evaluate
        train_mse, _, _ = evaluate_rollout(model, test_data, dt=dt, rollout_steps=train_steps)
        extrap_mse, pred, target = evaluate_rollout(model, test_data, dt=dt, rollout_steps=extrap_steps)

        results[name] = {
            'best_val': best_val,
            'train_mse': train_mse,
            'extrap_mse': extrap_mse,
            'pred': pred,
            'target': target
        }

        print(f"\n  Val loss:          {best_val:.6f}")
        print(f"  Test (100 steps):  {train_mse:.6f}")
        print(f"  Test (200 steps):  {extrap_mse:.6f}")
        print(f"  Extrap ratio:      {extrap_mse / train_mse:.2f}x")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Val':<12} {'100-step':<12} {'200-step':<12} {'Extrap':<10}")
    print("-"*70)
    for name, r in results.items():
        extrap = r['extrap_mse'] / r['train_mse'] if r['train_mse'] > 0 else float('inf')
        print(f"{name:<20} {r['best_val']:<12.4f} {r['train_mse']:<12.4f} {r['extrap_mse']:<12.4f} {extrap:<10.2f}x")

    # Find winner
    best_train = min(results.items(), key=lambda x: x[1]['train_mse'])
    best_extrap = min(results.items(), key=lambda x: x[1]['extrap_mse'])

    print(f"\nBest at 100-step: {best_train[0]} ({best_train[1]['train_mse']:.4f})")
    print(f"Best at 200-step: {best_extrap[0]} ({best_extrap[1]['extrap_mse']:.4f})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, r) in enumerate(results.items()):
        ax = axes.flat[idx]
        ax.plot(r['target'][0, :, 0], 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(r['pred'][0, :, 0], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax.axvline(x=train_steps, color='g', linestyle=':', label='Train horizon', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('x')
        ax.set_title(f'{name} (MSE: {r["extrap_mse"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dynamics_comparison.png', dpi=150)
    plt.close()
    print("\nSaved dynamics_comparison.png")


if __name__ == "__main__":
    main()
