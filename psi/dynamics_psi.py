"""
Dynamics PSI - A neural ODE alternative that learns vector fields
and uses cumsum for parallel trajectory integration.

Key idea: Instead of sequential ODE solving, we:
1. Learn a vector field f(x) that predicts dx/dt
2. Compute all dx/dt values in parallel for a trajectory
3. Use cumsum to integrate: x(t) = x(0) + cumsum(dx/dt * dt)

The trick for parallelization: We iterate the vector field evaluation
a fixed number of times, allowing the model to refine its predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class VectorFieldNet(nn.Module):
    """Learns the vector field dx/dt = f(x)."""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Per-dimension integration scale (learned dt per dimension)
        self.integration_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x):
        """Compute dx/dt for state x."""
        return self.net(x)


class DynamicsPSI(nn.Module):
    """
    Dynamics PSI: Parallel trajectory generation via vector field learning.

    Instead of sequential ODE solving, we:
    1. Start with initial condition x0
    2. Predict entire trajectory of dx/dt values
    3. Integrate via cumsum

    For training parallelization, we use "teacher forcing" on the trajectory:
    given ground truth trajectory, predict dx/dt at each point.
    """

    def __init__(self, state_dim, hidden_dim=64, num_refine_steps=3):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_refine_steps = num_refine_steps

        # Vector field network
        self.vector_field = VectorFieldNet(state_dim, hidden_dim)

        # Refinement: iteratively improve dx/dt predictions
        # This allows the model to "look ahead" and correct predictions
        self.refine_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # current dx/dt + state
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward_trajectory(self, x0, num_steps, dt=0.01):
        """
        Generate trajectory from initial condition.

        Args:
            x0: Initial state [batch, state_dim]
            num_steps: Number of integration steps
            dt: Time step

        Returns:
            trajectory: [batch, num_steps+1, state_dim]
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Initialize trajectory with x0
        trajectory = torch.zeros(batch_size, num_steps + 1, self.state_dim, device=device)
        trajectory[:, 0] = x0

        # Sequential generation (for inference/rollout)
        x = x0
        for t in range(num_steps):
            dx_dt = self.vector_field(x)
            # Apply learned per-dimension scaling
            dx = dx_dt * self.vector_field.integration_scale * dt
            x = x + dx
            trajectory[:, t + 1] = x

        return trajectory

    def forward_parallel(self, trajectory_input):
        """
        Parallel training mode: given input trajectory, predict dx/dt at each point.

        This is like "teacher forcing" - we use the ground truth trajectory
        to compute what dx/dt should be at each point.

        Args:
            trajectory_input: [batch, seq_len, state_dim] - input trajectory

        Returns:
            dx_dt_pred: [batch, seq_len, state_dim] - predicted derivatives
        """
        # Compute vector field at each point in parallel
        dx_dt = self.vector_field(trajectory_input)

        # Optional: refinement steps
        for _ in range(self.num_refine_steps):
            refine_input = torch.cat([dx_dt, trajectory_input], dim=-1)
            dx_dt = dx_dt + self.refine_net(refine_input)

        return dx_dt

    def forward_cumsum(self, x0, dx_dt_sequence, dt=0.01):
        """
        Integrate dx/dt sequence using cumsum (parallel integration).

        Args:
            x0: Initial state [batch, state_dim]
            dx_dt_sequence: [batch, seq_len, state_dim] - derivatives
            dt: Time step

        Returns:
            trajectory: [batch, seq_len+1, state_dim]
        """
        # Scale by dt and learned integration scale
        scaled_dx = dx_dt_sequence * self.vector_field.integration_scale * dt

        # Cumulative sum for integration
        cumulative = torch.cumsum(scaled_dx, dim=1)

        # Add initial condition
        trajectory = x0.unsqueeze(1) + torch.cat([
            torch.zeros_like(x0).unsqueeze(1),
            cumulative
        ], dim=1)

        return trajectory


class DynamicsPSIv2(nn.Module):
    """
    Version 2: Fully parallel trajectory prediction.

    Key insight: If we know the trajectory length, we can predict
    all dx/dt values at once by conditioning on position in time.

    x(t) = x(0) + integral_0^t f(x(s), s) ds

    We approximate by predicting f at discrete times and using cumsum.
    """

    def __init__(self, state_dim, hidden_dim=64, max_steps=200):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Vector field conditioned on time and initial state
        self.vector_field = nn.Sequential(
            nn.Linear(state_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Initial state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Learned integration scale
        self.integration_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x0, num_steps, dt=0.01):
        """
        Generate trajectory in parallel.

        Args:
            x0: [batch, state_dim] initial conditions
            num_steps: number of steps to predict
            dt: time step

        Returns:
            trajectory: [batch, num_steps+1, state_dim]
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Encode initial state
        x0_encoded = self.state_encoder(x0)  # [batch, hidden]

        # Create time embeddings for all steps
        times = torch.arange(num_steps, device=device).float() * dt
        times = times.view(1, -1, 1).expand(batch_size, -1, -1)  # [batch, steps, 1]
        time_embeds = self.time_embed(times)  # [batch, steps, hidden]

        # Expand x0 encoding for all time steps
        x0_expanded = x0_encoded.unsqueeze(1).expand(-1, num_steps, -1)  # [batch, steps, hidden]
        x0_state = x0.unsqueeze(1).expand(-1, num_steps, -1)  # [batch, steps, state_dim]

        # Predict dx/dt at all times (conditioned on x0 and t)
        vf_input = torch.cat([x0_state, x0_expanded, time_embeds], dim=-1)
        dx_dt = self.vector_field(vf_input)  # [batch, steps, state_dim]

        # Integrate via cumsum
        scaled_dx = dx_dt * self.integration_scale * dt
        cumulative = torch.cumsum(scaled_dx, dim=1)

        # Build trajectory: x0, x0+dx1, x0+dx1+dx2, ...
        trajectory = torch.cat([
            x0.unsqueeze(1),
            x0.unsqueeze(1) + cumulative
        ], dim=1)

        return trajectory


class DynamicsPSIv3(nn.Module):
    """
    Version 3: Iterative refinement with parallel integration.

    Idea: Start with a rough trajectory estimate, then iteratively
    refine using the vector field. Each iteration uses cumsum.

    This is like implicit Euler or fixed-point iteration.
    """

    def __init__(self, state_dim, hidden_dim=64, num_iterations=3):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        # Vector field: dx/dt = f(x)
        self.vector_field = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Learned integration scale
        self.integration_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x0, num_steps, dt=0.01):
        """
        Generate trajectory via iterative refinement.

        1. Initialize with linear extrapolation from x0
        2. Compute vector field at all points
        3. Re-integrate via cumsum
        4. Repeat
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Initial guess: just repeat x0
        trajectory = x0.unsqueeze(1).expand(-1, num_steps + 1, -1).clone()

        for iteration in range(self.num_iterations):
            # Compute vector field at all trajectory points (except last)
            dx_dt = self.vector_field(trajectory[:, :-1, :])  # [batch, steps, state_dim]

            # Scale and integrate
            scaled_dx = dx_dt * self.integration_scale * dt
            cumulative = torch.cumsum(scaled_dx, dim=1)

            # Update trajectory (keep x0 fixed)
            trajectory = torch.cat([
                x0.unsqueeze(1),
                x0.unsqueeze(1) + cumulative
            ], dim=1)

        return trajectory


# =============================================================================
# Benchmark against Neural ODE
# =============================================================================

def lorenz_dynamics(state, t, sigma=10.0, rho=28.0, beta=8/3):
    """True Lorenz dynamics for ground truth."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def generate_lorenz_data(n_trajectories, n_steps, dt=0.01):
    """Generate Lorenz trajectories for training."""
    trajectories = []

    for _ in range(n_trajectories):
        # Random initial condition
        x0 = np.random.randn(3) * 5 + np.array([0, 0, 25])

        # Integrate
        t = np.linspace(0, n_steps * dt, n_steps + 1)
        traj = odeint(lorenz_dynamics, x0, t)
        trajectories.append(traj)

    return torch.tensor(np.array(trajectories), dtype=torch.float32)


def train_dynamics_model(model, train_data, val_data, epochs=100, lr=1e-3, dt=0.01):
    """Train a dynamics model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    device = next(model.parameters()).device
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()

        # Get initial conditions and full trajectories
        x0 = train_data[:, 0, :]
        target_traj = train_data
        num_steps = train_data.shape[1] - 1

        # Forward pass
        optimizer.zero_grad()
        pred_traj = model(x0, num_steps, dt=dt)

        # Loss on full trajectory
        loss = F.mse_loss(pred_traj, target_traj)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['train'].append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            x0_val = val_data[:, 0, :]
            pred_val = model(x0_val, val_data.shape[1] - 1, dt=dt)
            val_loss = F.mse_loss(pred_val, val_data)
            history['val'].append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train={loss.item():.6f}, val={val_loss.item():.6f}")

    return best_val_loss, history


def test_rollout(model, test_data, dt=0.01, rollout_steps=None):
    """Test rollout accuracy."""
    model.eval()
    device = next(model.parameters()).device
    test_data = test_data.to(device)

    if rollout_steps is None:
        rollout_steps = test_data.shape[1] - 1

    x0 = test_data[:, 0, :]
    target = test_data[:, :rollout_steps + 1, :]

    with torch.no_grad():
        pred = model(x0, rollout_steps, dt=dt)

    mse = F.mse_loss(pred, target).item()
    return mse, pred.cpu(), target.cpu()


def main():
    print("="*70)
    print("DYNAMICS PSI BENCHMARK")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating Lorenz data...")
    dt = 0.01
    train_steps = 100
    test_steps = 200  # Longer for extrapolation test

    train_data = generate_lorenz_data(500, train_steps, dt=dt)
    val_data = generate_lorenz_data(100, train_steps, dt=dt)
    test_data = generate_lorenz_data(100, test_steps, dt=dt)

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Test different model versions
    models = {
        'DynamicsPSI_v2': DynamicsPSIv2(state_dim=3, hidden_dim=64).to(device),
        'DynamicsPSI_v3': DynamicsPSIv3(state_dim=3, hidden_dim=64, num_iterations=3).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Train
        best_val, history = train_dynamics_model(
            model, train_data, val_data,
            epochs=100, lr=1e-3, dt=dt
        )

        # Test on training length
        train_len_mse, _, _ = test_rollout(model, test_data, dt=dt, rollout_steps=train_steps)

        # Test on longer rollout (extrapolation)
        extrap_mse, pred, target = test_rollout(model, test_data, dt=dt, rollout_steps=test_steps)

        results[name] = {
            'best_val': best_val,
            'train_len_mse': train_len_mse,
            'extrap_mse': extrap_mse,
            'pred': pred,
            'target': target
        }

        print(f"\nResults for {name}:")
        print(f"  Val loss:        {best_val:.6f}")
        print(f"  Test (100 steps): {train_len_mse:.6f}")
        print(f"  Test (200 steps): {extrap_mse:.6f}")
        print(f"  Extrapolation ratio: {extrap_mse / train_len_mse:.2f}x")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Val Loss':<12} {'100-step':<12} {'200-step':<12} {'Extrap':<10}")
    print("-"*70)
    for name, r in results.items():
        extrap_ratio = r['extrap_mse'] / r['train_len_mse']
        print(f"{name:<20} {r['best_val']:<12.6f} {r['train_len_mse']:<12.6f} {r['extrap_mse']:<12.6f} {extrap_ratio:<10.2f}x")

    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, r) in enumerate(results.items()):
        ax = axes.flat[idx]
        # Plot first trajectory, x dimension
        ax.plot(r['target'][0, :, 0], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(r['pred'][0, :, 0], 'r--', label='Predicted', alpha=0.7)
        ax.axvline(x=train_steps, color='g', linestyle=':', label='Train horizon')
        ax.set_xlabel('Time step')
        ax.set_ylabel('x')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3D phase space for last model
    ax = axes.flat[-1]
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    name = list(results.keys())[-1]
    r = results[name]
    ax.plot(r['target'][0, :, 0], r['target'][0, :, 1], r['target'][0, :, 2],
            'b-', label='Ground Truth', alpha=0.7)
    ax.plot(r['pred'][0, :, 0], r['pred'][0, :, 1], r['pred'][0, :, 2],
            'r--', label='Predicted', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'{name} - Phase Space')
    ax.legend()

    plt.tight_layout()
    plt.savefig('dynamics_psi_results.png', dpi=150)
    plt.close()
    print("\nSaved dynamics_psi_results.png")


if __name__ == "__main__":
    main()
