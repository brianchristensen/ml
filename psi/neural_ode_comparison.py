"""
Neural ODE vs PSI Dynamics - Direct Comparison

Neural ODE: Sequential solver (what we want to replace)
PSI Dynamics: Parallel cumsum-based integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint
import time
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# DATA: Lorenz system
# =============================================================================

def lorenz(state, t):
    x, y, z = state
    return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]


def gen_data(n, steps, dt=0.01):
    trajs = []
    for _ in range(n):
        x0 = np.random.randn(3)*5 + [0,0,25]
        t = np.linspace(0, steps*dt, steps+1)
        trajs.append(odeint(lorenz, x0, t))
    return torch.tensor(np.array(trajs), dtype=torch.float32)


# =============================================================================
# NEURAL ODE (baseline we want to replace)
# =============================================================================

class NeuralODEFunc(nn.Module):
    """The vector field f(x) that Neural ODE learns."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        return self.net(x)


class NeuralODE(nn.Module):
    """
    Neural ODE: learns dx/dt = f(x), integrates via Euler.
    This is what we want to REPLACE with PSI.
    Sequential - each step depends on previous.
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.func = NeuralODEFunc(state_dim, hidden_dim)
        self.dt_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x0, steps, dt=0.01):
        """Generate trajectory from initial condition - SEQUENTIAL."""
        traj = [x0]
        x = x0
        for _ in range(steps):
            dx = self.func(x) * self.dt_scale * dt
            x = x + dx
            traj.append(x)
        return torch.stack(traj, dim=1)


# =============================================================================
# PSI DYNAMICS - Multi-step prediction (our advantage)
# =============================================================================

class PSIMultiStepDynamics(nn.Module):
    """
    PSI-based dynamics with MULTI-STEP prediction.

    Key advantage: Given history, predict K future steps in ONE forward pass.
    - History encoding: PARALLEL (cumsum)
    - Future prediction: PARALLEL (cumsum for integration)
    """
    def __init__(self, state_dim, hidden_dim=64, num_layers=2, max_future_steps=200):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_future_steps = max_future_steps

        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.ModuleList([PSIBlock(hidden_dim) for _ in range(num_layers)])

        # Predict ALL future derivatives at once
        self.future_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_future_steps * state_dim)
        )
        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x0, steps, dt=0.01):
        """
        Generate trajectory from x0.
        Uses x0 as minimal "history" and predicts future.
        """
        batch = x0.shape[0]

        # Minimal history: just x0
        history = x0.unsqueeze(1)  # [B, 1, state_dim]

        # Process through PSI (parallel even for length 1)
        h = self.input_proj(history)
        for block in self.blocks:
            h = block(h)

        h_final = h[:, -1, :]  # [B, hidden_dim]

        # Predict all future derivatives at once
        dx_flat = self.future_proj(h_final)  # [B, max_steps * state_dim]
        dx_all = dx_flat.view(batch, self.max_future_steps, self.state_dim)

        # Take only the steps we need
        dx = dx_all[:, :steps, :] * self.dt_scale * dt

        # Integrate via CUMSUM (parallel!)
        cumulative = torch.cumsum(dx, dim=1)
        future = x0.unsqueeze(1) + cumulative

        # Prepend x0
        traj = torch.cat([x0.unsqueeze(1), future], dim=1)

        return traj

    def forward_with_history(self, history, future_steps, dt=0.01):
        """
        Given history sequence, predict future.
        History processing AND future integration are both PARALLEL.
        """
        batch = history.shape[0]

        # Process history through PSI (PARALLEL via cumsum!)
        h = self.input_proj(history)
        for block in self.blocks:
            h = block(h)

        h_final = h[:, -1, :]  # [B, hidden_dim]

        # Predict all future derivatives at once
        dx_flat = self.future_proj(h_final)
        dx_all = dx_flat.view(batch, self.max_future_steps, self.state_dim)
        dx = dx_all[:, :future_steps, :] * self.dt_scale * dt

        # Integrate via CUMSUM (parallel!)
        x_last = history[:, -1, :]
        cumulative = torch.cumsum(dx, dim=1)
        future = x_last.unsqueeze(1) + cumulative

        return future


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_data, val_data, epochs=100, dt=0.01):
    """Train model to predict trajectories from x0."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    steps = train_data.shape[1] - 1
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x0 = train_data[:, 0, :]
        target = train_data

        pred = model(x0, steps, dt=dt)
        loss = F.mse_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            x0_val = val_data[:, 0, :]
            pred_val = model(x0_val, val_data.shape[1]-1, dt=dt)
            val_loss = F.mse_loss(pred_val, val_data)
            if val_loss < best_val:
                best_val = val_loss

        if (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

    return best_val.item()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("NEURAL ODE vs PSI DYNAMICS")
    print("="*70)
    print(f"Device: {device}")
    print()

    # Generate data
    dt = 0.01
    train_steps = 100
    test_steps = 200

    train_data = gen_data(300, train_steps, dt=dt)
    val_data = gen_data(50, train_steps, dt=dt)
    test_data = gen_data(50, test_steps, dt=dt)

    print(f"Train: {train_data.shape}")
    print(f"Training horizon: {train_steps} steps")
    print(f"Test horizon: {test_steps} steps (extrapolation)")
    print()

    # Models
    models = {
        'Neural ODE': NeuralODE(state_dim=3, hidden_dim=64).to(device),
        'PSI Multi-Step': PSIMultiStepDynamics(state_dim=3, hidden_dim=64,
                                                num_layers=2, max_future_steps=test_steps).to(device),
    }

    results = {}

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name} ({n_params:,} params)")
        print("-"*40)

        # Train
        best_val = train_model(model, train_data, val_data, epochs=100, dt=dt)

        # Test
        model.eval()
        test_data_gpu = test_data.to(device)
        with torch.no_grad():
            x0 = test_data_gpu[:, 0, :]
            pred_train = model(x0, train_steps, dt=dt)
            mse_train = F.mse_loss(pred_train, test_data_gpu[:, :train_steps+1]).item()

            pred_test = model(x0, test_steps, dt=dt)
            mse_test = F.mse_loss(pred_test, test_data_gpu).item()

        # Timing
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(20):
            with torch.no_grad():
                _ = model(x0[:16], test_steps, dt=dt)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) / 20 * 1000

        results[name] = {
            'val': best_val,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'time': elapsed,
            'pred': pred_test.cpu()
        }

        print(f"  Val: {best_val:.4f}")
        print(f"  Test ({train_steps} steps): {mse_train:.4f}")
        print(f"  Test ({test_steps} steps): {mse_test:.4f}")
        print(f"  Time ({test_steps} steps): {elapsed:.2f} ms")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'100-step MSE':<15} {'200-step MSE':<15} {'Time (ms)':<12}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<20} {r['mse_train']:<15.4f} {r['mse_test']:<15.4f} {r['time']:<12.2f}")

    # Speedup
    ode_time = results['Neural ODE']['time']
    psi_time = results['PSI Multi-Step']['time']
    speedup = ode_time / psi_time
    print(f"\nPSI Speedup: {speedup:.1f}x faster than Neural ODE")

    # Accuracy comparison
    ode_mse = results['Neural ODE']['mse_test']
    psi_mse = results['PSI Multi-Step']['mse_test']
    if psi_mse < ode_mse:
        print(f"PSI Accuracy: {ode_mse/psi_mse:.1f}x better MSE")
    else:
        print(f"PSI Accuracy: {psi_mse/ode_mse:.1f}x worse MSE")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trajectory comparison
    ax = axes[0]
    target = test_data[0, :, 0].numpy()
    ax.plot(target, 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
    for name, r in results.items():
        ax.plot(r['pred'][0, :, 0].numpy(), '--', label=name, linewidth=2, alpha=0.7)
    ax.axvline(x=train_steps, color='gray', linestyle=':', label='Train horizon')
    ax.set_xlabel('Time step')
    ax.set_ylabel('x')
    ax.set_title('Trajectory Prediction (x dimension)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speed comparison
    ax = axes[1]
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    colors = ['steelblue', 'coral']
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Inference Speed ({test_steps} steps)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f}ms', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('neural_ode_vs_psi.png', dpi=150)
    plt.close()
    print("\nSaved neural_ode_vs_psi.png")


if __name__ == "__main__":
    main()
