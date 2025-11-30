"""
Multi-step Dynamics Prediction

Key insight: Instead of predicting one step at a time (slow),
predict K steps at once and then use cumsum.

This gives us:
- Parallel training (predict all K-step blocks)
- Parallel inference (generate K steps per forward pass)
- O(N/K) sequential operations instead of O(N)

The model learns: Given x(t), predict [dx(1), dx(2), ..., dx(K)]
Then: x(t:t+K) = x(t) + cumsum([dx(1), ..., dx(K)])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiStepPredictor(nn.Module):
    """
    Predict K future steps at once.

    Given current state x, output [dx_1, dx_2, ..., dx_K]
    Then integrate via cumsum: x_{1:K} = x + cumsum(dx)
    """

    def __init__(self, state_dim, hidden_dim=64, K=10):
        super().__init__()
        self.state_dim = state_dim
        self.K = K

        # Encoder: current state -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Predict K steps of derivatives
        self.predictor = nn.Linear(hidden_dim, K * state_dim)

        # Learnable integration scale
        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x):
        """
        Args:
            x: [batch, state_dim] current state

        Returns:
            next_states: [batch, K, state_dim] next K states
        """
        h = self.encoder(x)

        # Predict all K derivatives at once
        dx_flat = self.predictor(h)  # [batch, K * state_dim]
        dx = dx_flat.view(-1, self.K, self.state_dim)  # [batch, K, state_dim]

        # Scale derivatives
        dx_scaled = dx * self.dt_scale

        # Integrate via cumsum
        trajectory = x.unsqueeze(1) + torch.cumsum(dx_scaled, dim=1)

        return trajectory

    def rollout(self, x0, num_steps):
        """Generate trajectory by chaining K-step predictions."""
        chunks = []
        x = x0

        steps_remaining = num_steps
        while steps_remaining > 0:
            # Predict K steps
            pred = self.forward(x)  # [batch, K, state_dim]

            # Take what we need
            take = min(self.K, steps_remaining)
            chunks.append(pred[:, :take, :])

            # Update x to last predicted state
            x = pred[:, take - 1, :]
            steps_remaining -= take

        # Concatenate all chunks
        trajectory = torch.cat([x0.unsqueeze(1)] + chunks, dim=1)
        return trajectory


class SingleStepBaseline(nn.Module):
    """Single-step predictor for comparison."""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

    def forward(self, x):
        dx = self.net(x) * self.dt_scale
        return x + dx

    def rollout(self, x0, num_steps):
        trajectory = [x0]
        x = x0
        for _ in range(num_steps):
            x = self.forward(x)
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

def train_multistep(model, train_data, val_data, epochs=100, lr=1e-3):
    """Train multi-step predictor."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    K = model.K
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Sample random starting points and predict K steps
        seq_len = train_data.shape[1]
        n_samples = train_data.shape[0]

        # Get random starting indices
        max_start = seq_len - K - 1
        starts = torch.randint(0, max_start, (n_samples,))

        # Extract windows
        x_batch = train_data[torch.arange(n_samples), starts, :]  # [batch, state_dim]
        target_batch = torch.stack([
            train_data[i, starts[i]+1:starts[i]+K+1, :]
            for i in range(n_samples)
        ])  # [batch, K, state_dim]

        optimizer.zero_grad()
        pred = model(x_batch)  # [batch, K, state_dim]
        loss = F.mse_loss(pred, target_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation: full rollout
        model.eval()
        with torch.no_grad():
            x0 = val_data[:, 0, :]
            pred_traj = model.rollout(x0, val_data.shape[1] - 1)
            val_loss = F.mse_loss(pred_traj, val_data)

            if val_loss < best_val:
                best_val = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train={loss.item():.6f}, val_rollout={val_loss.item():.6f}")

    return best_val.item()


def train_singlestep(model, train_data, val_data, epochs=100, lr=1e-3):
    """Train single-step predictor."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()

        # Train on all consecutive pairs
        x = train_data[:, :-1, :].reshape(-1, 3)  # [batch*seq, state_dim]
        y = train_data[:, 1:, :].reshape(-1, 3)

        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation: full rollout
        model.eval()
        with torch.no_grad():
            x0 = val_data[:, 0, :]
            pred_traj = model.rollout(x0, val_data.shape[1] - 1)
            val_loss = F.mse_loss(pred_traj, val_data)

            if val_loss < best_val:
                best_val = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train={loss.item():.6f}, val_rollout={val_loss.item():.6f}")

    return best_val.item()


def benchmark_speed(model, x0, num_steps, n_trials=10):
    """Benchmark rollout speed."""
    model.eval()
    x0 = x0.to(device)

    # Warmup
    with torch.no_grad():
        _ = model.rollout(x0, num_steps)

    torch.cuda.synchronize() if device == 'cuda' else None

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            _ = model.rollout(x0, num_steps)
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.time() - start)

    return np.mean(times), np.std(times)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("MULTI-STEP DYNAMICS PREDICTION")
    print("="*70)
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating Lorenz data...")
    dt = 0.01
    train_steps = 100
    test_steps = 200

    train_data = generate_lorenz_data(500, train_steps, dt=dt)
    val_data = generate_lorenz_data(100, train_steps, dt=dt)
    test_data = generate_lorenz_data(100, test_steps, dt=dt)

    print(f"Train: {train_data.shape}")

    # Test different K values
    K_values = [1, 5, 10, 20, 50]
    results = {}

    for K in K_values:
        print(f"\n{'='*50}")
        print(f"K = {K} (predict {K} steps at once)")
        print(f"{'='*50}")

        if K == 1:
            model = SingleStepBaseline(state_dim=3, hidden_dim=64).to(device)
            best_val = train_singlestep(model, train_data, val_data, epochs=100)
        else:
            model = MultiStepPredictor(state_dim=3, hidden_dim=64, K=K).to(device)
            best_val = train_multistep(model, train_data, val_data, epochs=100)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Evaluate
        model.eval()
        test_data_gpu = test_data.to(device)
        with torch.no_grad():
            x0 = test_data_gpu[:, 0, :]

            # 100-step rollout
            pred_100 = model.rollout(x0, train_steps)
            mse_100 = F.mse_loss(pred_100, test_data_gpu[:, :train_steps+1, :]).item()

            # 200-step rollout
            pred_200 = model.rollout(x0, test_steps)
            mse_200 = F.mse_loss(pred_200, test_data_gpu[:, :test_steps+1, :]).item()

        # Speed benchmark
        x0_bench = test_data[:32, 0, :].to(device)
        mean_time, std_time = benchmark_speed(model, x0_bench, test_steps)

        results[K] = {
            'val': best_val,
            'mse_100': mse_100,
            'mse_200': mse_200,
            'time': mean_time,
            'time_std': std_time,
            'n_params': n_params,
            'pred': pred_200.cpu(),
            'target': test_data[:, :test_steps+1, :]
        }

        print(f"\n  Val loss:    {best_val:.4f}")
        print(f"  100-step:    {mse_100:.4f}")
        print(f"  200-step:    {mse_200:.4f}")
        print(f"  Extrap:      {mse_200/mse_100:.2f}x")
        print(f"  Time (200):  {mean_time*1000:.2f} +/- {std_time*1000:.2f} ms")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'K':<6} {'Val':<10} {'100-step':<10} {'200-step':<10} {'Extrap':<8} {'Time(ms)':<12}")
    print("-"*70)

    for K, r in results.items():
        extrap = r['mse_200'] / r['mse_100']
        print(f"{K:<6} {r['val']:<10.4f} {r['mse_100']:<10.4f} {r['mse_200']:<10.4f} {extrap:<8.2f}x {r['time']*1000:<12.2f}")

    # Speedup analysis
    baseline_time = results[1]['time']
    print(f"\n{'K':<6} {'Speedup vs K=1':<15}")
    print("-"*25)
    for K, r in results.items():
        speedup = baseline_time / r['time']
        print(f"{K:<6} {speedup:<15.2f}x")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, K in enumerate(K_values[:3]):
        ax = axes[0, idx]
        r = results[K]
        ax.plot(r['target'][0, :, 0], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(r['pred'][0, :, 0], 'r--', label='Predicted', alpha=0.7)
        ax.axvline(x=train_steps, color='g', linestyle=':', label='Train horizon')
        ax.set_title(f'K={K} (MSE: {r["mse_200"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Speed vs accuracy plot
    ax = axes[1, 0]
    Ks = list(results.keys())
    times = [results[k]['time'] * 1000 for k in Ks]
    mses = [results[k]['mse_200'] for k in Ks]
    ax.scatter(times, mses, s=100)
    for i, K in enumerate(Ks):
        ax.annotate(f'K={K}', (times[i], mses[i]), textcoords="offset points", xytext=(5,5))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('MSE (200-step)')
    ax.set_title('Speed vs Accuracy Tradeoff')
    ax.grid(True, alpha=0.3)

    # Speedup vs K
    ax = axes[1, 1]
    speedups = [baseline_time / results[k]['time'] for k in Ks]
    ax.bar([str(k) for k in Ks], speedups)
    ax.set_xlabel('K (steps per prediction)')
    ax.set_ylabel('Speedup vs K=1')
    ax.set_title('Speedup from Multi-step Prediction')
    ax.grid(True, alpha=0.3)

    # Theoretical vs actual speedup
    ax = axes[1, 2]
    theoretical = [200/k for k in Ks]  # 200 steps / K steps per call
    ax.plot(Ks, theoretical, 'b-o', label='Theoretical (200/K)')
    ax.plot(Ks, speedups, 'r-o', label='Actual')
    ax.set_xlabel('K')
    ax.set_ylabel('Speedup')
    ax.set_title('Theoretical vs Actual Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multistep_dynamics.png', dpi=150)
    plt.close()
    print("\nSaved multistep_dynamics.png")


if __name__ == "__main__":
    main()
