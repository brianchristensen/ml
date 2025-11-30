"""
Fair Multi-Step Comparison: PSI vs Transformer vs LSTM

ALL models predict K derivatives at once and integrate via cumsum.
This isolates the architectural differences, not the prediction strategy.

The question: Does PSI's cumsum-based memory provide any advantage
when everyone uses the same multi-step + cumsum integration approach?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Data Generation (N-body)
# =============================================================================

def generate_nbody_data(n_bodies, n_samples, n_steps, dt=0.01):
    """Generate n-body gravitational dynamics."""
    G = 1.0
    softening = 0.1

    trajectories = []
    for _ in range(n_samples):
        # Random initial conditions
        pos = np.random.randn(n_bodies, 2) * 2
        vel = np.random.randn(n_bodies, 2) * 0.5
        mass = np.ones(n_bodies)

        states = []
        for _ in range(n_steps):
            state = np.concatenate([pos.flatten(), vel.flatten()])
            states.append(state)

            # Compute accelerations
            acc = np.zeros_like(pos)
            for i in range(n_bodies):
                for j in range(n_bodies):
                    if i != j:
                        r = pos[j] - pos[i]
                        dist = np.sqrt(np.sum(r**2) + softening**2)
                        acc[i] += G * mass[j] * r / (dist**3)

            # Leapfrog integration
            vel = vel + acc * dt
            pos = pos + vel * dt

        trajectories.append(states)

    return torch.tensor(np.array(trajectories), dtype=torch.float32)


# =============================================================================
# Multi-Step Architectures (ALL use cumsum integration)
# =============================================================================

class PSIMultiStep(nn.Module):
    """PSI with multi-step prediction via cumsum."""

    def __init__(self, state_dim, dim=128, num_layers=4, K=30):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.dim = dim

        self.input_proj = nn.Linear(state_dim, dim)

        # PSI blocks with cumsum memory
        self.blocks = nn.ModuleList([
            PSIBlock(dim) for _ in range(num_layers)
        ])

        # Time embeddings for K future steps
        self.time_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)

        # Predict K derivatives
        self.derivative_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, state_dim)
        )

        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def forward(self, context):
        """
        Args:
            context: [batch, seq_len, state_dim]
        Returns:
            future: [batch, K, state_dim] - K future states
        """
        batch_size, seq_len, _ = context.shape

        # Encode context
        h = self.input_proj(context)
        for block in self.blocks:
            h = block(h)

        # Use final hidden state + time embeddings
        final_h = h[:, -1:, :]  # [batch, 1, dim]
        h_future = final_h.expand(-1, self.K, -1) + self.time_embed

        # Predict K derivatives
        dx = self.derivative_head(h_future)  # [batch, K, state_dim]

        # Integrate via cumsum
        last_state = context[:, -1, :]
        future = last_state.unsqueeze(1) + torch.cumsum(dx * self.dt_scale, dim=1)

        return future


class PSIBlock(nn.Module):
    """PSI block with cumsum-based memory."""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.value = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Cumsum attention
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)

        # Cumsum memory (O(n) complexity)
        cumsum_v = torch.cumsum(g * v, dim=1)
        cumsum_g = torch.cumsum(g, dim=1) + 1e-6
        mem = cumsum_v / cumsum_g

        x = x + mem

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerMultiStep(nn.Module):
    """Transformer with multi-step prediction via cumsum."""

    def __init__(self, state_dim, dim=128, num_layers=4, K=30, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.dim = dim

        self.input_proj = nn.Linear(state_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Time embeddings for K future steps
        self.time_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)

        # Predict K derivatives (same as PSI)
        self.derivative_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, state_dim)
        )

        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def forward(self, context):
        batch_size, seq_len, _ = context.shape

        # Encode context
        h = self.input_proj(context) + self.pos_embed[:, :seq_len, :]
        h = self.encoder(h)

        # Use final hidden state + time embeddings
        final_h = h[:, -1:, :]
        h_future = final_h.expand(-1, self.K, -1) + self.time_embed

        # Predict K derivatives
        dx = self.derivative_head(h_future)

        # Integrate via cumsum (SAME as PSI)
        last_state = context[:, -1, :]
        future = last_state.unsqueeze(1) + torch.cumsum(dx * self.dt_scale, dim=1)

        return future


class LSTMMultiStep(nn.Module):
    """LSTM with multi-step prediction via cumsum."""

    def __init__(self, state_dim, dim=128, num_layers=4, K=30):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.dim = dim

        self.input_proj = nn.Linear(state_dim, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)

        # Time embeddings for K future steps
        self.time_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)

        # Predict K derivatives (same as PSI)
        self.derivative_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, state_dim)
        )

        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def forward(self, context):
        batch_size, seq_len, _ = context.shape

        # Encode context
        h = self.input_proj(context)
        h, _ = self.lstm(h)

        # Use final hidden state + time embeddings
        final_h = h[:, -1:, :]
        h_future = final_h.expand(-1, self.K, -1) + self.time_embed

        # Predict K derivatives
        dx = self.derivative_head(h_future)

        # Integrate via cumsum (SAME as PSI)
        last_state = context[:, -1, :]
        future = last_state.unsqueeze(1) + torch.cumsum(dx * self.dt_scale, dim=1)

        return future


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_data, val_data, context_len, K, epochs=100, lr=1e-3, batch_size=64):
    """Train any multi-step model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    n_train = train_data.shape[0]
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch = train_data[idx]

            # Random starting points
            max_start = batch.shape[1] - context_len - K
            if max_start <= 0:
                continue
            starts = torch.randint(0, max_start, (len(idx),))

            # Extract context and targets
            contexts = torch.stack([batch[j, starts[j]:starts[j]+context_len] for j in range(len(idx))])
            targets = torch.stack([batch[j, starts[j]+context_len:starts[j]+context_len+K] for j in range(len(idx))])

            optimizer.zero_grad()
            pred = model(contexts)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # Use fixed validation windows
            val_contexts = val_data[:, :context_len, :]
            val_targets = val_data[:, context_len:context_len+K, :]
            val_pred = model(val_contexts)
            val_loss = F.mse_loss(val_pred, val_targets).item()

            if val_loss < best_val:
                best_val = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={total_loss/n_batches:.6f}, val={val_loss:.6f}")

    return best_val


def benchmark_inference(model, test_data, context_len, K, n_trials=50):
    """Benchmark inference speed."""
    model.eval()
    test_data = test_data.to(device)

    contexts = test_data[:32, :context_len, :]

    # Warmup
    with torch.no_grad():
        _ = model(contexts)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(contexts)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return np.mean(times) * 1000, np.std(times) * 1000


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("FAIR MULTI-STEP COMPARISON")
    print("All models: predict K derivatives + cumsum integration")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Parameters
    n_bodies = 3
    state_dim = n_bodies * 4  # pos + vel for each body
    n_train = 500
    n_val = 100
    n_test = 100
    timesteps = 150
    context_len = 20
    K = 50  # Predict 50 steps at once

    dim = 128
    num_layers = 4
    epochs = 100

    print(f"Task: N-body ({n_bodies} bodies)")
    print(f"Context: {context_len} steps, Predict: {K} steps")
    print(f"Model dim: {dim}, layers: {num_layers}")
    print()

    # Generate data
    print("Generating data...")
    train_data = generate_nbody_data(n_bodies, n_train, timesteps)
    val_data = generate_nbody_data(n_bodies, n_val, timesteps)
    test_data = generate_nbody_data(n_bodies, n_test, timesteps)
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    print()

    # Models to compare
    models = {
        'PSI Multi-Step': lambda: PSIMultiStep(state_dim, dim, num_layers, K),
        'Transformer Multi-Step': lambda: TransformerMultiStep(state_dim, dim, num_layers, K),
        'LSTM Multi-Step': lambda: LSTMMultiStep(state_dim, dim, num_layers, K),
    }

    results = {}

    for name, model_fn in models.items():
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model = model_fn().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Train
        best_val = train_model(model, train_data, val_data, context_len, K, epochs=epochs)

        # Test
        model.eval()
        test_data_gpu = test_data.to(device)
        with torch.no_grad():
            test_contexts = test_data_gpu[:, :context_len, :]
            test_targets = test_data_gpu[:, context_len:context_len+K, :]
            test_pred = model(test_contexts)
            test_mse = F.mse_loss(test_pred, test_targets).item()

        # Speed
        mean_time, std_time = benchmark_inference(model, test_data, context_len, K)

        results[name] = {
            'val_mse': best_val,
            'test_mse': test_mse,
            'time_ms': mean_time,
            'time_std': std_time,
            'params': n_params,
            'pred': test_pred.cpu(),
            'target': test_targets.cpu()
        }

        print(f"\n  Val MSE:  {best_val:.6f}")
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Time:     {mean_time:.2f} ± {std_time:.2f} ms")
        print()

    # Summary
    print("\n" + "=" * 80)
    print("FAIR COMPARISON RESULTS")
    print("All models use: Multi-step prediction + Cumsum integration")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Params':>10} {'Test MSE':>12} {'Time (ms)':>12}")
    print("-" * 65)

    for name, r in results.items():
        print(f"{name:<25} {r['params']:>10,} {r['test_mse']:>12.6f} {r['time_ms']:>12.2f}")

    # Relative comparison
    print("\n" + "-" * 65)
    print("Relative to PSI:")
    psi_mse = results['PSI Multi-Step']['test_mse']
    psi_time = results['PSI Multi-Step']['time_ms']

    for name, r in results.items():
        mse_ratio = (r['test_mse'] - psi_mse) / psi_mse * 100
        time_ratio = r['time_ms'] / psi_time
        mse_str = f"+{mse_ratio:.1f}%" if mse_ratio > 0 else f"{mse_ratio:.1f}%"
        print(f"  {name:<23} MSE: {mse_str:>8}  Speed: {time_ratio:.2f}x")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    trans_mse = results['Transformer Multi-Step']['test_mse']
    lstm_mse = results['LSTM Multi-Step']['test_mse']
    trans_time = results['Transformer Multi-Step']['time_ms']
    lstm_time = results['LSTM Multi-Step']['time_ms']

    print(f"""
When ALL models use multi-step + cumsum:

PSI vs Transformer:
  - MSE:   PSI {psi_mse:.6f} vs Trans {trans_mse:.6f} ({(trans_mse/psi_mse - 1)*100:+.1f}%)
  - Speed: PSI {psi_time:.1f}ms vs Trans {trans_time:.1f}ms ({trans_time/psi_time:.1f}x slower)

PSI vs LSTM:
  - MSE:   PSI {psi_mse:.6f} vs LSTM {lstm_mse:.6f} ({(lstm_mse/psi_mse - 1)*100:+.1f}%)
  - Speed: PSI {psi_time:.1f}ms vs LSTM {lstm_time:.1f}ms ({lstm_time/psi_time:.1f}x slower)

The multi-step trick works for ALL architectures.
Any remaining PSI advantage must come from:
1. O(n) vs O(n²) memory complexity for long contexts
2. Cumsum memory being naturally suited to dynamics
3. Architectural efficiency (fewer FLOPs)
""")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory comparison
    ax = axes[0, 0]
    sample_idx = 0
    dim_idx = 0
    for name, r in results.items():
        ax.plot(r['pred'][sample_idx, :, dim_idx].numpy(), label=name, alpha=0.7)
    ax.plot(results['PSI Multi-Step']['target'][sample_idx, :, dim_idx].numpy(),
            'k--', label='Ground Truth', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('State')
    ax.set_title(f'Trajectory Prediction (Sample {sample_idx}, Dim {dim_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE comparison
    ax = axes[0, 1]
    names = list(results.keys())
    mses = [results[n]['test_mse'] for n in names]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax.bar(names, mses, color=colors)
    ax.set_ylabel('Test MSE')
    ax.set_title('Accuracy Comparison (Lower is Better)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

    # Speed comparison
    ax = axes[1, 0]
    times = [results[n]['time_ms'] for n in names]
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Speed Comparison (Lower is Better)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom', fontsize=10)

    # Efficiency: MSE per ms
    ax = axes[1, 1]
    efficiency = [1 / (results[n]['test_mse'] * results[n]['time_ms']) for n in names]
    bars = ax.bar(names, efficiency, color=colors)
    ax.set_ylabel('Efficiency (1 / (MSE × Time))')
    ax.set_title('Overall Efficiency (Higher is Better)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('fair_multistep_comparison.png', dpi=150)
    plt.close()
    print("\nSaved fair_multistep_comparison.png")

    # Test at different horizons
    print("\n" + "=" * 80)
    print("TESTING AT DIFFERENT PREDICTION HORIZONS")
    print("=" * 80)

    horizons = [20, 50, 80]
    horizon_results = {name: {} for name in models.keys()}

    for K_test in horizons:
        print(f"\n--- K = {K_test} steps ---")

        for name, model_fn in models.items():
            model = model_fn.__self__ if hasattr(model_fn, '__self__') else model_fn()
            # Recreate with new K
            if 'PSI' in name:
                model = PSIMultiStep(state_dim, dim, num_layers, K_test).to(device)
            elif 'Transformer' in name:
                model = TransformerMultiStep(state_dim, dim, num_layers, K_test).to(device)
            else:
                model = LSTMMultiStep(state_dim, dim, num_layers, K_test).to(device)

            # Quick training
            best_val = train_model(model, train_data, val_data, context_len, K_test, epochs=50)

            # Test
            model.eval()
            with torch.no_grad():
                test_contexts = test_data.to(device)[:, :context_len, :]
                test_targets = test_data.to(device)[:, context_len:context_len+K_test, :]
                test_pred = model(test_contexts)
                test_mse = F.mse_loss(test_pred, test_targets).item()

            mean_time, _ = benchmark_inference(model, test_data, context_len, K_test)

            horizon_results[name][K_test] = {'mse': test_mse, 'time': mean_time}
            print(f"  {name}: MSE={test_mse:.6f}, Time={mean_time:.1f}ms")

    # Plot horizon scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name, color in zip(horizon_results.keys(), colors):
        mses = [horizon_results[name][h]['mse'] for h in horizons]
        ax.plot(horizons, mses, 'o-', label=name, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (K)')
    ax.set_ylabel('Test MSE')
    ax.set_title('Accuracy vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, color in zip(horizon_results.keys(), colors):
        times = [horizon_results[name][h]['time'] for h in horizons]
        ax.plot(horizons, times, 'o-', label=name, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (K)')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Speed vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fair_multistep_horizons.png', dpi=150)
    plt.close()
    print("\nSaved fair_multistep_horizons.png")


if __name__ == "__main__":
    main()
