"""
Sensor Fusion and Dropout Benchmark

Tests how well PSI, Transformer, and LSTM handle:
1. Multi-sensor fusion (combining multiple noisy sensor streams)
2. Sensor dropout (graceful degradation when sensors fail)
3. Varying noise levels
4. Partial observability

This is a realistic test for robotics/autonomous systems applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, '.')
from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Data Generation - Multi-Sensor System
# =============================================================================

def lorenz_dynamics(state, t, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_multisensor_data(n_trajectories, seq_len, n_sensors=4, noise_levels=None, dt=0.01):
    """
    Generate multi-sensor observations of Lorenz system.

    Each sensor observes a noisy, partial view of the true state.
    Sensors have different noise characteristics and observe different aspects.

    Sensors:
    - Sensor 0: Observes x with noise
    - Sensor 1: Observes y with noise
    - Sensor 2: Observes z with noise
    - Sensor 3: Observes x+y (combined) with noise
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.15, 0.2, 0.1]  # Different noise per sensor

    trajectories = []
    sensor_data = []

    for _ in range(n_trajectories):
        # Random initial condition
        x0 = np.random.randn(3) * 5 + np.array([0, 0, 25])
        t = np.linspace(0, seq_len * dt, seq_len)
        traj = odeint(lorenz_dynamics, x0, t)

        # Normalize
        traj_norm = traj / np.array([20.0, 25.0, 25.0])
        trajectories.append(traj_norm)

        # Generate sensor observations
        sensors = np.zeros((seq_len, n_sensors))
        sensors[:, 0] = traj_norm[:, 0] + np.random.randn(seq_len) * noise_levels[0]  # x
        sensors[:, 1] = traj_norm[:, 1] + np.random.randn(seq_len) * noise_levels[1]  # y
        sensors[:, 2] = traj_norm[:, 2] + np.random.randn(seq_len) * noise_levels[2]  # z
        sensors[:, 3] = (traj_norm[:, 0] + traj_norm[:, 1]) / 2 + np.random.randn(seq_len) * noise_levels[3]  # combined

        sensor_data.append(sensors)

    return (torch.tensor(np.array(trajectories), dtype=torch.float32),
            torch.tensor(np.array(sensor_data), dtype=torch.float32))


def apply_sensor_dropout(sensor_data, dropout_mask):
    """
    Apply sensor dropout - set dropped sensors to zero.

    dropout_mask: [n_sensors] boolean, True = sensor available
    """
    masked = sensor_data.clone()
    for i, available in enumerate(dropout_mask):
        if not available:
            masked[:, :, i] = 0.0
    return masked


# =============================================================================
# Models
# =============================================================================

class PSISensorFusion(nn.Module):
    """
    PSI-based sensor fusion model.

    Matches the architecture from sensor_fusion_experiment.py:
    - Direct projection of all sensor inputs into hidden_dim
    - PSI blocks for temporal processing
    - Output head for state estimation
    """

    def __init__(self, n_sensors, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()
        self.n_sensors = n_sensors

        # Project all sensors directly into hidden space (matches original PSI approach)
        self.input_proj = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # PSI blocks
        self.blocks = nn.ModuleList([PSIBlock(hidden_dim) for _ in range(num_layers)])

        # Output head (matches original)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, sensors):
        """
        sensors: [batch, seq_len, n_sensors]
        """
        # Project all sensor inputs together
        h = self.input_proj(sensors)

        # PSI processing
        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)


class TransformerSensorFusion(nn.Module):
    """Transformer-based sensor fusion model - same architecture pattern as PSI."""

    def __init__(self, n_sensors, hidden_dim=64, num_layers=4, output_dim=3, num_heads=4):
        super().__init__()
        self.n_sensors = n_sensors

        # Direct projection (same as PSI)
        self.input_proj = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                       dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Causal mask
        self.register_buffer('causal_mask', None)

        # Output head (same as PSI)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, sensors):
        batch, seq_len, n_sensors = sensors.shape

        # Project all sensors together
        h = self.input_proj(sensors)
        h = h + self.pos_encoding[:, :seq_len, :]

        # Create causal mask
        if self.causal_mask is None or self.causal_mask.shape[0] != seq_len:
            self.causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=sensors.device), diagonal=1
            ).bool()

        for block in self.blocks:
            h = block(h, src_mask=self.causal_mask)

        return self.output_proj(h)


class ManualLSTMCell(nn.Module):
    """Pure PyTorch LSTM cell - no cuDNN."""

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


class LSTMSensorFusion(nn.Module):
    """Manual LSTM sensor fusion - same architecture pattern as PSI."""

    def __init__(self, n_sensors, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Direct projection (same as PSI)
        self.input_proj = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Manual LSTM cells
        self.cells = nn.ModuleList([
            ManualLSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output head (same as PSI)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, sensors):
        batch, seq_len, n_sensors = sensors.shape

        # Project all sensors together
        x = self.input_proj(sensors)

        # Initialize hidden states
        h = [torch.zeros(batch, self.hidden_dim, device=sensors.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_dim, device=sensors.device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            outputs.append(h[-1])

        out = torch.stack(outputs, dim=1)
        return self.output_proj(out)


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_sensors, train_targets, val_sensors, val_targets,
                epochs=50, lr=1e-3, verbose=True):
    """Train a sensor fusion model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_sensors = train_sensors.to(device)
    train_targets = train_targets.to(device)
    val_sensors = val_sensors.to(device)
    val_targets = val_targets.to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(train_sensors)
        loss = F.mse_loss(pred, train_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_sensors)
            val_loss = F.mse_loss(val_pred, val_targets)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.6f}, val={val_loss.item():.6f}")

    return best_val_loss.item()


def evaluate_with_dropout(model, test_sensors, test_targets, dropout_configs):
    """
    Evaluate model under different sensor dropout scenarios.

    dropout_configs: list of (name, dropout_mask) tuples
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for name, mask in dropout_configs:
            masked_sensors = apply_sensor_dropout(test_sensors, mask)
            pred = model(masked_sensors)
            mse = F.mse_loss(pred, test_targets).item()
            results[name] = mse

    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("="*70)
    print("SENSOR FUSION AND DROPOUT BENCHMARK")
    print("="*70)
    print(f"Device: {device}")
    print()

    # Generate data
    print("Generating multi-sensor data...")
    n_sensors = 4
    seq_len = 100

    train_traj, train_sensors = generate_multisensor_data(500, seq_len, n_sensors)
    val_traj, val_sensors = generate_multisensor_data(100, seq_len, n_sensors)
    test_traj, test_sensors = generate_multisensor_data(100, seq_len, n_sensors)

    print(f"Train: {train_sensors.shape}")
    print(f"Sensors: {n_sensors} (x, y, z, combined)")
    print()

    # Models
    hidden_dim = 64
    num_layers = 4

    models = {
        'PSI': PSISensorFusion(n_sensors, hidden_dim, num_layers).to(device),
        'Transformer': TransformerSensorFusion(n_sensors, hidden_dim, num_layers).to(device),
        'LSTM': LSTMSensorFusion(n_sensors, hidden_dim, num_layers).to(device),
    }

    # Print parameter counts
    print("Model Parameters:")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,}")
    print()

    # Dropout configurations to test
    dropout_configs = [
        ("All sensors", [True, True, True, True]),
        ("Drop sensor 0 (x)", [False, True, True, True]),
        ("Drop sensor 1 (y)", [True, False, True, True]),
        ("Drop sensor 2 (z)", [True, True, False, True]),
        ("Drop sensor 3 (combined)", [True, True, True, False]),
        ("Drop 2 sensors (x,y)", [False, False, True, True]),
        ("Drop 2 sensors (x,z)", [False, True, False, True]),
        ("Only sensor 0 (x)", [True, False, False, False]),
        ("Only sensor 3 (combined)", [False, False, False, True]),
    ]

    # Train and evaluate each model
    results = {}
    train_times = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)

        start_time = time.time()
        val_loss = train_model(model, train_sensors, train_traj,
                               val_sensors, val_traj, epochs=50, lr=1e-3)
        train_time = time.time() - start_time
        train_times[name] = train_time

        print(f"\nBest val loss: {val_loss:.6f}")
        print(f"Training time: {train_time:.1f}s")

        # Evaluate with dropout
        print("\nEvaluating sensor dropout scenarios...")
        test_sensors_gpu = test_sensors.to(device)
        test_traj_gpu = test_traj.to(device)

        dropout_results = evaluate_with_dropout(model, test_sensors_gpu,
                                                 test_traj_gpu, dropout_configs)
        results[name] = dropout_results

    # Compute degradation metrics
    print("\n" + "="*70)
    print("SENSOR DROPOUT RESULTS")
    print("="*70)

    # Header
    print(f"\n{'Scenario':<30} {'PSI':<12} {'Transformer':<12} {'LSTM':<12} {'Winner':<12}")
    print("-"*78)

    for config_name, _ in dropout_configs:
        psi_mse = results['PSI'][config_name]
        trans_mse = results['Transformer'][config_name]
        lstm_mse = results['LSTM'][config_name]

        # Determine winner
        min_mse = min(psi_mse, trans_mse, lstm_mse)
        if psi_mse == min_mse:
            winner = "PSI"
        elif trans_mse == min_mse:
            winner = "Transformer"
        else:
            winner = "LSTM"

        print(f"{config_name:<30} {psi_mse:<12.6f} {trans_mse:<12.6f} {lstm_mse:<12.6f} {winner:<12}")

    # Compute graceful degradation score
    print("\n" + "="*70)
    print("GRACEFUL DEGRADATION ANALYSIS")
    print("="*70)

    for name in models.keys():
        baseline = results[name]["All sensors"]
        degradations = []

        for config_name, mask in dropout_configs[1:]:  # Skip "All sensors"
            n_dropped = sum(1 for m in mask if not m)
            mse = results[name][config_name]
            degradation = mse / baseline
            degradations.append((config_name, n_dropped, degradation))

        print(f"\n{name}:")
        print(f"  Baseline (all sensors): {baseline:.6f}")

        # Average degradation by number of dropped sensors
        for n_drop in [1, 2, 3]:
            avg_deg = np.mean([d for _, n, d in degradations if n == n_drop])
            if not np.isnan(avg_deg):
                print(f"  Avg degradation ({n_drop} sensor dropped): {avg_deg:.2f}x")

    # Speed comparison
    print("\n" + "="*70)
    print("SPEED COMPARISON")
    print("="*70)

    # Inference timing
    print("\nInference time (100 samples, seq_len=100):")
    for name, model in models.items():
        model.eval()
        test_batch = test_sensors[:100].to(device)

        # Warmup
        with torch.no_grad():
            _ = model(test_batch)

        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(20):
            with torch.no_grad():
                _ = model(test_batch)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) / 20 * 1000

        print(f"  {name}: {elapsed:.2f} ms")

    print(f"\nTraining time:")
    for name, t in train_times.items():
        print(f"  {name}: {t:.1f}s")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Count wins
    wins = {'PSI': 0, 'Transformer': 0, 'LSTM': 0}
    for config_name, _ in dropout_configs:
        psi_mse = results['PSI'][config_name]
        trans_mse = results['Transformer'][config_name]
        lstm_mse = results['LSTM'][config_name]

        min_mse = min(psi_mse, trans_mse, lstm_mse)
        if psi_mse == min_mse:
            wins['PSI'] += 1
        elif trans_mse == min_mse:
            wins['Transformer'] += 1
        else:
            wins['LSTM'] += 1

    print(f"\nWins across {len(dropout_configs)} dropout scenarios:")
    for name, w in wins.items():
        print(f"  {name}: {w} wins")

    # Robustness score (average performance across all scenarios)
    print(f"\nAverage MSE across all scenarios:")
    for name in models.keys():
        avg_mse = np.mean(list(results[name].values()))
        print(f"  {name}: {avg_mse:.6f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All scenarios comparison
    ax = axes[0, 0]
    scenarios = [c[0] for c in dropout_configs]
    x = np.arange(len(scenarios))
    width = 0.25

    ax.bar(x - width, [results['PSI'][s] for s in scenarios], width, label='PSI')
    ax.bar(x, [results['Transformer'][s] for s in scenarios], width, label='Transformer')
    ax.bar(x + width, [results['LSTM'][s] for s in scenarios], width, label='LSTM')

    ax.set_ylabel('MSE')
    ax.set_title('Sensor Fusion: Dropout Robustness')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Degradation by number of dropped sensors
    ax = axes[0, 1]
    for name in models.keys():
        baseline = results[name]["All sensors"]
        x_vals = []
        y_vals = []

        for config_name, mask in dropout_configs:
            n_dropped = sum(1 for m in mask if not m)
            mse = results[name][config_name]
            x_vals.append(n_dropped)
            y_vals.append(mse / baseline)

        # Sort by x
        sorted_pairs = sorted(zip(x_vals, y_vals))
        x_sorted = [p[0] for p in sorted_pairs]
        y_sorted = [p[1] for p in sorted_pairs]

        ax.scatter(x_sorted, y_sorted, label=name, alpha=0.7, s=60)

    ax.set_xlabel('Number of Dropped Sensors')
    ax.set_ylabel('Degradation Factor (MSE / baseline)')
    ax.set_title('Graceful Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Training time
    ax = axes[1, 0]
    names = list(train_times.keys())
    times = [train_times[n] for n in names]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Speed')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f}s', ha='center', va='bottom')

    # Plot 4: Win distribution
    ax = axes[1, 1]
    win_counts = [wins[n] for n in names]
    ax.pie(win_counts, labels=names, autopct='%1.1f%%', colors=colors)
    ax.set_title(f'Wins Across {len(dropout_configs)} Scenarios')

    plt.tight_layout()
    plt.savefig('sensor_fusion_benchmark.png', dpi=150)
    plt.close()
    print("\nSaved sensor_fusion_benchmark.png")


if __name__ == "__main__":
    main()
