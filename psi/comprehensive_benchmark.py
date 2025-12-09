"""
Comprehensive PSI vs Transformer Benchmark

Tests four key dimensions where PSI might have genuine advantages:
1. Sample Efficiency - Learning from limited data
2. Extrapolation - Generalizing to longer sequences
3. Out-of-Distribution - Generalizing to unseen parameters
4. Computational Efficiency - Wall-clock time and memory

This is the rigorous comparison that should have been done from the start.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import models
import sys
sys.path.insert(0, '.')
from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_lorenz_trajectory(x0, y0, z0, sigma=10.0, rho=28.0, beta=8/3,
                                dt=0.01, n_steps=1000):
    """Generate Lorenz attractor trajectory."""
    trajectory = np.zeros((n_steps, 3))
    x, y, z = x0, y0, z0

    for i in range(n_steps):
        trajectory[i] = [x, y, z]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    return trajectory


def create_lorenz_dataset(n_trajectories, seq_len, sigma=10.0, rho=28.0, beta=8/3):
    """Create dataset of Lorenz sequences for next-step prediction."""
    inputs = []
    targets = []

    for _ in range(n_trajectories):
        # Random initial conditions
        x0 = np.random.uniform(-20, 20)
        y0 = np.random.uniform(-20, 20)
        z0 = np.random.uniform(5, 45)

        traj = generate_lorenz_trajectory(x0, y0, z0, sigma, rho, beta,
                                          n_steps=seq_len + 1)

        # Normalize
        traj = traj / np.array([20.0, 25.0, 25.0])

        inputs.append(traj[:-1])
        targets.append(traj[1:])

    return (torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


# =============================================================================
# MODELS
# =============================================================================

class PSIModel(nn.Module):
    """PSI model for sequence prediction."""

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.psi_blocks = nn.ModuleList([
            PSIBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.psi_blocks:
            h = block(h)
        return self.output_proj(h)


class TransformerBlock(nn.Module):
    """Standard Transformer block."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.register_buffer('causal_mask', None)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        if self.causal_mask is None or self.causal_mask.shape[0] != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.causal_mask = mask

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=self.causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    """Transformer model for sequence prediction."""

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :seq_len, :]
        for block in self.transformer_blocks:
            h = block(h)
        return self.output_proj(h)


class ManualLSTMCell(nn.Module):
    """Pure PyTorch LSTM cell - no cuDNN optimization."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Combined gates for efficiency
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


class LSTMModel(nn.Module):
    """Manual LSTM baseline - no cuDNN, fair comparison with PSI."""

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=4, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            ManualLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states
        h = [torch.zeros(batch, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_dim, device=device) for _ in range(self.num_layers)]

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
# TRAINING UTILITIES
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, verbose=True):
    """Train a model and return training history."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = nn.functional.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs)
                val_loss += nn.functional.mse_loss(pred, targets).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return history, best_val_loss


def evaluate_model(model, test_loader):
    """Evaluate model RMSE."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            pred = model(inputs)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return rmse


# =============================================================================
# BENCHMARK 1: SAMPLE EFFICIENCY
# =============================================================================

def benchmark_sample_efficiency(args):
    """Test how well models learn from limited data."""
    print("\n" + "="*80)
    print("BENCHMARK 1: SAMPLE EFFICIENCY")
    print("="*80)
    print("Question: Does PSI learn better from limited training data?")
    print()

    # Generate full dataset
    print("Generating dataset...")
    full_inputs, full_targets = create_lorenz_dataset(
        n_trajectories=1000, seq_len=100
    )

    # Split into train/val/test
    n_train = 800
    n_val = 100

    train_inputs, train_targets = full_inputs[:n_train], full_targets[:n_train]
    val_inputs, val_targets = full_inputs[n_train:n_train+n_val], full_targets[n_train:n_train+n_val]
    test_inputs, test_targets = full_inputs[n_train+n_val:], full_targets[n_train+n_val:]

    val_dataset = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=32)

    test_dataset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Test with different data fractions
    fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
    results = {
        'PSI': [], 'Transformer': [], 'LSTM': []
    }

    for frac in fractions:
        n_samples = int(n_train * frac)
        print(f"\n--- Training with {frac*100:.0f}% data ({n_samples} samples) ---")

        train_subset = TensorDataset(train_inputs[:n_samples], train_targets[:n_samples])
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

        for model_name, ModelClass in [('PSI', PSIModel),
                                        ('Transformer', TransformerModel),
                                        ('LSTM', LSTMModel)]:
            print(f"\n  Training {model_name}...")
            model = ModelClass(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
            _, _ = train_model(model, train_loader, val_loader,
                              epochs=args.epochs, verbose=False)
            rmse = evaluate_model(model, test_loader)
            results[model_name].append(rmse)
            print(f"    Test RMSE: {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name, rmses in results.items():
        plt.plot([f*100 for f in fractions], rmses, 'o-', label=model_name, linewidth=2, markersize=8)

    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title('Sample Efficiency: Learning from Limited Data', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_sample_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*40)
    print("SAMPLE EFFICIENCY RESULTS")
    print("-"*40)
    print(f"{'Data %':<10} {'PSI':<12} {'Transformer':<12} {'LSTM':<12}")
    for i, frac in enumerate(fractions):
        print(f"{frac*100:<10.0f} {results['PSI'][i]:<12.4f} {results['Transformer'][i]:<12.4f} {results['LSTM'][i]:<12.4f}")

    # Determine winner
    avg_psi = np.mean(results['PSI'])
    avg_trans = np.mean(results['Transformer'])
    avg_lstm = np.mean(results['LSTM'])

    winner = min([('PSI', avg_psi), ('Transformer', avg_trans), ('LSTM', avg_lstm)], key=lambda x: x[1])
    print(f"\nBest average performance: {winner[0]} ({winner[1]:.4f})")

    return results


# =============================================================================
# BENCHMARK 2: EXTRAPOLATION
# =============================================================================

def benchmark_extrapolation(args):
    """Test generalization to longer sequences than training."""
    print("\n" + "="*80)
    print("BENCHMARK 2: EXTRAPOLATION")
    print("="*80)
    print("Question: Can models generalize to longer sequences than seen during training?")
    print()

    # Train on short sequences
    train_seq_len = 50
    print(f"Training on sequences of length {train_seq_len}...")

    train_inputs, train_targets = create_lorenz_dataset(800, train_seq_len)
    val_inputs, val_targets = create_lorenz_dataset(100, train_seq_len)

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=32)

    # Train models
    models = {}
    for model_name, ModelClass in [('PSI', PSIModel),
                                    ('Transformer', TransformerModel),
                                    ('LSTM', LSTMModel)]:
        print(f"\n  Training {model_name}...")
        model = ModelClass(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        _, _ = train_model(model, train_loader, val_loader, epochs=args.epochs, verbose=False)
        models[model_name] = model

    # Test on progressively longer sequences
    test_seq_lens = [50, 100, 200, 400, 800]
    results = {name: [] for name in models.keys()}

    print("\nEvaluating on different sequence lengths...")
    for seq_len in test_seq_lens:
        print(f"\n  Sequence length: {seq_len}")
        test_inputs, test_targets = create_lorenz_dataset(100, seq_len)
        test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=16)

        for model_name, model in models.items():
            rmse = evaluate_model(model, test_loader)
            results[model_name].append(rmse)
            print(f"    {model_name}: RMSE = {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name, rmses in results.items():
        plt.plot(test_seq_lens, rmses, 'o-', label=model_name, linewidth=2, markersize=8)

    plt.axvline(x=train_seq_len, color='gray', linestyle='--', label='Training length')
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Test RMSE', fontsize=12)
    plt.title(f'Extrapolation: Trained on length {train_seq_len}, tested on longer', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig('benchmark_extrapolation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*40)
    print("EXTRAPOLATION RESULTS")
    print("-"*40)
    print(f"{'Seq Len':<10} {'PSI':<12} {'Transformer':<12} {'LSTM':<12}")
    for i, seq_len in enumerate(test_seq_lens):
        marker = " (train)" if seq_len == train_seq_len else ""
        print(f"{seq_len:<10} {results['PSI'][i]:<12.4f} {results['Transformer'][i]:<12.4f} {results['LSTM'][i]:<12.4f}{marker}")

    # Compute degradation ratio (RMSE at longest / RMSE at train length)
    print("\nDegradation ratio (longest / training length):")
    for model_name in results:
        ratio = results[model_name][-1] / results[model_name][0]
        print(f"  {model_name}: {ratio:.2f}x")

    return results


# =============================================================================
# BENCHMARK 3: OUT-OF-DISTRIBUTION GENERALIZATION
# =============================================================================

def benchmark_ood_generalization(args):
    """Test generalization to different Lorenz parameters."""
    print("\n" + "="*80)
    print("BENCHMARK 3: OUT-OF-DISTRIBUTION GENERALIZATION")
    print("="*80)
    print("Question: Can models generalize to unseen dynamical regimes?")
    print()

    # Training parameters (standard Lorenz)
    train_sigma, train_rho, train_beta = 10.0, 28.0, 8/3
    print(f"Training on: sigma={train_sigma}, rho={train_rho}, beta={train_beta:.2f}")

    train_inputs, train_targets = create_lorenz_dataset(
        800, 100, sigma=train_sigma, rho=train_rho, beta=train_beta
    )
    val_inputs, val_targets = create_lorenz_dataset(
        100, 100, sigma=train_sigma, rho=train_rho, beta=train_beta
    )

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=32)

    # Train models
    models = {}
    for model_name, ModelClass in [('PSI', PSIModel),
                                    ('Transformer', TransformerModel),
                                    ('LSTM', LSTMModel)]:
        print(f"\n  Training {model_name}...")
        model = ModelClass(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        _, _ = train_model(model, train_loader, val_loader, epochs=args.epochs, verbose=False)
        models[model_name] = model

    # Test on different parameter combinations
    test_params = [
        (10.0, 28.0, 8/3, "Standard (train)"),
        (12.0, 28.0, 8/3, "sigma=12 (OOD)"),
        (10.0, 35.0, 8/3, "rho=35 (OOD)"),
        (10.0, 28.0, 3.0, "beta=3.0 (OOD)"),
        (14.0, 32.0, 2.5, "All different (OOD)"),
    ]

    results = {name: [] for name in models.keys()}
    param_labels = []

    print("\nEvaluating on different parameters...")
    for sigma, rho, beta, label in test_params:
        print(f"\n  {label}: sigma={sigma}, rho={rho}, beta={beta:.2f}")
        param_labels.append(label.split(" ")[0] if "train" not in label else "Train")

        test_inputs, test_targets = create_lorenz_dataset(
            100, 100, sigma=sigma, rho=rho, beta=beta
        )
        test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=32)

        for model_name, model in models.items():
            rmse = evaluate_model(model, test_loader)
            results[model_name].append(rmse)
            print(f"    {model_name}: RMSE = {rmse:.4f}")

    # Plot results
    x = np.arange(len(param_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, rmses) in enumerate(results.items()):
        ax.bar(x + i*width, rmses, width, label=model_name)

    ax.set_xlabel('Parameter Setting', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Out-of-Distribution Generalization: Different Lorenz Parameters', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(param_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('benchmark_ood_generalization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*40)
    print("OOD GENERALIZATION RESULTS")
    print("-"*40)
    print(f"{'Setting':<20} {'PSI':<12} {'Transformer':<12} {'LSTM':<12}")
    for i, (_, _, _, label) in enumerate(test_params):
        print(f"{label:<20} {results['PSI'][i]:<12.4f} {results['Transformer'][i]:<12.4f} {results['LSTM'][i]:<12.4f}")

    # Compute average OOD degradation
    print("\nAverage OOD RMSE (excluding train):")
    for model_name in results:
        ood_avg = np.mean(results[model_name][1:])
        train_rmse = results[model_name][0]
        print(f"  {model_name}: {ood_avg:.4f} ({ood_avg/train_rmse:.2f}x train)")

    return results


# =============================================================================
# BENCHMARK 4: COMPUTATIONAL EFFICIENCY
# =============================================================================

def benchmark_computational_efficiency(args):
    """Test wall-clock time and memory at different sequence lengths."""
    print("\n" + "="*80)
    print("BENCHMARK 4: COMPUTATIONAL EFFICIENCY")
    print("="*80)
    print("Question: How do models scale with sequence length?")
    print()

    seq_lens = [50, 100, 200, 400, 800]
    batch_size = 32
    n_trials = 10

    results = {
        'PSI': {'time': [], 'memory': []},
        'Transformer': {'time': [], 'memory': []},
        'LSTM': {'time': [], 'memory': []}
    }

    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")

        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, 3).to(device)

        for model_name, ModelClass in [('PSI', PSIModel),
                                        ('Transformer', TransformerModel),
                                        ('LSTM', LSTMModel)]:
            model = ModelClass(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)

            # Synchronize before timing
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # Time forward passes
            times = []
            for _ in range(n_trials):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    _ = model(dummy_input)

                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = np.mean(times) * 1000  # ms

            # Memory (only meaningful on CUDA)
            if device == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                peak_memory = 0

            results[model_name]['time'].append(avg_time)
            results[model_name]['memory'].append(peak_memory)

            print(f"  {model_name}: {avg_time:.2f} ms, {peak_memory:.1f} MB")

            # Clean up
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time plot
    ax = axes[0]
    for model_name in results:
        ax.plot(seq_lens, results[model_name]['time'], 'o-', label=model_name, linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Forward Pass Time (ms)', fontsize=12)
    ax.set_title('Computational Time vs Sequence Length', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Memory plot (only if CUDA)
    ax = axes[1]
    if device == 'cuda':
        for model_name in results:
            ax.plot(seq_lens, results[model_name]['memory'], 'o-', label=model_name, linewidth=2, markersize=8)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_title('Memory Usage vs Sequence Length', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Memory tracking\nrequires CUDA', ha='center', va='center', fontsize=14)
        ax.set_title('Memory Usage (CUDA only)', fontsize=14)

    plt.tight_layout()
    plt.savefig('benchmark_computational_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "-"*40)
    print("COMPUTATIONAL EFFICIENCY RESULTS")
    print("-"*40)
    print(f"{'Seq Len':<10} {'PSI (ms)':<12} {'Trans (ms)':<12} {'LSTM (ms)':<12}")
    for i, seq_len in enumerate(seq_lens):
        print(f"{seq_len:<10} {results['PSI']['time'][i]:<12.2f} {results['Transformer']['time'][i]:<12.2f} {results['LSTM']['time'][i]:<12.2f}")

    # Compute scaling
    print("\nScaling (time at longest / time at shortest):")
    for model_name in results:
        ratio = results[model_name]['time'][-1] / results[model_name]['time'][0]
        expected_linear = seq_lens[-1] / seq_lens[0]
        expected_quadratic = (seq_lens[-1] / seq_lens[0]) ** 2
        print(f"  {model_name}: {ratio:.2f}x (linear would be {expected_linear:.1f}x, quadratic {expected_quadratic:.1f}x)")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive PSI vs Transformer Benchmark')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per run')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'sample', 'extrapolation', 'ood', 'efficiency'],
                        help='Which benchmark to run')
    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE PSI vs TRANSFORMER BENCHMARK")
    print("="*80)
    print(f"Device: {device}")
    print(f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}, Epochs: {args.epochs}")

    all_results = {}

    if args.benchmark in ['all', 'sample']:
        all_results['sample_efficiency'] = benchmark_sample_efficiency(args)

    if args.benchmark in ['all', 'extrapolation']:
        all_results['extrapolation'] = benchmark_extrapolation(args)

    if args.benchmark in ['all', 'ood']:
        all_results['ood_generalization'] = benchmark_ood_generalization(args)

    if args.benchmark in ['all', 'efficiency']:
        all_results['computational_efficiency'] = benchmark_computational_efficiency(args)

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("""
This benchmark tested PSI against Transformer and LSTM on four dimensions:

1. SAMPLE EFFICIENCY
   - How well does each model learn from limited data?
   - Winner indicated by lowest average RMSE across data fractions

2. EXTRAPOLATION
   - Can models generalize to longer sequences than training?
   - Winner indicated by smallest degradation ratio

3. OUT-OF-DISTRIBUTION GENERALIZATION
   - Can models handle unseen dynamical regimes?
   - Winner indicated by smallest average OOD RMSE

4. COMPUTATIONAL EFFICIENCY
   - How does inference time scale with sequence length?
   - PSI should show O(n) scaling, Transformer O(n²)

See generated PNG files for visualizations.
""")

    print("Saved plots:")
    print("  - benchmark_sample_efficiency.png")
    print("  - benchmark_extrapolation.png")
    print("  - benchmark_ood_generalization.png")
    print("  - benchmark_computational_efficiency.png")


if __name__ == "__main__":
    main()
