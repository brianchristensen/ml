"""
Integration/Accumulation Benchmark

Tests whether PSI's cumsum-based memory gives it an advantage on tasks
that require perfect integration.

Tasks:
1. Running sum - predict cumsum of input sequence
2. Running average - predict cumulative mean
3. Counting - count occurrences of a pattern
4. Integration - approximate integral of a function
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import sys
sys.path.insert(0, '.')
from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([PSIBlock(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_enc[:, :seq_len, :]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)

        return self.output_proj(h)


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


class LSTMModel(nn.Module):
    """Manual LSTM - no cuDNN, fair comparison with PSI."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, output_dim=1):
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
        h = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

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
# Data Generation
# =============================================================================

def generate_running_sum_data(n_samples, seq_len, noise_std=0.0):
    """
    Task: Given sequence x, predict cumsum(x) at each position.
    This is exactly what PSI's memory does internally.
    """
    # Random inputs
    x = torch.randn(n_samples, seq_len, 1)

    # Add noise to inputs if specified
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    # Target is cumulative sum
    y = torch.cumsum(x, dim=1)

    return x, y


def generate_running_average_data(n_samples, seq_len):
    """
    Task: Given sequence x, predict cumulative mean at each position.
    """
    x = torch.randn(n_samples, seq_len, 1)

    # Cumulative mean: cumsum(x) / position
    cumsum = torch.cumsum(x, dim=1)
    positions = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, -1, 1)
    y = cumsum / positions

    return x, y


def generate_counting_data(n_samples, seq_len, threshold=0.5):
    """
    Task: Count how many elements exceed threshold so far.
    """
    x = torch.rand(n_samples, seq_len, 1)

    # Count elements > threshold
    above_threshold = (x > threshold).float()
    y = torch.cumsum(above_threshold, dim=1)

    return x, y


def generate_integration_data(n_samples, seq_len, dt=0.1):
    """
    Task: Approximate integral of a function.
    Given f(t) at discrete points, predict integral from 0 to t.
    """
    # Generate smooth functions (sum of sinusoids)
    t = torch.linspace(0, seq_len * dt, seq_len).view(1, -1, 1)
    freqs = torch.rand(n_samples, 1, 3) * 2 + 0.5  # Random frequencies
    phases = torch.rand(n_samples, 1, 3) * 2 * np.pi

    # f(t) = sum of sinusoids
    x = torch.sin(freqs[:, :, 0:1] * t + phases[:, :, 0:1])
    x = x + 0.5 * torch.sin(freqs[:, :, 1:2] * t + phases[:, :, 1:2])
    x = x + 0.3 * torch.sin(freqs[:, :, 2:3] * t + phases[:, :, 2:3])

    # Integral approximation (trapezoidal would be better, but cumsum * dt is close)
    y = torch.cumsum(x, dim=1) * dt

    return x, y


# =============================================================================
# Training
# =============================================================================

def train_and_evaluate(model, train_loader, test_loader, epochs=30, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_test_loss = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += nn.functional.mse_loss(pred, y).item()
        test_loss /= len(test_loader)

        history.append((train_loss, test_loss))

        if test_loss < best_test_loss:
            best_test_loss = test_loss

    return best_test_loss, history


def run_task(task_name, data_generator, seq_lens, args):
    """Run a task across different sequence lengths."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")

    results = {'PSI': [], 'Transformer': [], 'LSTM': []}

    for seq_len in seq_lens:
        print(f"\n  Sequence length: {seq_len}")

        # Generate data
        x_train, y_train = data_generator(800, seq_len)
        x_test, y_test = data_generator(200, seq_len)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)

        for model_name, ModelClass in [('PSI', PSIModel),
                                        ('Transformer', TransformerModel),
                                        ('LSTM', LSTMModel)]:
            model = ModelClass(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
            test_loss, _ = train_and_evaluate(model, train_loader, test_loader,
                                               epochs=args.epochs, lr=args.lr)
            results[model_name].append(test_loss)
            print(f"    {model_name}: {test_loss:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    print("="*60)
    print("INTEGRATION/ACCUMULATION BENCHMARK")
    print("="*60)
    print(f"Device: {device}")
    print(f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")

    seq_lens = [50, 100, 200, 400]

    # Task 1: Running Sum
    results_sum = run_task(
        "Running Sum (cumsum)",
        generate_running_sum_data,
        seq_lens, args
    )

    # Task 2: Running Average
    results_avg = run_task(
        "Running Average (cumsum / position)",
        generate_running_average_data,
        seq_lens, args
    )

    # Task 3: Counting
    results_count = run_task(
        "Counting (cumsum of indicator)",
        generate_counting_data,
        seq_lens, args
    )

    # Task 4: Integration
    results_int = run_task(
        "Integration (cumsum * dt)",
        generate_integration_data,
        seq_lens, args
    )

    # Summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tasks = [
        ("Running Sum", results_sum),
        ("Running Average", results_avg),
        ("Counting", results_count),
        ("Integration", results_int)
    ]

    for ax, (task_name, results) in zip(axes.flat, tasks):
        for model_name in results:
            ax.plot(seq_lens, results[model_name], 'o-', label=model_name, linewidth=2)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Test MSE')
        ax.set_title(task_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('integration_benchmark.png', dpi=150)
    plt.close()
    print("\nSaved integration_benchmark.png")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Task':<20} {'PSI':<12} {'Transformer':<12} {'LSTM':<12} {'Winner':<10}")
    print("-"*60)

    for task_name, results in tasks:
        avg_psi = np.mean(results['PSI'])
        avg_trans = np.mean(results['Transformer'])
        avg_lstm = np.mean(results['LSTM'])
        winner = min([('PSI', avg_psi), ('Transformer', avg_trans), ('LSTM', avg_lstm)], key=lambda x: x[1])[0]
        print(f"{task_name:<20} {avg_psi:<12.6f} {avg_trans:<12.6f} {avg_lstm:<12.6f} {winner:<10}")


if __name__ == "__main__":
    main()
