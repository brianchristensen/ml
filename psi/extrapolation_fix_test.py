"""
Test different normalization strategies for PSI extrapolation.

Compare:
1. Original (1/n normalization)
2. Sqrt normalization (1/sqrt(n))
3. EMA-style (exponential moving average)
4. No normalization (just cumsum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# PSI VARIANTS
# =============================================================================

class PSIBlock_Original(nn.Module):
    """Original PSI with 1/n normalization - from control_systems_benchmark."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        cumsum_v = torch.cumsum(g * v, dim=1)
        cumsum_g = torch.cumsum(g, dim=1) + 1e-6
        mem = cumsum_v / cumsum_g  # This is the running average
        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIBlock_SqrtNorm(nn.Module):
    """PSI with sqrt(n) normalization for slower decay."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        cumsum_v = torch.cumsum(g * v, dim=1)
        cumsum_g = torch.cumsum(g, dim=1) + 1e-6

        # Sqrt normalization instead of linear
        position = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        mem = cumsum_v / (torch.sqrt(cumsum_g) * torch.sqrt(position / seq_len + 0.1))

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIBlock_EMA(nn.Module):
    """PSI with exponential moving average - bounded magnitude."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.ones(dim) * 0.1)  # Learned decay
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)

        # EMA: bounded magnitude regardless of length
        alpha = torch.sigmoid(self.alpha)  # [0, 1]

        # Compute EMA efficiently using cumsum trick
        # EMA[t] = alpha * value[t] + (1-alpha) * EMA[t-1]
        # Can be computed as: cumsum(alpha * (1-alpha)^(t-i) * value[i])

        # Simpler: just use a fixed decay
        decay = 0.95
        weights = decay ** torch.arange(seq_len, 0, -1, device=x.device, dtype=x.dtype)
        weights = weights.view(1, -1, 1)

        # Weighted cumsum (approximation to EMA)
        gv = g * v
        # For each position, sum weighted past values
        # This is O(n^2) but we can approximate with cumsum
        mem = torch.zeros_like(x)
        running = torch.zeros(batch, dim, device=x.device)
        for t in range(seq_len):
            running = decay * running + (1 - decay) * gv[:, t, :]
            mem[:, t, :] = running

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIBlock_Chunked(nn.Module):
    """PSI with chunk-based normalization - reset every chunk_size steps."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)

        # Process in chunks
        mem = torch.zeros_like(x)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_v = g[:, start:end] * v[:, start:end]
            chunk_g = g[:, start:end]
            cumsum_v = torch.cumsum(chunk_v, dim=1)
            cumsum_g = torch.cumsum(chunk_g, dim=1) + 1e-6
            mem[:, start:end] = cumsum_v / cumsum_g

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MODELS
# =============================================================================

class PSIModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=64, num_layers=4, block_class=PSIBlock_Original):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([block_class(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


# =============================================================================
# DATA
# =============================================================================

def generate_lorenz_trajectory(x0, y0, z0, sigma=10.0, rho=28.0, beta=8/3, dt=0.01, n_steps=1000):
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


def create_lorenz_dataset(n_trajectories, seq_len):
    inputs, targets = [], []
    for _ in range(n_trajectories):
        x0, y0, z0 = np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(5, 45)
        traj = generate_lorenz_trajectory(x0, y0, z0, n_steps=seq_len + 1)
        traj = traj / np.array([20.0, 25.0, 25.0])
        inputs.append(traj[:-1])
        targets.append(traj[1:])
    return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)


# =============================================================================
# EXPERIMENT
# =============================================================================

def train_and_evaluate(model, train_loader, test_loaders, epochs=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluate at all lengths
    model.eval()
    results = {}
    with torch.no_grad():
        for name, loader in test_loaders.items():
            total_mse = 0
            count = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs)
                total_mse += F.mse_loss(pred, targets).item() * inputs.shape[0]
                count += inputs.shape[0]
            results[name] = np.sqrt(total_mse / count)  # RMSE

    return results


def main():
    print("=" * 70)
    print("EXTRAPOLATION FIX TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    train_seq_len = 50
    test_seq_lens = [50, 100, 200, 400]

    print(f"Training on length {train_seq_len}, testing on {test_seq_lens}")
    print()

    # Create datasets
    train_inputs, train_targets = create_lorenz_dataset(500, train_seq_len)
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=32, shuffle=True)

    test_loaders = {}
    for seq_len in test_seq_lens:
        test_inputs, test_targets = create_lorenz_dataset(100, seq_len)
        test_loaders[f"len_{seq_len}"] = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=16)

    # Test each variant
    variants = [
        ("Original (1/n)", PSIBlock_Original),
        ("Chunked (64)", lambda dim: PSIBlock_Chunked(dim, chunk_size=64)),
        ("EMA", PSIBlock_EMA),
    ]

    all_results = {}

    for name, block_class in variants:
        print(f"\nTraining {name}...")
        model = PSIModel(3, 3, dim=64, num_layers=4, block_class=block_class).to(device)
        results = train_and_evaluate(model, train_loader, test_loaders, epochs=30)
        all_results[name] = results

        print(f"  Results: ", end="")
        for seq_name, rmse in results.items():
            print(f"{seq_name}={rmse:.4f}  ", end="")
        print()

    # Summary
    print("\n" + "=" * 70)
    print("EXTRAPOLATION COMPARISON (RMSE)")
    print("=" * 70)
    print(f"{'Variant':<20}", end="")
    for seq_len in test_seq_lens:
        print(f"{'len_' + str(seq_len):<12}", end="")
    print("Degradation")
    print("-" * 70)

    for name, results in all_results.items():
        print(f"{name:<20}", end="")
        base = results["len_50"]
        for seq_len in test_seq_lens:
            print(f"{results[f'len_{seq_len}']:<12.4f}", end="")
        degradation = results[f"len_{test_seq_lens[-1]}"] / base
        print(f"{degradation:.2f}x")

    print()
    print("Lower degradation = better extrapolation")


if __name__ == "__main__":
    main()
