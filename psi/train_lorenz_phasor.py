"""
Lorenz Attractor Dynamics Learning with Optimal Phasor Model

Testing the orthogonal phasor (O(n) associative memory) on chaotic dynamics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Optimal Phasor Model (from phasor_optimal.py)
# ============================================================================

class PureOrthoPhasor(nn.Module):
    """Orthogonal phasor with random fixed phases."""
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.dim = dim
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)

        value = self.to_value(x)

        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)

        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)

        return x + self.to_out(retrieved)


class PhasorDynamicsPredictor(nn.Module):
    """
    Optimal Phasor model for learning dynamical systems.

    Architecture:
    - Input: Sequence of states [batch, seq_len, state_dim]
    - 2 Phasor layers (minimal architecture that works)
    - Output: Next state [batch, state_dim]
    """

    def __init__(self, state_dim=3, dim=128, max_len=50, device='cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.dim = dim
        self.max_len = max_len
        self.device = device

        # State embedding (project 3D state to model dim)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # 2 Phasor layers (minimal architecture)
        self.norm1 = nn.LayerNorm(dim)
        self.phasor1 = PureOrthoPhasor(dim, max_len)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor2 = PureOrthoPhasor(dim, max_len)

        # Output: predict next state
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, state_dim)
        )

    def forward(self, states):
        """
        Args:
            states: [batch, seq_len, state_dim] - sequence of states

        Returns:
            prediction: [batch, state_dim] - predicted next state
        """
        # Embed states
        x = self.state_embedding(states)  # [batch, seq_len, dim]

        # Apply phasor layers
        x = x + self.phasor1(self.norm1(x))
        x = x + self.phasor2(self.norm2(x))

        # Predict next state from last position
        last_state = x[:, -1, :]  # [batch, dim]
        prediction = self.output_head(last_state)  # [batch, state_dim]

        return prediction

    def predict_trajectory(self, initial_context, num_steps):
        """Autoregressive multi-step prediction."""
        self.eval()
        predictions = []
        current_sequence = initial_context.clone()

        with torch.no_grad():
            for step in range(num_steps):
                next_state = self.forward(current_sequence)
                predictions.append(next_state)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    next_state.unsqueeze(1)
                ], dim=1)

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Lorenz Attractor Generation (same as train_lorenz.py)
# ============================================================================

def lorenz_step(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx * dt, y + dy * dt, z + dz * dt


def generate_lorenz_trajectory(length=200, dt=0.01, x0=None, y0=None, z0=None):
    if x0 is None:
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(5, 40)
    else:
        x, y, z = x0, y0, z0

    trajectory = np.zeros((length, 3), dtype=np.float32)
    for i in range(length):
        trajectory[i] = [x, y, z]
        x, y, z = lorenz_step(x, y, z, dt=dt)

    return trajectory


def generate_lorenz_dataset(num_trajectories=10000, trajectory_length=200, dt=0.01, normalize=True):
    print(f"Generating {num_trajectories} Lorenz trajectories...")
    trajectories = np.zeros((num_trajectories, trajectory_length, 3), dtype=np.float32)

    for i in range(num_trajectories):
        trajectories[i] = generate_lorenz_trajectory(trajectory_length, dt=dt)

    print("Dataset generation complete!")

    if normalize:
        mean = trajectories.mean(axis=(0, 1))
        std = trajectories.std(axis=(0, 1))
        trajectories = (trajectories - mean) / (std + 1e-8)
        print(f"  Normalized - Mean: {mean}, Std: {std}")
        return trajectories, (mean, std)

    return trajectories, None


class LorenzDataset(Dataset):
    def __init__(self, trajectories, context_len=20):
        self.trajectories = trajectories
        self.context_len = context_len
        self.trajectory_length = trajectories.shape[1]
        self.seqs_per_traj = self.trajectory_length - context_len - 1

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


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        context = batch['context'].to(device)
        target = batch['target'].to(device)

        prediction = model(context)
        loss = criterion(prediction, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)

            prediction = model(context)
            loss = nn.functional.mse_loss(prediction, target)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def evaluate_trajectory_prediction(model, trajectories, context_len=20, predict_len=50, num_samples=100, device='cuda'):
    model.eval()

    step_losses = np.zeros(predict_len)

    with torch.no_grad():
        for i in range(num_samples):
            traj_idx = np.random.randint(0, len(trajectories))
            trajectory = trajectories[traj_idx]

            start_idx = np.random.randint(0, len(trajectory) - context_len - predict_len)

            context = trajectory[start_idx:start_idx + context_len]
            gt_future = trajectory[start_idx + context_len:start_idx + context_len + predict_len]

            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model.predict_trajectory(context_tensor, predict_len)
            predictions = predictions[0].cpu().numpy()

            for step in range(predict_len):
                loss = np.mean((predictions[step] - gt_future[step]) ** 2)
                step_losses[step] += loss

    step_losses /= num_samples
    return step_losses


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("OPTIMAL PHASOR - Lorenz Attractor Dynamics")
    print("Testing O(n) associative memory on chaotic dynamics")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    print()

    # Hyperparameters
    context_len = 20
    trajectory_length = 200
    dt = 0.01

    batch_size = 256  # Optimal batch size
    n_epochs = 10
    learning_rate = 5e-3  # Optimal LR

    # Model config
    state_dim = 3
    dim = 128

    print("Hyperparameters:")
    print(f"  Context length: {context_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model dim: {dim}")
    print()

    # Generate dataset
    print("=" * 80)
    print("Dataset Generation")
    print("=" * 80)

    train_trajectories, train_stats = generate_lorenz_dataset(
        num_trajectories=8000,
        trajectory_length=trajectory_length,
        dt=dt,
        normalize=True
    )

    val_trajectories, _ = generate_lorenz_dataset(
        num_trajectories=1000,
        trajectory_length=trajectory_length,
        dt=dt,
        normalize=True
    )

    test_trajectories, _ = generate_lorenz_dataset(
        num_trajectories=1000,
        trajectory_length=trajectory_length,
        dt=dt,
        normalize=True
    )

    train_dataset = LorenzDataset(train_trajectories, context_len=context_len)
    val_dataset = LorenzDataset(val_trajectories, context_len=context_len)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("=" * 80)
    print("Model Creation")
    print("=" * 80)
    model = PhasorDynamicsPredictor(
        state_dim=state_dim,
        dim=dim,
        max_len=context_len + 10,
        device=device
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lorenz_phasor_best.pt')

        # Trajectory eval every 5 epochs
        if (epoch + 1) % 5 == 0:
            step_losses = evaluate_trajectory_prediction(
                model, val_trajectories,
                context_len=context_len,
                predict_len=50,
                num_samples=100,
                device=device
            )
            print(f"  Multi-step errors: step1={step_losses[0]:.4f}, step10={step_losses[9]:.4f}, step50={step_losses[49]:.4f}")

    # Final evaluation
    print()
    print("=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    print()

    model.load_state_dict(torch.load('lorenz_phasor_best.pt'))

    test_dataset = LorenzDataset(test_trajectories, context_len=context_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss (single-step): {test_loss:.6f}")

    # Multi-step trajectory prediction
    step_losses = evaluate_trajectory_prediction(
        model, test_trajectories,
        context_len=context_len,
        predict_len=100,
        num_samples=200,
        device=device
    )

    print()
    print("Per-step prediction errors (MSE):")
    for i in [0, 9, 19, 49, 99]:
        if i < len(step_losses):
            print(f"  Step +{i+1}: {step_losses[i]:.6f}")

    # Error growth analysis
    print()
    print("Error Growth Analysis:")
    print(f"Step 1 error: {step_losses[0]:.6f} (baseline)")
    for step in [9, 19, 49, 99]:
        if step < len(step_losses):
            error_growth = (step_losses[step] / step_losses[0] - 1) * 100
            print(f"Step {step+1} error: {step_losses[step]:.6f} (+{error_growth:.1f}%)")

    # Verdict
    final_growth = (step_losses[-1] / step_losses[0] - 1) * 100
    print()
    if final_growth < 100:
        print("EXCELLENT: Error growth <100% - Phasor learned chaotic dynamics!")
    elif final_growth < 300:
        print("GOOD: Error growth 100-300% - Captured attractor structure")
    elif final_growth < 1000:
        print("FAIR: Error growth 300-1000% - Learned short-term dynamics")
    else:
        print("CHALLENGING: Error growth >1000% - Chaos is hard!")


if __name__ == "__main__":
    main()
