"""
Lorenz Attractor Dynamics Learning with TPI

THE ULTIMATE TEST: Can TPI learn chaotic dynamics?

Lorenz system (weather chaos):
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

Properties:
- Nonlinear coupled equations
- Chaotic (sensitive to initial conditions)
- Strange attractor (butterfly shape)
- Multi-scale (fast oscillations + slow drift)

If TPI can predict Lorenz trajectories:
✅ Proves it learns arbitrary differential equations
✅ Handles chaos (exponential sensitivity!)
✅ Multi-scale integration captures attractor geometry
✅ Can generalize nonlinear dynamics

This is the smoking gun for universal dynamics learning!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from pathlib import Path

from psi import PhaseSpaceIntegrator

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Lorenz Attractor Generation
# ============================================================================

def lorenz_step(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    """
    One step of Lorenz system using Euler integration.

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return x + dx * dt, y + dy * dt, z + dz * dt


def generate_lorenz_trajectory(length=200, dt=0.01, x0=None, y0=None, z0=None):
    """
    Generate a single Lorenz attractor trajectory.

    Args:
        length: Number of time steps
        dt: Time step size
        x0, y0, z0: Initial conditions (random if None)

    Returns:
        trajectory: [length, 3] array of (x, y, z) states
    """
    # Random initial conditions if not provided
    if x0 is None:
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(5, 40)
    else:
        x, y, z = x0, y0, z0

    trajectory = np.zeros((length, 3), dtype=np.float32)

    # Integrate
    for i in range(length):
        trajectory[i] = [x, y, z]
        x, y, z = lorenz_step(x, y, z, dt=dt)

    return trajectory


def generate_lorenz_dataset(num_trajectories=10000, trajectory_length=200, dt=0.01, normalize=True):
    """
    Generate dataset of Lorenz trajectories.

    Returns:
        trajectories: [num_trajectories, trajectory_length, 3]
        stats: (mean, std) if normalize=True, else None
    """
    print(f"Generating {num_trajectories} Lorenz trajectories...")
    print(f"  Length: {trajectory_length} steps")
    print(f"  Time step: {dt}")
    print()

    trajectories = np.zeros((num_trajectories, trajectory_length, 3), dtype=np.float32)

    for i in range(num_trajectories):
        trajectories[i] = generate_lorenz_trajectory(trajectory_length, dt=dt)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{num_trajectories} trajectories")

    print("Dataset generation complete!")

    # Normalize to zero mean, unit variance
    if normalize:
        mean = trajectories.mean(axis=(0, 1))
        std = trajectories.std(axis=(0, 1))
        trajectories = (trajectories - mean) / (std + 1e-8)
        print(f"  Normalized - Mean: {mean}, Std: {std}")
        print()
        return trajectories, (mean, std)

    print()
    return trajectories, None


# ============================================================================
# Dataset
# ============================================================================

class LorenzDataset(Dataset):
    """
    Dataset for Lorenz dynamics learning.

    Given states [t-context_len:t], predict state at t+1.
    """

    def __init__(self, trajectories, context_len=20):
        """
        Args:
            trajectories: [num_trajectories, trajectory_length, 3]
            context_len: Number of past states to use as context
        """
        self.trajectories = trajectories
        self.context_len = context_len
        self.trajectory_length = trajectories.shape[1]

        # Number of valid sequences per trajectory
        self.seqs_per_traj = self.trajectory_length - context_len - 1

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        # Which trajectory?
        traj_idx = idx // self.seqs_per_traj
        # Which position in trajectory?
        pos = idx % self.seqs_per_traj

        trajectory = self.trajectories[traj_idx]

        # Context states and target state
        context = trajectory[pos:pos + self.context_len]  # [context_len, 3]
        target = trajectory[pos + self.context_len]  # [3]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


# ============================================================================
# TPI Dynamics Model
# ============================================================================

class TPIDynamicsPredictor(nn.Module):
    """
    TPI model for learning dynamical systems.

    Architecture:
    - Input: Sequence of states [batch, seq_len, state_dim]
    - TPI layers learn the dynamics (dx/dt = f(x))
    - Output: Next state [batch, state_dim]
    """

    def __init__(self, state_dim=3, dim=128, num_layers=6, max_len=50, device='cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.dim = dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device

        # State embedding (project 3D state to model dim)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        # TPI blocks (learn the dynamics!)
        from psi import PSIBlock
        self.blocks = nn.ModuleList([
            PSIBlock(dim=dim)
            for _ in range(num_layers)
        ])

        # Output: predict next state
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, state_dim)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        """Sinusoidal position encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, states):
        """
        Args:
            states: [batch, seq_len, state_dim] - sequence of states

        Returns:
            prediction: [batch, state_dim] - predicted next state
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        x = self.state_embedding(states)  # [batch, seq_len, dim]

        # Add positional encoding
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        # Apply TPI blocks (learn dynamics!)
        for block in self.blocks:
            x = block(x)

        # Predict next state from last position
        last_state = x[:, -1, :]  # [batch, dim]
        prediction = self.output_head(last_state)  # [batch, state_dim]

        return prediction

    def predict_trajectory(self, initial_context, num_steps):
        """
        AUTOREGRESSIVE multi-step prediction.

        Args:
            initial_context: [batch, context_len, state_dim]
            num_steps: Number of future states to predict

        Returns:
            predictions: [batch, num_steps, state_dim]
        """
        self.eval()
        predictions = []

        # Start with context
        current_sequence = initial_context.clone()

        with torch.no_grad():
            for step in range(num_steps):
                # Predict next state
                next_state = self.forward(current_sequence)  # [batch, state_dim]
                predictions.append(next_state)

                # Append prediction to sequence
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],  # Drop oldest state
                    next_state.unsqueeze(1)       # Add new prediction
                ], dim=1)

        predictions = torch.stack(predictions, dim=1)  # [batch, num_steps, state_dim]
        return predictions

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        context = batch['context'].to(device)  # [batch, context_len, 3]
        target = batch['target'].to(device)    # [batch, 3]

        # Forward
        prediction = model(context)

        # MSE loss
        loss = criterion(prediction, target)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed

            print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                  f"Loss: {avg_loss:.6f} - "
                  f"Speed: {samples_per_sec:.0f} samples/s")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate single-step prediction."""
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

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_trajectory_prediction(model, trajectories, context_len=20, predict_len=50, num_samples=100, device='cuda'):
    """
    Evaluate multi-step trajectory prediction.

    Tests how long TPI can maintain coherent predictions on chaotic attractor.
    """
    model.eval()

    print("Evaluating multi-step trajectory prediction...")
    print(f"  Predicting {predict_len} steps ahead")
    print()

    step_losses = np.zeros(predict_len)

    with torch.no_grad():
        for i in range(num_samples):
            # Random trajectory
            traj_idx = np.random.randint(0, len(trajectories))
            trajectory = trajectories[traj_idx]

            # Random starting point (with enough room for prediction)
            start_idx = np.random.randint(0, len(trajectory) - context_len - predict_len)

            # Context and ground truth
            context = trajectory[start_idx:start_idx + context_len]
            gt_future = trajectory[start_idx + context_len:start_idx + context_len + predict_len]

            # Predict
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model.predict_trajectory(context_tensor, predict_len)
            predictions = predictions[0].cpu().numpy()  # [predict_len, 3]

            # Compute loss per step
            for step in range(predict_len):
                loss = np.mean((predictions[step] - gt_future[step]) ** 2)
                step_losses[step] += loss

    # Average losses
    step_losses /= num_samples

    # Print results
    print("Per-step prediction errors (MSE):")
    for i in [0, 4, 9, 19, 49]:
        if i < len(step_losses):
            print(f"  Step +{i+1}: {step_losses[i]:.6f}")
    print()

    return step_losses


def visualize_trajectories(model, trajectories, context_len=20, predict_len=100, num_samples=3, device='cuda'):
    """Visualize predicted vs ground truth trajectories in 3D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()

    fig = plt.figure(figsize=(15, 5 * num_samples))

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Random trajectory
            traj_idx = np.random.randint(0, len(trajectories))
            trajectory = trajectories[traj_idx]

            # Random starting point
            start_idx = np.random.randint(0, len(trajectory) - context_len - predict_len)

            # Context and ground truth
            context = trajectory[start_idx:start_idx + context_len]
            gt_future = trajectory[start_idx + context_len:start_idx + context_len + predict_len]

            # Predict
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model.predict_trajectory(context_tensor, predict_len)
            predictions = predictions[0].cpu().numpy()

            # Plot
            ax = fig.add_subplot(num_samples, 1, sample_idx + 1, projection='3d')

            # Context (black)
            ax.plot(context[:, 0], context[:, 1], context[:, 2], 'k-', linewidth=2, label='Context')

            # Ground truth (blue)
            ax.plot(gt_future[:, 0], gt_future[:, 1], gt_future[:, 2], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)

            # Prediction (red)
            ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'r--', linewidth=2, label='TPI Prediction', alpha=0.7)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {sample_idx + 1}: Lorenz Trajectory Prediction')
            ax.legend()

    plt.tight_layout()
    plt.savefig('lorenz_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory visualization to lorenz_trajectories.png")
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("=" * 80)
    print("TPI Chaotic Dynamics Learning - Lorenz Attractor")
    print("THE ULTIMATE TEST: Can TPI learn chaos?")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    context_len = 20      # Use 20 past states to predict next
    trajectory_length = 200
    dt = 0.01

    batch_size = 64
    n_epochs = 2
    learning_rate = 1e-3

    # Model config
    state_dim = 3  # (x, y, z)
    dim = 128
    num_layers = 6

    print("Hyperparameters:")
    print(f"  State dimension: {state_dim} (x, y, z)")
    print(f"  Context length: {context_len} states")
    print(f"  Time step (dt): {dt}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Generate dataset
    print("=" * 80)
    print("Dataset Generation")
    print("=" * 80)
    print()

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

    # Create datasets
    train_dataset = LorenzDataset(train_trajectories, context_len=context_len)
    val_dataset = LorenzDataset(val_trajectories, context_len=context_len)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("=" * 80)
    print("Model Creation")
    print("=" * 80)
    model = TPIDynamicsPredictor(
        state_dim=state_dim,
        dim=dim,
        num_layers=num_layers,
        max_len=context_len + 10,
        device=device
    ).to(device)

    print(f"Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    print()

    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * n_epochs, eta_min=learning_rate * 0.1
    )
    criterion = nn.MSELoss()

    # Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)

        print()
        print(f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'state_dim': state_dim,
                    'dim': dim,
                    'num_layers': num_layers,
                    'context_len': context_len
                }
            }, 'lorenz_best.pt')
            print(f"  → Saved best model (Val Loss: {val_loss:.6f})")
            print()

        # Evaluate trajectory prediction every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("-" * 80)
            evaluate_trajectory_prediction(model, val_trajectories, context_len=context_len, predict_len=50, device=device)
            print("-" * 80)
            print()

    # Final evaluation
    print("=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    print()

    # Load best model
    checkpoint = torch.load('lorenz_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Single-step accuracy
    test_dataset = LorenzDataset(test_trajectories, context_len=context_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss (single-step): {test_loss:.6f}")
    print()

    # Multi-step trajectory prediction
    print("-" * 80)
    step_losses = evaluate_trajectory_prediction(
        model, test_trajectories,
        context_len=context_len,
        predict_len=100,
        num_samples=200,
        device=device
    )

    # Analyze error growth
    print("-" * 80)
    print("Error Growth Analysis:")
    print("-" * 80)
    print(f"Step 1 error: {step_losses[0]:.6f} (baseline)")
    for step in [9, 19, 49, 99]:
        if step < len(step_losses):
            error_growth = (step_losses[step] / step_losses[0] - 1) * 100
            print(f"Step {step+1} error: {step_losses[step]:.6f} (+{error_growth:.1f}% from step 1)")
    print()

    # Chaos analysis
    final_growth = (step_losses[-1] / step_losses[0] - 1) * 100
    if final_growth < 100:
        print("✅ REMARKABLE: Errors grow slowly (<100% over 100 steps)")
        print("   TPI learned chaotic dynamics!")
    elif final_growth < 300:
        print("✅ GOOD: Moderate error growth (100-300%)")
        print("   TPI captured chaotic attractor structure")
    elif final_growth < 1000:
        print("⚠️  FAIR: Significant error growth (300-1000%)")
        print("   TPI learned short-term dynamics")
    else:
        print("⚠️  CHALLENGING: Large error growth (>1000%)")
        print("   Chaos is hard! But TPI tried.")
    print()

    # Visualizations
    print("Generating trajectory visualizations...")
    visualize_trajectories(model, test_trajectories, context_len=context_len, predict_len=100, num_samples=5, device=device)
    print()

    print("=" * 80)
    print("CHAOS LEARNING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print()
    print("If TPI maintained trajectories on the attractor:")
    print("  → Learned dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz")
    print("  → Phase integration = learning differential equations")
    print("  → TPI is a universal dynamics learner!")
    print()


if __name__ == "__main__":
    main()
