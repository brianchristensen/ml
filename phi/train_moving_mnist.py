"""
Moving MNIST Video Prediction with TPI

Test TPI's temporal dynamics learning on pure motion prediction.

Dataset: Moving MNIST (2 digits moving in 64x64 frame)
Task: Given frames 1-10, predict frame 11
Goal: Prove TPI can learn velocity fields and motion dynamics

This tests TPI's core strength:
- Temporal dynamics (motion is pure dynamics!)
- Multi-scale integration (pixel jitter vs object motion)
- Phase trajectories encode motion states

If this works, TPI can learn any temporal dynamical system!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from pathlib import Path

from phi import ParallelHolographicIntegrator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Moving MNIST Dataset Generation
# ============================================================================

def generate_moving_mnist(num_sequences=10000, seq_len=20, image_size=64, num_digits=2):
    """
    Generate Moving MNIST dataset on the fly.

    Args:
        num_sequences: Number of sequences to generate
        seq_len: Length of each sequence
        image_size: Size of frames (image_size x image_size)
        num_digits: Number of moving digits per sequence

    Returns:
        sequences: [num_sequences, seq_len, image_size, image_size]
    """
    from torchvision import datasets

    print(f"Generating {num_sequences} Moving MNIST sequences...")

    # Load MNIST digits
    mnist = datasets.MNIST('data/mnist', train=True, download=True)
    digit_images = mnist.data.numpy()  # [60000, 28, 28]

    sequences = np.zeros((num_sequences, seq_len, image_size, image_size), dtype=np.float32)

    for seq_idx in range(num_sequences):
        # Sample random digits
        digit_indices = np.random.choice(len(digit_images), num_digits, replace=False)
        digits = digit_images[digit_indices]  # [num_digits, 28, 28]

        # Random initial positions and velocities
        positions = np.random.rand(num_digits, 2) * (image_size - 28)  # [num_digits, 2]
        velocities = (np.random.rand(num_digits, 2) - 0.5) * 4  # [-2, 2] pixels/frame

        # Generate sequence
        for frame_idx in range(seq_len):
            frame = np.zeros((image_size, image_size), dtype=np.float32)

            for digit_idx in range(num_digits):
                # Current position
                x, y = positions[digit_idx].astype(int)

                # Add digit to frame (with bounds checking)
                x_end = min(x + 28, image_size)
                y_end = min(y + 28, image_size)
                digit_crop = digits[digit_idx][:x_end-x, :y_end-y]
                frame[x:x_end, y:y_end] = np.maximum(
                    frame[x:x_end, y:y_end],
                    digit_crop
                )

                # Update position with velocity
                positions[digit_idx] += velocities[digit_idx]

                # Bounce off walls
                for dim in range(2):
                    if positions[digit_idx, dim] < 0:
                        positions[digit_idx, dim] = 0
                        velocities[digit_idx, dim] = -velocities[digit_idx, dim]
                    elif positions[digit_idx, dim] > image_size - 28:
                        positions[digit_idx, dim] = image_size - 28
                        velocities[digit_idx, dim] = -velocities[digit_idx, dim]

            sequences[seq_idx, frame_idx] = frame / 255.0  # Normalize to [0, 1]

        if (seq_idx + 1) % 1000 == 0:
            print(f"  Generated {seq_idx + 1}/{num_sequences} sequences")

    print("Dataset generation complete!")
    return sequences


class MovingMNISTDataset(Dataset):
    """
    Dataset for Moving MNIST video prediction.

    Given frames [0:context_len], predict frame [context_len].
    """

    def __init__(self, sequences, context_len=10):
        """
        Args:
            sequences: [num_sequences, seq_len, H, W]
            context_len: Number of frames to use as context
        """
        self.sequences = sequences
        self.context_len = context_len
        self.seq_len = sequences.shape[1]

        # Only use sequences long enough
        assert self.seq_len > context_len, "Sequences must be longer than context_len"

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # [seq_len, H, W]

        # Random starting point
        start_idx = np.random.randint(0, self.seq_len - self.context_len)

        # Context frames and target frame
        context_frames = sequence[start_idx:start_idx + self.context_len]  # [context_len, H, W]
        target_frame = sequence[start_idx + self.context_len]  # [H, W]

        # Flatten frames to vectors
        H, W = context_frames.shape[1:]
        context_flat = context_frames.reshape(self.context_len, H * W)  # [context_len, H*W]
        target_flat = target_frame.reshape(H * W)  # [H*W]

        return {
            'context': torch.tensor(context_flat, dtype=torch.float32),
            'target': torch.tensor(target_flat, dtype=torch.float32)
        }


# ============================================================================
# TPI Video Model
# ============================================================================

class TPIVideoPredictor(nn.Module):
    """
    TPI model adapted for video prediction.

    Architecture:
    - Input: Sequence of flattened frames [batch, seq_len, H*W]
    - TPI layers learn motion dynamics
    - Output: Predicted next frame [batch, H*W]
    """

    def __init__(self, frame_size, dim=256, num_layers=8, max_len=20, device='cuda'):
        super().__init__()

        self.frame_size = frame_size  # H*W
        self.dim = dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device

        # Frame embedding (project flattened frame to model dim)
        self.frame_embedding = nn.Sequential(
            nn.Linear(frame_size, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # Sinusoidal positional encoding
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        # TPI blocks (these learn the motion dynamics!)
        from phi import PHIBlock
        self.blocks = nn.ModuleList([
            PHIBlock(dim=dim)
            for _ in range(num_layers)
        ])

        # Output: predict next frame
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, frame_size)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        """Sinusoidal position encoding."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, frames):
        """
        Args:
            frames: [batch, seq_len, frame_size] - sequence of flattened frames

        Returns:
            prediction: [batch, frame_size] - predicted next frame
        """
        batch_size, seq_len, _ = frames.shape

        # Embed frames
        x = self.frame_embedding(frames)  # [batch, seq_len, dim]

        # Add positional encoding
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        # Apply TPI blocks (learn motion dynamics!)
        for block in self.blocks:
            x = block(x)

        # Predict next frame from last position
        last_state = x[:, -1, :]  # [batch, dim]
        prediction = self.output_head(last_state)  # [batch, frame_size]

        return prediction

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

        context = batch['context'].to(device)  # [batch, context_len, H*W]
        target = batch['target'].to(device)  # [batch, H*W]

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
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed

            print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                  f"Loss: {avg_loss:.6f} - "
                  f"Speed: {samples_per_sec:.0f} samples/s")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model."""
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


def visualize_predictions(model, dataset, num_samples=5, device='cuda'):
    """Visualize predictions vs ground truth."""
    import matplotlib.pyplot as plt

    model.eval()

    image_size = int(np.sqrt(dataset.sequences.shape[-1] * dataset.sequences.shape[-2]))

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            sample = dataset[i]
            context = sample['context'].unsqueeze(0).to(device)  # [1, context_len, H*W]
            target = sample['target'].numpy()  # [H*W]

            # Predict
            prediction = model(context).cpu().numpy()[0]  # [H*W]

            # Reshape for visualization
            H = W = image_size
            last_frame = context[0, -1].cpu().numpy().reshape(H, W)
            target_frame = target.reshape(H, W)
            pred_frame = prediction.reshape(H, W)

            # Plot
            axes[i, 0].imshow(last_frame, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Last Context Frame')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(target_frame, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth Next Frame')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('TPI Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('moving_mnist_predictions.png', dpi=150)
    print("Saved predictions to moving_mnist_predictions.png")
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("=" * 80)
    print("TPI Video Prediction - Moving MNIST")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    image_size = 64
    context_len = 10  # Use 10 frames to predict frame 11
    num_digits = 2
    seq_len = 20

    batch_size = 32
    n_epochs = 20
    learning_rate = 1e-3

    # Model config
    dim = 256
    num_layers = 8

    print("Hyperparameters:")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Context length: {context_len} frames")
    print(f"  Number of digits: {num_digits}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Generate dataset
    train_sequences = generate_moving_mnist(
        num_sequences=8000,
        seq_len=seq_len,
        image_size=image_size,
        num_digits=num_digits
    )

    val_sequences = generate_moving_mnist(
        num_sequences=1000,
        seq_len=seq_len,
        image_size=image_size,
        num_digits=num_digits
    )

    test_sequences = generate_moving_mnist(
        num_sequences=1000,
        seq_len=seq_len,
        image_size=image_size,
        num_digits=num_digits
    )

    print()

    # Create datasets
    train_dataset = MovingMNISTDataset(train_sequences, context_len=context_len)
    val_dataset = MovingMNISTDataset(val_sequences, context_len=context_len)
    test_dataset = MovingMNISTDataset(test_sequences, context_len=context_len)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    frame_size = image_size * image_size
    model = TPIVideoPredictor(
        frame_size=frame_size,
        dim=dim,
        num_layers=num_layers,
        max_len=context_len + 5,  # Slightly more than needed
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
                    'frame_size': frame_size,
                    'dim': dim,
                    'num_layers': num_layers,
                    'context_len': context_len,
                    'image_size': image_size
                }
            }, 'moving_mnist_best.pt')
            print(f"  → Saved best model (Val Loss: {val_loss:.6f})")
            print()

        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Generating visualization...")
            visualize_predictions(model, val_dataset, num_samples=5, device=device)
            print()

    # Final test evaluation
    print("=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)
    print()

    # Load best model
    checkpoint = torch.load('moving_mnist_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print()

    # Final visualization
    print("Generating final predictions...")
    visualize_predictions(model, test_dataset, num_samples=10, device=device)

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print()
    print("If TPI learned motion dynamics, predictions should show:")
    print("  - Correct digit positions")
    print("  - Correct velocities")
    print("  - Smooth trajectories")
    print()
    print("This proves TPI can learn temporal dynamical systems!")
    print()


if __name__ == "__main__":
    main()
