"""
Moving MNIST Multi-Step Video Prediction with TPI

THE SMOKING GUN TEST:
- Given frames 1-10, predict frames 11-15 (5 steps into future)
- Autoregressive: each prediction feeds into next
- Tests if TPI learned the ACTUAL dynamical system

If TPI truly learned dx/dt = v, then:
✅ Multi-step predictions should maintain velocity
✅ Trajectories should stay coherent over 5 steps
✅ Errors should grow slowly (not explode)

If TPI just memorized 1-step transitions:
❌ Errors explode after 2-3 steps
❌ Trajectories diverge wildly
❌ Predicted motion becomes incoherent

This experiment proves whether TPI is a true dynamics learner!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from pathlib import Path

from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Moving MNIST Dataset Generation (same as before)
# ============================================================================

def generate_moving_mnist(num_sequences=10000, seq_len=20, image_size=64, num_digits=2):
    """Generate Moving MNIST dataset on the fly."""
    from torchvision import datasets

    print(f"Generating {num_sequences} Moving MNIST sequences...")

    mnist = datasets.MNIST('data/mnist', train=True, download=True)
    digit_images = mnist.data.numpy()

    sequences = np.zeros((num_sequences, seq_len, image_size, image_size), dtype=np.float32)

    for seq_idx in range(num_sequences):
        digit_indices = np.random.choice(len(digit_images), num_digits, replace=False)
        digits = digit_images[digit_indices]

        positions = np.random.rand(num_digits, 2) * (image_size - 28)
        velocities = (np.random.rand(num_digits, 2) - 0.5) * 4

        for frame_idx in range(seq_len):
            frame = np.zeros((image_size, image_size), dtype=np.float32)

            for digit_idx in range(num_digits):
                x, y = positions[digit_idx].astype(int)
                x_end = min(x + 28, image_size)
                y_end = min(y + 28, image_size)
                digit_crop = digits[digit_idx][:x_end-x, :y_end-y]
                frame[x:x_end, y:y_end] = np.maximum(
                    frame[x:x_end, y:y_end],
                    digit_crop
                )

                positions[digit_idx] += velocities[digit_idx]

                for dim in range(2):
                    if positions[digit_idx, dim] < 0:
                        positions[digit_idx, dim] = 0
                        velocities[digit_idx, dim] = -velocities[digit_idx, dim]
                    elif positions[digit_idx, dim] > image_size - 28:
                        positions[digit_idx, dim] = image_size - 28
                        velocities[digit_idx, dim] = -velocities[digit_idx, dim]

            sequences[seq_idx, frame_idx] = frame / 255.0

        if (seq_idx + 1) % 1000 == 0:
            print(f"  Generated {seq_idx + 1}/{num_sequences} sequences")

    print("Dataset generation complete!")
    return sequences


class MovingMNISTMultiStepDataset(Dataset):
    """
    Dataset for MULTI-STEP video prediction.

    Given frames [0:context_len], predict frames [context_len:context_len+predict_len].
    """

    def __init__(self, sequences, context_len=10, predict_len=5):
        """
        Args:
            sequences: [num_sequences, seq_len, H, W]
            context_len: Number of frames to use as context
            predict_len: Number of frames to predict
        """
        self.sequences = sequences
        self.context_len = context_len
        self.predict_len = predict_len
        self.seq_len = sequences.shape[1]

        assert self.seq_len >= context_len + predict_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Random starting point
        max_start = self.seq_len - self.context_len - self.predict_len
        start_idx = np.random.randint(0, max_start + 1)

        # Context frames and target frames
        context_frames = sequence[start_idx:start_idx + self.context_len]
        target_frames = sequence[start_idx + self.context_len:start_idx + self.context_len + self.predict_len]

        # Flatten
        H, W = context_frames.shape[1:]
        context_flat = context_frames.reshape(self.context_len, H * W)
        target_flat = target_frames.reshape(self.predict_len, H * W)

        return {
            'context': torch.tensor(context_flat, dtype=torch.float32),
            'target': torch.tensor(target_flat, dtype=torch.float32)
        }


# ============================================================================
# TPI Video Model (same as before)
# ============================================================================

class TPIVideoPredictor(nn.Module):
    """TPI model for video prediction."""

    def __init__(self, frame_size, dim=256, num_layers=8, max_len=20, device='cuda'):
        super().__init__()

        self.frame_size = frame_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device

        # Frame embedding
        self.frame_embedding = nn.Sequential(
            nn.Linear(frame_size, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        # TPI blocks
        from novel_attention import TPIBlock
        self.blocks = nn.ModuleList([
            TPIBlock(dim=dim)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, frame_size)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, frames):
        """
        Args:
            frames: [batch, seq_len, frame_size]
        Returns:
            prediction: [batch, frame_size]
        """
        batch_size, seq_len, _ = frames.shape

        x = self.frame_embedding(frames)
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        for block in self.blocks:
            x = block(x)

        last_state = x[:, -1, :]
        prediction = self.output_head(last_state)

        return prediction

    def predict_multistep(self, context_frames, num_steps):
        """
        AUTOREGRESSIVE multi-step prediction.

        Args:
            context_frames: [batch, context_len, frame_size]
            num_steps: Number of future frames to predict

        Returns:
            predictions: [batch, num_steps, frame_size]
        """
        self.eval()
        batch_size = context_frames.shape[0]
        predictions = []

        # Start with context
        current_sequence = context_frames.clone()

        with torch.no_grad():
            for step in range(num_steps):
                # Predict next frame
                next_frame = self.forward(current_sequence)  # [batch, frame_size]
                predictions.append(next_frame)

                # Append prediction to sequence (keep last context_len frames)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],  # Drop oldest frame
                    next_frame.unsqueeze(1)       # Add new prediction
                ], dim=1)

        predictions = torch.stack(predictions, dim=1)  # [batch, num_steps, frame_size]
        return predictions

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train one epoch with MULTI-STEP prediction.

    For each step, we train on single-step prediction to avoid
    accumulating errors during training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        context = batch['context'].to(device)  # [batch, context_len, H*W]
        targets = batch['target'].to(device)   # [batch, predict_len, H*W]

        # Train on all prediction steps
        # Use teacher forcing: always use ground truth for context
        loss = 0.0
        for step in range(targets.shape[1]):
            # Predict step
            prediction = model(context)

            # Loss for this step
            step_loss = criterion(prediction, targets[:, step, :])
            loss += step_loss

            # Update context for next step (teacher forcing)
            if step < targets.shape[1] - 1:
                context = torch.cat([
                    context[:, 1:, :],
                    targets[:, step, :].unsqueeze(1)
                ], dim=1)

        # Average loss over steps
        loss = loss / targets.shape[1]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed

            print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                  f"Loss: {avg_loss:.6f} - "
                  f"Speed: {samples_per_sec:.0f} samples/s")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_multistep(model, dataloader, device, num_steps):
    """
    Evaluate MULTI-STEP prediction with autoregressive generation.

    Returns per-step losses to see error accumulation.
    """
    model.eval()
    step_losses = [0.0] * num_steps
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            targets = batch['target'].to(device)

            # Autoregressive prediction
            predictions = model.predict_multistep(context, num_steps)

            # Compute loss per step
            for step in range(num_steps):
                loss = nn.functional.mse_loss(predictions[:, step, :], targets[:, step, :])
                step_losses[step] += loss.item()

            num_batches += 1

    # Average losses
    step_losses = [loss / num_batches for loss in step_losses]
    avg_loss = sum(step_losses) / len(step_losses)

    return avg_loss, step_losses


def visualize_multistep_predictions(model, dataset, num_samples=5, device='cuda'):
    """Visualize multi-step predictions."""
    import matplotlib.pyplot as plt

    model.eval()
    image_size = int(np.sqrt(dataset.sequences.shape[2] * dataset.sequences.shape[3]))
    predict_len = dataset.predict_len

    fig, axes = plt.subplots(num_samples, predict_len + 2, figsize=(2 * (predict_len + 2), 2 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            context = sample['context'].unsqueeze(0).to(device)
            targets = sample['target'].numpy()

            # Multi-step prediction
            predictions = model.predict_multistep(context, predict_len).cpu().numpy()[0]

            # Last context frame
            last_frame = context[0, -1].cpu().numpy().reshape(image_size, image_size)
            axes[i, 0].imshow(last_frame, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Last Context' if i == 0 else '')
            axes[i, 0].axis('off')

            # Ground truth future frames
            for step in range(predict_len):
                gt_frame = targets[step].reshape(image_size, image_size)
                axes[i, step + 1].imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
                axes[i, step + 1].set_title(f'GT +{step+1}' if i == 0 else '')
                axes[i, step + 1].axis('off')

            # Predicted trajectory (last predicted frame)
            pred_frame = predictions[-1].reshape(image_size, image_size)
            axes[i, -1].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
            axes[i, -1].set_title(f'Pred +{predict_len}' if i == 0 else '')
            axes[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig('moving_mnist_multistep_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved predictions to moving_mnist_multistep_predictions.png")
    plt.close()

    # Also create trajectory comparison plot
    fig, axes = plt.subplots(num_samples, predict_len * 2, figsize=(2 * predict_len * 2, 2 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            context = sample['context'].unsqueeze(0).to(device)
            targets = sample['target'].numpy()

            predictions = model.predict_multistep(context, predict_len).cpu().numpy()[0]

            for step in range(predict_len):
                # Ground truth
                gt_frame = targets[step].reshape(image_size, image_size)
                axes[i, step * 2].imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
                axes[i, step * 2].set_title(f'GT +{step+1}' if i == 0 else '')
                axes[i, step * 2].axis('off')

                # Prediction
                pred_frame = predictions[step].reshape(image_size, image_size)
                axes[i, step * 2 + 1].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
                axes[i, step * 2 + 1].set_title(f'Pred +{step+1}' if i == 0 else '')
                axes[i, step * 2 + 1].axis('off')

    plt.tight_layout()
    plt.savefig('moving_mnist_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory comparison to moving_mnist_trajectory_comparison.png")
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    print("=" * 80)
    print("TPI Multi-Step Video Prediction - Moving MNIST")
    print("THE SMOKING GUN: Does TPI learn the dynamical system?")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Hyperparameters
    image_size = 64
    context_len = 10
    predict_len = 5  # Predict 5 frames into future!
    num_digits = 2
    seq_len = 25  # Need longer sequences for multi-step

    batch_size = 32
    n_epochs = 30  # More epochs for multi-step
    learning_rate = 1e-3

    # Model config
    dim = 256
    num_layers = 8

    print("Hyperparameters:")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Context length: {context_len} frames")
    print(f"  Predict length: {predict_len} frames (MULTI-STEP!)")
    print(f"  Number of digits: {num_digits}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model: {num_layers}L / {dim}D")
    print()

    # Generate dataset
    print("=" * 80)
    print("Dataset Generation")
    print("=" * 80)

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
    train_dataset = MovingMNISTMultiStepDataset(
        train_sequences, context_len=context_len, predict_len=predict_len
    )
    val_dataset = MovingMNISTMultiStepDataset(
        val_sequences, context_len=context_len, predict_len=predict_len
    )
    test_dataset = MovingMNISTMultiStepDataset(
        test_sequences, context_len=context_len, predict_len=predict_len
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("=" * 80)
    print("Model Creation")
    print("=" * 80)
    frame_size = image_size * image_size
    model = TPIVideoPredictor(
        frame_size=frame_size,
        dim=dim,
        num_layers=num_layers,
        max_len=context_len + predict_len,
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
    print("Training (Multi-Step with Teacher Forcing)")
    print("=" * 80)
    print()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate (autoregressive multi-step)
        val_loss, val_step_losses = evaluate_multistep(model, val_loader, device, predict_len)

        print()
        print(f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        print(f"Per-step Val Losses:")
        for step, step_loss in enumerate(val_step_losses, 1):
            print(f"  Step +{step}: {step_loss:.6f}")
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
                'val_step_losses': val_step_losses,
                'config': {
                    'frame_size': frame_size,
                    'dim': dim,
                    'num_layers': num_layers,
                    'context_len': context_len,
                    'predict_len': predict_len,
                    'image_size': image_size
                }
            }, 'moving_mnist_multistep_best.pt')
            print(f"  → Saved best model (Val Loss: {val_loss:.6f})")
            print()

        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Generating visualization...")
            visualize_multistep_predictions(model, val_dataset, num_samples=5, device=device)
            print()

    # Final test evaluation
    print("=" * 80)
    print("Final Test Evaluation (Autoregressive Multi-Step)")
    print("=" * 80)
    print()

    # Load best model
    checkpoint = torch.load('moving_mnist_multistep_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_step_losses = evaluate_multistep(model, test_loader, device, predict_len)
    print(f"Test Loss (Average): {test_loss:.6f}")
    print()
    print(f"Per-step Test Losses:")
    for step, step_loss in enumerate(test_step_losses, 1):
        print(f"  Step +{step}: {step_loss:.6f}")
    print()

    # Analyze error growth
    print("-" * 80)
    print("Error Growth Analysis:")
    print("-" * 80)
    print(f"Step 1 error: {test_step_losses[0]:.6f} (baseline)")
    for step in range(1, len(test_step_losses)):
        error_growth = (test_step_losses[step] / test_step_losses[0] - 1) * 100
        print(f"Step {step+1} error: {test_step_losses[step]:.6f} (+{error_growth:.1f}% from step 1)")
    print()

    final_growth = (test_step_losses[-1] / test_step_losses[0] - 1) * 100
    if final_growth < 50:
        print("✅ EXCELLENT: Errors grow slowly (<50% increase over 5 steps)")
        print("   TPI learned the dynamical system!")
    elif final_growth < 100:
        print("✅ GOOD: Errors grow moderately (50-100% increase)")
        print("   TPI captured most of the dynamics")
    elif final_growth < 200:
        print("⚠️  FAIR: Errors double (100-200% increase)")
        print("   TPI learned short-term dynamics but struggles long-term")
    else:
        print("❌ POOR: Errors explode (>200% increase)")
        print("   TPI did not learn the dynamical system")
    print()

    # Final visualization
    print("Generating final predictions...")
    visualize_multistep_predictions(model, test_dataset, num_samples=10, device=device)

    print()
    print("=" * 80)
    print("MULTI-STEP TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print()
    print("Check the visualizations:")
    print("  - moving_mnist_multistep_predictions.png")
    print("  - moving_mnist_trajectory_comparison.png")
    print()
    print("If trajectories stay coherent over 5 steps:")
    print("  → TPI learned dx/dt = v(x)")
    print("  → Phase integration = velocity field integration")
    print("  → TPI is a universal temporal dynamics learner!")
    print()


if __name__ == "__main__":
    main()
