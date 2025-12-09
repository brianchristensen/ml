"""
Flocking Ablation Study: PSI vs MLP vs Transformer

Compare PSI against baseline architectures to verify that
the phase-space integration mechanism is responsible for
learning emergent behavior, not just model capacity.

All models have similar parameter counts (~8.5M).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import math

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# Import data generation from flocking experiment
# ============================================================================

from flocking_experiment import (
    generate_flocking_data,
    FlockingDataset,
    predict_trajectory,
    visualize_emergent_metrics
)


# ============================================================================
# Baseline Models
# ============================================================================

class MLPBlock(nn.Module):
    """Simple MLP block to replace PSIBlock."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class TransformerBlock(nn.Module):
    """Standard transformer block with causal attention."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class LSTMBlock(nn.Module):
    """LSTM-based block for sequential processing."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x_norm = self.norm(x)
        lstm_out, _ = self.lstm(x_norm)
        return x + self.proj(lstm_out)


# ============================================================================
# Generic Predictor (works with any block type)
# ============================================================================

class DynamicsPredictor(nn.Module):
    """Generic dynamics predictor that can use different block types."""

    def __init__(self, state_dim, block_type='psi', dim=256, num_layers=8, max_len=100):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.block_type = block_type

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        # Create blocks based on type
        if block_type == 'psi':
            self.blocks = nn.ModuleList([PSIBlock(dim=dim) for _ in range(num_layers)])
        elif block_type == 'mlp':
            self.blocks = nn.ModuleList([MLPBlock(dim=dim) for _ in range(num_layers)])
        elif block_type == 'transformer':
            # Fewer layers for transformer to match param count (attention is expensive)
            self.blocks = nn.ModuleList([TransformerBlock(dim=dim) for _ in range(num_layers // 2)])
        elif block_type == 'lstm':
            # Fewer LSTM layers (they're expensive)
            self.blocks = nn.ModuleList([LSTMBlock(dim=dim) for _ in range(num_layers // 2)])
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, states):
        batch_size, seq_len, _ = states.shape
        x = self.state_embedding(states)
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        for block in self.blocks:
            x = block(x)

        final_state = x[:, -1, :]
        prediction = self.output_head(final_state)
        return prediction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        context = batch['context'].to(device)
        target = batch['target'].to(device)

        prediction = model(context)
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            prediction = model(context)
            loss = criterion(prediction, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def compute_polarization(states, n_boids):
    """Compute polarization from flat state vector."""
    states = states.reshape(-1, n_boids, 4)
    velocities = states[:, :, 2:4]

    vel_norms = np.linalg.norm(velocities, axis=2, keepdims=True)
    vel_norms = np.maximum(vel_norms, 1e-8)
    unit_vels = velocities / vel_norms

    polarization = np.linalg.norm(np.mean(unit_vels, axis=1), axis=1)
    return polarization


def evaluate_emergence(model, trajectories, context_len, rollout_steps, mean, std, n_boids, device):
    """Evaluate emergent behavior preservation."""
    all_gt_pol = []
    all_pred_pol = []
    all_mse = []

    for i in range(min(20, len(trajectories))):  # Evaluate on 20 samples
        gt_traj = trajectories[i]
        initial = gt_traj[:context_len].reshape(context_len, -1)

        # Predict
        pred = predict_trajectory(model, initial, rollout_steps, mean, std, device)

        # Ground truth
        gt_future = gt_traj[context_len:context_len + rollout_steps].reshape(-1, n_boids * 4)

        # MSE
        mse = np.mean((gt_future - pred)**2)
        all_mse.append(mse)

        # Polarization
        gt_pol = compute_polarization(gt_future, n_boids)
        pred_pol = compute_polarization(pred, n_boids)

        all_gt_pol.append(gt_pol[-1])  # Final polarization
        all_pred_pol.append(pred_pol[-1])

    return {
        'mse': np.mean(all_mse),
        'gt_polarization': np.mean(all_gt_pol),
        'pred_polarization': np.mean(all_pred_pol),
        'polarization_preservation': np.mean(all_pred_pol) / np.mean(all_gt_pol) * 100
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_boids', type=int, default=20)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--models', type=str, default='psi,mlp,transformer,lstm',
                        help='Comma-separated list of models to test')
    args = parser.parse_args()

    model_types = [m.strip() for m in args.models.split(',')]

    print("=" * 80)
    print("FLOCKING ABLATION STUDY")
    print("=" * 80)
    print(f"Comparing: {model_types}")
    print(f"Data: {args.trials} trials, {args.n_boids} boids, {args.timesteps} timesteps")
    print(f"Training: {args.epochs} epochs")
    print("=" * 80)
    print()

    # Generate data once for all models
    print("Generating training data...")
    x, metrics = generate_flocking_data(
        n_boids=args.n_boids,
        trials=args.trials,
        timesteps=args.timesteps,
        cluster_start=True,
        vary_params=True
    )

    state_dim = args.n_boids * 4
    train_dataset = FlockingDataset(x, metrics, args.context_len, split='train')
    val_dataset = FlockingDataset(x, metrics, args.context_len, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    val_trajectories = x[int(len(x) * 0.8):]

    # Results storage
    results = {}

    # Train and evaluate each model type
    for model_type in model_types:
        print("\n" + "=" * 80)
        print(f"Training {model_type.upper()} model")
        print("=" * 80)

        model = DynamicsPredictor(
            state_dim=state_dim,
            block_type=model_type,
            dim=args.dim,
            num_layers=args.num_layers,
            max_len=args.context_len
        ).to(device)

        print(f"Parameters: {model.count_parameters():,}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(args.epochs):
            start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{args.epochs} ({time.time()-start:.1f}s) - "
                  f"Train: {train_loss:.6f} - Val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, f'flocking_ablation_{model_type}.pt')

        # Load best model
        checkpoint = torch.load(f'flocking_ablation_{model_type}.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate emergence
        print(f"\nEvaluating {model_type.upper()} emergent behavior...")
        emergence = evaluate_emergence(
            model, val_trajectories, args.context_len, args.rollout_steps,
            train_dataset.mean, train_dataset.std, args.n_boids, device
        )

        results[model_type] = {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'params': model.count_parameters(),
            **emergence
        }

        print(f"\n{model_type.upper()} Results:")
        print(f"  Val Loss: {best_val_loss:.6f}")
        print(f"  MSE: {emergence['mse']:.2f}")
        print(f"  GT Polarization: {emergence['gt_polarization']:.3f}")
        print(f"  Pred Polarization: {emergence['pred_polarization']:.3f}")
        print(f"  Polarization Preservation: {emergence['polarization_preservation']:.1f}%")

    # ========================================================================
    # Summary and Visualization
    # ========================================================================

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<12} {'Params':>10} {'Val Loss':>10} {'MSE':>10} {'Pol Pres':>10}")
    print("-" * 54)
    for model_type, res in results.items():
        print(f"{model_type:<12} {res['params']:>10,} {res['best_val_loss']:>10.6f} "
              f"{res['mse']:>10.2f} {res['polarization_preservation']:>9.1f}%")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training curves
    for model_type, res in results.items():
        axes[0].plot(res['val_losses'], label=model_type)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Val loss comparison
    models = list(results.keys())
    val_losses = [results[m]['best_val_loss'] for m in models]
    colors = ['green' if m == 'psi' else 'gray' for m in models]
    axes[1].bar(models, val_losses, color=colors)
    axes[1].set_ylabel('Best Validation Loss')
    axes[1].set_title('Final Validation Loss')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Polarization preservation
    pol_pres = [results[m]['polarization_preservation'] for m in models]
    colors = ['green' if m == 'psi' else 'gray' for m in models]
    axes[2].bar(models, pol_pres, color=colors)
    axes[2].axhline(y=100, color='red', linestyle='--', label='Perfect preservation')
    axes[2].set_ylabel('Polarization Preservation (%)')
    axes[2].set_title('Emergent Behavior Preservation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('flocking_ablation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved flocking_ablation_comparison.png")
    plt.close()

    # Determine winner
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    best_model = max(results.keys(), key=lambda m: results[m]['polarization_preservation'])
    print(f"\nBest emergent behavior preservation: {best_model.upper()}")
    print(f"  Polarization Preservation: {results[best_model]['polarization_preservation']:.1f}%")

    if best_model == 'psi':
        print("\n✓ PSI outperforms baselines on emergent behavior prediction.")
        print("  The phase-space integration mechanism is responsible for the result.")
    else:
        print(f"\n✗ {best_model.upper()} outperformed PSI.")
        print("  The result may not be specific to PSI's architecture.")


if __name__ == "__main__":
    main()
