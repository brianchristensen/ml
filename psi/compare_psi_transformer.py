"""
PSI vs Transformer Comparison

This script rigorously compares PSI and Transformer on the same tasks to determine
whether the "phase space" properties we observed are unique to PSI or are
simply properties of any trained neural network.

Tests:
1. Task performance (RMSE)
2. Hidden state correlation with true state
3. Untrained model correlation (control)
4. Graceful degradation under sensor dropout

If Transformer shows similar correlations to PSI, then the findings are about
the TASK, not the architecture. If PSI shows uniquely higher correlations,
then there may be something special about the phase space formulation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from transformer_baseline import (
    TransformerLorenzFusionModel,
    TransformerWithHooks,
    train_transformer_lorenz_fusion,
    extract_transformer_representations,
    compute_state_correlations
)

from lorenz_fusion_experiment import (
    PSILorenzFusionModel,
    LorenzFusionDataset,
    SENSORS,
    compute_input_dim,
)

from lorenz_fusion_phase_viz import (
    LorenzFusionPSIWithHooks,
    extract_phase_data
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def extract_psi_final_layer(model, dataset, device, n_samples=500):
    """Extract final layer phi from PSI model."""
    hooked_model = LorenzFusionPSIWithHooks(model)
    hooked_model.eval()

    all_phi = []
    all_targets = []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting PSI phi"):
            inputs, target = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)

            _ = hooked_model(inputs)
            activations = hooked_model.activations

            # Get final layer phi at last timestep
            num_layers = len(model.psi_blocks)
            phi = activations[f'layer_{num_layers-1}_phi'].cpu().numpy()[0, -1, :]

            all_phi.append(phi)
            all_targets.append(target.numpy())

    return np.array(all_phi), np.array(all_targets)


def extract_transformer_final_layer(model, dataset, device, n_samples=500):
    """Extract final layer hidden states from Transformer model."""
    hooked_model = TransformerWithHooks(model)
    hooked_model.eval()

    all_hidden = []
    all_targets = []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting Transformer hidden"):
            inputs, target = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)

            _ = hooked_model(inputs)
            activations = hooked_model.get_activations()

            # Get final layer hidden at last timestep
            num_layers = len(model.transformer_blocks)
            hidden = activations[f'layer_{num_layers-1}_hidden'].cpu().numpy()[0, -1, :]

            all_hidden.append(hidden)
            all_targets.append(target.numpy())

    return np.array(all_hidden), np.array(all_targets)


def compute_correlations(representations, targets):
    """Compute max correlation per state variable."""
    state_names = ['x', 'y', 'z']
    n_samples, dim = representations.shape

    correlations = {}
    best_dims = {}

    for s_idx, s_name in enumerate(state_names):
        corrs = []
        for d in range(dim):
            c = np.corrcoef(representations[:, d], targets[:, s_idx])[0, 1]
            corrs.append(c if not np.isnan(c) else 0)

        best_dim = np.argmax(np.abs(corrs))
        correlations[s_name] = corrs[best_dim]
        best_dims[s_name] = best_dim

    return correlations, best_dims


def evaluate_model_performance(model, test_loader, device):
    """Evaluate model RMSE on test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            pred = model(inputs)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    per_dim_rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))

    return rmse, per_dim_rmse


def main():
    parser = argparse.ArgumentParser(description='Compare PSI vs Transformer')
    parser.add_argument('--n_trajectories', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and load existing models')
    args = parser.parse_args()

    print("=" * 80)
    print("PSI vs TRANSFORMER COMPARISON")
    print("=" * 80)
    print()
    print("This test determines whether PSI's 'phase space' properties are unique")
    print("or simply properties of any trained neural network.")
    print()
    print("If Transformer shows similar state correlations → findings are about the TASK")
    print("If PSI shows uniquely higher correlations → PSI may have special properties")
    print("=" * 80)

    # Create datasets
    print("\n1. CREATING DATASETS")
    print("-" * 40)

    train_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories,
        seq_len=50,
        split='train'
    )

    val_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories // 5,
        seq_len=50,
        split='val'
    )

    test_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories // 5,
        seq_len=50,
        split='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_dim = compute_input_dim(len(SENSORS))
    print(f"Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Train or load models
    print("\n2. TRAINING MODELS")
    print("-" * 40)

    # PSI Model
    psi_model = PSILorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    # Transformer Model
    transformer_model = TransformerLorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    psi_params = sum(p.numel() for p in psi_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"PSI parameters: {psi_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")

    if not args.skip_training:
        # Train PSI
        print("\nTraining PSI model...")
        psi_optimizer = torch.optim.AdamW(psi_model.parameters(), lr=1e-4, weight_decay=1e-4)
        psi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(psi_optimizer, args.epochs)

        best_psi_loss = float('inf')
        for epoch in range(args.epochs):
            psi_model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"PSI Epoch {epoch+1}/{args.epochs}"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                psi_optimizer.zero_grad()
                # PSI Lorenz fusion outputs [batch, state_dim] directly
                pred = psi_model(inputs)
                loss = nn.functional.mse_loss(pred, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(psi_model.parameters(), 1.0)
                psi_optimizer.step()
                train_loss += loss.item()

            psi_scheduler.step()
            train_loss /= len(train_loader)

            # Validation
            psi_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pred = psi_model(inputs)
                    val_loss += nn.functional.mse_loss(pred, targets).item()
            val_loss /= len(val_loader)

            if val_loss < best_psi_loss:
                best_psi_loss = val_loss
                torch.save({'model_state_dict': psi_model.state_dict()}, 'psi_comparison_model.pt')

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        # Train Transformer
        print("\nTraining Transformer model...")
        transformer_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=1e-4, weight_decay=1e-4)
        transformer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(transformer_optimizer, args.epochs)

        best_transformer_loss = float('inf')
        for epoch in range(args.epochs):
            transformer_model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Transformer Epoch {epoch+1}/{args.epochs}"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                transformer_optimizer.zero_grad()
                pred = transformer_model(inputs)
                loss = nn.functional.mse_loss(pred, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.0)
                transformer_optimizer.step()
                train_loss += loss.item()

            transformer_scheduler.step()
            train_loss /= len(train_loader)

            # Validation
            transformer_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    pred = transformer_model(inputs)
                    val_loss += nn.functional.mse_loss(pred, targets).item()
            val_loss /= len(val_loader)

            if val_loss < best_transformer_loss:
                best_transformer_loss = val_loss
                torch.save({'model_state_dict': transformer_model.state_dict()}, 'transformer_comparison_model.pt')

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    else:
        print("Loading existing models...")
        psi_model.load_state_dict(torch.load('psi_comparison_model.pt', weights_only=True)['model_state_dict'])
        transformer_model.load_state_dict(torch.load('transformer_comparison_model.pt', weights_only=True)['model_state_dict'])

    # Evaluate performance
    print("\n3. EVALUATING PERFORMANCE")
    print("-" * 40)

    psi_rmse, psi_per_dim = evaluate_model_performance(psi_model, test_loader, device)
    transformer_rmse, transformer_per_dim = evaluate_model_performance(transformer_model, test_loader, device)

    print(f"PSI RMSE:         {psi_rmse:.4f} (x={psi_per_dim[0]:.4f}, y={psi_per_dim[1]:.4f}, z={psi_per_dim[2]:.4f})")
    print(f"Transformer RMSE: {transformer_rmse:.4f} (x={transformer_per_dim[0]:.4f}, y={transformer_per_dim[1]:.4f}, z={transformer_per_dim[2]:.4f})")

    # Extract representations
    print("\n4. EXTRACTING INTERNAL REPRESENTATIONS")
    print("-" * 40)

    psi_phi, psi_targets = extract_psi_final_layer(psi_model, test_dataset, device, args.n_samples)
    transformer_hidden, transformer_targets = extract_transformer_final_layer(transformer_model, test_dataset, device, args.n_samples)

    print(f"PSI phi shape: {psi_phi.shape}")
    print(f"Transformer hidden shape: {transformer_hidden.shape}")

    # Compute correlations
    print("\n5. COMPUTING STATE CORRELATIONS (TRAINED MODELS)")
    print("-" * 40)

    psi_corr, psi_dims = compute_correlations(psi_phi, psi_targets)
    transformer_corr, transformer_dims = compute_correlations(transformer_hidden, transformer_targets)

    print("\nTrained Model Correlations:")
    print(f"{'State':<8} {'PSI':<12} {'Transformer':<12} {'Difference':<12}")
    print("-" * 44)
    for state in ['x', 'y', 'z']:
        psi_val = abs(psi_corr[state])
        trans_val = abs(transformer_corr[state])
        diff = psi_val - trans_val
        print(f"{state:<8} {psi_val:<12.3f} {trans_val:<12.3f} {diff:+.3f}")

    psi_avg = np.mean([abs(psi_corr[s]) for s in ['x', 'y', 'z']])
    trans_avg = np.mean([abs(transformer_corr[s]) for s in ['x', 'y', 'z']])
    print("-" * 44)
    print(f"{'Average':<8} {psi_avg:<12.3f} {trans_avg:<12.3f} {psi_avg - trans_avg:+.3f}")

    # CRITICAL TEST: Untrained model correlations
    print("\n6. CONTROL TEST: UNTRAINED MODEL CORRELATIONS")
    print("-" * 40)

    # Create fresh untrained models
    untrained_psi = PSILorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    untrained_transformer = TransformerLorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    untrained_psi_phi, _ = extract_psi_final_layer(untrained_psi, test_dataset, device, args.n_samples)
    untrained_transformer_hidden, _ = extract_transformer_final_layer(untrained_transformer, test_dataset, device, args.n_samples)

    untrained_psi_corr, _ = compute_correlations(untrained_psi_phi, psi_targets)
    untrained_transformer_corr, _ = compute_correlations(untrained_transformer_hidden, transformer_targets)

    print("\nUntrained Model Correlations:")
    print(f"{'State':<8} {'PSI':<12} {'Transformer':<12}")
    print("-" * 32)
    for state in ['x', 'y', 'z']:
        print(f"{state:<8} {abs(untrained_psi_corr[state]):<12.3f} {abs(untrained_transformer_corr[state]):<12.3f}")

    untrained_psi_avg = np.mean([abs(untrained_psi_corr[s]) for s in ['x', 'y', 'z']])
    untrained_trans_avg = np.mean([abs(untrained_transformer_corr[s]) for s in ['x', 'y', 'z']])
    print("-" * 32)
    print(f"{'Average':<8} {untrained_psi_avg:<12.3f} {untrained_trans_avg:<12.3f}")

    # Visualization
    print("\n7. GENERATING COMPARISON VISUALIZATION")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PSI vs Transformer: Hidden State Correlation with True State', fontsize=14)

    state_names = ['x', 'y', 'z']

    # Row 1: Trained models
    for idx, state in enumerate(state_names):
        ax = axes[0, idx]

        # PSI scatter
        psi_dim = psi_dims[state]
        ax.scatter(psi_phi[:, psi_dim], psi_targets[:, idx],
                  alpha=0.3, s=5, label=f'PSI (r={abs(psi_corr[state]):.3f})', c='blue')

        # Transformer scatter
        trans_dim = transformer_dims[state]
        ax.scatter(transformer_hidden[:, trans_dim], transformer_targets[:, idx],
                  alpha=0.3, s=5, label=f'Transformer (r={abs(transformer_corr[state]):.3f})', c='red')

        ax.set_xlabel('Hidden Dimension Value')
        ax.set_ylabel(f'True {state}')
        ax.set_title(f'{state.upper()} (Trained)')
        ax.legend()

    # Row 2: Bar comparison
    ax = axes[1, 0]
    x_pos = np.arange(3)
    width = 0.35
    ax.bar(x_pos - width/2, [abs(psi_corr[s]) for s in state_names], width, label='PSI', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, [abs(transformer_corr[s]) for s in state_names], width, label='Transformer', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_ylabel('|Correlation|')
    ax.set_ylim(0, 1)
    ax.set_title('Trained Model Correlations')
    ax.legend()

    # Untrained comparison
    ax = axes[1, 1]
    ax.bar(x_pos - width/2, [abs(untrained_psi_corr[s]) for s in state_names], width, label='PSI (untrained)', color='blue', alpha=0.3)
    ax.bar(x_pos + width/2, [abs(untrained_transformer_corr[s]) for s in state_names], width, label='Transformer (untrained)', color='red', alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_ylabel('|Correlation|')
    ax.set_ylim(0, 1)
    ax.set_title('Untrained Model Correlations (Control)')
    ax.legend()

    # Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
SUMMARY
========

Task Performance (RMSE):
  PSI:         {psi_rmse:.4f}
  Transformer: {transformer_rmse:.4f}

Trained State Correlation (avg):
  PSI:         {psi_avg:.3f}
  Transformer: {trans_avg:.3f}

Untrained Correlation (avg):
  PSI:         {untrained_psi_avg:.3f}
  Transformer: {untrained_trans_avg:.3f}

CONCLUSION:
"""

    if abs(psi_avg - trans_avg) < 0.05:
        summary_text += "Correlations are SIMILAR.\n→ Findings are about the TASK,\n   not PSI specifically."
        conclusion = "SIMILAR"
    elif psi_avg > trans_avg:
        summary_text += f"PSI shows HIGHER correlations\n(+{psi_avg - trans_avg:.3f}).\n→ PSI may have unique properties."
        conclusion = "PSI_BETTER"
    else:
        summary_text += f"Transformer shows HIGHER correlations\n(+{trans_avg - psi_avg:.3f}).\n→ PSI is NOT special."
        conclusion = "TRANSFORMER_BETTER"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('psi_vs_transformer_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved psi_vs_transformer_comparison.png")
    plt.close()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    print(f"\nTask Performance:")
    print(f"  PSI RMSE:         {psi_rmse:.4f}")
    print(f"  Transformer RMSE: {transformer_rmse:.4f}")

    print(f"\nTrained Model State Correlations (average):")
    print(f"  PSI:         {psi_avg:.3f}")
    print(f"  Transformer: {trans_avg:.3f}")

    print(f"\nUntrained Model Correlations (average):")
    print(f"  PSI:         {untrained_psi_avg:.3f}")
    print(f"  Transformer: {untrained_trans_avg:.3f}")

    print(f"\nCorrelation Improvement from Training:")
    print(f"  PSI:         {psi_avg - untrained_psi_avg:+.3f}")
    print(f"  Transformer: {trans_avg - untrained_trans_avg:+.3f}")

    print("\n" + "-" * 40)

    if abs(psi_avg - trans_avg) < 0.05:
        print("CONCLUSION: The high state correlations are NOT unique to PSI.")
        print("            Both architectures learn similar representations.")
        print("            The findings are about the TASK, not the architecture.")
    elif psi_avg > trans_avg + 0.05:
        print("CONCLUSION: PSI shows meaningfully HIGHER state correlations.")
        print("            This suggests PSI may have unique representational properties.")
        print("            Further investigation warranted.")
    else:
        print("CONCLUSION: Transformer shows HIGHER state correlations than PSI.")
        print("            PSI's 'phase space' framing does NOT provide an advantage.")

    print()


if __name__ == "__main__":
    main()
