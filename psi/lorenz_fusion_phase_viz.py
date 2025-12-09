"""
Phase Space Visualization for Lorenz Fusion Model

Compare learned phase space from:
1. Direct Lorenz (state input → next state prediction)
2. Lorenz Fusion (noisy observations → state reconstruction)

Hypothesis: Fusion model should show HIGH state correlation (like sensor fusion)
because the state is hidden and must be reconstructed.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm

from lorenz_fusion_experiment import (
    PSILorenzFusionModel,
    LorenzFusionDataset,
    SENSORS,
    compute_input_dim,
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class LorenzFusionPSIWithHooks(nn.Module):
    """Wrapper to extract internal PSI representations."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        self._setup_hooks()

    def _setup_hooks(self):
        for i, block in enumerate(self.model.psi_blocks):
            psi_module = block.integration

            def make_capturing_forward(layer_idx, psi_mod):
                def capturing_forward(x):
                    omega = psi_mod.to_omega(x)
                    magnitude = torch.sigmoid(psi_mod.to_magnitude(x)) * 5.0
                    phi_init = psi_mod.to_phase_init(x)
                    gate = torch.sigmoid(psi_mod.to_gate(x))

                    omega_scaled = omega * psi_mod.integration_scale.abs()
                    gated_omega = gate * omega_scaled
                    phi = phi_init + torch.cumsum(gated_omega, dim=1)

                    self.activations[f'layer_{layer_idx}_phi'] = phi.detach()
                    self.activations[f'layer_{layer_idx}_omega'] = omega.detach()
                    self.activations[f'layer_{layer_idx}_gated_omega'] = gated_omega.detach()

                    trajectory_real = torch.cos(phi)
                    trajectory_imag = torch.sin(phi)

                    weighted_content = magnitude * x
                    memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
                    memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

                    accumulated_magnitude = torch.cumsum(magnitude, dim=1)
                    sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
                    memory_real_normalized = memory_real / sqrt_magnitude
                    memory_imag_normalized = memory_imag / sqrt_magnitude

                    query_offset = psi_mod.to_query_offset(x)
                    phi_query = phi + query_offset

                    cos_phi_q = torch.cos(phi_query)
                    sin_phi_q = torch.sin(phi_query)

                    retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
                    retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

                    content_modulated_real = x * trajectory_real
                    content_modulated_imag = x * trajectory_imag

                    context = torch.cat([
                        content_modulated_real,
                        content_modulated_imag,
                        retrieved_real,
                        retrieved_imag
                    ], dim=-1)

                    phase_contribution = psi_mod.to_out(context)
                    return x + phase_contribution
                return capturing_forward

            psi_module.forward = make_capturing_forward(i, psi_module)

    def forward(self, x):
        self.activations = {}
        return self.model(x)


def extract_phase_data(model, dataset, device, n_samples=500):
    """Extract phase space data from model."""
    hooked_model = LorenzFusionPSIWithHooks(model)
    hooked_model.eval()

    all_data = {
        'states': [],  # True states (targets)
    }

    num_layers = len(model.psi_blocks)
    for layer_idx in range(num_layers):
        all_data[f'phi_layer_{layer_idx}'] = []
        all_data[f'gated_omega_layer_{layer_idx}'] = []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting phase space"):
            inputs, target = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)

            _ = hooked_model(inputs)
            activations = hooked_model.activations

            all_data['states'].append(target.numpy())

            for layer_idx in range(num_layers):
                # Take last timestep
                all_data[f'phi_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_phi'].cpu().numpy()[0, -1, :]
                )
                all_data[f'gated_omega_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_gated_omega'].cpu().numpy()[0, -1, :]
                )

    for key in all_data:
        all_data[key] = np.array(all_data[key])

    return all_data, num_layers


def visualize_phase_state_correlation(data, num_layers, save_prefix='lorenz_fusion_phase'):
    """Analyze correlation between phase dimensions and true state."""
    states = data['states']
    state_names = ['x', 'y', 'z']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lorenz FUSION: Phase-State Correlation Analysis\n(State reconstructed from noisy observations)', fontsize=14)

    for layer_idx, ax_row in zip([0, num_layers-1], axes):
        phi = data[f'phi_layer_{layer_idx}']
        n_samples, dim = phi.shape

        correlations = np.zeros((dim, 3))
        for d in range(dim):
            for s in range(3):
                corr = np.corrcoef(phi[:, d], states[:, s])[0, 1]
                correlations[d, s] = corr if not np.isnan(corr) else 0

        # Heatmap
        ax = ax_row[0]
        im = ax.imshow(np.abs(correlations[:50, :]), aspect='auto', cmap='hot')
        ax.set_xlabel('State Variable')
        ax.set_ylabel('Phase Dimension')
        ax.set_xticks(range(3))
        ax.set_xticklabels(state_names)
        ax.set_title(f'Layer {layer_idx}: |Correlation| (first 50 dims)')
        plt.colorbar(im, ax=ax)

        # Max correlation per state
        ax = ax_row[1]
        max_corr = np.max(np.abs(correlations), axis=0)
        best_dims = np.argmax(np.abs(correlations), axis=0)
        bars = ax.bar(state_names, max_corr, color=['#e41a1c', '#377eb8', '#4daf4a'])
        ax.set_ylabel('Max |Correlation|')
        ax.set_title(f'Layer {layer_idx}: Best Correlating Dimension')
        ax.set_ylim(0, 1)
        for bar, val, dim_idx in zip(bars, max_corr, best_dims):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}\n(dim {dim_idx})', ha='center', va='bottom', fontsize=9)

        # Scatter plot
        ax = ax_row[2]
        best_x = np.argmax(np.abs(correlations[:, 0]))
        best_y = np.argmax(np.abs(correlations[:, 1]))
        best_z = np.argmax(np.abs(correlations[:, 2]))

        ax.scatter(phi[:, best_x], states[:, 0], alpha=0.3, s=1, label=f'X (dim {best_x})')
        ax.scatter(phi[:, best_y], states[:, 1], alpha=0.3, s=1, label=f'Y (dim {best_y})')
        ax.scatter(phi[:, best_z], states[:, 2], alpha=0.3, s=1, label=f'Z (dim {best_z})')
        ax.set_xlabel('Phase Dimension Value')
        ax.set_ylabel('State Value')
        ax.set_title(f'Layer {layer_idx}: Best Dims vs State')
        ax.legend(markerscale=10)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_correlation.png")
    plt.close()

    return correlations


def visualize_phase_space_3d(data, num_layers, save_prefix='lorenz_fusion_phase'):
    """Visualize learned phase space in 3D."""
    states = data['states']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Lorenz FUSION: Learned Phase Space vs True State Space', fontsize=14)

    # True state space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(states[:, 0], states[:, 1], states[:, 2],
               c=np.arange(len(states)), cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True Lorenz State Space')

    # Phase space for different layers
    for idx, layer_idx in enumerate([0, num_layers//2, num_layers-1]):
        phi = data[f'phi_layer_{layer_idx}']

        pca = PCA(n_components=3)
        phi_3d = pca.fit_transform(phi)

        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        scatter = ax.scatter(phi_3d[:, 0], phi_3d[:, 1], phi_3d[:, 2],
                           c=states[:, 0], cmap='coolwarm', s=1, alpha=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Layer {layer_idx} Phase Space\n(colored by X)')
        plt.colorbar(scatter, ax=ax, shrink=0.5)

    # 2D projections of final layer
    phi = data[f'phi_layer_{num_layers-1}']
    pca = PCA(n_components=2)
    phi_2d = pca.fit_transform(phi)

    ax = fig.add_subplot(2, 3, 5)
    scatter = ax.scatter(phi_2d[:, 0], phi_2d[:, 1], c=states[:, 0], cmap='coolwarm', s=1, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Layer {num_layers-1}: Colored by X')
    plt.colorbar(scatter, ax=ax)

    ax = fig.add_subplot(2, 3, 6)
    scatter = ax.scatter(phi_2d[:, 0], phi_2d[:, 1], c=states[:, 2], cmap='coolwarm', s=1, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Layer {num_layers-1}: Colored by Z')
    plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_3d.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_3d.png")
    plt.close()


def visualize_comparison_summary(data, num_layers, save_prefix='lorenz_fusion_phase'):
    """Create summary comparison visualization."""
    states = data['states']
    phi = data[f'phi_layer_{num_layers-1}']

    # Compute correlations
    correlations = {}
    best_dims = {}
    for s_idx, s_name in enumerate(['x', 'y', 'z']):
        corrs = [np.corrcoef(phi[:, d], states[:, s_idx])[0, 1] for d in range(phi.shape[1])]
        corrs = [c if not np.isnan(c) else 0 for c in corrs]
        best_dim = np.argmax(np.abs(corrs))
        correlations[s_name] = corrs[best_dim]
        best_dims[s_name] = best_dim

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Lorenz FUSION vs Direct Lorenz: Phase-State Correlation Comparison', fontsize=14)

    # Bar comparison
    ax = axes[0]
    x_pos = np.arange(3)
    width = 0.35

    # Direct Lorenz results (from previous experiment)
    direct_corr = [0.73, 0.76, 0.84]  # x, y, z from direct Lorenz
    fusion_corr = [correlations['x'], correlations['y'], correlations['z']]

    ax.bar(x_pos - width/2, np.abs(direct_corr), width, label='Direct Lorenz', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, np.abs(fusion_corr), width, label='Lorenz Fusion', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_ylabel('|Correlation|')
    ax.set_ylim(0, 1)
    ax.set_title('State Correlation: Direct vs Fusion')
    ax.legend()

    # Add values on bars
    for i, (d_val, f_val) in enumerate(zip(direct_corr, fusion_corr)):
        ax.text(i - width/2, np.abs(d_val) + 0.02, f'{np.abs(d_val):.2f}', ha='center', fontsize=9)
        ax.text(i + width/2, np.abs(f_val) + 0.02, f'{np.abs(f_val):.2f}', ha='center', fontsize=9)

    # Scatter: best X dimension vs X
    ax = axes[1]
    ax.scatter(phi[:, best_dims['x']], states[:, 0], alpha=0.3, s=1)
    ax.set_xlabel(f'Phase dim {best_dims["x"]}')
    ax.set_ylabel('True X')
    ax.set_title(f'X encoding (r={correlations["x"]:.3f})')

    # Scatter: best Z dimension vs Z
    ax = axes[2]
    ax.scatter(phi[:, best_dims['z']], states[:, 2], alpha=0.3, s=1)
    ax.set_xlabel(f'Phase dim {best_dims["z"]}')
    ax.set_ylabel('True Z')
    ax.set_title(f'Z encoding (r={correlations["z"]:.3f})')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_comparison.png")
    plt.close()

    return correlations, best_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_trajectories', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--save_prefix', type=str, default='lorenz_fusion_phase')
    args = parser.parse_args()

    print("=" * 80)
    print("LORENZ FUSION PHASE SPACE VISUALIZATION")
    print("=" * 80)
    print()
    print("Comparing learned representations:")
    print("  - Direct Lorenz: State given directly → predict next state")
    print("  - Lorenz Fusion: Noisy observations → reconstruct hidden state")
    print()
    print("Hypothesis: Fusion should show HIGH state correlation")
    print("because the model must RECONSTRUCT the hidden state.")
    print("=" * 80)

    # Load model
    print("\nLoading trained model...")
    input_dim = compute_input_dim(len(SENSORS))
    model = PSILorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    try:
        checkpoint = torch.load('lorenz_fusion_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded lorenz_fusion_model.pt")
    except FileNotFoundError:
        print("ERROR: lorenz_fusion_model.pt not found!")
        print("Please train the model first: python lorenz_fusion_experiment.py")
        return

    # Generate test data
    print(f"\nGenerating test data ({args.n_trajectories} trajectories)...")
    test_dataset = LorenzFusionDataset(
        n_trajectories=args.n_trajectories,
        seq_len=50,
        split='test',
        seed=999
    )

    # Extract phase space
    print(f"\nExtracting phase space ({args.n_samples} samples)...")
    data, num_layers = extract_phase_data(model, test_dataset, device, args.n_samples)

    print(f"\nExtracted shapes:")
    print(f"  States: {data['states'].shape}")
    print(f"  Phi (layer 0): {data['phi_layer_0'].shape}")

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n1. Phase-State Correlation Analysis...")
    correlations = visualize_phase_state_correlation(data, num_layers, args.save_prefix)

    print("\n2. 3D Phase Space Visualization...")
    visualize_phase_space_3d(data, num_layers, args.save_prefix)

    print("\n3. Comparison Summary...")
    final_corr, best_dims = visualize_comparison_summary(data, num_layers, args.save_prefix)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Lorenz Fusion Phase-State Correlations")
    print("=" * 80)

    phi = data[f'phi_layer_{num_layers-1}']
    states = data['states']

    print("\nFinal layer correlations:")
    for s_idx, s_name in enumerate(['x', 'y', 'z']):
        corrs = [np.corrcoef(phi[:, d], states[:, s_idx])[0, 1] for d in range(phi.shape[1])]
        corrs = [c if not np.isnan(c) else 0 for c in corrs]
        best_dim = np.argmax(np.abs(corrs))
        best_corr = corrs[best_dim]
        print(f"  {s_name}: dim {best_dim} (r = {best_corr:.3f})")

    print("\n" + "-" * 40)
    print("COMPARISON:")
    print("-" * 40)
    print("                    Direct Lorenz    Lorenz Fusion")
    print(f"  X correlation:        0.73            {np.abs(final_corr['x']):.2f}")
    print(f"  Y correlation:        0.76            {np.abs(final_corr['y']):.2f}")
    print(f"  Z correlation:        0.84            {np.abs(final_corr['z']):.2f}")
    print()

    avg_direct = np.mean([0.73, 0.76, 0.84])
    avg_fusion = np.mean([np.abs(final_corr['x']), np.abs(final_corr['y']), np.abs(final_corr['z'])])
    print(f"  Average:              {avg_direct:.2f}            {avg_fusion:.2f}")

    if avg_fusion > avg_direct + 0.05:
        print("\n✅ HYPOTHESIS CONFIRMED: Fusion shows higher state correlation!")
        print("   When state is hidden, PSI learns to RECONSTRUCT it.")
    elif avg_fusion > avg_direct - 0.05:
        print("\n~ INCONCLUSIVE: Similar correlations.")
    else:
        print("\n❌ HYPOTHESIS REJECTED: Direct Lorenz shows higher correlation.")

    print("\nDone! Check the generated PNG files.")


if __name__ == "__main__":
    main()
