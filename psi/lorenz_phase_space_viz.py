"""
Phase Space Visualization for Lorenz Attractor PSI Model

Extract and visualize the learned phase space representation from a PSI model
trained on the Lorenz attractor. This is especially interesting because:

1. Lorenz has a KNOWN 3D phase space (x, y, z)
2. The attractor has famous geometric structure (butterfly)
3. We can directly compare learned vs true phase portraits
4. Chaos means local divergence but global structure - can PSI capture both?

Key questions:
- Does PSI learn dimensions that correspond to x, y, z?
- Does the learned flow field match the Lorenz equations?
- Does the phase portrait show the butterfly attractor?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm

from train_lorenz import (
    TPIDynamicsPredictor,
    generate_lorenz_trajectory,
    LorenzDataset,
    generate_lorenz_dataset
)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class LorenzPSIWithHooks(nn.Module):
    """
    Wrapper that extracts internal PSI representations during forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up forward hooks on PSI layers."""
        for i, block in enumerate(self.model.blocks):
            psi_module = block.integration
            original_forward = psi_module.forward

            def make_capturing_forward(layer_idx, psi_mod):
                def capturing_forward(x):
                    # Capture PSI internals
                    omega = psi_mod.to_omega(x)
                    magnitude_scale = 5.0
                    magnitude = torch.sigmoid(psi_mod.to_magnitude(x)) * magnitude_scale
                    phi_init = psi_mod.to_phase_init(x)
                    gate = torch.sigmoid(psi_mod.to_gate(x))

                    omega_scaled = omega * psi_mod.integration_scale.abs()
                    gated_omega = gate * omega_scaled

                    phi = phi_init + torch.cumsum(gated_omega, dim=1)

                    # Store activations
                    self.activations[f'layer_{layer_idx}_omega'] = omega.detach()
                    self.activations[f'layer_{layer_idx}_phi'] = phi.detach()
                    self.activations[f'layer_{layer_idx}_gate'] = gate.detach()
                    self.activations[f'layer_{layer_idx}_gated_omega'] = gated_omega.detach()

                    trajectory_real = torch.cos(phi)
                    trajectory_imag = torch.sin(phi)

                    self.activations[f'layer_{layer_idx}_traj_real'] = trajectory_real.detach()
                    self.activations[f'layer_{layer_idx}_traj_imag'] = trajectory_imag.detach()

                    # Continue with normal forward
                    weighted_content = magnitude * x
                    memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
                    memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

                    accumulated_magnitude = torch.cumsum(magnitude, dim=1)
                    sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
                    memory_real_normalized = memory_real / sqrt_magnitude
                    memory_imag_normalized = memory_imag / sqrt_magnitude

                    self.activations[f'layer_{layer_idx}_memory_real'] = memory_real_normalized.detach()
                    self.activations[f'layer_{layer_idx}_memory_imag'] = memory_imag_normalized.detach()

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
                    output = x + phase_contribution

                    return output
                return capturing_forward

            psi_module.forward = make_capturing_forward(i, psi_module)

    def forward(self, x):
        self.activations = {}
        return self.model(x)

    def get_activations(self):
        return self.activations


def extract_lorenz_phase_data(model, n_trajectories=100, trajectory_length=200, context_len=20):
    """
    Extract phase space representations for Lorenz trajectories.
    """
    hooked_model = LorenzPSIWithHooks(model)
    hooked_model.eval()

    # Generate fresh trajectories
    print(f"Generating {n_trajectories} Lorenz trajectories...")
    trajectories, _ = generate_lorenz_dataset(
        num_trajectories=n_trajectories,
        trajectory_length=trajectory_length,
        normalize=True
    )

    all_data = {
        'states': [],  # Ground truth (x, y, z)
        'inputs': [],
    }

    num_layers = len(model.blocks)
    for layer_idx in range(num_layers):
        all_data[f'phi_layer_{layer_idx}'] = []
        all_data[f'omega_layer_{layer_idx}'] = []
        all_data[f'gated_omega_layer_{layer_idx}'] = []

    print("Extracting phase space representations...")
    with torch.no_grad():
        for traj_idx in tqdm(range(n_trajectories)):
            trajectory = trajectories[traj_idx]

            # Use sliding windows through trajectory
            for start in range(0, trajectory_length - context_len - 1, 5):  # Step by 5 to reduce redundancy
                context = trajectory[start:start + context_len]
                target_state = trajectory[start + context_len]

                context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

                # Forward pass captures activations
                _ = hooked_model(context_tensor)
                activations = hooked_model.get_activations()

                # Store the last timestep's activations (prediction point)
                all_data['states'].append(target_state)
                all_data['inputs'].append(context[-1])  # Last context state

                for layer_idx in range(num_layers):
                    # Take last timestep
                    all_data[f'phi_layer_{layer_idx}'].append(
                        activations[f'layer_{layer_idx}_phi'].cpu().numpy()[0, -1, :]
                    )
                    all_data[f'omega_layer_{layer_idx}'].append(
                        activations[f'layer_{layer_idx}_omega'].cpu().numpy()[0, -1, :]
                    )
                    all_data[f'gated_omega_layer_{layer_idx}'].append(
                        activations[f'layer_{layer_idx}_gated_omega'].cpu().numpy()[0, -1, :]
                    )

    # Stack arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])

    return all_data, num_layers


def visualize_learned_vs_true_phase_space(data, num_layers, save_prefix='lorenz_phase'):
    """
    Compare learned phase space structure to true Lorenz phase space.
    """
    states = data['states']  # [n_samples, 3] - true (x, y, z)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Learned Phase Space vs True Lorenz Phase Space', fontsize=14)

    # True phase space (3D Lorenz attractor)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(states[:, 0], states[:, 1], states[:, 2], c=np.arange(len(states)),
               cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True Lorenz Phase Space')

    # Learned phase space (PCA of phi) for different layers
    for idx, layer_idx in enumerate([0, num_layers//2, num_layers-1]):
        phi = data[f'phi_layer_{layer_idx}']

        # PCA to 3D
        pca = PCA(n_components=3)
        phi_3d = pca.fit_transform(phi)

        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')

        # Color by true X position
        scatter = ax.scatter(phi_3d[:, 0], phi_3d[:, 1], phi_3d[:, 2],
                           c=states[:, 0], cmap='coolwarm', s=1, alpha=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Learned Phase Space (Layer {layer_idx})\nColored by X')
        plt.colorbar(scatter, ax=ax, shrink=0.5)

    # 2D projections colored by different state variables
    ax4 = fig.add_subplot(2, 3, 4)
    phi = data[f'phi_layer_{num_layers-1}']
    pca = PCA(n_components=2)
    phi_2d = pca.fit_transform(phi)

    scatter = ax4.scatter(phi_2d[:, 0], phi_2d[:, 1], c=states[:, 0], cmap='coolwarm', s=1, alpha=0.5)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title(f'Layer {num_layers-1} Phase Space (colored by X)')
    plt.colorbar(scatter, ax=ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(phi_2d[:, 0], phi_2d[:, 1], c=states[:, 1], cmap='coolwarm', s=1, alpha=0.5)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title(f'Layer {num_layers-1} Phase Space (colored by Y)')
    plt.colorbar(scatter, ax=ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    scatter = ax6.scatter(phi_2d[:, 0], phi_2d[:, 1], c=states[:, 2], cmap='coolwarm', s=1, alpha=0.5)
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.set_title(f'Layer {num_layers-1} Phase Space (colored by Z)')
    plt.colorbar(scatter, ax=ax6)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_3d_structure.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_3d_structure.png")
    plt.close()


def visualize_phase_state_correlation_lorenz(data, num_layers, save_prefix='lorenz_phase'):
    """
    Analyze correlation between learned phase dimensions and Lorenz state variables.
    """
    states = data['states']  # [n_samples, 3]
    state_names = ['x', 'y', 'z']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase-State Correlation Analysis (Lorenz)', fontsize=14)

    for layer_idx, ax_row in zip([0, num_layers-1], axes):
        phi = data[f'phi_layer_{layer_idx}']
        n_samples, dim = phi.shape

        # Compute correlations
        correlations = np.zeros((dim, 3))
        for d in range(dim):
            for s in range(3):
                corr = np.corrcoef(phi[:, d], states[:, s])[0, 1]
                correlations[d, s] = corr if not np.isnan(corr) else 0

        # Heatmap of correlations (first 50 dims)
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

        # Scatter of best dimensions vs states
        ax = ax_row[2]
        best_x_dim = np.argmax(np.abs(correlations[:, 0]))
        best_y_dim = np.argmax(np.abs(correlations[:, 1]))
        best_z_dim = np.argmax(np.abs(correlations[:, 2]))

        ax.scatter(phi[:, best_x_dim], states[:, 0], alpha=0.3, s=1, label=f'X (dim {best_x_dim})')
        ax.scatter(phi[:, best_y_dim], states[:, 1], alpha=0.3, s=1, label=f'Y (dim {best_y_dim})')
        ax.scatter(phi[:, best_z_dim], states[:, 2], alpha=0.3, s=1, label=f'Z (dim {best_z_dim})')
        ax.set_xlabel('Phase Dimension Value')
        ax.set_ylabel('State Value')
        ax.set_title(f'Layer {layer_idx}: Best Dims vs State')
        ax.legend(markerscale=10)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_correlation.png")
    plt.close()

    return correlations


def visualize_flow_field_lorenz(data, num_layers, save_prefix='lorenz_phase'):
    """
    Visualize the learned flow field and compare to true Lorenz flow.
    """
    states = data['states']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Learned Flow Field (Phase Velocities)', fontsize=14)

    # True Lorenz flow field (compute derivatives)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    # Note: states are normalized, so we work in normalized space
    true_dx = sigma * (states[:, 1] - states[:, 0])
    true_dy = states[:, 0] * (rho - states[:, 2]) - states[:, 1]  # Approximate
    true_dz = states[:, 0] * states[:, 1] - beta * states[:, 2]

    # For each layer, show flow field
    for idx, layer_idx in enumerate([0, num_layers//2, num_layers-1]):
        phi = data[f'phi_layer_{layer_idx}']
        omega = data[f'gated_omega_layer_{layer_idx}']

        # PCA on phi for visualization
        pca = PCA(n_components=2)
        phi_2d = pca.fit_transform(phi)

        # Project omega to same space
        omega_2d = pca.transform(omega)

        # Subsample for quiver plot
        step = max(1, len(phi_2d) // 500)
        ax = axes[0, idx]
        ax.quiver(
            phi_2d[::step, 0], phi_2d[::step, 1],
            omega_2d[::step, 0], omega_2d[::step, 1],
            alpha=0.5, scale=50
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Layer {layer_idx} Flow Field')

    # Flow magnitude vs position in attractor
    ax = axes[1, 0]
    omega = data[f'gated_omega_layer_{num_layers-1}']
    omega_mag = np.linalg.norm(omega, axis=-1)
    ax.scatter(states[:, 0], omega_mag, alpha=0.3, s=1, c=states[:, 2], cmap='viridis')
    ax.set_xlabel('X state')
    ax.set_ylabel('Flow Magnitude')
    ax.set_title('Flow Magnitude vs X (colored by Z)')

    # Flow magnitude distribution
    ax = axes[1, 1]
    for layer_idx in [0, num_layers//2, num_layers-1]:
        omega = data[f'gated_omega_layer_{layer_idx}']
        omega_mag = np.linalg.norm(omega, axis=-1)
        ax.hist(omega_mag, bins=50, alpha=0.5, label=f'Layer {layer_idx}', density=True)
    ax.set_xlabel('Flow Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Flow Magnitudes')
    ax.legend()

    # Correlation between flow magnitude and true derivative magnitude
    ax = axes[1, 2]
    omega = data[f'gated_omega_layer_{num_layers-1}']
    omega_mag = np.linalg.norm(omega, axis=-1)
    true_deriv_mag = np.sqrt(true_dx**2 + true_dy**2 + true_dz**2)
    ax.scatter(true_deriv_mag[:5000], omega_mag[:5000], alpha=0.2, s=1)
    corr = np.corrcoef(true_deriv_mag, omega_mag)[0, 1]
    ax.set_xlabel('True |dState/dt|')
    ax.set_ylabel('Learned Flow Magnitude')
    ax.set_title(f'Flow vs True Derivative (r={corr:.3f})')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_flow_field.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_flow_field.png")
    plt.close()


def visualize_attractor_structure(data, num_layers, save_prefix='lorenz_phase'):
    """
    Check if learned phase space preserves attractor structure.

    The Lorenz attractor has two lobes - does PSI's representation preserve this?
    """
    states = data['states']
    phi = data[f'phi_layer_{num_layers-1}']

    # Identify which lobe each point is on (X > 0 or X < 0 roughly)
    lobe = (states[:, 0] > 0).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Attractor Structure in Learned Phase Space', fontsize=14)

    # True attractor colored by lobe
    ax = axes[0, 0]
    ax.scatter(states[lobe==0, 0], states[lobe==0, 2], c='blue', s=1, alpha=0.3, label='Left lobe')
    ax.scatter(states[lobe==1, 0], states[lobe==1, 2], c='red', s=1, alpha=0.3, label='Right lobe')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('True Lorenz: X-Z projection')
    ax.legend()

    # Learned phase space colored by lobe
    pca = PCA(n_components=2)
    phi_2d = pca.fit_transform(phi)

    ax = axes[0, 1]
    ax.scatter(phi_2d[lobe==0, 0], phi_2d[lobe==0, 1], c='blue', s=1, alpha=0.3, label='Left lobe')
    ax.scatter(phi_2d[lobe==1, 0], phi_2d[lobe==1, 1], c='red', s=1, alpha=0.3, label='Right lobe')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Learned Phase Space: PCA projection')
    ax.legend()

    # Check separability of lobes in learned space
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(phi_2d, lobe)
    lobe_accuracy = clf.score(phi_2d, lobe)

    ax = axes[1, 0]
    ax.text(0.5, 0.6, f'Lobe Classification\nAccuracy: {lobe_accuracy*100:.1f}%',
            ha='center', va='center', fontsize=24, transform=ax.transAxes)
    ax.text(0.5, 0.3, '(Using 2D PCA of learned phase space)',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Attractor Structure Preservation')

    # Z-coordinate encoding
    ax = axes[1, 1]
    best_z_dim = np.argmax([np.abs(np.corrcoef(phi[:, d], states[:, 2])[0, 1])
                           for d in range(phi.shape[1])])
    z_corr = np.corrcoef(phi[:, best_z_dim], states[:, 2])[0, 1]

    ax.scatter(phi[:, best_z_dim], states[:, 2], c=lobe, cmap='coolwarm', s=1, alpha=0.3)
    ax.set_xlabel(f'Phase dim {best_z_dim}')
    ax.set_ylabel('Z state')
    ax.set_title(f'Z encoding (r={z_corr:.3f})')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_attractor.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_attractor.png")
    plt.close()

    return lobe_accuracy


def visualize_butterfly_reconstruction(data, num_layers, save_prefix='lorenz_phase'):
    """
    Attempt to reconstruct the butterfly shape from learned phase space.
    """
    states = data['states']
    phi = data[f'phi_layer_{num_layers-1}']

    # Find 3 phase dimensions that best correlate with x, y, z
    correlations = np.zeros((phi.shape[1], 3))
    for d in range(phi.shape[1]):
        for s in range(3):
            corr = np.corrcoef(phi[:, d], states[:, s])[0, 1]
            correlations[d, s] = corr if not np.isnan(corr) else 0

    best_x = np.argmax(np.abs(correlations[:, 0]))
    best_y = np.argmax(np.abs(correlations[:, 1]))
    best_z = np.argmax(np.abs(correlations[:, 2]))

    # Flip signs if negative correlation
    sign_x = np.sign(correlations[best_x, 0])
    sign_y = np.sign(correlations[best_y, 1])
    sign_z = np.sign(correlations[best_z, 2])

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('Butterfly Attractor: True vs Reconstructed from Phase Space', fontsize=14)

    # True attractor
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(states[:, 0], states[:, 1], states[:, 2],
               c=np.arange(len(states)), cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True Lorenz Attractor')

    # Reconstructed from best-correlating phase dimensions
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(sign_x * phi[:, best_x],
               sign_y * phi[:, best_y],
               sign_z * phi[:, best_z],
               c=np.arange(len(phi)), cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel(f'Phase dim {best_x} (X)')
    ax2.set_ylabel(f'Phase dim {best_y} (Y)')
    ax2.set_zlabel(f'Phase dim {best_z} (Z)')
    ax2.set_title(f'Reconstructed from Phase Space\nr_x={correlations[best_x,0]:.2f}, r_y={correlations[best_y,1]:.2f}, r_z={correlations[best_z,2]:.2f}')

    # PCA reconstruction
    pca = PCA(n_components=3)
    phi_3d = pca.fit_transform(phi)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(phi_3d[:, 0], phi_3d[:, 1], phi_3d[:, 2],
               c=np.arange(len(phi_3d)), cmap='viridis', s=1, alpha=0.5)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    ax3.set_title(f'PCA of Phase Space\nExplained var: {sum(pca.explained_variance_ratio_)*100:.1f}%')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_butterfly.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_butterfly.png")
    plt.close()

    return {
        'best_x': best_x, 'best_y': best_y, 'best_z': best_z,
        'corr_x': correlations[best_x, 0],
        'corr_y': correlations[best_y, 1],
        'corr_z': correlations[best_z, 2]
    }


def main():
    parser = argparse.ArgumentParser(description='Phase Space Visualization for Lorenz PSI')
    parser.add_argument('--n_trajectories', type=int, default=100,
                        help='Number of trajectories to analyze')
    parser.add_argument('--trajectory_length', type=int, default=200,
                        help='Length of each trajectory')
    parser.add_argument('--context_len', type=int, default=20,
                        help='Context length for model')
    parser.add_argument('--dim', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of PSI layers')
    parser.add_argument('--save_prefix', type=str, default='lorenz_phase',
                        help='Prefix for saved figures')
    args = parser.parse_args()

    print("=" * 80)
    print("LORENZ ATTRACTOR PHASE SPACE VISUALIZATION")
    print("=" * 80)
    print()
    print("Extracting learned phase space representation from PSI model")
    print("trained on chaotic Lorenz dynamics.")
    print()
    print("Key questions:")
    print("  - Does PSI learn dimensions corresponding to x, y, z?")
    print("  - Does the learned flow field match Lorenz equations?")
    print("  - Does the phase portrait preserve the butterfly structure?")
    print("=" * 80)
    print()

    # Load model
    print("Loading trained model...")
    model = TPIDynamicsPredictor(
        state_dim=3,
        dim=args.dim,
        num_layers=args.num_layers,
        max_len=args.context_len + 10,
        device=device
    ).to(device)

    try:
        checkpoint = torch.load('lorenz_best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from lorenz_best.pt")
        print(f"  Config: {checkpoint.get('config', 'N/A')}")
    except FileNotFoundError:
        print("ERROR: lorenz_best.pt not found!")
        print("Please train the Lorenz model first with: python train_lorenz.py")
        return

    # Extract phase space data
    print()
    data, num_layers = extract_lorenz_phase_data(
        model,
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
        context_len=args.context_len
    )

    print(f"\nExtracted data shapes:")
    print(f"  States (x,y,z): {data['states'].shape}")
    print(f"  Phi (layer 0): {data['phi_layer_0'].shape}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n1. Learned vs True Phase Space (3D)...")
    visualize_learned_vs_true_phase_space(data, num_layers, args.save_prefix)

    print("\n2. Phase-State Correlation Analysis...")
    correlations = visualize_phase_state_correlation_lorenz(data, num_layers, args.save_prefix)

    print("\n3. Flow Field Analysis...")
    visualize_flow_field_lorenz(data, num_layers, args.save_prefix)

    print("\n4. Attractor Structure Preservation...")
    lobe_accuracy = visualize_attractor_structure(data, num_layers, args.save_prefix)

    print("\n5. Butterfly Reconstruction...")
    reconstruction = visualize_butterfly_reconstruction(data, num_layers, args.save_prefix)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    phi = data[f'phi_layer_{num_layers-1}']
    states = data['states']

    print("\nPhase-State Correlations (final layer):")
    for s_idx, s_name in enumerate(['x', 'y', 'z']):
        correlations = [np.corrcoef(phi[:, d], states[:, s_idx])[0, 1]
                       for d in range(phi.shape[1])]
        correlations = [c if not np.isnan(c) else 0 for c in correlations]
        max_corr_idx = np.argmax(np.abs(correlations))
        max_corr = correlations[max_corr_idx]
        print(f"  {s_name}: dim {max_corr_idx} (r = {max_corr:.3f})")

    print(f"\nAttractor structure preservation:")
    print(f"  Lobe classification accuracy: {lobe_accuracy*100:.1f}%")

    print(f"\nButterfly reconstruction:")
    print(f"  X correlation: {reconstruction['corr_x']:.3f} (dim {reconstruction['best_x']})")
    print(f"  Y correlation: {reconstruction['corr_y']:.3f} (dim {reconstruction['best_y']})")
    print(f"  Z correlation: {reconstruction['corr_z']:.3f} (dim {reconstruction['best_z']})")

    print("\nDone! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
