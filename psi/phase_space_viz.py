"""
Phase Space Visualization for PSI Models

Extract and visualize the learned phase space representation from a trained PSI model.
This allows us to see if PSI learned the true dynamical structure of the system.

Key visualizations:
1. Phase trajectories colored by ground truth state
2. Flow field (phase velocities)
3. Phase space structure across layers
4. Correlation between learned phases and physical state variables
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm

from sensor_fusion_experiment import (
    SensorFusionDataset,
    PSISensorFusionModel,
    RADAR, LIDAR, CAMERA, IMU,
    compute_input_dim,
    device
)


class PSIWithHooks(nn.Module):
    """
    Wrapper that extracts internal PSI representations during forward pass.

    Captures:
    - phi: integrated phase (the trajectory in phase space)
    - omega: phase velocity (the flow field)
    - gate: gating values (which dimensions are active)
    - trajectory: complex representation (cos(phi), sin(phi))
    - memory: accumulated holographic memory
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on PSI layers to capture internal state."""

        def make_hook(name):
            def hook(module, input, output):
                # Store the input to PSI (after norm)
                if hasattr(module, 'integration'):
                    # This is a PSIBlock - we want to hook into the PSI module inside
                    pass
            return hook

        # We need to hook into the PSI modules themselves
        self.hooks = []
        for i, block in enumerate(self.model.psi_blocks):
            # Hook into the PSI integration module
            psi_module = block.integration

            # Create a custom forward that captures internals
            original_forward = psi_module.forward

            def make_capturing_forward(layer_idx, orig_fwd, psi_mod):
                def capturing_forward(x):
                    # Replicate PSI forward but capture intermediates
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
                    self.activations[f'layer_{layer_idx}_magnitude'] = magnitude.detach()

                    # Continue with normal forward
                    trajectory_real = torch.cos(phi)
                    trajectory_imag = torch.sin(phi)

                    self.activations[f'layer_{layer_idx}_traj_real'] = trajectory_real.detach()
                    self.activations[f'layer_{layer_idx}_traj_imag'] = trajectory_imag.detach()

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

            # Replace forward method
            psi_module.forward = make_capturing_forward(i, original_forward, psi_module)

    def forward(self, x):
        self.activations = {}  # Clear previous
        return self.model(x)

    def get_activations(self):
        return self.activations


def extract_phase_space_data(model, dataset, device, n_samples=100):
    """
    Extract phase space representations for a set of samples.

    Returns dict with:
    - phases: [n_samples, seq_len, dim] for each layer
    - omegas: [n_samples, seq_len, dim] for each layer
    - targets: [n_samples, seq_len, 6] ground truth states
    """
    hooked_model = PSIWithHooks(model)
    hooked_model.eval()

    all_data = {
        'targets': [],
        'inputs': [],
    }

    # Initialize storage for each layer
    num_layers = len(model.psi_blocks)
    for layer_idx in range(num_layers):
        all_data[f'phi_layer_{layer_idx}'] = []
        all_data[f'omega_layer_{layer_idx}'] = []
        all_data[f'gated_omega_layer_{layer_idx}'] = []
        all_data[f'traj_real_layer_{layer_idx}'] = []
        all_data[f'traj_imag_layer_{layer_idx}'] = []
        all_data[f'memory_real_layer_{layer_idx}'] = []
        all_data[f'memory_imag_layer_{layer_idx}'] = []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting phase space"):
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.numpy()

            # Forward pass captures activations
            _ = hooked_model(inputs)
            activations = hooked_model.get_activations()

            all_data['targets'].append(targets)
            all_data['inputs'].append(inputs.cpu().numpy()[0])

            for layer_idx in range(num_layers):
                all_data[f'phi_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_phi'].cpu().numpy()[0]
                )
                all_data[f'omega_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_omega'].cpu().numpy()[0]
                )
                all_data[f'gated_omega_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_gated_omega'].cpu().numpy()[0]
                )
                all_data[f'traj_real_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_traj_real'].cpu().numpy()[0]
                )
                all_data[f'traj_imag_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_traj_imag'].cpu().numpy()[0]
                )
                all_data[f'memory_real_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_memory_real'].cpu().numpy()[0]
                )
                all_data[f'memory_imag_layer_{layer_idx}'].append(
                    activations[f'layer_{layer_idx}_memory_imag'].cpu().numpy()[0]
                )

    # Stack arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])

    return all_data, num_layers


def visualize_phase_trajectories(data, num_layers, save_prefix='phase_space'):
    """
    Visualize phase trajectories colored by ground truth state.

    Creates plots showing how the learned phase representation
    correlates with the true physical state.
    """
    targets = data['targets']  # [n_samples, seq_len, 6]

    # Use PCA to reduce phase dimensions for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase Space Trajectories Colored by Ground Truth', fontsize=14)

    # Focus on first and last layer
    for layer_idx, ax_row in zip([0, num_layers-1], axes):
        phi = data[f'phi_layer_{layer_idx}']  # [n_samples, seq_len, dim]

        # Flatten for PCA
        n_samples, seq_len, dim = phi.shape
        phi_flat = phi.reshape(-1, dim)

        # PCA to 2D
        pca = PCA(n_components=2)
        phi_2d = pca.fit_transform(phi_flat)
        phi_2d = phi_2d.reshape(n_samples, seq_len, 2)

        # Color by position x, position y, velocity magnitude
        targets_flat = targets.reshape(n_samples, seq_len, 6)

        # Plot 1: Color by x position
        ax = ax_row[0]
        for i in range(min(20, n_samples)):
            colors = targets_flat[i, :, 0]  # x position
            scatter = ax.scatter(phi_2d[i, :, 0], phi_2d[i, :, 1],
                               c=colors, cmap='coolwarm', s=5, alpha=0.5)
            ax.plot(phi_2d[i, :, 0], phi_2d[i, :, 1], alpha=0.2, linewidth=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Layer {layer_idx}: Colored by X position')
        plt.colorbar(scatter, ax=ax, label='X (m)')

        # Plot 2: Color by y position
        ax = ax_row[1]
        for i in range(min(20, n_samples)):
            colors = targets_flat[i, :, 1]  # y position
            scatter = ax.scatter(phi_2d[i, :, 0], phi_2d[i, :, 1],
                               c=colors, cmap='coolwarm', s=5, alpha=0.5)
            ax.plot(phi_2d[i, :, 0], phi_2d[i, :, 1], alpha=0.2, linewidth=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Layer {layer_idx}: Colored by Y position')
        plt.colorbar(scatter, ax=ax, label='Y (m)')

        # Plot 3: Color by velocity magnitude
        ax = ax_row[2]
        for i in range(min(20, n_samples)):
            vel_mag = np.sqrt(targets_flat[i, :, 3]**2 + targets_flat[i, :, 4]**2)
            scatter = ax.scatter(phi_2d[i, :, 0], phi_2d[i, :, 1],
                               c=vel_mag, cmap='viridis', s=5, alpha=0.5)
            ax.plot(phi_2d[i, :, 0], phi_2d[i, :, 1], alpha=0.2, linewidth=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Layer {layer_idx}: Colored by velocity magnitude')
        plt.colorbar(scatter, ax=ax, label='|v| (m/s)')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_trajectories.png")
    plt.close()


def visualize_flow_field(data, num_layers, save_prefix='phase_space'):
    """
    Visualize the learned flow field (phase velocities).

    This shows how the model "pushes" states through phase space.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Learned Flow Field (Phase Velocities)', fontsize=14)

    for layer_idx, ax in zip([0, num_layers//2, num_layers-1], axes.flat[:3]):
        phi = data[f'phi_layer_{layer_idx}']
        omega = data[f'gated_omega_layer_{layer_idx}']  # Use gated omega (actual flow)

        n_samples, seq_len, dim = phi.shape

        # PCA on phi
        phi_flat = phi.reshape(-1, dim)
        pca = PCA(n_components=2)
        phi_2d = pca.fit_transform(phi_flat).reshape(n_samples, seq_len, 2)

        # Project omega to same space
        omega_flat = omega.reshape(-1, dim)
        omega_2d = pca.transform(omega_flat).reshape(n_samples, seq_len, 2)

        # Subsample for quiver plot
        step = 5
        for i in range(min(10, n_samples)):
            ax.quiver(
                phi_2d[i, ::step, 0], phi_2d[i, ::step, 1],
                omega_2d[i, ::step, 0], omega_2d[i, ::step, 1],
                alpha=0.5, scale=50
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Layer {layer_idx} Flow Field')

    # Flow magnitude histogram
    ax = axes.flat[3]
    for layer_idx in [0, num_layers-1]:
        omega = data[f'gated_omega_layer_{layer_idx}']
        omega_mag = np.linalg.norm(omega, axis=-1).flatten()
        ax.hist(omega_mag, bins=50, alpha=0.5, label=f'Layer {layer_idx}', density=True)
    ax.set_xlabel('Flow Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Flow Magnitudes')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_flow_field.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_flow_field.png")
    plt.close()


def visualize_phase_state_correlation(data, num_layers, save_prefix='phase_space'):
    """
    Analyze correlation between learned phase dimensions and physical state.

    If PSI learned the true dynamics, some phase dimensions should
    strongly correlate with position/velocity.
    """
    targets = data['targets']  # [n_samples, seq_len, 6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Phase-State Correlation Analysis', fontsize=14)

    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    for layer_idx, ax_row in zip([0, num_layers-1], axes):
        phi = data[f'phi_layer_{layer_idx}']  # [n_samples, seq_len, dim]

        n_samples, seq_len, dim = phi.shape

        # Compute correlation between each phase dim and each state dim
        phi_flat = phi.reshape(-1, dim)
        targets_flat = targets.reshape(-1, 6)

        # Correlation matrix
        correlations = np.zeros((dim, 6))
        for d in range(dim):
            for s in range(6):
                correlations[d, s] = np.corrcoef(phi_flat[:, d], targets_flat[:, s])[0, 1]

        # Find top correlated dimensions for each state
        ax = ax_row[0]
        im = ax.imshow(np.abs(correlations[:50, :]), aspect='auto', cmap='hot')
        ax.set_xlabel('State Variable')
        ax.set_ylabel('Phase Dimension')
        ax.set_xticks(range(6))
        ax.set_xticklabels(state_names)
        ax.set_title(f'Layer {layer_idx}: |Correlation| (first 50 dims)')
        plt.colorbar(im, ax=ax)

        # Max correlation per state
        ax = ax_row[1]
        max_corr = np.max(np.abs(correlations), axis=0)
        bars = ax.bar(state_names, max_corr)
        ax.set_ylabel('Max |Correlation|')
        ax.set_title(f'Layer {layer_idx}: Best Correlating Dimension')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, max_corr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # Scatter of best correlating dim vs position x
        ax = ax_row[2]
        best_dim = np.argmax(np.abs(correlations[:, 0]))  # Best for x position
        ax.scatter(phi_flat[:5000, best_dim], targets_flat[:5000, 0],
                  alpha=0.1, s=1)
        ax.set_xlabel(f'Phase dim {best_dim}')
        ax.set_ylabel('X position (m)')
        ax.set_title(f'Layer {layer_idx}: Best dim for X (r={correlations[best_dim, 0]:.3f})')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_correlation.png")
    plt.close()

    return correlations


def visualize_complex_trajectory(data, num_layers, save_prefix='phase_space'):
    """
    Visualize the complex trajectory (cos(phi), sin(phi)) on the unit circle.

    Shows how different dimensions oscillate through phase space.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Complex Trajectories on Unit Circle', fontsize=14)

    targets = data['targets']

    for layer_idx, ax_row in zip([0, num_layers-1], axes):
        traj_real = data[f'traj_real_layer_{layer_idx}']  # cos(phi)
        traj_imag = data[f'traj_imag_layer_{layer_idx}']  # sin(phi)

        n_samples, seq_len, dim = traj_real.shape

        # Pick a few dimensions to visualize
        dims_to_show = [0, dim//4, dim//2, 3*dim//4]

        for d_idx, (dim_idx, ax) in enumerate(zip(dims_to_show, ax_row)):
            # Plot unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

            # Plot trajectories for a few samples, colored by time
            for i in range(min(5, n_samples)):
                real = traj_real[i, :, dim_idx]
                imag = traj_imag[i, :, dim_idx]
                colors = np.arange(seq_len)
                scatter = ax.scatter(real, imag, c=colors, cmap='viridis', s=3, alpha=0.7)
                ax.plot(real, imag, alpha=0.3, linewidth=0.5)

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.set_xlabel('cos(phi)')
            ax.set_ylabel('sin(phi)')
            ax.set_title(f'Layer {layer_idx}, Dim {dim_idx}')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_unit_circle.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_unit_circle.png")
    plt.close()


def visualize_memory_accumulation(data, num_layers, save_prefix='phase_space'):
    """
    Visualize how holographic memory accumulates over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Holographic Memory Accumulation', fontsize=14)

    targets = data['targets']

    # Memory magnitude over time for different layers
    ax = axes[0, 0]
    for layer_idx in [0, num_layers//2, num_layers-1]:
        mem_real = data[f'memory_real_layer_{layer_idx}']
        mem_imag = data[f'memory_imag_layer_{layer_idx}']
        mem_mag = np.sqrt(mem_real**2 + mem_imag**2)

        # Average over samples and dimensions
        mean_mag = mem_mag.mean(axis=(0, 2))  # [seq_len]
        std_mag = mem_mag.std(axis=(0, 2))

        timesteps = np.arange(len(mean_mag))
        ax.plot(timesteps, mean_mag, label=f'Layer {layer_idx}')
        ax.fill_between(timesteps, mean_mag - std_mag, mean_mag + std_mag, alpha=0.2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Memory Magnitude')
    ax.set_title('Memory Accumulation Over Time')
    ax.legend()

    # Memory PCA colored by position
    ax = axes[0, 1]
    layer_idx = num_layers - 1
    mem_real = data[f'memory_real_layer_{layer_idx}']
    mem_imag = data[f'memory_imag_layer_{layer_idx}']

    n_samples, seq_len, dim = mem_real.shape

    # Combine real and imag for PCA
    memory = np.concatenate([mem_real, mem_imag], axis=-1)
    memory_flat = memory.reshape(-1, dim * 2)

    pca = PCA(n_components=2)
    mem_2d = pca.fit_transform(memory_flat).reshape(n_samples, seq_len, 2)

    for i in range(min(15, n_samples)):
        colors = targets[i, :, 0]  # x position
        ax.scatter(mem_2d[i, :, 0], mem_2d[i, :, 1], c=colors, cmap='coolwarm', s=3, alpha=0.5)
        ax.plot(mem_2d[i, :, 0], mem_2d[i, :, 1], alpha=0.2, linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Memory Space (Layer {layer_idx}) Colored by X')

    # Memory vs velocity
    ax = axes[1, 0]
    mem_mag_final = np.sqrt(mem_real[:, -1, :]**2 + mem_imag[:, -1, :]**2).mean(axis=-1)
    vel_mag_final = np.sqrt(targets[:, -1, 3]**2 + targets[:, -1, 4]**2)
    ax.scatter(vel_mag_final, mem_mag_final, alpha=0.5, s=10)
    ax.set_xlabel('Final Velocity Magnitude (m/s)')
    ax.set_ylabel('Final Memory Magnitude')
    ax.set_title('Memory vs Velocity at Sequence End')

    # Phase velocity vs actual velocity
    ax = axes[1, 1]
    omega = data[f'gated_omega_layer_{num_layers-1}']
    omega_mag = np.linalg.norm(omega, axis=-1).mean(axis=-1)  # [n_samples]
    vel_mag_mean = np.sqrt(targets[:, :, 3]**2 + targets[:, :, 4]**2).mean(axis=-1)
    ax.scatter(vel_mag_mean, omega_mag, alpha=0.5, s=10)
    ax.set_xlabel('Mean Velocity Magnitude (m/s)')
    ax.set_ylabel('Mean Phase Velocity (flow)')
    ax.set_title('Phase Flow vs Physical Velocity')

    # Add correlation
    corr = np.corrcoef(vel_mag_mean, omega_mag)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_memory.png', dpi=150, bbox_inches='tight')
    print(f"Saved {save_prefix}_memory.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Phase Space Visualization for PSI')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples to analyze')
    parser.add_argument('--n_trajectories', type=int, default=500,
                        help='Number of trajectories for test dataset')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Sequence length')
    parser.add_argument('--dim', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of PSI layers')
    parser.add_argument('--save_prefix', type=str, default='phase_space',
                        help='Prefix for saved figures')
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE SPACE VISUALIZATION")
    print("=" * 80)
    print()
    print("Extracting learned phase space representation from trained PSI model.")
    print("This reveals what dynamical structure the model discovered.")
    print("=" * 80)
    print()

    # Load model
    print("Loading trained model...")
    sensors = [RADAR, LIDAR, CAMERA, IMU]
    input_dim = compute_input_dim(len(sensors))

    model = PSISensorFusionModel(
        input_dim=input_dim,
        state_dim=6,
        hidden_dim=args.dim,
        num_layers=args.num_layers
    ).to(device)

    try:
        checkpoint = torch.load('sensor_fusion_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint")
    except FileNotFoundError:
        print("ERROR: sensor_fusion_model.pt not found!")
        print("Please train the sensor fusion model first.")
        return

    # Generate test data
    print(f"\nGenerating test data ({args.n_trajectories} trajectories)...")
    test_dataset = SensorFusionDataset(
        n_trajectories=args.n_trajectories,
        duration=10.0,
        sensors=sensors,
        seq_len=args.seq_len,
        split='test',
        seed=999
    )

    # Extract phase space data
    print(f"\nExtracting phase space representations ({args.n_samples} samples)...")
    data, num_layers = extract_phase_space_data(model, test_dataset, device, args.n_samples)

    print(f"\nExtracted data shapes:")
    print(f"  Targets: {data['targets'].shape}")
    print(f"  Phi (layer 0): {data['phi_layer_0'].shape}")
    print(f"  Omega (layer 0): {data['omega_layer_0'].shape}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n1. Phase trajectories colored by ground truth...")
    visualize_phase_trajectories(data, num_layers, args.save_prefix)

    print("\n2. Flow field visualization...")
    visualize_flow_field(data, num_layers, args.save_prefix)

    print("\n3. Phase-state correlation analysis...")
    correlations = visualize_phase_state_correlation(data, num_layers, args.save_prefix)

    print("\n4. Complex trajectories on unit circle...")
    visualize_complex_trajectory(data, num_layers, args.save_prefix)

    print("\n5. Memory accumulation analysis...")
    visualize_memory_accumulation(data, num_layers, args.save_prefix)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Report strongest correlations
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    print("\nStrongest phase-state correlations (final layer):")
    phi = data[f'phi_layer_{num_layers-1}']
    targets = data['targets']

    phi_flat = phi.reshape(-1, phi.shape[-1])
    targets_flat = targets.reshape(-1, 6)

    for s_idx, s_name in enumerate(state_names):
        correlations = [np.corrcoef(phi_flat[:, d], targets_flat[:, s_idx])[0, 1]
                       for d in range(phi.shape[-1])]
        max_corr_idx = np.argmax(np.abs(correlations))
        max_corr = correlations[max_corr_idx]
        print(f"  {s_name}: dim {max_corr_idx} (r = {max_corr:.3f})")

    print("\nDone! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
