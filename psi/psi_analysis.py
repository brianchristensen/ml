"""
PSI Deep Analysis: Understanding What PSI Learns

This script analyzes PSI's internal representations to understand:
1. Phase trajectory evolution during different tasks
2. Memory accumulation patterns
3. Why PSI succeeds at dynamics but fails at long-range dependencies
4. What makes PSI generate coherent text despite worse perplexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, TensorDataset
import math
from psi import PSI, PSIBlock

# ============================================================================
# INSTRUMENTED PSI - Captures internal states for analysis
# ============================================================================

class InstrumentedPSI(nn.Module):
    """PSI with hooks to capture internal states."""

    def __init__(self, dim, init_scale=0.1):
        super().__init__()
        self.dim = dim
        self.to_omega = nn.Linear(dim, dim)
        # Learned per-dimension integration scale (log-space for stability)
        self.log_scale = nn.Parameter(torch.ones(dim) * math.log(init_scale))
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # Storage for internal states
        self.captured_states = {}

    def forward(self, x, capture=False):
        batch_size, seq_len, dim = x.shape

        # Compute velocity field
        omega = self.to_omega(x)

        # Position-dependent scale: 1/sqrt(position) decay prevents unbounded phase accumulation
        position = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        pos_scale = 1.0 / torch.sqrt(position)

        # Learned scale with position decay
        scale = torch.exp(self.log_scale)
        omega_scaled = omega * scale * pos_scale

        # Integrate: phi = cumsum(omega)
        phi = torch.cumsum(omega_scaled, dim=1)

        # Phase trajectory
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Phase-bound memory accumulation
        memory_real = torch.cumsum(x * cos_phi, dim=1)
        memory_imag = torch.cumsum(x * sin_phi, dim=1)

        # Position-based normalization
        memory_real_normalized = memory_real / position
        memory_imag_normalized = memory_imag / position

        # Retrieve at current phase
        retrieved_real = memory_real_normalized * cos_phi + memory_imag_normalized * sin_phi
        retrieved_imag = memory_imag_normalized * cos_phi - memory_real_normalized * sin_phi

        # Content modulation
        content_modulated_real = x * cos_phi
        content_modulated_imag = x * sin_phi

        if capture:
            self.captured_states = {
                'omega': omega.detach(),
                'omega_scaled': omega_scaled.detach(),
                'phi': phi.detach(),
                'cos_phi': cos_phi.detach(),
                'sin_phi': sin_phi.detach(),
                'memory_real': memory_real.detach(),
                'memory_imag': memory_imag.detach(),
                'memory_real_normalized': memory_real_normalized.detach(),
                'memory_imag_normalized': memory_imag_normalized.detach(),
                'retrieved_real': retrieved_real.detach(),
                'retrieved_imag': retrieved_imag.detach(),
                'content_modulated_real': content_modulated_real.detach(),
                'content_modulated_imag': content_modulated_imag.detach(),
                'scale': scale.detach(),
                'pos_scale': pos_scale.detach(),
            }

        # Combine all signals
        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        phase_contribution = self.to_out(context)

        return x + phase_contribution


class InstrumentedPSIModel(nn.Module):
    """Full PSI model with instrumentation."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(8192, hidden_dim))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'integration': InstrumentedPSI(hidden_dim)
            })
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.layer_states = {}

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x, capture=False):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, seq_len, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_encoding[:seq_len].unsqueeze(0)

        self.layer_states = {}

        for i, block in enumerate(self.blocks):
            h_normed = block['norm'](h)
            h = h + block['integration'](h_normed, capture=capture)

            if capture:
                self.layer_states[f'layer_{i}'] = block['integration'].captured_states.copy()

        h = self.norm(h)
        return self.output(h)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_phase_trajectories(model, X, title="Phase Trajectories"):
    """Visualize how phase evolves across sequence positions."""

    model.eval()
    with torch.no_grad():
        _ = model(X, capture=True)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    for layer_idx in range(min(4, len(model.layer_states))):
        states = model.layer_states[f'layer_{layer_idx}']
        phi = states['phi'][0].cpu().numpy()  # First sample

        seq_len, dim = phi.shape

        # Plot phase evolution for first 8 dimensions
        ax = fig.add_subplot(gs[0, layer_idx])
        for d in range(min(8, dim)):
            ax.plot(phi[:, d], alpha=0.7, label=f'dim {d}')
        ax.set_title(f'Layer {layer_idx}: Phase φ over time')
        ax.set_xlabel('Position')
        ax.set_ylabel('Phase (radians)')
        if layer_idx == 0:
            ax.legend(fontsize=6)

        # Plot omega (velocity field)
        omega = states['omega'][0].cpu().numpy()
        ax = fig.add_subplot(gs[1, layer_idx])
        im = ax.imshow(omega.T[:32, :], aspect='auto', cmap='RdBu_r')
        ax.set_title(f'Layer {layer_idx}: ω (velocity field)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax)

        # Plot phase distribution at different positions
        ax = fig.add_subplot(gs[2, layer_idx])
        positions = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
        for pos in positions:
            ax.hist(phi[pos, :], bins=30, alpha=0.5, label=f'pos {pos}')
        ax.set_title(f'Layer {layer_idx}: Phase distribution')
        ax.set_xlabel('Phase')
        ax.legend(fontsize=6)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_phase_trajectories.png', dpi=150)
    plt.close()
    print("Saved: analysis_phase_trajectories.png")


def analyze_memory_persistence(model, X, title="Memory Persistence"):
    """Analyze how memory accumulates and persists over sequence."""

    model.eval()
    with torch.no_grad():
        _ = model(X, capture=True)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    for layer_idx in range(min(4, len(model.layer_states))):
        states = model.layer_states[f'layer_{layer_idx}']

        mem_real = states['memory_real_normalized'][0].cpu().numpy()
        mem_imag = states['memory_imag_normalized'][0].cpu().numpy()

        # Memory magnitude over time
        mem_magnitude = np.sqrt(mem_real**2 + mem_imag**2)

        ax = fig.add_subplot(gs[0, layer_idx])
        ax.plot(mem_magnitude.mean(axis=1), label='Mean magnitude')
        ax.fill_between(range(len(mem_magnitude)),
                       mem_magnitude.min(axis=1),
                       mem_magnitude.max(axis=1), alpha=0.3)
        ax.set_title(f'Layer {layer_idx}: Memory magnitude')
        ax.set_xlabel('Position')
        ax.set_ylabel('|Memory|')

        # Memory heatmap
        ax = fig.add_subplot(gs[1, layer_idx])
        im = ax.imshow(mem_magnitude.T[:32, :], aspect='auto', cmap='viridis')
        ax.set_title(f'Layer {layer_idx}: Memory heatmap')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax)

        # Retrieved vs original content correlation
        retrieved_real = states['retrieved_real'][0].cpu().numpy()
        content_real = states['content_modulated_real'][0].cpu().numpy()

        correlations = []
        for t in range(mem_real.shape[0]):
            corr = np.corrcoef(retrieved_real[t], content_real[t])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)

        ax = fig.add_subplot(gs[2, layer_idx])
        ax.plot(correlations)
        ax.set_title(f'Layer {layer_idx}: Retrieved-Content correlation')
        ax.set_xlabel('Position')
        ax.set_ylabel('Correlation')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_memory_persistence.png', dpi=150)
    plt.close()
    print("Saved: analysis_memory_persistence.png")


def analyze_information_flow(model, X_success, X_failure, task_names=('Success', 'Failure')):
    """Compare information flow between successful and failed tasks."""

    model.eval()

    fig = plt.figure(figsize=(16, 8))

    for task_idx, (X, task_name) in enumerate([(X_success, task_names[0]), (X_failure, task_names[1])]):
        with torch.no_grad():
            _ = model(X, capture=True)

        # Collect metrics across layers
        layer_metrics = []

        for layer_idx in range(len(model.layer_states)):
            states = model.layer_states[f'layer_{layer_idx}']

            phi = states['phi'][0].cpu().numpy()
            omega = states['omega'][0].cpu().numpy()
            mem_real = states['memory_real_normalized'][0].cpu().numpy()
            mem_imag = states['memory_imag_normalized'][0].cpu().numpy()

            # Compute metrics
            phase_velocity = np.abs(omega).mean()
            phase_spread = phi.std()
            memory_magnitude = np.sqrt(mem_real**2 + mem_imag**2).mean()

            # Information at start vs end
            start_info = np.sqrt(mem_real[0]**2 + mem_imag[0]**2).mean()
            end_info = np.sqrt(mem_real[-1]**2 + mem_imag[-1]**2).mean()
            info_ratio = end_info / (start_info + 1e-8)

            layer_metrics.append({
                'phase_velocity': phase_velocity,
                'phase_spread': phase_spread,
                'memory_magnitude': memory_magnitude,
                'info_ratio': info_ratio
            })

        # Plot comparisons
        n_layers = len(layer_metrics)

        ax = fig.add_subplot(2, 4, task_idx * 4 + 1)
        ax.bar(range(n_layers), [m['phase_velocity'] for m in layer_metrics])
        ax.set_title(f'{task_name}: Phase velocity')
        ax.set_xlabel('Layer')

        ax = fig.add_subplot(2, 4, task_idx * 4 + 2)
        ax.bar(range(n_layers), [m['phase_spread'] for m in layer_metrics])
        ax.set_title(f'{task_name}: Phase spread')
        ax.set_xlabel('Layer')

        ax = fig.add_subplot(2, 4, task_idx * 4 + 3)
        ax.bar(range(n_layers), [m['memory_magnitude'] for m in layer_metrics])
        ax.set_title(f'{task_name}: Memory magnitude')
        ax.set_xlabel('Layer')

        ax = fig.add_subplot(2, 4, task_idx * 4 + 4)
        ax.bar(range(n_layers), [m['info_ratio'] for m in layer_metrics])
        ax.set_title(f'{task_name}: Info preservation (end/start)')
        ax.set_xlabel('Layer')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('analysis_information_flow.png', dpi=150)
    plt.close()
    print("Saved: analysis_information_flow.png")


def analyze_first_last_token_memory(model, seq_len=512):
    """
    Critical analysis: Does PSI remember the first token at the end of the sequence?
    This is exactly what the long-range task requires.
    """

    # Create input where first and last tokens have distinct signals
    device = next(model.parameters()).device
    input_dim = model.input_proj.in_features  # Get actual input dim from model
    X = torch.randn(1, seq_len, input_dim, device=device) * 0.1  # Small noise
    X[0, 0, :] = 1.0  # Strong signal at start
    X[0, -1, :] = -1.0  # Opposite signal at end

    model.eval()
    with torch.no_grad():
        _ = model(X, capture=True)

    fig = plt.figure(figsize=(16, 12))

    for layer_idx in range(min(4, len(model.layer_states))):
        states = model.layer_states[f'layer_{layer_idx}']

        mem_real = states['memory_real_normalized'][0].cpu().numpy()
        mem_imag = states['memory_imag_normalized'][0].cpu().numpy()

        # Track how the first token's information persists
        # The first token contributes to memory at all positions
        # Let's see how much of position 0's signal remains at position -1

        first_token_contribution = mem_real[0, :]  # What was stored at t=0

        # At each position, compute how much of the original signal is retrievable
        signal_at_positions = []
        for t in range(seq_len):
            # Correlation between what we stored first and current memory
            corr = np.corrcoef(first_token_contribution, mem_real[t, :])[0, 1]
            signal_at_positions.append(corr if not np.isnan(corr) else 0)

        ax = fig.add_subplot(2, 4, layer_idx + 1)
        ax.plot(signal_at_positions)
        ax.set_title(f'Layer {layer_idx}: First token signal over time')
        ax.set_xlabel('Position')
        ax.set_ylabel('Correlation with t=0')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Also show the raw memory magnitude decay
        mem_magnitude = np.sqrt(mem_real**2 + mem_imag**2)

        ax = fig.add_subplot(2, 4, layer_idx + 5)
        # Compare memory content at position 0 vs position -1
        ax.scatter(mem_magnitude[0, :32], mem_magnitude[-1, :32], alpha=0.5)
        ax.plot([0, mem_magnitude.max()], [0, mem_magnitude.max()], 'r--', alpha=0.5)
        ax.set_title(f'Layer {layer_idx}: Memory t=0 vs t=-1')
        ax.set_xlabel('Memory at start')
        ax.set_ylabel('Memory at end')

    plt.suptitle('First Token Memory Persistence Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_first_token_memory.png', dpi=150)
    plt.close()
    print("Saved: analysis_first_token_memory.png")

    # Print numerical summary
    print("\n=== First Token Memory Persistence ===")
    for layer_idx in range(min(4, len(model.layer_states))):
        states = model.layer_states[f'layer_{layer_idx}']
        mem_real = states['memory_real_normalized'][0].cpu().numpy()

        first_signal = mem_real[0, :]
        last_memory = mem_real[-1, :]

        correlation = np.corrcoef(first_signal, last_memory)[0, 1]
        print(f"Layer {layer_idx}: Correlation(mem[0], mem[-1]) = {correlation:.4f}")


def analyze_phase_separation(model, X, title="Phase Separation Analysis"):
    """
    Analyze how phase separates different types of information.
    Key question: Do different dimensions evolve to different phase regions?
    """

    model.eval()
    with torch.no_grad():
        _ = model(X, capture=True)

    fig = plt.figure(figsize=(16, 12))

    for layer_idx in range(min(4, len(model.layer_states))):
        states = model.layer_states[f'layer_{layer_idx}']

        phi = states['phi'][0].cpu().numpy()  # (seq_len, dim)
        cos_phi = states['cos_phi'][0].cpu().numpy()
        sin_phi = states['sin_phi'][0].cpu().numpy()

        seq_len, dim = phi.shape

        # 1. Phase at end of sequence - are dimensions separated?
        ax = fig.add_subplot(3, 4, layer_idx + 1)
        final_phase = phi[-1, :] % (2 * np.pi)  # Wrap to [0, 2π]
        ax.hist(final_phase, bins=50, edgecolor='black')
        ax.set_title(f'Layer {layer_idx}: Final phase distribution')
        ax.set_xlabel('Phase (radians)')
        ax.set_ylabel('Count')

        # 2. Phase trajectory in 2D (first 2 dims)
        ax = fig.add_subplot(3, 4, layer_idx + 5)
        ax.plot(cos_phi[:, 0], sin_phi[:, 0], 'b-', alpha=0.7, label='dim 0')
        ax.plot(cos_phi[:, 1], sin_phi[:, 1], 'r-', alpha=0.7, label='dim 1')
        ax.scatter([cos_phi[0, 0]], [sin_phi[0, 0]], c='green', s=100, marker='o', zorder=5)
        ax.scatter([cos_phi[-1, 0]], [sin_phi[-1, 0]], c='blue', s=100, marker='x', zorder=5)
        ax.set_title(f'Layer {layer_idx}: Phase trajectory (dims 0,1)')
        ax.set_xlabel('cos(φ)')
        ax.set_ylabel('sin(φ)')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.legend()

        # 3. Phase velocity correlation between dimensions
        omega = states['omega'][0].cpu().numpy()
        omega_corr = np.corrcoef(omega.T)

        ax = fig.add_subplot(3, 4, layer_idx + 9)
        im = ax.imshow(omega_corr[:32, :32], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Layer {layer_idx}: ω correlation')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_phase_separation.png', dpi=150)
    plt.close()
    print("Saved: analysis_phase_separation.png")


def run_comparative_analysis():
    """Run analysis comparing dynamics task vs long-range task."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = InstrumentedPSIModel(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=4
    ).to(device)

    print("\n" + "=" * 80)
    print("PSI DEEP ANALYSIS: What is PSI Actually Learning?")
    print("=" * 80)

    # ========================================================================
    # ANALYSIS 1: Dynamics task (where PSI succeeds)
    # ========================================================================
    print("\n--- Analysis 1: Dynamics Task (PSI succeeds here) ---")

    # Generate simple oscillator data
    t = torch.linspace(0, 10, 200).unsqueeze(0).unsqueeze(-1)
    X_dynamics = torch.sin(t * 2) + 0.5 * torch.sin(t * 5)  # Multi-frequency
    X_dynamics = X_dynamics.to(device)

    analyze_phase_trajectories(model, X_dynamics, "Phase Trajectories - Dynamics Task")
    analyze_memory_persistence(model, X_dynamics, "Memory Persistence - Dynamics Task")
    analyze_phase_separation(model, X_dynamics, "Phase Separation - Dynamics Task")

    # ========================================================================
    # ANALYSIS 2: Long-range task (where PSI fails)
    # ========================================================================
    print("\n--- Analysis 2: Long-Range Task (PSI fails here) ---")

    # Generate long-range dependency data
    seq_len = 512
    X_longrange = torch.randn(1, seq_len, 1).to(device) * 0.1
    X_longrange[0, 0, 0] = 1.0  # Signal at start
    X_longrange[0, -1, 0] = 1.0  # Signal at end

    analyze_phase_trajectories(model, X_longrange, "Phase Trajectories - Long-Range Task")
    analyze_memory_persistence(model, X_longrange, "Memory Persistence - Long-Range Task")

    # ========================================================================
    # ANALYSIS 3: Critical - First token memory persistence
    # ========================================================================
    print("\n--- Analysis 3: First Token Memory Persistence ---")
    analyze_first_last_token_memory(model, seq_len=512)

    # ========================================================================
    # ANALYSIS 4: Compare trained models on both tasks
    # ========================================================================
    print("\n--- Analysis 4: Training and comparing on both tasks ---")

    # Train on dynamics
    print("\nTraining model on dynamics task...")
    model_dynamics = InstrumentedPSIModel(1, 64, 1, 4).to(device)
    optimizer = torch.optim.AdamW(model_dynamics.parameters(), lr=1e-3)

    for epoch in range(50):
        t = torch.linspace(0, 10, 200).unsqueeze(0).unsqueeze(-1).to(device)
        X = torch.sin(t * 2) + 0.5 * torch.sin(t * 5)
        Y = torch.sin((t + 0.1) * 2) + 0.5 * torch.sin((t + 0.1) * 5)  # Shifted

        optimizer.zero_grad()
        pred = model_dynamics(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()

    print(f"  Final dynamics loss: {loss.item():.6f}")

    # Analyze trained dynamics model
    with torch.no_grad():
        _ = model_dynamics(X, capture=True)

    print("\n=== Trained Dynamics Model - Phase Analysis ===")
    for layer_idx in range(4):
        states = model_dynamics.layer_states[f'layer_{layer_idx}']
        phi = states['phi'][0].cpu().numpy()
        omega = states['omega'][0].cpu().numpy()

        print(f"Layer {layer_idx}:")
        print(f"  Phase range: [{phi.min():.2f}, {phi.max():.2f}]")
        print(f"  Omega mean: {omega.mean():.4f}, std: {omega.std():.4f}")

    # Train on long-range
    print("\nTraining model on long-range task...")
    model_longrange = InstrumentedPSIModel(1, 64, 2, 4).to(device)
    optimizer = torch.optim.AdamW(model_longrange.parameters(), lr=1e-3)

    for epoch in range(100):
        # Create long-range classification task
        X = torch.randn(32, 256, 1).to(device) * 0.1
        labels = torch.randint(0, 2, (32,)).to(device)

        for i in range(32):
            if labels[i] == 0:
                X[i, 0, 0] = 1.0
                X[i, -1, 0] = 1.0
            else:
                X[i, 0, 0] = 1.0
                X[i, -1, 0] = -1.0

        optimizer.zero_grad()
        pred = model_longrange(X)
        pred_class = pred.mean(dim=1)  # Pool
        loss = F.cross_entropy(pred_class, labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    model_longrange.eval()
    with torch.no_grad():
        X_test = torch.randn(100, 256, 1).to(device) * 0.1
        labels_test = torch.randint(0, 2, (100,)).to(device)

        for i in range(100):
            if labels_test[i] == 0:
                X_test[i, 0, 0] = 1.0
                X_test[i, -1, 0] = 1.0
            else:
                X_test[i, 0, 0] = 1.0
                X_test[i, -1, 0] = -1.0

        pred = model_longrange(X_test, capture=True)
        pred_class = pred.mean(dim=1)
        accuracy = (pred_class.argmax(dim=1) == labels_test).float().mean()

    print(f"  Long-range accuracy: {accuracy.item():.4f}")

    print("\n=== Trained Long-Range Model - Phase Analysis ===")
    for layer_idx in range(4):
        states = model_longrange.layer_states[f'layer_{layer_idx}']
        phi = states['phi'][0].cpu().numpy()
        omega = states['omega'][0].cpu().numpy()

        print(f"Layer {layer_idx}:")
        print(f"  Phase range: [{phi.min():.2f}, {phi.max():.2f}]")
        print(f"  Omega mean: {omega.mean():.4f}, std: {omega.std():.4f}")

    # ========================================================================
    # KEY INSIGHT ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("""
    The analysis should reveal:

    1. MEMORY DECAY: With position-based normalization (dividing by position),
       early information gets diluted. At position 512, the first token's
       contribution is divided by 512, making it 0.2% of its original strength.

    2. PHASE ACCUMULATION: Cumsum causes phase to grow unboundedly. By the end
       of a long sequence, phases may have wrapped around multiple times, losing
       the ability to selectively retrieve early memories.

    3. WHY DYNAMICS WORKS: For dynamics, we don't need to retrieve specific
       early memories - we need smooth temporal evolution. The cumsum naturally
       provides this temporal coherence.

    4. WHY LONG-RANGE FAILS: Retrieving specific tokens (first and last) requires
       precise phase alignment, which is lost after cumulative phase accumulation.

    POTENTIAL FIXES:
    - Selective memory (like Mamba) - don't accumulate everything
    - Phase reset/attention mechanisms
    - Learned forgetting rates
    - Skip connections in phase space
    """)


if __name__ == '__main__':
    run_comparative_analysis()
