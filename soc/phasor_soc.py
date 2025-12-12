"""
Phasor-SOC Mind: Self-Organizing Critical dynamics with Phase-based binding
===========================================================================

Combines:
- SOC dynamics: Avalanches, criticality, power laws
- Phasor binding: Phase coherence for memory and computation

Each unit is an oscillator with:
- A_i: Amplitude (activation strength)
- θ_i: Phase (binding/addressing)

Key insight: Resonance = phase coherence
- Matching patterns → phases synchronize → constructive interference → avalanche
- Non-matching patterns → destructive interference → dies out

The sparse distributed pattern matching memory IS the computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import networkx as nx
from scipy import sparse
import threading
import queue
import sys


class PhasorSOCMind:
    """
    Self-Organizing Critical system with Phasor-based binding.

    Units are oscillators with amplitude and phase.
    Weights encode both magnitude and phase offset.
    Resonance through phase coherence enables pattern recognition.
    """

    def __init__(self, n_units=1024, k_neighbors=10, rewire_prob=0.1,
                 n_input=128, n_output=128, seed=42):
        """
        Initialize the Phasor-SOC mind.

        Args:
            n_units: Total units in network
            k_neighbors: Small-world connectivity
            rewire_prob: Watts-Strogatz rewiring
            n_input: Size of input region (first n_input units)
            n_output: Size of output region (last n_output units)
            seed: Random seed
        """
        np.random.seed(seed)
        self.n = n_units
        self.t = 0.0

        # === Region definitions ===
        self.n_input = n_input
        self.n_output = n_output
        self.input_region = np.arange(0, n_input)
        self.output_region = np.arange(n_units - n_output, n_units)
        self.association_region = np.arange(n_input, n_units - n_output)

        # === State Variables ===
        # Amplitude (activation magnitude)
        self.A = np.zeros(n_units)
        # Phase (binding/addressing) - random initialization
        self.theta = np.random.uniform(0, 2 * np.pi, n_units)
        # Energy/resources
        self.E = np.ones(n_units)

        # === Network Topology (Small-World + Cross-Region Connections) ===
        G = nx.watts_strogatz_graph(n_units, k_neighbors, rewire_prob, seed=seed)

        # Add DENSE connections between input and output regions
        # Every input unit connects to multiple output units
        n_connections_per_input = 16  # Each input connects to 16 outputs
        for i, inp_unit in enumerate(self.input_region):
            # Connect to spread of output units based on input index
            for j in range(n_connections_per_input):
                out_idx = (i * 7 + j * 13) % n_output  # Spread across outputs
                out_unit = self.output_region[out_idx]
                G.add_edge(inp_unit, out_unit)

            # Also connect through association region
            assoc_idx = (i * 11) % (n_units - n_input - n_output)
            assoc_unit = n_input + assoc_idx
            G.add_edge(inp_unit, assoc_unit)

        # Also connect association to all outputs
        for i in range(0, n_units - n_input - n_output, 8):
            assoc_unit = n_input + i
            for j in range(0, n_output, 8):
                out_unit = self.output_region[j]
                G.add_edge(assoc_unit, out_unit)

        self.adj = nx.to_scipy_sparse_array(G, format='csr')

        # === Weights: Magnitude and Phase Offset ===
        # W_mag: connection strength (like original W)
        # W_phase: learned phase offset for binding
        self.W_mag = sparse.lil_matrix((n_units, n_units))
        self.W_phase = sparse.lil_matrix((n_units, n_units))

        input_set = set(self.input_region)
        output_set = set(self.output_region)

        for i, j in G.edges():
            # Stronger initial weights for cross-region connections
            if (i in input_set and j in output_set) or \
               (j in input_set and i in output_set):
                w = np.random.uniform(0.3, 0.5)  # Stronger
            else:
                w = np.random.uniform(0.1, 0.3)

            phi = np.random.uniform(-np.pi/4, np.pi/4)
            self.W_mag[i, j] = w
            self.W_mag[j, i] = w
            self.W_phase[i, j] = phi
            self.W_phase[j, i] = -phi

        self.W_mag = self.W_mag.tocsr()
        self.W_phase = self.W_phase.tocsr()

        # === Parameters ===
        # Amplitude dynamics
        self.theta_thresh = 0.3    # Firing threshold
        self.decay = 0.1           # Amplitude decay
        self.noise_std = 0.005     # Amplitude noise

        # Phase dynamics
        self.phase_coupling = 0.3  # How strongly phases couple
        self.phase_noise = 0.02    # Phase diffusion

        # Plasticity
        self.eta_mag = 0.002       # Magnitude learning rate
        self.eta_phase = 0.01      # Phase learning rate
        self.alpha = 0.01          # Depression coefficient
        self.w_max = 1.0           # Maximum weight magnitude

        # Resource dynamics
        self.r_income = 0.05
        self.r_cost = 0.1
        self.r_diffusion = 0.01

        # === Event Queue ===
        self.event_queue = []

        # === History ===
        self.history = {
            'time': [],
            'mean_amplitude': [],
            'active_count': [],
            'phase_coherence': [],
            'resonance': [],
            'input_amplitude': [],
            'output_amplitude': []
        }

        # === Training state ===
        self.training_mode = False
        self.trained_patterns = []

        # === Visualization positions ===
        grid_size = int(np.ceil(np.sqrt(n_units)))
        self.grid_size = grid_size
        self.positions = np.array([
            [np.cos(2 * np.pi * i / n_units), np.sin(2 * np.pi * i / n_units)]
            for i in range(n_units)
        ])

    def step(self, dt=0.1):
        """Advance the system by one timestep."""
        self.t += dt

        # Process external events
        for event in self.event_queue:
            self._inject_pattern(event)
        self.event_queue.clear()

        # Compute dynamics
        dA, dtheta = self._oscillator_dynamics(dt)
        dW_mag, dW_phase = self._plasticity_dynamics(dt)
        dE = self._resource_dynamics(dt)

        # Update state
        self.A = np.clip(self.A + dt * dA, 0, 1)
        self.theta = (self.theta + dt * dtheta) % (2 * np.pi)

        # Update weights
        self.W_mag = self.W_mag + dt * dW_mag
        self.W_mag.data = np.clip(self.W_mag.data, 0, self.w_max)

        self.W_phase = self.W_phase + dt * dW_phase
        self.W_phase.data = np.clip(self.W_phase.data, -np.pi, np.pi)

        self.E = np.clip(self.E + dt * dE, 0, 1)

        # Record history
        self._record_history()

    def _oscillator_dynamics(self, dt):
        """
        Compute amplitude and phase dynamics.

        Key: Input strength depends on phase alignment!
        input_i = Σ_j W_mag_ij * A_j * cos(θ_i - θ_j - φ_ij) * E_j

        This creates resonance: aligned phases → strong input → avalanche
        """
        # Which units are firing?
        firing = (self.A > self.theta_thresh).astype(float)

        # Get sparse structure
        rows, cols = self.W_mag.nonzero()

        # Phase differences: θ_i - θ_j - φ_ij
        phase_diff = self.theta[rows] - self.theta[cols] - self.W_phase.data

        # Phase-modulated input (resonance term!)
        # cos(phase_diff) ∈ [-1, 1]: aligned phases amplify, misaligned suppress
        phase_factor = np.cos(phase_diff)

        # Build input signal with phase modulation
        weighted_input = self.W_mag.data * firing[cols] * self.E[cols] * phase_factor

        # Sum inputs for each unit
        input_signal = np.zeros(self.n)
        np.add.at(input_signal, rows, weighted_input)

        # Normalize by in-degree
        in_degree = np.array(self.W_mag.sum(axis=1)).flatten()
        in_degree = np.maximum(in_degree, 1)
        input_signal = input_signal / in_degree

        # Amplitude dynamics: input - decay + noise
        dA = input_signal - self.decay * self.A + np.random.normal(0, self.noise_std, self.n)

        # === Phase dynamics ===
        # Phases couple toward alignment when both units are active
        # Kuramoto-like coupling: dθ_i/dt = ω_i + Σ_j K_ij * A_j * sin(θ_j + φ_ij - θ_i)

        phase_pull = self.W_mag.data * self.A[cols] * np.sin(
            self.theta[cols] + self.W_phase.data - self.theta[rows]
        )

        dtheta = np.zeros(self.n)
        np.add.at(dtheta, rows, phase_pull)
        dtheta = self.phase_coupling * dtheta / in_degree

        # Phase noise (prevents perfect lock)
        dtheta += np.random.normal(0, self.phase_noise, self.n)

        return dA, dtheta

    def _plasticity_dynamics(self, dt):
        """
        Hebbian plasticity for both magnitude and phase.

        Magnitude: Strengthen connections between co-active units
        Phase: Adjust phase offset to encode binding relationship
        """
        rows, cols = self.W_mag.nonzero()

        # Hebbian: strengthen when both active
        hebbian = self.eta_mag * self.A[rows] * self.A[cols]

        # Depression: weaken when pre is very active (homeostasis)
        depression = self.alpha * self.W_mag.data * (self.A[cols] ** 2)

        # Recovery toward baseline
        recovery = (0.2 - self.W_mag.data) / 100.0

        dW_mag_data = hebbian - depression + recovery
        dW_mag = sparse.csr_matrix((dW_mag_data, (rows, cols)), shape=self.W_mag.shape)

        # === Phase plasticity ===
        # When two units are co-active, adjust phase offset to match their current relationship
        # This encodes the binding: "these phases go together"

        co_activation = self.A[rows] * self.A[cols]
        current_phase_diff = self.theta[rows] - self.theta[cols]
        target_offset = current_phase_diff  # Learn to expect this relationship

        # Move W_phase toward the observed phase difference
        phase_error = target_offset - self.W_phase.data
        # Wrap to [-π, π]
        phase_error = np.arctan2(np.sin(phase_error), np.cos(phase_error))

        dW_phase_data = self.eta_phase * co_activation * phase_error
        dW_phase = sparse.csr_matrix((dW_phase_data, (rows, cols)), shape=self.W_phase.shape)

        return dW_mag, dW_phase

    def _resource_dynamics(self, dt):
        """Energy dynamics (same as before)."""
        income = self.r_income * (1 - self.E)
        cost = self.r_cost * self.A

        neighbor_energy = self.adj.dot(self.E)
        degree = np.array(self.adj.sum(axis=1)).flatten()
        degree = np.maximum(degree, 1)
        diffusion = self.r_diffusion * (neighbor_energy / degree - self.E)

        return income - cost + diffusion

    def _inject_pattern(self, event):
        """Inject a pattern with both amplitude and phase."""
        units = event.get('units', [])
        amplitudes = event.get('amplitudes', 0.5)
        phases = event.get('phases', None)

        if isinstance(amplitudes, (int, float)):
            amplitudes = np.full(len(units), amplitudes)

        for i, idx in enumerate(units):
            self.A[idx] = min(1.0, self.A[idx] + amplitudes[i])
            if phases is not None:
                self.theta[idx] = phases[i]

    def _record_history(self):
        """Record system state."""
        self.history['time'].append(self.t)
        self.history['mean_amplitude'].append(np.mean(self.A))
        self.history['active_count'].append(np.sum(self.A > self.theta_thresh))
        self.history['phase_coherence'].append(self._compute_phase_coherence())
        self.history['resonance'].append(self._compute_resonance())
        self.history['input_amplitude'].append(np.mean(self.A[self.input_region]))
        self.history['output_amplitude'].append(np.mean(self.A[self.output_region]))

    def _compute_phase_coherence(self):
        """
        Compute global phase coherence (Kuramoto order parameter).
        R = |1/N * Σ exp(i*θ_j)|

        R ≈ 1: all phases aligned (high coherence)
        R ≈ 0: phases random (no coherence)
        """
        # Weight by amplitude - only active units contribute
        weighted_phases = self.A * np.exp(1j * self.theta)
        if np.sum(self.A) > 0:
            order_param = np.abs(np.sum(weighted_phases)) / (np.sum(self.A) + 1e-8)
        else:
            order_param = 0.0
        return order_param

    def _compute_resonance(self):
        """
        Compute resonance strength.
        High when: phases aligned AND amplitudes high
        """
        coherence = self._compute_phase_coherence()
        mean_amp = np.mean(self.A)
        return coherence * mean_amp

    # === High-level API for training and testing ===

    def create_pattern(self, region='input', n_active=20, seed=None):
        """
        Create a random sparse pattern.

        Returns dict with units, amplitudes, phases
        """
        if seed is not None:
            np.random.seed(seed)

        if region == 'input':
            available = self.input_region
        elif region == 'output':
            available = self.output_region
        else:
            available = np.arange(self.n)

        units = np.random.choice(available, min(n_active, len(available)), replace=False)
        amplitudes = np.random.uniform(0.5, 1.0, len(units))
        # Coherent phases for the pattern (with some variation)
        base_phase = np.random.uniform(0, 2 * np.pi)
        phases = base_phase + np.random.normal(0, 0.3, len(units))
        phases = phases % (2 * np.pi)

        return {
            'units': units,
            'amplitudes': amplitudes,
            'phases': phases,
            'base_phase': base_phase
        }

    def inject_pattern(self, pattern, strength=1.0):
        """Inject a pattern into the network."""
        self.event_queue.append({
            'units': pattern['units'],
            'amplitudes': pattern['amplitudes'] * strength,
            'phases': pattern['phases']
        })

    def train_association(self, input_pattern, output_pattern, n_steps=100, dt=0.1):
        """
        Train an association between input and output patterns.

        Present both simultaneously, let Hebbian learning bind them.
        """
        self.training_mode = True

        # Store for later testing
        self.trained_patterns.append({
            'input': input_pattern,
            'output': output_pattern
        })

        # Temporarily boost learning rate during training
        old_eta_mag = self.eta_mag
        old_eta_phase = self.eta_phase
        self.eta_mag = 0.01  # 5x boost
        self.eta_phase = 0.05  # 5x boost

        # Present both patterns together repeatedly
        for step in range(n_steps):
            # Inject both patterns frequently to maintain strong co-activation
            if step % 5 == 0:  # More frequent injection
                self.inject_pattern(input_pattern, strength=1.0)
                self.inject_pattern(output_pattern, strength=1.0)

            self.step(dt)

        # Restore learning rates
        self.eta_mag = old_eta_mag
        self.eta_phase = old_eta_phase

        # Let it settle
        for _ in range(50):
            self.step(dt)

        self.training_mode = False
        print(f"Trained association: {len(input_pattern['units'])} input -> {len(output_pattern['units'])} output units")

    def test_recall(self, input_pattern, n_steps=100, dt=0.1):
        """
        Test recall: inject input pattern only, measure output response.

        Returns resonance and output activation over time.
        """
        # Reset state
        self.A = np.zeros(self.n)
        self.theta = np.random.uniform(0, 2 * np.pi, self.n)
        self.E = np.ones(self.n)

        # Clear history for this test
        test_history = {
            'output_amplitude': [],
            'resonance': [],
            'phase_coherence': []
        }

        # Inject input pattern
        self.inject_pattern(input_pattern, strength=1.0)

        # Run and record
        for step in range(n_steps):
            self.step(dt)
            test_history['output_amplitude'].append(np.mean(self.A[self.output_region]))
            test_history['resonance'].append(self._compute_resonance())
            test_history['phase_coherence'].append(self._compute_phase_coherence())

        return test_history

    def measure_pattern_match(self, target_pattern):
        """
        Measure how well current output matches a target pattern.

        Uses both amplitude overlap and phase alignment.
        """
        target_units = target_pattern['units']
        target_phases = target_pattern['phases']

        # Amplitude overlap: are the right units active?
        target_mask = np.zeros(self.n)
        target_mask[target_units] = 1.0

        output_amp = self.A[self.output_region]
        target_amp = target_mask[self.output_region]

        amp_overlap = np.corrcoef(output_amp, target_amp)[0, 1] if np.std(output_amp) > 0 else 0

        # Phase alignment: are phases correct for active units?
        active_target = target_units[self.A[target_units] > 0.1]
        if len(active_target) > 0:
            current_phases = self.theta[active_target]
            expected_phases = target_phases[np.isin(target_units, active_target)]

            # Circular correlation
            phase_diff = current_phases - expected_phases
            phase_alignment = np.mean(np.cos(phase_diff))
        else:
            phase_alignment = 0.0

        return {
            'amplitude_overlap': amp_overlap,
            'phase_alignment': phase_alignment,
            'combined': (amp_overlap + phase_alignment) / 2
        }

    def reset_state(self):
        """Reset activations and phases."""
        self.A = np.zeros(self.n)
        self.theta = np.random.uniform(0, 2 * np.pi, self.n)
        self.E = np.ones(self.n)

    def inject_text(self, text, strength=0.6):
        """Convert text to pattern and inject into input region."""
        if not text:
            return

        # Hash text to select input units and phases
        units = []
        phases = []

        for i, char in enumerate(text):
            base = ord(char)
            # Select units in input region
            unit_idx = self.input_region[base % len(self.input_region)]
            units.append(unit_idx)
            # Phase based on character and position
            phase = (base * 0.1 + i * 0.5) % (2 * np.pi)
            phases.append(phase)

        # Add some spread
        expanded_units = list(units)
        expanded_phases = list(phases)
        for u, p in zip(units, phases):
            for offset in [-1, 1]:
                neighbor = (u + offset - self.input_region[0]) % len(self.input_region) + self.input_region[0]
                expanded_units.append(neighbor)
                expanded_phases.append(p + np.random.normal(0, 0.2))

        pattern = {
            'units': np.array(expanded_units),
            'amplitudes': np.full(len(expanded_units), strength),
            'phases': np.array(expanded_phases) % (2 * np.pi)
        }

        self.inject_pattern(pattern)


class InteractivePhasorSOC:
    """Interactive visualization for Phasor-SOC system."""

    def __init__(self, mind):
        self.mind = mind
        self.input_queue = queue.Queue()
        self.input_listener = None
        self.last_input = ""

        # Create figure
        self.fig = plt.figure(figsize=(18, 12))
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.4],
                                   hspace=0.3, wspace=0.3,
                                   left=0.05, right=0.95, top=0.95, bottom=0.05)

        # === Top row ===
        # Amplitude heatmap
        self.ax_amp = self.fig.add_subplot(gs[0, 0])
        self.ax_amp.set_title('Amplitude Pattern')
        padded = np.zeros(mind.grid_size ** 2)
        padded[:mind.n] = mind.A
        self.amp_img = self.ax_amp.imshow(
            padded.reshape(mind.grid_size, mind.grid_size),
            cmap='hot', vmin=0, vmax=1
        )
        self.fig.colorbar(self.amp_img, ax=self.ax_amp, fraction=0.046)

        # Phase heatmap (HSV: phase = hue, amplitude = value)
        self.ax_phase = self.fig.add_subplot(gs[0, 1])
        self.ax_phase.set_title('Phase Pattern (hue=phase, brightness=amp)')
        self.phase_img = self.ax_phase.imshow(
            self._make_phase_image(), vmin=0, vmax=1
        )

        # Resonance over time
        self.ax_resonance = self.fig.add_subplot(gs[0, 2])
        self.ax_resonance.set_title('Resonance & Coherence')
        self.line_resonance, = self.ax_resonance.plot([], [], 'r-', lw=1.5, label='Resonance')
        self.line_coherence, = self.ax_resonance.plot([], [], 'b-', lw=1, label='Coherence')
        self.ax_resonance.legend()
        self.ax_resonance.set_xlabel('Time')

        # Input/Output amplitude
        self.ax_io = self.fig.add_subplot(gs[0, 3])
        self.ax_io.set_title('Input/Output Activity')
        self.line_input, = self.ax_io.plot([], [], 'g-', lw=1, label='Input')
        self.line_output, = self.ax_io.plot([], [], 'm-', lw=1, label='Output')
        self.ax_io.legend()
        self.ax_io.set_xlabel('Time')

        # === Middle row ===
        # Network view (circular)
        self.ax_network = self.fig.add_subplot(gs[1, 0])
        self.ax_network.set_title('Network (color=phase)')
        self.ax_network.set_xlim(-1.3, 1.3)
        self.ax_network.set_ylim(-1.3, 1.3)
        self.ax_network.set_aspect('equal')
        # Color by phase, size by amplitude
        self.scatter = self.ax_network.scatter(
            mind.positions[:, 0], mind.positions[:, 1],
            c=mind.theta, cmap='hsv', s=mind.A * 50 + 5,
            vmin=0, vmax=2*np.pi, alpha=0.7
        )

        # Region indicators
        theta_vals = np.linspace(0, 2*np.pi, 100)
        # Mark input region
        input_start = 2 * np.pi * mind.input_region[0] / mind.n
        input_end = 2 * np.pi * mind.input_region[-1] / mind.n
        self.ax_network.annotate('INPUT', xy=(1.15, 0), fontsize=8, color='green')
        # Mark output region
        self.ax_network.annotate('OUTPUT', xy=(-1.15, 0), fontsize=8, color='purple', ha='right')

        # Active units bar
        self.ax_active = self.fig.add_subplot(gs[1, 1])
        self.ax_active.set_title('Active Units Over Time')
        self.line_active, = self.ax_active.plot([], [], 'orange', lw=1)

        # Mean amplitude
        self.ax_mean = self.fig.add_subplot(gs[1, 2])
        self.ax_mean.set_title('Mean Amplitude')
        self.line_mean, = self.ax_mean.plot([], [], 'b-', lw=1)

        # Stats
        self.ax_stats = self.fig.add_subplot(gs[1, 3])
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.9, '', ha='left', va='top',
                                              fontsize=9, family='monospace',
                                              transform=self.ax_stats.transAxes)

        # === Sliders ===
        self.sliders = {}
        slider_configs = [
            ('theta_thresh', 'Threshold', 0.0, 1.0, mind.theta_thresh),
            ('decay', 'Decay', 0.01, 0.5, mind.decay),
            ('phase_coupling', 'Phase Coupling', 0.0, 1.0, mind.phase_coupling),
            ('eta_mag', 'Learn Rate', 0.0, 0.01, mind.eta_mag),
            ('eta_phase', 'Phase Learn', 0.0, 0.1, mind.eta_phase),
            ('noise_std', 'Noise', 0.0, 0.05, mind.noise_std),
        ]

        n_sliders = len(slider_configs)
        slider_width = 0.9 / n_sliders

        for i, (param, label, vmin, vmax, vinit) in enumerate(slider_configs):
            ax_slider = self.fig.add_axes([0.05 + i * slider_width, 0.08,
                                           slider_width * 0.85, 0.02])
            slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit,
                           valstep=(vmax - vmin) / 100)
            slider.on_changed(self._make_slider_callback(param))
            self.sliders[param] = slider

        # Buttons
        ax_reset = self.fig.add_axes([0.85, 0.01, 0.1, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        ax_burst = self.fig.add_axes([0.73, 0.01, 0.1, 0.03])
        self.btn_burst = Button(ax_burst, 'Burst')
        self.btn_burst.on_clicked(self._on_burst)

        ax_train = self.fig.add_axes([0.61, 0.01, 0.1, 0.03])
        self.btn_train = Button(ax_train, 'Train Demo')
        self.btn_train.on_clicked(self._on_train_demo)

    def _make_phase_image(self):
        """Create RGB image where hue=phase, brightness=amplitude."""
        from matplotlib.colors import hsv_to_rgb

        padded_phase = np.zeros(self.mind.grid_size ** 2)
        padded_phase[:self.mind.n] = self.mind.theta / (2 * np.pi)

        padded_amp = np.zeros(self.mind.grid_size ** 2)
        padded_amp[:self.mind.n] = self.mind.A

        h = padded_phase.reshape(self.mind.grid_size, self.mind.grid_size)
        s = np.ones_like(h)
        v = padded_amp.reshape(self.mind.grid_size, self.mind.grid_size)

        hsv = np.stack([h, s, v], axis=-1)
        rgb = hsv_to_rgb(hsv)
        return rgb

    def _make_slider_callback(self, param):
        def callback(val):
            setattr(self.mind, param, val)
        return callback

    def _on_reset(self, event):
        self.mind.reset_state()

    def _on_burst(self, event):
        # Random burst in input region
        pattern = self.mind.create_pattern('input', n_active=30)
        self.mind.inject_pattern(pattern, strength=0.9)

    def _on_train_demo(self, event):
        """Demo: train a simple association."""
        print("\nTraining demo association...")
        input_pat = self.mind.create_pattern('input', n_active=20, seed=42)
        output_pat = self.mind.create_pattern('output', n_active=20, seed=43)
        self.mind.train_association(input_pat, output_pat, n_steps=200)
        print("Training complete!")
        print("Type 'test' to inject the trained input pattern and watch the output.")

    def _train_text_association(self, input_word, output_word):
        """Train association between two text strings."""
        print(f"\nTraining: '{input_word}' -> '{output_word}'")

        # Create input pattern from input_word
        input_pat = self._text_to_pattern(input_word, 'input')

        # Create output pattern from output_word
        output_pat = self._text_to_pattern(output_word, 'output')

        # Store the word mapping for later reference
        if not hasattr(self.mind, 'word_associations'):
            self.mind.word_associations = {}
        self.mind.word_associations[input_word] = output_word

        # Train
        self.mind.train_association(input_pat, output_pat, n_steps=200)
        print(f"Learned! Type 'recall {input_word}' to test.")
        sys.stdout.flush()

    def _text_to_pattern(self, text, region):
        """Convert text to a pattern in the specified region."""
        if region == 'input':
            available = self.mind.input_region
        else:
            available = self.mind.output_region

        units = []
        phases = []

        for i, char in enumerate(text):
            base = ord(char)
            # Deterministic mapping based on character
            unit_idx = available[base % len(available)]
            units.append(unit_idx)
            # Coherent phase for the word
            phase = (hash(text) * 0.1 + base * 0.05) % (2 * np.pi)
            phases.append(phase)

            # Add neighbors for spread
            for offset in [-1, 1, 2]:
                neighbor = available[(base + offset) % len(available)]
                if neighbor not in units:
                    units.append(neighbor)
                    phases.append(phase + offset * 0.1)

        return {
            'units': np.array(units),
            'amplitudes': np.ones(len(units)) * 0.8,
            'phases': np.array(phases) % (2 * np.pi)
        }

    def _diagnose(self):
        """Diagnose network connectivity and learning state."""
        print("\n=== DIAGNOSTIC ===")

        # Check connectivity between input and output regions
        input_units = set(self.mind.input_region)
        output_units = set(self.mind.output_region)

        # Find paths from input to output via weight matrix
        rows, cols = self.mind.W_mag.nonzero()

        # Direct connections input -> output
        direct = 0
        for r, c in zip(rows, cols):
            if c in input_units and r in output_units:
                direct += 1
            if r in input_units and c in output_units:
                direct += 1

        print(f"Input region: units 0-{self.mind.n_input-1}")
        print(f"Output region: units {self.mind.n - self.mind.n_output}-{self.mind.n-1}")
        print(f"Direct input->output connections: {direct}")

        # Check weight statistics
        print(f"\nWeight magnitude: mean={np.mean(self.mind.W_mag.data):.4f}, max={np.max(self.mind.W_mag.data):.4f}")
        print(f"Weight phase: mean={np.mean(np.abs(self.mind.W_phase.data)):.4f}")

        # Check if training changed weights
        if hasattr(self.mind, 'word_associations') and self.mind.word_associations:
            print(f"\nTrained associations: {len(self.mind.word_associations)}")
            for inp, out in self.mind.word_associations.items():
                print(f"  '{inp}' -> '{out}'")

        print("=================")
        sys.stdout.flush()

    def _do_recall(self, word):
        """Perform recall and decode the output."""
        print(f"\n--- Recalling '{word}' ---")

        # Reset state for clean test
        self.mind.A = np.zeros(self.mind.n)
        self.mind.E = np.ones(self.mind.n)

        # Inject the input word
        self.mind.inject_text(word, strength=0.9)

        # Let it propagate
        print("Propagating activation...")
        for step in range(50):
            self.mind.step(dt=0.1)

        # Read output region
        output_amp = self.mind.A[self.mind.output_region]
        output_phases = self.mind.theta[self.mind.output_region]

        mean_output = np.mean(output_amp)
        max_output = np.max(output_amp)
        n_active = np.sum(output_amp > self.mind.theta_thresh)

        print(f"Output region: mean={mean_output:.4f}, max={max_output:.4f}, active={n_active}")

        # Compare against all trained output patterns
        if hasattr(self.mind, 'word_associations') and self.mind.word_associations:
            print("\nPattern matching:")
            best_match = None
            best_score = -1

            for input_word, output_word in self.mind.word_associations.items():
                # Get the expected output pattern
                expected_pat = self._text_to_pattern(output_word, 'output')

                # Calculate overlap score
                score = self._pattern_similarity(expected_pat, output_amp, output_phases)
                print(f"  '{output_word}': score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_match = output_word

            print(f"\n>>> DETECTED OUTPUT: '{best_match}' (score={best_score:.4f})")

            # Check if it matches expected
            if word in self.mind.word_associations:
                expected = self.mind.word_associations[word]
                if best_match == expected:
                    print(f">>> CORRECT! '{word}' -> '{expected}'")
                else:
                    print(f">>> Expected '{expected}', got '{best_match}'")

            self.last_input = f"[{word} -> {best_match}]"
        else:
            print("No trained patterns to compare against.")
            self.last_input = f"[RECALL {word}]"

        print("---")
        sys.stdout.flush()

    def _pattern_similarity(self, pattern, current_amp, current_phases):
        """Calculate similarity between expected pattern and current output state."""
        # Get the units in the pattern
        units = pattern['units']
        expected_phases = pattern['phases']

        # Translate unit indices to output region indices
        output_start = self.mind.output_region[0]
        local_indices = units - output_start
        valid_mask = (local_indices >= 0) & (local_indices < len(current_amp))
        local_indices = local_indices[valid_mask]
        expected_phases = expected_phases[valid_mask]

        if len(local_indices) == 0:
            return 0.0

        # Amplitude score: are the right units active?
        target_amp = np.zeros(len(current_amp))
        target_amp[local_indices] = 1.0

        # Normalize
        if np.max(current_amp) > 0:
            norm_current = current_amp / (np.max(current_amp) + 1e-8)
        else:
            norm_current = current_amp

        amp_score = np.sum(norm_current * target_amp) / (np.sum(target_amp) + 1e-8)

        # Phase score: are phases aligned?
        active_mask = current_amp[local_indices] > 0.1
        if np.sum(active_mask) > 0:
            phase_diff = current_phases[local_indices[active_mask]] - expected_phases[active_mask]
            phase_score = np.mean(np.cos(phase_diff))
            phase_score = (phase_score + 1) / 2  # Map from [-1,1] to [0,1]
        else:
            phase_score = 0.0

        # Combined score
        combined = 0.6 * amp_score + 0.4 * phase_score
        return combined

    def _process_input(self):
        try:
            while True:
                text = self.input_queue.get_nowait()
                text = text.strip()

                if text.lower() in ('quit', 'exit'):
                    plt.close(self.fig)
                    return
                elif text.lower() == 'reset':
                    self.mind.reset_state()
                    self.last_input = "[RESET]"
                elif text.lower() == 'burst':
                    self._on_burst(None)
                    self.last_input = "[BURST]"
                elif text.lower() == 'train':
                    self._on_train_demo(None)
                    self.last_input = "[TRAINING...]"
                elif text.lower() == 'test':
                    if self.mind.trained_patterns:
                        pat = self.mind.trained_patterns[-1]['input']
                        self.mind.inject_pattern(pat, strength=1.0)
                        self.last_input = "[TEST RECALL]"
                        print("Injected trained input pattern. Watch the output region!")
                    else:
                        self.last_input = "[No trained patterns]"
                        print("No trained patterns yet. Type 'train' first.")
                elif text.lower().startswith('learn '):
                    # Format: "learn hello -> world"
                    parts = text[6:].split('->')
                    if len(parts) == 2:
                        input_word = parts[0].strip()
                        output_word = parts[1].strip()
                        self._train_text_association(input_word, output_word)
                        self.last_input = f"[LEARNED {input_word}->{output_word}]"
                    else:
                        print("Usage: learn word1 -> word2")
                        self.last_input = "[LEARN ERROR]"
                elif text.lower().startswith('recall '):
                    # Inject just the input word of a learned association
                    word = text[7:].strip()
                    self._do_recall(word)
                elif text.lower() == 'diagnose':
                    self._diagnose()
                    self.last_input = "[DIAGNOSE]"
                else:
                    self.mind.inject_text(text, strength=0.7)
                    self.last_input = text
        except queue.Empty:
            pass

    def update(self, frame):
        self._process_input()

        # Run steps
        for _ in range(10):
            if np.random.random() < 0.005:
                pattern = self.mind.create_pattern('input', n_active=5)
                self.mind.inject_pattern(pattern, strength=0.3)
            self.mind.step(dt=0.1)

        # Update amplitude heatmap
        padded = np.zeros(self.mind.grid_size ** 2)
        padded[:self.mind.n] = self.mind.A
        self.amp_img.set_data(padded.reshape(self.mind.grid_size, self.mind.grid_size))

        # Update phase image
        self.phase_img.set_data(self._make_phase_image())

        # Update network scatter
        self.scatter.set_array(self.mind.theta)
        self.scatter.set_sizes(self.mind.A * 100 + 5)

        # Update time series
        times = self.mind.history['time'][-500:]

        resonance = self.mind.history['resonance'][-500:]
        coherence = self.mind.history['phase_coherence'][-500:]
        self.line_resonance.set_data(times, resonance)
        self.line_coherence.set_data(times, coherence)
        if len(times) > 0:
            self.ax_resonance.set_xlim(times[0], times[-1])
            self.ax_resonance.set_ylim(0, max(0.1, max(max(resonance), max(coherence)) * 1.1))

        input_amp = self.mind.history['input_amplitude'][-500:]
        output_amp = self.mind.history['output_amplitude'][-500:]
        self.line_input.set_data(times, input_amp)
        self.line_output.set_data(times, output_amp)
        if len(times) > 0:
            self.ax_io.set_xlim(times[0], times[-1])
            self.ax_io.set_ylim(0, max(0.1, max(max(input_amp), max(output_amp)) * 1.1))

        active = self.mind.history['active_count'][-500:]
        self.line_active.set_data(times, active)
        if len(times) > 0:
            self.ax_active.set_xlim(times[0], times[-1])
            self.ax_active.set_ylim(0, max(10, max(active) * 1.1))

        mean_amp = self.mind.history['mean_amplitude'][-500:]
        self.line_mean.set_data(times, mean_amp)
        if len(times) > 0:
            self.ax_mean.set_xlim(times[0], times[-1])
            self.ax_mean.set_ylim(0, max(0.1, max(mean_amp) * 1.1))

        # Stats
        stats = f"""Time: {self.mind.t:.1f}
Coherence: {self.mind._compute_phase_coherence():.3f}
Resonance: {self.mind._compute_resonance():.4f}
Active: {np.sum(self.mind.A > self.mind.theta_thresh)}
Input amp: {np.mean(self.mind.A[self.mind.input_region]):.3f}
Output amp: {np.mean(self.mind.A[self.mind.output_region]):.3f}
Trained: {len(self.mind.trained_patterns)} patterns

Last: "{self.last_input[:20]}..." """
        self.stats_text.set_text(stats)

        return [self.amp_img, self.phase_img, self.scatter]

    def animate(self, interval=50):
        # Start input listener
        self.input_listener = InputListener(self.input_queue)
        self.input_listener.start()

        self.anim = FuncAnimation(self.fig, self.update, frames=None,
                                  interval=interval, blit=False, cache_frame_data=False)
        plt.show()

        if self.input_listener:
            self.input_listener.stop()


class InputListener(threading.Thread):
    """Background thread for command-line input."""

    def __init__(self, input_queue):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.running = True

    def run(self):
        print("\n" + "=" * 60)
        print("PHASOR-SOC MIND - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  <text>         - Inject text as input pattern")
        print("  learn A -> B   - Train association (e.g., 'learn cat -> meow')")
        print("  recall A       - Test recall of learned word")
        print("  train          - Train a demo (random) association")
        print("  test           - Test recall of last trained pattern")
        print("  burst          - Random activation burst")
        print("  reset          - Reset network state")
        print("  quit           - Exit")
        print("=" * 60)
        sys.stdout.flush()

        while self.running:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if line:
                    line = line.strip()
                    if line:  # Only queue non-empty lines
                        self.input_queue.put(line)
            except EOFError:
                break
            except:
                break

    def stop(self):
        self.running = False


def run_learning_test():
    """Test the learning capabilities."""
    print("=" * 60)
    print("Phasor-SOC Learning Test")
    print("=" * 60)

    mind = PhasorSOCMind(n_units=1024, n_input=128, n_output=128)

    # Create patterns
    print("\nCreating patterns...")
    input_A = mind.create_pattern('input', n_active=25, seed=100)
    output_A = mind.create_pattern('output', n_active=25, seed=101)

    input_B = mind.create_pattern('input', n_active=25, seed=200)
    output_B = mind.create_pattern('output', n_active=25, seed=201)

    # Train associations
    print("\nTraining A -> A'...")
    mind.train_association(input_A, output_A, n_steps=300)

    print("Training B -> B'...")
    mind.train_association(input_B, output_B, n_steps=300)

    # Test recall
    print("\n" + "=" * 60)
    print("Testing recall...")
    print("=" * 60)

    # Test A
    print("\nInjecting pattern A (trained):")
    result_A = mind.test_recall(input_A, n_steps=100)
    match_A = mind.measure_pattern_match(output_A)
    print(f"  Output amplitude: {np.mean(result_A['output_amplitude'][-20:]):.4f}")
    print(f"  Peak resonance: {max(result_A['resonance']):.4f}")
    print(f"  Pattern match: {match_A}")

    # Test B
    print("\nInjecting pattern B (trained):")
    result_B = mind.test_recall(input_B, n_steps=100)
    match_B = mind.measure_pattern_match(output_B)
    print(f"  Output amplitude: {np.mean(result_B['output_amplitude'][-20:]):.4f}")
    print(f"  Peak resonance: {max(result_B['resonance']):.4f}")
    print(f"  Pattern match: {match_B}")

    # Test novel pattern (should show less resonance)
    print("\nInjecting novel pattern C (untrained):")
    input_C = mind.create_pattern('input', n_active=25, seed=999)
    result_C = mind.test_recall(input_C, n_steps=100)
    print(f"  Output amplitude: {np.mean(result_C['output_amplitude'][-20:]):.4f}")
    print(f"  Peak resonance: {max(result_C['resonance']):.4f}")

    print("\n" + "=" * 60)
    print("If learning works, trained patterns should show:")
    print("  - Higher output amplitude")
    print("  - Higher resonance")
    print("  - Better pattern match")
    print("than the novel pattern.")
    print("=" * 60)

    return mind


def run_interactive():
    """Run interactive visualization."""
    print("=" * 60)
    print("Phasor-SOC Mind - Interactive Mode")
    print("=" * 60)

    mind = PhasorSOCMind(n_units=1024, n_input=128, n_output=128)

    print("Warming up...")
    for _ in range(100):
        mind.step(dt=0.1)

    print("Starting visualization...")
    viz = InteractivePhasorSOC(mind)
    viz.animate(interval=50)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_learning_test()
    else:
        run_interactive()
