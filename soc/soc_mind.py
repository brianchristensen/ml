"""
Self-Organizing Critical Mind (SOC-Mind)
========================================

A mathematical model of mind based on self-organized criticality.

Core principle: Power laws emerge from the interaction of:
- Exponential growth: activation spreads proportionally to connection strength
- Exponential decay: activity-dependent depression suppresses high-activity units

The system self-tunes to the critical point between order and chaos through
local plasticity rules - no external optimizer needed.

Mathematical Model:
------------------
State variables:
  A_i(t) = activation of unit i (bounded [0, 1])
  W_ij(t) = connection strength from j to i
  E_i(t) = energy/resource available to unit i

Dynamics:
  1. Activation: dA/dt = input - decay + noise
     - Input from neighbors (thresholded for avalanche behavior)
     - Exponential decay toward resting state

  2. Plasticity: dW/dt = Hebbian growth - activity-dependent depression
     - Connections strengthen when pre and post are active
     - High activity suppresses connections (homeostasis)

  3. Resources: dE/dt = income - cost_of_firing + diffusion
     - Units have limited energy
     - Firing depletes energy, recovery over time

The threshold creates discrete avalanches.
The quadratic depression creates power-law distribution of avalanche sizes.

Interactive Mode:
-----------------
Run from command line and type text to inject activation patterns.
Use sliders to adjust parameters in real-time.

Commands:
  - Type any text and press Enter to inject activation
  - 'quit' or 'exit' to stop
  - 'reset' to reset the network state
  - 'burst' for a large random activation burst
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from collections import deque
import networkx as nx
from scipy import sparse
import threading
import queue
import sys


class SOCMind:
    """
    Self-Organizing Critical cognitive system.

    Implements coupled differential equations with:
    - Small-world network topology
    - Threshold-based activation (creates avalanches)
    - Hebbian plasticity with homeostatic depression
    - Resource/energy constraints
    """

    def __init__(self, n_units=1000, k_neighbors=10, rewire_prob=0.1, seed=42):
        """
        Initialize the SOC mind.

        Args:
            n_units: Number of units in the network
            k_neighbors: Each node connects to k nearest neighbors (small-world)
            rewire_prob: Probability of rewiring (Watts-Strogatz parameter)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n = n_units
        self.t = 0.0

        # === State Variables ===
        self.A = np.zeros(n_units)  # Activations [0, 1]
        self.E = np.ones(n_units)   # Energy/resources [0, 1]

        # === Network Topology (Small-World) ===
        G = nx.watts_strogatz_graph(n_units, k_neighbors, rewire_prob, seed=seed)
        self.adj = nx.to_scipy_sparse_array(G, format='csr')

        # Connection weights (initialized near critical point)
        # Start with uniform weights, will self-tune
        self.W = sparse.lil_matrix((n_units, n_units))
        for i, j in G.edges():
            w = np.random.uniform(0.1, 0.3)
            self.W[i, j] = w
            self.W[j, i] = w  # Symmetric for now
        self.W = self.W.tocsr()

        # Store positions for visualization (circular layout)
        self.positions = np.array([
            [np.cos(2 * np.pi * i / n_units), np.sin(2 * np.pi * i / n_units)]
            for i in range(n_units)
        ])

        # === Parameters (can be adjusted via sliders) ===
        # Activation dynamics
        self.theta = 0.3          # Firing threshold
        self.decay = 0.1          # Activation decay rate
        self.noise_std = 0.01     # Spontaneous noise

        # Plasticity dynamics
        self.eta = 0.001          # Hebbian learning rate
        self.alpha = 0.01         # Depression coefficient
        self.tau_w = 100.0        # Weight recovery timescale
        self.w_max = 1.0          # Maximum weight

        # Resource dynamics
        self.r_income = 0.05      # Energy recovery rate
        self.r_cost = 0.1         # Cost of firing
        self.r_diffusion = 0.01   # Energy diffusion rate

        # === Event Queue ===
        self.event_queue = []

        # === History for Analysis ===
        self.avalanche_sizes = []
        self.current_avalanche = 0
        self.in_avalanche = False
        self.history = {
            'time': [],
            'mean_activation': [],
            'active_count': [],
            'mean_weight': [],
            'total_energy': [],
            'activity_rate': []  # Rate of change of activity
        }

    def step(self, dt=0.1):
        """
        Advance the system by one timestep.

        Uses Euler integration for the coupled differential equations.
        """
        self.t += dt

        # === Process External Events ===
        for event in self.event_queue:
            self._inject_activation(event)
        self.event_queue.clear()

        # === Compute Derivatives ===
        dA = self._activation_dynamics(dt)
        dW = self._plasticity_dynamics(dt)
        dE = self._resource_dynamics(dt)

        # === Update State ===
        self.A = np.clip(self.A + dt * dA, 0, 1)

        # Update weights (sparse update)
        self.W = self.W + dt * dW
        self.W.data = np.clip(self.W.data, 0, self.w_max)

        self.E = np.clip(self.E + dt * dE, 0, 1)

        # === Avalanche Detection ===
        self._detect_avalanche()

        # === Record History ===
        self._record_history()

    def _activation_dynamics(self, dt):
        """
        Compute activation change: dA/dt = input - decay + noise

        Key features:
        - Thresholded input creates discrete avalanches
        - Input weighted by connection strength AND source energy
        """
        # Which units are firing (above threshold)?
        firing = (self.A > self.theta).astype(float)

        # Input from neighbors: sum of W_ij * firing_j * E_j
        # Energy-weighted: depleted units contribute less
        weighted_firing = firing * self.E
        input_signal = self.W.dot(weighted_firing)

        # Normalize by number of inputs to prevent explosion
        in_degree = np.array(self.W.sum(axis=1)).flatten()
        in_degree = np.maximum(in_degree, 1)  # Avoid division by zero
        input_signal = input_signal / in_degree

        # Decay toward resting state (0)
        decay = -self.decay * self.A

        # Spontaneous noise (keeps system from dying)
        noise = np.random.normal(0, self.noise_std, self.n)

        # Combined dynamics
        dA = input_signal + decay + noise

        return dA

    def _plasticity_dynamics(self, dt):
        """
        Compute weight change: dW/dt = Hebbian - depression + recovery

        This is where the power law emerges:
        - Hebbian: W grows when pre and post are active
        - Depression: W shrinks proportionally to pre activity SQUARED
        - This creates the exponential growth vs exponential decay balance
        """
        # Get activations as column and row vectors for outer product
        A_post = self.A.reshape(-1, 1)  # [n, 1]
        A_pre = self.A.reshape(1, -1)   # [1, n]

        # Hebbian term: eta * A_post * A_pre (only where connections exist)
        # Depression term: alpha * W * A_pre^2
        # Recovery term: 1/tau_w (slow drift back to baseline)

        # For efficiency, work with sparse structure
        rows, cols = self.W.nonzero()

        hebbian = self.eta * self.A[rows] * self.A[cols]
        depression = self.alpha * self.W.data * (self.A[cols] ** 2)
        recovery = (0.2 - self.W.data) / self.tau_w  # Drift toward 0.2

        dW_data = hebbian - depression + recovery

        # Create sparse matrix with same structure
        dW = sparse.csr_matrix((dW_data, (rows, cols)), shape=self.W.shape)

        return dW

    def _resource_dynamics(self, dt):
        """
        Compute energy change: dE/dt = income - cost + diffusion

        Units have limited resources:
        - Firing costs energy
        - Energy recovers over time
        - Energy diffuses between neighbors
        """
        # Income: constant recovery
        income = self.r_income * (1 - self.E)  # Faster recovery when depleted

        # Cost: proportional to activation
        cost = self.r_cost * self.A

        # Diffusion: energy flows from high to low
        # Laplacian diffusion: dE_i = sum_j (E_j - E_i)
        neighbor_energy = self.adj.dot(self.E)
        degree = np.array(self.adj.sum(axis=1)).flatten()
        degree = np.maximum(degree, 1)
        diffusion = self.r_diffusion * (neighbor_energy / degree - self.E)

        dE = income - cost + diffusion

        return dE

    def _inject_activation(self, event):
        """Inject external activation into the system."""
        unit_idx = event.get('unit', np.random.randint(self.n))
        strength = event.get('strength', 0.5)

        # Can inject into multiple units
        if isinstance(unit_idx, (list, np.ndarray)):
            for idx in unit_idx:
                self.A[idx] = min(1.0, self.A[idx] + strength)
        else:
            self.A[unit_idx] = min(1.0, self.A[unit_idx] + strength)

    def _detect_avalanche(self):
        """
        Detect and measure avalanches.

        An avalanche is a cascade of activations above threshold.
        We track the total number of firing events in each avalanche.
        """
        n_active = np.sum(self.A > self.theta)

        if n_active > 0:
            if not self.in_avalanche:
                # Start of new avalanche
                self.in_avalanche = True
                self.current_avalanche = n_active
            else:
                # Continue avalanche
                self.current_avalanche += n_active
        else:
            if self.in_avalanche:
                # End of avalanche - record size
                self.avalanche_sizes.append(self.current_avalanche)
                self.in_avalanche = False
                self.current_avalanche = 0

    def _record_history(self):
        """Record system state for analysis."""
        self.history['time'].append(self.t)
        self.history['mean_activation'].append(np.mean(self.A))
        active_count = np.sum(self.A > self.theta)
        self.history['active_count'].append(active_count)
        self.history['mean_weight'].append(np.mean(self.W.data))
        self.history['total_energy'].append(np.sum(self.E))

        # Compute activity rate (change from previous)
        if len(self.history['active_count']) > 1:
            prev_count = self.history['active_count'][-2]
            rate = active_count - prev_count
        else:
            rate = 0
        self.history['activity_rate'].append(rate)

    def inject_event(self, unit=None, strength=0.5):
        """Queue an external event for injection."""
        if unit is None:
            unit = np.random.randint(self.n)
        self.event_queue.append({'unit': unit, 'strength': strength})

    def inject_random_events(self, n_events=5, strength=0.3):
        """Inject multiple random events."""
        units = np.random.choice(self.n, n_events, replace=False)
        for u in units:
            self.inject_event(u, strength)

    def inject_text(self, text, strength=0.6):
        """
        Convert text input to activation pattern and inject it.

        Uses a hash-based mapping to select units based on text content.
        This creates reproducible but distributed activation patterns.
        """
        if not text:
            return

        # Convert text to unit indices using character-based hashing
        units = set()
        for i, char in enumerate(text):
            # Create multiple activations per character for richer patterns
            base = ord(char)
            # Primary unit based on character
            units.add(base % self.n)
            # Secondary units based on position and character combination
            units.add((base * (i + 1)) % self.n)
            units.add((base + i * 7) % self.n)

        # Also add some neighboring units for spatial coherence
        expanded_units = set()
        for u in units:
            expanded_units.add(u)
            # Add a few neighbors (wrapping around)
            expanded_units.add((u + 1) % self.n)
            expanded_units.add((u - 1) % self.n)

        # Inject with varying strengths based on position
        for i, u in enumerate(expanded_units):
            s = strength * (0.5 + 0.5 * np.random.random())
            self.inject_event(u, s)

    def reset_state(self):
        """Reset activations and energy to initial state."""
        self.A = np.zeros(self.n)
        self.E = np.ones(self.n)

    def run(self, n_steps=1000, dt=0.1, inject_rate=0.01):
        """
        Run the simulation for n_steps.

        Args:
            n_steps: Number of timesteps
            dt: Time step size
            inject_rate: Probability of injecting a random event each step
        """
        for _ in range(n_steps):
            # Random external stimulation
            if np.random.random() < inject_rate:
                self.inject_random_events(n_events=3, strength=0.4)

            self.step(dt)

    def get_avalanche_distribution(self):
        """
        Compute avalanche size distribution.

        Returns (sizes, counts) for plotting.
        Power law should appear as straight line on log-log plot.
        """
        if len(self.avalanche_sizes) == 0:
            return np.array([]), np.array([])

        sizes = np.array(self.avalanche_sizes)
        unique_sizes, counts = np.unique(sizes, return_counts=True)

        # Convert to probability
        probs = counts / len(sizes)

        return unique_sizes, probs

    def compute_susceptibility(self):
        """
        Compute susceptibility (variance of avalanche sizes).

        Peaks at critical point.
        """
        if len(self.avalanche_sizes) < 10:
            return 0.0

        sizes = np.array(self.avalanche_sizes)
        return np.var(sizes)


class InputListener(threading.Thread):
    """Background thread that listens for command-line input."""

    def __init__(self, input_queue):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.running = True

    def run(self):
        """Continuously read input and put it in the queue."""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE - Type and press Enter to inject activation")
        print("=" * 60)
        print("Commands:")
        print("  <text>  - Type anything to inject activation pattern")
        print("  burst   - Large random activation burst")
        print("  reset   - Reset network state")
        print("  quit    - Exit the program")
        print("=" * 60 + "\n")

        while self.running:
            try:
                line = input("> ")
                if line:
                    self.input_queue.put(line)
            except EOFError:
                break
            except Exception:
                break

    def stop(self):
        self.running = False


class InteractiveSOCVisualizer:
    """Real-time visualization of SOC dynamics with parameter sliders and input."""

    def __init__(self, mind):
        self.mind = mind
        self.input_queue = queue.Queue()
        self.input_listener = None
        self.last_input = ""

        # Create figure with custom layout for sliders
        self.fig = plt.figure(figsize=(18, 12))

        # Use GridSpec for flexible layout
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.4],
                                   hspace=0.3, wspace=0.3,
                                   left=0.05, right=0.95, top=0.95, bottom=0.05)

        # === Main Visualization Panels ===

        # Network activation (top left)
        self.ax_network = self.fig.add_subplot(gs[0, 0])
        self.ax_network.set_title('Network Activation')
        self.ax_network.set_xlim(-1.2, 1.2)
        self.ax_network.set_ylim(-1.2, 1.2)
        self.ax_network.set_aspect('equal')
        self.scatter = self.ax_network.scatter(
            mind.positions[:, 0], mind.positions[:, 1],
            c=mind.A, cmap='hot', s=10, vmin=0, vmax=1
        )

        # Activation time series (top middle)
        self.ax_activation = self.fig.add_subplot(gs[0, 1])
        self.ax_activation.set_title('Mean Activation Over Time')
        self.ax_activation.set_xlabel('Time')
        self.ax_activation.set_ylabel('Mean Activation')
        self.line_activation, = self.ax_activation.plot([], [], 'b-', lw=1)

        # Active count time series (top right)
        self.ax_active = self.fig.add_subplot(gs[0, 2])
        self.ax_active.set_title('Active Units Over Time')
        self.ax_active.set_xlabel('Time')
        self.ax_active.set_ylabel('# Active (A > theta)')
        self.line_active, = self.ax_active.plot([], [], 'r-', lw=1)

        # Input display (top far right)
        self.ax_input = self.fig.add_subplot(gs[0, 3])
        self.ax_input.set_title('Last Input')
        self.ax_input.axis('off')
        self.input_text = self.ax_input.text(0.5, 0.5, '', ha='center', va='center',
                                              fontsize=14, wrap=True,
                                              transform=self.ax_input.transAxes)

        # Activation heatmap (middle left) - shows fractal spatial patterns
        self.ax_heatmap = self.fig.add_subplot(gs[1, 0])
        self.ax_heatmap.set_title('Activation Pattern')
        self.ax_heatmap.set_xlabel('x')
        self.ax_heatmap.set_ylabel('y')
        # Reshape activations into 2D grid for visualization
        self.grid_size = int(np.ceil(np.sqrt(mind.n)))
        self.padded_size = self.grid_size * self.grid_size
        padded_A = np.zeros(self.padded_size)
        padded_A[:mind.n] = mind.A
        self.heatmap_img = self.ax_heatmap.imshow(
            padded_A.reshape(self.grid_size, self.grid_size),
            cmap='hot', vmin=0, vmax=1, interpolation='nearest'
        )
        self.fig.colorbar(self.heatmap_img, ax=self.ax_heatmap, fraction=0.046)

        # Weight distribution (middle center-left)
        self.ax_weights = self.fig.add_subplot(gs[1, 1])
        self.ax_weights.set_title('Weight Distribution')
        self.ax_weights.set_xlabel('Weight')
        self.ax_weights.set_ylabel('Count')

        # Energy distribution (middle center-right)
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.ax_energy.set_title('Energy Distribution')
        self.ax_energy.set_xlabel('Energy')
        self.ax_energy.set_ylabel('Count')

        # Stats panel (middle right)
        self.ax_stats = self.fig.add_subplot(gs[1, 3])
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.9, '', ha='left', va='top',
                                              fontsize=10, family='monospace',
                                              transform=self.ax_stats.transAxes)

        # === Sliders (bottom row) ===
        self.sliders = {}
        slider_configs = [
            ('theta', 'Threshold', 0.0, 1.0, mind.theta),
            ('decay', 'Decay', 0.01, 0.5, mind.decay),
            ('noise_std', 'Noise', 0.0, 0.1, mind.noise_std),
            ('eta', 'Hebbian LR', 0.0, 0.01, mind.eta),
            ('alpha', 'Depression', 0.0, 0.1, mind.alpha),
            ('tau_w', 'W Recovery', 10.0, 500.0, mind.tau_w),
            ('r_income', 'E Income', 0.0, 0.2, mind.r_income),
            ('r_cost', 'E Cost', 0.0, 0.3, mind.r_cost),
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

        # Reset button
        ax_reset = self.fig.add_axes([0.85, 0.01, 0.1, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        # Burst button
        ax_burst = self.fig.add_axes([0.73, 0.01, 0.1, 0.03])
        self.btn_burst = Button(ax_burst, 'Burst')
        self.btn_burst.on_clicked(self._on_burst)

    def _make_slider_callback(self, param):
        """Create a callback function for a parameter slider."""
        def callback(val):
            setattr(self.mind, param, val)
        return callback

    def _on_reset(self, event):
        """Reset button callback."""
        self.mind.reset_state()

    def _on_burst(self, event):
        """Burst button callback."""
        self.mind.inject_random_events(n_events=50, strength=0.8)

    def _process_input(self):
        """Process any pending input from the queue."""
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
                    self.mind.inject_random_events(n_events=50, strength=0.8)
                    self.last_input = "[BURST]"
                else:
                    self.mind.inject_text(text, strength=0.7)
                    self.last_input = text

        except queue.Empty:
            pass

    def update(self, frame):
        """Update visualization for animation."""
        # Process any pending input
        self._process_input()

        # Run several steps per frame
        for _ in range(10):
            if np.random.random() < 0.01:  # Lower spontaneous rate
                self.mind.inject_random_events(n_events=2, strength=0.3)
            self.mind.step(dt=0.1)

        # Update network scatter
        self.scatter.set_array(self.mind.A)

        # Update input display
        display_text = self.last_input if self.last_input else "(waiting for input)"
        if len(display_text) > 50:
            display_text = display_text[:47] + "..."
        self.input_text.set_text(f'"{display_text}"')

        # Update activation time series
        times = self.mind.history['time'][-500:]
        activations = self.mind.history['mean_activation'][-500:]
        self.line_activation.set_data(times, activations)
        if len(times) > 0:
            self.ax_activation.set_xlim(times[0], times[-1])
            self.ax_activation.set_ylim(0, max(0.1, max(activations) * 1.1))

        # Update active count time series
        active_counts = self.mind.history['active_count'][-500:]
        self.line_active.set_data(times, active_counts)
        if len(times) > 0:
            self.ax_active.set_xlim(times[0], times[-1])
            self.ax_active.set_ylim(0, max(10, max(active_counts) * 1.1))

        # Update activation heatmap
        padded_A = np.zeros(self.padded_size)
        padded_A[:self.mind.n] = self.mind.A
        self.heatmap_img.set_data(padded_A.reshape(self.grid_size, self.grid_size))

        # Update weight histogram
        self.ax_weights.clear()
        self.ax_weights.hist(self.mind.W.data, bins=50, color='green', alpha=0.7)
        self.ax_weights.set_title(f'Weight Distribution (mean={np.mean(self.mind.W.data):.3f})')
        self.ax_weights.set_xlabel('Weight')
        self.ax_weights.set_ylabel('Count')

        # Update energy histogram
        self.ax_energy.clear()
        self.ax_energy.hist(self.mind.E, bins=50, color='purple', alpha=0.7)
        self.ax_energy.set_title(f'Energy Distribution (mean={np.mean(self.mind.E):.3f})')
        self.ax_energy.set_xlabel('Energy')
        self.ax_energy.set_ylabel('Count')

        # Update statistics
        stats = f"""Time: {self.mind.t:.1f}
Avalanches: {len(self.mind.avalanche_sizes)}
Mean Act: {np.mean(self.mind.A):.4f}
Active: {np.sum(self.mind.A > self.mind.theta)}
Mean W: {np.mean(self.mind.W.data):.4f}
Mean E: {np.mean(self.mind.E):.4f}
Suscept: {self.mind.compute_susceptibility():.2f}"""
        self.stats_text.set_text(stats)

        return [self.scatter, self.line_activation, self.line_active, self.heatmap_img]

    def animate(self, interval=50):
        """Run the animation with input listening."""
        # Start input listener thread
        self.input_listener = InputListener(self.input_queue)
        self.input_listener.start()

        # Run animation indefinitely
        self.anim = FuncAnimation(self.fig, self.update, frames=None,
                                  interval=interval, blit=False, cache_frame_data=False)
        plt.show()

        # Cleanup
        if self.input_listener:
            self.input_listener.stop()


def run_interactive():
    """Run the SOC model in interactive mode."""
    print("=" * 70)
    print("Self-Organizing Critical Mind (SOC-Mind) - Interactive Mode")
    print("=" * 70)
    print()

    # Create the mind
    print("Creating SOC mind with 1000 units, small-world topology...")
    mind = SOCMind(n_units=1000, k_neighbors=10, rewire_prob=0.1)

    # Brief warmup
    print("Warming up simulation (500 steps)...")
    mind.run(n_steps=500, dt=0.1, inject_rate=0.02)

    print()
    print("Starting interactive visualization...")
    print("The visualization window will open. Type in this terminal to interact.")
    print()

    viz = InteractiveSOCVisualizer(mind)
    viz.animate(interval=50)


def run_experiment():
    """Run a quick experiment to test the SOC model (non-interactive)."""
    print("=" * 70)
    print("Self-Organizing Critical Mind (SOC-Mind)")
    print("=" * 70)
    print()

    # Create the mind
    print("Creating SOC mind with 1000 units, small-world topology...")
    mind = SOCMind(n_units=1000, k_neighbors=10, rewire_prob=0.1)

    # Run for a while to let it self-organize
    print("Running simulation for 2000 steps...")
    mind.run(n_steps=2000, dt=0.1, inject_rate=0.02)

    # Print statistics
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total time: {mind.t:.1f}")
    print(f"Number of avalanches: {len(mind.avalanche_sizes)}")

    if len(mind.avalanche_sizes) > 0:
        sizes = np.array(mind.avalanche_sizes)
        print(f"Avalanche sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
        print(f"Susceptibility: {mind.compute_susceptibility():.2f}")

    print(f"Mean weight: {np.mean(mind.W.data):.4f}")
    print(f"Mean energy: {np.mean(mind.E):.4f}")
    print()

    # Check for power law
    sizes, probs = mind.get_avalanche_distribution()
    if len(sizes) > 5:
        log_sizes = np.log10(sizes[sizes > 0])
        log_probs = np.log10(probs[sizes > 0])
        if len(log_sizes) > 2:
            slope, _ = np.polyfit(log_sizes, log_probs, 1)
            print(f"Power law exponent (slope): {slope:.2f}")
            print("(Should be around -1.5 to -2.5 for SOC)")

    return mind


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Non-interactive batch mode
        mind = run_experiment()
    else:
        # Interactive mode (default)
        run_interactive()
