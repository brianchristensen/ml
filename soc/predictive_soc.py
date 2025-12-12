"""
Predictive SOC Mind
====================

A self-organizing critical system that learns by prediction.

Core Principles:
1. SELF-ORGANIZED CRITICALITY - System tunes itself to the edge of chaos
2. PREDICTION IS COMPUTATION - Each unit predicts its next state
3. LEARNING = MINIMIZING PREDICTION ERROR - Weights adjust to improve predictions
4. EMERGENT STRUCTURE - No hard-coded regions, structure emerges from input statistics

The mind is a prediction engine operating at criticality.
Representations emerge from what's useful for prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy import sparse
import threading
import queue
import sys


class PredictiveSOCMind:
    """
    A self-organizing critical system that learns through prediction.

    Each unit:
    - Has activation A and phase θ
    - Predicts its own next state
    - Learns by minimizing prediction error

    The network:
    - Self-tunes to criticality via homeostatic plasticity
    - Forms representations that are useful for prediction
    - Has no hard-coded structure - everything emerges
    """

    def __init__(self, n_units=1024, connectivity=0.1, seed=42):
        np.random.seed(seed)
        self.n = n_units
        self.t = 0.0

        # === State Variables ===
        self.A = np.zeros(n_units)           # Activation [0, 1]
        self.theta = np.random.uniform(0, 2*np.pi, n_units)  # Phase
        self.E = np.ones(n_units)            # Energy

        # Prediction state - each unit's prediction of its next activation
        self.A_pred = np.zeros(n_units)
        self.prediction_error = np.zeros(n_units)

        # === Connectivity (sparse random) ===
        # Start with random sparse connectivity - structure will emerge from learning
        n_connections = int(n_units * n_units * connectivity)
        rows = np.random.randint(0, n_units, n_connections)
        cols = np.random.randint(0, n_units, n_connections)

        # Remove self-connections and duplicates
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        # Remove duplicates by converting to set of tuples
        edges = list(set(zip(rows, cols)))
        rows = np.array([e[0] for e in edges])
        cols = np.array([e[1] for e in edges])

        self._conn_rows = rows
        self._conn_cols = cols
        self._n_connections = len(rows)

        # TWO WEIGHT ARRAYS - stored as flat arrays, same indexing as rows/cols
        # W_dyn: Controls actual dynamics (how activation spreads)
        # W_pred: Used for prediction (the internal model)

        self.W_dyn_data = np.random.randn(self._n_connections) * 0.1
        self.W_pred_data = np.random.randn(self._n_connections) * 0.1
        self.W_phase_data = np.random.uniform(-np.pi/4, np.pi/4, self._n_connections)

        # Track prediction confidence for consolidation
        self.prediction_confidence = np.zeros(n_units)  # Running average of prediction accuracy

        # === SOC Parameters ===
        self.threshold = 0.3       # Firing threshold
        self.decay = 0.1           # Activation decay
        self.noise = 0.01          # Spontaneous noise

        # === Learning Parameters ===
        self.lr_pred = 0.01        # Prediction weight learning rate (can be fast now!)
        self.lr_consolidate = 0.001  # Rate of transferring W_pred -> W_dyn
        self.lr_phase = 0.005      # Phase learning rate
        self.confidence_decay = 0.95  # How fast confidence decays

        # === Homeostatic Parameters (for SOC) ===
        self.target_activity = 0.05   # Target fraction of active units
        self.homeostatic_rate = 0.001 # How fast threshold adapts

        # === History ===
        self.history = {
            'time': [],
            'mean_activation': [],
            'active_fraction': [],
            'mean_pred_error': [],
            'threshold': [],
            'phase_coherence': [],
            'mean_confidence': [],
            'w_pred_mean': [],
            'w_dyn_mean': []
        }

        # For visualization
        self.grid_size = int(np.ceil(np.sqrt(n_units)))

        # Input queue for external stimuli
        self.input_queue = []

        # === Pattern Memory (Hebbian traces) ===
        # When we "learn" a pattern, we strengthen connections between co-active units
        # This creates attractors that familiar patterns fall into more easily
        self.pattern_traces = {}  # name -> activation pattern
        self.hebbian_trace = np.zeros(self._n_connections)  # Accumulated co-activation

    def step(self, dt=0.1):
        """One timestep of dynamics."""
        self.t += dt

        # 1. Process external input
        for inp in self.input_queue:
            self._inject(inp)
        self.input_queue.clear()

        # 2. Each unit predicts its next state
        self._make_predictions()

        # 3. Compute actual next state (SOC dynamics)
        A_next = self._soc_dynamics(dt)

        # 4. Compute prediction error
        self.prediction_error = A_next - self.A_pred

        # 5. Learn from prediction error
        self._learn_from_error(dt)

        # 6. Homeostatic threshold adjustment (maintains criticality)
        self._homeostatic_adjustment()

        # 7. Update state
        self.A = A_next
        self._update_phases(dt)
        self._update_energy(dt)

        # Record history
        self._record()

    def _make_predictions(self):
        """Each unit predicts its next activation using W_pred (the internal model)."""
        rows, cols = self._conn_rows, self._conn_cols

        # Phase-modulated input using PREDICTION weights
        phase_diff = self.theta[rows] - self.theta[cols] - self.W_phase_data
        phase_mod = np.cos(phase_diff)

        weighted_input = self.W_pred_data * self.A[cols] * phase_mod

        # Sum for each unit
        pred = np.zeros(self.n)
        np.add.at(pred, rows, weighted_input)

        # Normalize by connection count
        in_degree = np.zeros(self.n)
        np.add.at(in_degree, rows, np.abs(self.W_pred_data))
        in_degree = np.maximum(in_degree, 1)
        pred = pred / in_degree

        # Prediction includes decay term (model of dynamics)
        pred = pred - self.decay * self.A

        # Clip to valid range
        self.A_pred = np.clip(self.A + 0.1 * pred, 0, 1)  # Predict next state

    def _soc_dynamics(self, dt):
        """
        SOC dynamics using W_dyn (the actual physics).
        This is where avalanches emerge.
        W_dyn is separate from W_pred - dynamics are stable while predictions learn.
        """
        # Which units are firing?
        firing = (self.A > self.threshold).astype(float)

        # Input from firing neighbors using DYNAMICS weights
        rows, cols = self._conn_rows, self._conn_cols

        # Only positive weights contribute to excitation
        excitatory = np.maximum(self.W_dyn_data, 0)
        inhibitory = np.minimum(self.W_dyn_data, 0)

        # Excitatory input (energy-weighted)
        exc_input = excitatory * firing[cols] * self.E[cols]
        exc_sum = np.zeros(self.n)
        np.add.at(exc_sum, rows, exc_input)

        # Inhibitory input
        inh_input = inhibitory * firing[cols]
        inh_sum = np.zeros(self.n)
        np.add.at(inh_sum, rows, inh_input)

        # Normalize by in-degree
        in_degree = np.zeros(self.n)
        np.add.at(in_degree, rows, np.abs(self.W_dyn_data))
        in_degree = np.maximum(in_degree, 1)

        total_input = (exc_sum + inh_sum) / in_degree

        # dA/dt = input - decay + noise
        dA = total_input - self.decay * self.A + np.random.normal(0, self.noise, self.n)

        A_next = np.clip(self.A + dt * dA, 0, 1)
        return A_next

    def _learn_from_error(self, dt):
        """
        Two-stage learning:
        1. W_pred learns quickly from prediction errors (the model improves)
        2. W_dyn slowly consolidates from W_pred when predictions are good

        This separates fast learning (model) from slow consolidation (dynamics).

        === CREDIT ASSIGNMENT ===

        The learning rule is the Delta Rule (Widrow-Hoff):

            dW_ij = learning_rate * error_j * activation_i

        Where:
            - i is the PRE-synaptic (source) unit
            - j is the POST-synaptic (target) unit
            - error_j = actual_j - predicted_j (how wrong was j's prediction?)
            - activation_i = how active was unit i?

        Credit assignment: "Blame the active inputs"
        - If unit j made a prediction error, ALL active inputs get blamed/credited
        - The amount of blame is proportional to how active each input was
        - This is LOCAL - each synapse only sees its source and target

        Example: If unit Z predicts too low (error > 0) and receives input from A, B, C:
            - If A=0.8, B=0.3, C=0.0:
            - W(A→Z) increases by 0.8 * error (A gets most blame)
            - W(B→Z) increases by 0.3 * error (B gets some blame)
            - W(C→Z) unchanged (C was silent)

        Limitations:
        - No temporal credit: If the error was caused by a chain A→B→C→Z,
          only C gets blamed, not A or B (no backprop through time)
        - Shared blame: Active inputs share credit regardless of actual causation
        - Local only: No gradient computation through the dynamics

        Why it might still work:
        - Over many trials, consistent correlations emerge
        - Criticality provides rich dynamics that help learning
        - Two-stage design: W_pred explores while W_dyn stays stable
        """
        rows, cols = self._conn_rows, self._conn_cols

        # === Stage 1: Update W_pred (fast learning) ===
        error_post = self.prediction_error[rows]  # Signed error
        A_pre = self.A[cols]

        # Simple delta rule for prediction weights
        # If error > 0 (actual > predicted), increase weights from active inputs
        # If error < 0 (actual < predicted), decrease weights
        dW_pred = self.lr_pred * error_post * A_pre

        # Small weight decay for regularization
        dW_pred -= 0.001 * self.W_pred_data

        self.W_pred_data += dt * dW_pred
        self.W_pred_data = np.clip(self.W_pred_data, -1.0, 1.0)

        # === Update prediction confidence ===
        # Confidence is high when prediction error is low
        abs_error = np.abs(self.prediction_error)
        accuracy = 1.0 - np.clip(abs_error * 2, 0, 1)  # Map error to [0,1] accuracy
        self.prediction_confidence = (
            self.confidence_decay * self.prediction_confidence +
            (1 - self.confidence_decay) * accuracy
        )

        # === Stage 2: Consolidate W_pred -> W_dyn (slow, confidence-gated) ===
        # Only consolidate for units with high confidence
        confidence_threshold = 0.6
        confident_units = self.prediction_confidence > confidence_threshold

        # For connections where post-synaptic unit is confident, move W_dyn toward W_pred
        confident_connections = confident_units[rows]

        weight_diff = self.W_pred_data - self.W_dyn_data
        dW_dyn = self.lr_consolidate * confident_connections * weight_diff

        self.W_dyn_data += dt * dW_dyn
        self.W_dyn_data = np.clip(self.W_dyn_data, -1.0, 1.0)

        # === Phase learning ===
        # Adjust phases when units are co-active and confident
        A_post = self.A[rows]
        co_active = A_pre * A_post
        phase_diff = self.theta[rows] - self.theta[cols] - self.W_phase_data

        dW_phase = self.lr_phase * co_active * confident_connections * np.sin(phase_diff)
        self.W_phase_data += dt * dW_phase
        self.W_phase_data = np.clip(self.W_phase_data, -np.pi, np.pi)

    def _homeostatic_adjustment(self):
        """
        Adjust threshold to maintain target activity level.
        This is what keeps the system at criticality.

        Too much activity → raise threshold
        Too little activity → lower threshold
        """
        active_fraction = np.mean(self.A > self.threshold)
        error = active_fraction - self.target_activity

        # Adjust threshold
        self.threshold += self.homeostatic_rate * error
        self.threshold = np.clip(self.threshold, 0.1, 0.9)

    def _update_phases(self, dt):
        """Phase dynamics: Kuramoto-like coupling using W_dyn."""
        rows, cols = self._conn_rows, self._conn_cols

        # Phases couple toward alignment when both units active
        coupling = np.abs(self.W_dyn_data) * self.A[rows] * self.A[cols]
        phase_pull = coupling * np.sin(self.theta[cols] + self.W_phase_data - self.theta[rows])

        dtheta = np.zeros(self.n)
        np.add.at(dtheta, rows, phase_pull)

        # Normalize
        count = np.zeros(self.n)
        np.add.at(count, rows, np.abs(self.W_dyn_data))
        count = np.maximum(count, 1)
        dtheta = 0.5 * dtheta / count

        # Add noise
        dtheta += np.random.normal(0, 0.01, self.n)

        self.theta = (self.theta + dt * dtheta) % (2 * np.pi)

    def _update_energy(self, dt):
        """Energy dynamics: firing costs energy, recovery over time."""
        # Recovery
        dE = 0.05 * (1 - self.E)
        # Cost of activation
        dE -= 0.1 * self.A

        self.E = np.clip(self.E + dt * dE, 0, 1)

    def _inject(self, inp):
        """Inject external input."""
        units = inp.get('units', [])
        strength = inp.get('strength', 0.5)
        phases = inp.get('phases', None)

        for i, u in enumerate(units):
            self.A[u] = min(1.0, self.A[u] + strength)
            if phases is not None:
                self.theta[u] = phases[i]

    def _record(self):
        """Record history."""
        self.history['time'].append(self.t)
        self.history['mean_activation'].append(np.mean(self.A))
        self.history['active_fraction'].append(np.mean(self.A > self.threshold))
        self.history['mean_pred_error'].append(np.mean(np.abs(self.prediction_error)))
        self.history['threshold'].append(self.threshold)
        self.history['phase_coherence'].append(self._phase_coherence())
        self.history['mean_confidence'].append(np.mean(self.prediction_confidence))
        self.history['w_pred_mean'].append(np.mean(np.abs(self.W_pred_data)))
        self.history['w_dyn_mean'].append(np.mean(np.abs(self.W_dyn_data)))

    def _phase_coherence(self):
        """Kuramoto order parameter."""
        weighted = self.A * np.exp(1j * self.theta)
        if np.sum(self.A) > 0:
            return np.abs(np.sum(weighted)) / (np.sum(self.A) + 1e-8)
        return 0.0

    # === High-level API ===

    def inject_pattern(self, pattern_id, strength=0.7):
        """
        Inject a pattern. Pattern is determined by ID (reproducible).
        No designated regions - patterns can be anywhere.
        """
        np.random.seed(pattern_id)
        n_active = np.random.randint(20, 50)
        units = np.random.choice(self.n, n_active, replace=False)

        # Coherent phase for this pattern
        base_phase = (pattern_id * 1.234) % (2 * np.pi)
        phases = base_phase + np.random.normal(0, 0.2, n_active)

        np.random.seed(None)  # Reset seed

        self.input_queue.append({
            'units': units,
            'strength': strength,
            'phases': phases % (2 * np.pi)
        })

        return {'units': units, 'phases': phases, 'id': pattern_id}

    def inject_text(self, text, strength=0.6):
        """Convert text to pattern and inject."""
        # Hash text to get reproducible pattern
        pattern_id = hash(text) % 10000
        return self.inject_pattern(pattern_id, strength)

    def learn_pattern(self, name, n_presentations=5, steps_per=30):
        """
        Learn a pattern through repeated presentation with Hebbian strengthening.

        This creates an attractor for the pattern by:
        1. Storing the activation pattern
        2. Strengthening connections between co-active units
        """
        print(f"Learning '{name}'...")

        all_activations = []

        for i in range(n_presentations):
            self.reset()
            self.inject_text(name, 0.8)

            # Let pattern develop and collect activations
            for _ in range(steps_per):
                self.step(0.1)
                all_activations.append(self.A.copy())

                # Hebbian learning: strengthen connections between co-active units
                rows, cols = self._conn_rows, self._conn_cols
                co_activation = self.A[rows] * self.A[cols]
                self.hebbian_trace += 0.01 * co_activation

        # Store average activation pattern
        avg_pattern = np.mean(all_activations, axis=0)
        self.pattern_traces[name] = avg_pattern

        # Apply Hebbian trace to dynamics weights (make attractors)
        # Normalize and apply gently
        if np.max(self.hebbian_trace) > 0:
            normalized_trace = self.hebbian_trace / (np.max(self.hebbian_trace) + 1e-8)
            self.W_dyn_data += 0.05 * normalized_trace
            self.W_dyn_data = np.clip(self.W_dyn_data, -1.0, 1.0)

        print(f"  Stored pattern with {np.sum(avg_pattern > 0.1):.0f} active units")
        return avg_pattern

    def measure_familiarity(self, name, n_steps=30):
        """
        Measure how familiar a pattern is by:
        1. Overlap with stored patterns
        2. How quickly/coherently the network settles
        3. Attractor stability
        """
        self.reset()
        self.inject_text(name, 0.8)

        # Let it evolve
        activations = []
        for _ in range(n_steps):
            self.step(0.1)
            activations.append(self.A.copy())

        final_A = self.A.copy()

        # Measure 1: Overlap with stored patterns
        overlaps = {}
        for stored_name, stored_pattern in self.pattern_traces.items():
            # Cosine similarity
            dot = np.dot(final_A, stored_pattern)
            norm = np.linalg.norm(final_A) * np.linalg.norm(stored_pattern) + 1e-8
            overlaps[stored_name] = dot / norm

        # Measure 2: Attractor stability (variance of late activations)
        late_activations = np.array(activations[-10:])
        stability = 1.0 / (np.mean(np.var(late_activations, axis=0)) + 0.01)

        # Measure 3: Check if this exact pattern is stored
        is_stored = name in self.pattern_traces
        self_overlap = overlaps.get(name, 0.0)

        return {
            'is_stored': is_stored,
            'self_overlap': self_overlap,
            'all_overlaps': overlaps,
            'stability': stability,
            'final_activation': final_A
        }

    def inject_sequence(self, items, delay_steps=20):
        """
        Inject a sequence of items with delays.
        This is how the system learns temporal structure.
        """
        # This would be called over multiple timesteps
        # For now, just inject the first item
        if items:
            return self.inject_text(items[0])

    def read_state(self):
        """Read current network state."""
        return {
            'activation': self.A.copy(),
            'phase': self.theta.copy(),
            'energy': self.E.copy(),
            'active_units': np.where(self.A > self.threshold)[0],
            'prediction_error': self.prediction_error.copy()
        }

    def measure_response(self, pattern_id, n_steps=50):
        """
        Measure network response to a pattern.
        Returns metrics about how the network processes this input.
        """
        # Reset
        self.A = np.zeros(self.n)
        self.E = np.ones(self.n)

        # Inject pattern
        pattern = self.inject_pattern(pattern_id, strength=0.8)

        # Let it evolve
        responses = []
        for _ in range(n_steps):
            self.step(dt=0.1)
            responses.append({
                'mean_A': np.mean(self.A),
                'pred_error': np.mean(np.abs(self.prediction_error)),
                'coherence': self._phase_coherence(),
                'active': np.sum(self.A > self.threshold)
            })

        return {
            'pattern': pattern,
            'responses': responses,
            'final_state': self.read_state()
        }

    def train_on_sequence(self, sequence, n_epochs=10, steps_per_item=30):
        """
        Train on a sequence of patterns.
        The network learns to predict what comes next.
        """
        print(f"Training on sequence: {sequence}")

        for epoch in range(n_epochs):
            total_error = 0

            for i, item in enumerate(sequence):
                # Inject current item
                self.inject_text(item, strength=0.8)

                # Let it process
                for step in range(steps_per_item):
                    self.step(dt=0.1)
                    total_error += np.mean(np.abs(self.prediction_error))

            avg_error = total_error / (len(sequence) * steps_per_item)
            print(f"  Epoch {epoch+1}: avg prediction error = {avg_error:.4f}")

        print("Training complete.")

    def reset(self):
        """Reset network state (but keep learned weights)."""
        self.A = np.zeros(self.n)
        self.theta = np.random.uniform(0, 2*np.pi, self.n)
        self.E = np.ones(self.n)
        self.prediction_error = np.zeros(self.n)


class InteractiveVisualizer:
    """Interactive visualization for Predictive SOC."""

    def __init__(self, mind):
        self.mind = mind
        self.input_queue = queue.Queue()
        self.last_input = ""

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Activation heatmap
        self.ax_act = self.fig.add_subplot(gs[0, 0])
        self.ax_act.set_title('Activation')
        padded = np.zeros(mind.grid_size**2)
        padded[:mind.n] = mind.A
        self.act_img = self.ax_act.imshow(
            padded.reshape(mind.grid_size, mind.grid_size),
            cmap='hot', vmin=0, vmax=1
        )
        plt.colorbar(self.act_img, ax=self.ax_act)

        # Prediction error heatmap
        self.ax_err = self.fig.add_subplot(gs[0, 1])
        self.ax_err.set_title('Prediction Error')
        self.err_img = self.ax_err.imshow(
            padded.reshape(mind.grid_size, mind.grid_size),
            cmap='RdBu', vmin=-0.5, vmax=0.5
        )
        plt.colorbar(self.err_img, ax=self.ax_err)

        # Phase heatmap
        self.ax_phase = self.fig.add_subplot(gs[0, 2])
        self.ax_phase.set_title('Phase (active units)')
        self.phase_img = self.ax_phase.imshow(
            self._make_phase_img(), vmin=0, vmax=1
        )

        # Time series: activation and error
        self.ax_ts = self.fig.add_subplot(gs[0, 3])
        self.ax_ts.set_title('Activation & Error')
        self.line_act, = self.ax_ts.plot([], [], 'b-', label='Activation')
        self.line_err, = self.ax_ts.plot([], [], 'r-', label='Pred Error')
        self.ax_ts.legend()

        # Time series: threshold and activity
        self.ax_soc = self.fig.add_subplot(gs[1, 0])
        self.ax_soc.set_title('SOC Dynamics')
        self.line_thresh, = self.ax_soc.plot([], [], 'g-', label='Threshold')
        self.line_frac, = self.ax_soc.plot([], [], 'm-', label='Active Frac')
        self.ax_soc.legend()

        # Phase coherence
        self.ax_coh = self.fig.add_subplot(gs[1, 1])
        self.ax_coh.set_title('Phase Coherence')
        self.line_coh, = self.ax_coh.plot([], [], 'orange')

        # Weight distribution
        self.ax_weights = self.fig.add_subplot(gs[1, 2])
        self.ax_weights.set_title('Weight Distribution')

        # Stats
        self.ax_stats = self.fig.add_subplot(gs[1, 3])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.9, '', ha='left', va='top',
                                              fontsize=10, family='monospace',
                                              transform=self.ax_stats.transAxes)

        # Sliders
        slider_ax = self.fig.add_axes([0.1, 0.02, 0.2, 0.02])
        self.slider_lr = Slider(slider_ax, 'Pred LR', 0, 0.05, valinit=mind.lr_pred)
        self.slider_lr.on_changed(lambda v: setattr(mind, 'lr_pred', v))

        slider_ax2 = self.fig.add_axes([0.4, 0.02, 0.2, 0.02])
        self.slider_target = Slider(slider_ax2, 'Target Activity', 0.01, 0.2, valinit=mind.target_activity)
        self.slider_target.on_changed(lambda v: setattr(mind, 'target_activity', v))

        # Buttons
        btn_ax = self.fig.add_axes([0.7, 0.02, 0.08, 0.03])
        self.btn_reset = Button(btn_ax, 'Reset')
        self.btn_reset.on_clicked(lambda e: mind.reset())

        btn_ax2 = self.fig.add_axes([0.8, 0.02, 0.08, 0.03])
        self.btn_burst = Button(btn_ax2, 'Burst')
        self.btn_burst.on_clicked(lambda e: mind.inject_pattern(np.random.randint(10000), 0.9))

    def _make_phase_img(self):
        """Phase as HSV image."""
        from matplotlib.colors import hsv_to_rgb

        padded_phase = np.zeros(self.mind.grid_size**2)
        padded_phase[:self.mind.n] = self.mind.theta / (2 * np.pi)

        padded_amp = np.zeros(self.mind.grid_size**2)
        padded_amp[:self.mind.n] = self.mind.A

        h = padded_phase.reshape(self.mind.grid_size, self.mind.grid_size)
        s = np.ones_like(h)
        v = np.clip(padded_amp.reshape(self.mind.grid_size, self.mind.grid_size) * 3, 0, 1)

        return hsv_to_rgb(np.stack([h, s, v], axis=-1))

    def _process_input(self):
        """Process command queue."""
        try:
            while True:
                text = self.input_queue.get_nowait().strip()
                if not text:
                    continue

                if text.lower() in ('quit', 'exit'):
                    plt.close(self.fig)
                    return
                elif text.lower() == 'reset':
                    self.mind.reset()
                    self.last_input = "[RESET]"
                elif text.lower().startswith('train '):
                    # train a b c d  -> train on sequence
                    items = text[6:].split()
                    self.mind.train_on_sequence(items, n_epochs=5)
                    self.last_input = f"[TRAINED: {items}]"
                elif text.lower().startswith('learn '):
                    # learn hello world -> learn each pattern with Hebbian strengthening
                    words = text[6:].split()
                    for word in words:
                        self.mind.learn_pattern(word)
                    print(f"Learned {len(words)} patterns: {words}")
                    print(f"Stored patterns: {list(self.mind.pattern_traces.keys())}")
                    sys.stdout.flush()
                    self.last_input = f"[LEARNED: {words}]"
                elif text.lower().startswith('test '):
                    word = text[5:].strip()

                    # Use familiarity measurement
                    result = self.mind.measure_familiarity(word)

                    print(f"\n[TEST '{word}']")
                    print(f"  Known pattern: {result['is_stored']}")
                    if result['is_stored']:
                        print(f"  Self-overlap: {result['self_overlap']:.3f}")

                    if result['all_overlaps']:
                        print(f"  Overlaps with stored patterns:")
                        for name, overlap in sorted(result['all_overlaps'].items(),
                                                     key=lambda x: -x[1]):
                            marker = " <--" if name == word else ""
                            print(f"    {name}: {overlap:.3f}{marker}")
                    else:
                        print(f"  No patterns learned yet. Use 'learn <word>' first.")

                    print(f"  Stability: {result['stability']:.2f}")
                    sys.stdout.flush()

                    self.last_input = f"[TEST: {word}]"
                else:
                    self.mind.inject_text(text, 0.7)
                    self.last_input = text

        except queue.Empty:
            pass

    def update(self, frame):
        """Animation update."""
        self._process_input()

        # Run steps
        for _ in range(5):
            if np.random.random() < 0.005:
                self.mind.inject_pattern(np.random.randint(10000), 0.3)
            self.mind.step(0.1)

        # Update activation
        padded = np.zeros(self.mind.grid_size**2)
        padded[:self.mind.n] = self.mind.A
        self.act_img.set_data(padded.reshape(self.mind.grid_size, self.mind.grid_size))

        # Update error
        padded_err = np.zeros(self.mind.grid_size**2)
        padded_err[:self.mind.n] = self.mind.prediction_error
        self.err_img.set_data(padded_err.reshape(self.mind.grid_size, self.mind.grid_size))

        # Update phase
        self.phase_img.set_data(self._make_phase_img())

        # Update time series
        t = self.mind.history['time'][-500:]

        act = self.mind.history['mean_activation'][-500:]
        err = self.mind.history['mean_pred_error'][-500:]
        self.line_act.set_data(t, act)
        self.line_err.set_data(t, err)
        if t:
            self.ax_ts.set_xlim(t[0], t[-1])
            self.ax_ts.set_ylim(0, max(0.1, max(max(act), max(err)) * 1.1))

        thresh = self.mind.history['threshold'][-500:]
        frac = self.mind.history['active_fraction'][-500:]
        self.line_thresh.set_data(t, thresh)
        self.line_frac.set_data(t, frac)
        if t:
            self.ax_soc.set_xlim(t[0], t[-1])
            self.ax_soc.set_ylim(0, max(0.5, max(max(thresh), max(frac)) * 1.1))

        coh = self.mind.history['phase_coherence'][-500:]
        self.line_coh.set_data(t, coh)
        if t:
            self.ax_coh.set_xlim(t[0], t[-1])
            self.ax_coh.set_ylim(0, 1)

        # Update weight histogram - show BOTH W_pred and W_dyn
        self.ax_weights.clear()
        self.ax_weights.hist(self.mind.W_pred_data, bins=50, alpha=0.5, label='W_pred', color='blue')
        self.ax_weights.hist(self.mind.W_dyn_data, bins=50, alpha=0.5, label='W_dyn', color='green')
        self.ax_weights.set_title(f'Weights (pred={np.mean(np.abs(self.mind.W_pred_data)):.3f}, dyn={np.mean(np.abs(self.mind.W_dyn_data)):.3f})')
        self.ax_weights.axvline(0, color='r', linestyle='--', alpha=0.5)
        self.ax_weights.legend(fontsize=8)

        # Stats
        n_patterns = len(self.mind.pattern_traces)
        pattern_names = list(self.mind.pattern_traces.keys())[:3]
        patterns_str = ', '.join(pattern_names) if pattern_names else 'none'
        if n_patterns > 3:
            patterns_str += f'... (+{n_patterns-3})'

        stats = f"""Time: {self.mind.t:.1f}
Threshold: {self.mind.threshold:.3f}
Active: {np.sum(self.mind.A > self.mind.threshold)}
Mean Error: {np.mean(np.abs(self.mind.prediction_error)):.4f}
Confidence: {np.mean(self.mind.prediction_confidence):.3f}
Coherence: {self.mind._phase_coherence():.3f}
Learned: {patterns_str}

Last: {self.last_input[:25]}"""
        self.stats_text.set_text(stats)

        return [self.act_img, self.err_img, self.phase_img]

    def run(self):
        """Start interactive visualization."""
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.anim = FuncAnimation(self.fig, self.update, interval=50,
                                  blit=False, cache_frame_data=False)
        plt.show()

    def _input_loop(self):
        """Background input loop."""
        print("\n" + "=" * 60)
        print("PREDICTIVE SOC MIND")
        print("=" * 60)
        print("This system learns by predicting its own next state.")
        print("Structure emerges from prediction, not hard-coded regions.")
        print()
        print("Commands:")
        print("  <text>         - Inject pattern (just activates, no learning)")
        print("  learn a b c    - Learn patterns with Hebbian strengthening")
        print("  test <word>    - Test familiarity (overlap with learned patterns)")
        print("  train a b c    - Train on sequence (prediction learning)")
        print("  reset          - Reset state (keep weights)")
        print("  quit           - Exit")
        print()
        print("Try: learn hello world  -> then: test hello  vs  test dog")
        print("=" * 60)
        sys.stdout.flush()

        while True:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if line:
                    self.input_queue.put(line)
            except:
                break


def demo_prediction_learning():
    """Demo: show how prediction error decreases with repeated patterns."""
    print("=" * 60)
    print("TWO-STAGE LEARNING DEMO")
    print("=" * 60)
    print("W_pred learns quickly from errors, W_dyn consolidates slowly")
    print()

    mind = PredictiveSOCMind(n_units=512)

    # Inject same pattern repeatedly, measure prediction error
    pattern_id = 42

    print("Injecting same pattern 20 times...")
    print(f"{'Inj':>4} {'Error':>8} {'Confid':>8} {'W_pred':>8} {'W_dyn':>8}")
    print("-" * 44)

    errors = []
    confidences = []
    for i in range(20):
        mind.inject_pattern(pattern_id, 0.8)

        # Run for a bit
        step_errors = []
        for _ in range(30):
            mind.step(0.1)
            step_errors.append(np.mean(np.abs(mind.prediction_error)))

        avg_err = np.mean(step_errors)
        avg_conf = np.mean(mind.prediction_confidence)
        w_pred = np.mean(np.abs(mind.W_pred_data))
        w_dyn = np.mean(np.abs(mind.W_dyn_data))

        errors.append(avg_err)
        confidences.append(avg_conf)

        print(f"{i+1:>4} {avg_err:>8.4f} {avg_conf:>8.4f} {w_pred:>8.4f} {w_dyn:>8.4f}")

    print("-" * 44)
    print(f"\nError: {errors[0]:.4f} -> {errors[-1]:.4f} ({(1 - errors[-1]/errors[0])*100:+.1f}%)")
    print(f"Confidence: {confidences[0]:.4f} -> {confidences[-1]:.4f}")

    return mind, errors


def demo_sequence_learning():
    """Demo: learn temporal sequences."""
    print("=" * 60)
    print("SEQUENCE LEARNING DEMO")
    print("=" * 60)

    mind = PredictiveSOCMind(n_units=512)

    # Train on sequence A -> B -> C
    sequence = ['alpha', 'beta', 'gamma']

    print(f"\nTraining on sequence: {sequence}")
    mind.train_on_sequence(sequence, n_epochs=10, steps_per_item=50)

    print("\nTesting: inject 'alpha', measure response...")
    mind.reset()
    response = mind.measure_response(hash('alpha') % 10000, n_steps=100)

    print(f"  Final active units: {response['final_state']['active_units'][:10]}...")
    print(f"  Final mean activation: {np.mean(response['final_state']['activation']):.4f}")

    return mind


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_prediction_learning()
        elif sys.argv[1] == '--sequence':
            demo_sequence_learning()
    else:
        print("Starting interactive mode...")
        mind = PredictiveSOCMind(n_units=1024)

        # Brief warmup
        for _ in range(100):
            mind.step(0.1)

        viz = InteractiveVisualizer(mind)
        viz.run()
