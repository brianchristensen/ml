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

        # === Temporal/Associative Memory ===
        # For sequence learning: strengthen connections FROM past TO present
        # This allows pattern A to trigger pattern B
        self.temporal_trace = np.zeros(self._n_connections)  # A_prev[col] * A_now[row]
        self.associations = {}  # "A->B" -> strength

        # === SOC: Synaptic Scaling ===
        # Each unit tracks its running average activity
        # If too active: scale DOWN incoming weights (prevents runaway)
        # If too quiet: scale UP incoming weights (prevents death)
        # This creates criticality through weight homeostasis
        self.activity_trace = np.ones(n_units) * self.target_activity  # Running avg
        self.synaptic_scale = np.ones(n_units)  # Scaling factor per unit

        # === Avalanche Tracking ===
        self.avalanche_sizes = []  # History of avalanche sizes
        self.current_avalanche = 0  # Current avalanche size
        self.A_prev = np.zeros(n_units)  # Previous activation for avalanche detection
        self.in_avalanche = False

    def step(self, dt=0.1, dream_mode=False):
        """One timestep of dynamics."""
        self.t += dt

        # Store previous activation for temporal learning and avalanche tracking
        self.A_prev = self.A.copy()

        # 1. Process external input (skip in dream mode - only noise)
        if not dream_mode:
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

        # 7. === SOC: Synaptic Scaling ===
        self._synaptic_scaling(dt)

        # 8. === Avalanche Tracking ===
        self._track_avalanche(A_next)

        # 9. Update state
        self.A = A_next
        self._update_phases(dt)
        self._update_energy(dt)

        # 10. Dream mode: temporal Hebbian learning during free evolution
        if dream_mode:
            self._dream_learning(dt)

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

    def _synaptic_scaling(self, dt):
        """
        SOC through synaptic scaling (weight homeostasis).

        This is the core SOC mechanism:
        - GROWTH: Learning strengthens specific connections → cascades can spread
        - DECAY: If a unit is too active, scale DOWN its incoming weights

        This creates competition: learned pathways compete for limited "activation budget".
        The system self-organizes to criticality through weight regulation.

        Biological basis: Homeostatic synaptic plasticity (Turrigiano et al.)
        """
        # Update running average of each unit's activity
        activity_tau = 0.99  # Slow averaging
        self.activity_trace = activity_tau * self.activity_trace + (1 - activity_tau) * self.A

        # Compute scaling factor: if activity > target, scale < 1 (reduce); if < target, scale > 1
        # Using multiplicative scaling for stability
        ratio = self.target_activity / (self.activity_trace + 1e-8)
        # Gentle scaling - don't change too fast
        target_scale = np.clip(ratio, 0.5, 2.0)
        self.synaptic_scale += 0.001 * (target_scale - self.synaptic_scale)

        # Apply scaling to incoming weights (W_dyn)
        # Each connection's effective weight is W * scale[target_unit]
        rows = self._conn_rows
        scale_factors = self.synaptic_scale[rows]

        # Instead of modifying W_dyn directly, we'll use this in _soc_dynamics
        # But for now, let's apply a gentle push toward balanced weights
        scaling_rate = 0.0001
        self.W_dyn_data *= (1 - scaling_rate) + scaling_rate * scale_factors

    def _track_avalanche(self, A_next):
        """
        Track avalanche sizes for SOC analysis.

        An avalanche starts when activity increases and ends when it decreases.
        Power-law distributed avalanche sizes = system at criticality.
        """
        # Count newly activated units (crossed threshold this step)
        newly_active = np.sum((A_next > self.threshold) & (self.A_prev <= self.threshold))
        still_active = np.sum((A_next > self.threshold) & (self.A_prev > self.threshold))

        # Avalanche detection
        if newly_active > 0:
            if not self.in_avalanche:
                # Start new avalanche
                self.in_avalanche = True
                self.current_avalanche = newly_active
            else:
                # Continue avalanche
                self.current_avalanche += newly_active
        else:
            if self.in_avalanche:
                # Avalanche ended - record its size
                if self.current_avalanche > 0:
                    self.avalanche_sizes.append(self.current_avalanche)
                    # Keep only recent avalanches
                    if len(self.avalanche_sizes) > 1000:
                        self.avalanche_sizes = self.avalanche_sizes[-500:]
                self.in_avalanche = False
                self.current_avalanche = 0

    def _dream_learning(self, dt):
        """
        Temporal Hebbian learning during dream/free evolution.

        When the network runs freely (no external input), associations
        that naturally co-activate get strengthened. This:
        1. Consolidates learned chains (cat→meow→purr)
        2. Allows the network to "practice" associations
        3. Strengthens whatever patterns emerge from SOC dynamics

        Key insight: In dream mode, the network's own dynamics determine
        what gets learned. Strong associations activate each other,
        reinforcing themselves.
        """
        rows, cols = self._conn_rows, self._conn_cols

        # Temporal Hebbian: strengthen connections FROM previously-active TO currently-active
        source_was_active = self.A_prev[cols]  # How active was source last step?
        target_is_active = self.A[rows]         # How active is target now?

        # Only learn when there's a temporal sequence (not just co-activation)
        temporal_signal = source_was_active * target_is_active

        # Weight the learning by how "surprising" this activation is
        # (prediction error modulated)
        surprise = np.abs(self.prediction_error[rows])

        # Dream learning rate (slower than explicit training)
        dream_lr = 0.001
        dW = dream_lr * temporal_signal * (1 + surprise)

        # Apply to both temporal trace and dynamics weights
        self.temporal_trace += dt * dW

        # Gently apply temporal trace to W_dyn
        if np.max(self.temporal_trace) > 0:
            normalized = self.temporal_trace / (np.max(self.temporal_trace) + 1e-8)
            self.W_dyn_data += dt * 0.01 * normalized
            self.W_dyn_data = np.clip(self.W_dyn_data, -1.0, 1.0)

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

    def learn_association(self, pattern_a, pattern_b, n_reps=10, transition_steps=20):
        """
        Learn that pattern A triggers pattern B.

        Temporal Hebbian rule:
        - Present A, let it develop
        - Present B, strengthen connections FROM A-active units TO B-active units
        - Repeat to strengthen the association

        This creates a "pathway" from A's attractor to B's attractor.
        """
        print(f"Learning association: '{pattern_a}' -> '{pattern_b}'")

        # First ensure both patterns are learned individually
        if pattern_a not in self.pattern_traces:
            self.learn_pattern(pattern_a)
        if pattern_b not in self.pattern_traces:
            self.learn_pattern(pattern_b)

        rows, cols = self._conn_rows, self._conn_cols

        for rep in range(n_reps):
            # Phase 1: Present pattern A and let it develop
            self.reset()
            self.inject_text(pattern_a, 0.9)
            for _ in range(15):
                self.step(0.1)

            # Capture A's activation (these are the "source" units)
            A_source = self.A.copy()

            # Phase 2: Present pattern B (the target)
            self.inject_text(pattern_b, 0.9)
            for step in range(transition_steps):
                self.step(0.1)

                # Temporal Hebbian: strengthen FROM A-active TO currently-active
                # This creates pathways from A's representation to B's
                source_activity = A_source[cols]  # How active was source in pattern A?
                target_activity = self.A[rows]     # How active is target now?

                # Only strengthen where both source was active (in A) and target is active (in B)
                temporal_strength = source_activity * target_activity
                self.temporal_trace += 0.005 * temporal_strength

        # Apply temporal trace to dynamics weights
        if np.max(self.temporal_trace) > 0:
            normalized = self.temporal_trace / (np.max(self.temporal_trace) + 1e-8)
            # Add to dynamics weights - this creates the associative pathway
            self.W_dyn_data += 0.1 * normalized
            self.W_dyn_data = np.clip(self.W_dyn_data, -1.0, 1.0)

        # Record the association
        assoc_key = f"{pattern_a}->{pattern_b}"
        self.associations[assoc_key] = self.associations.get(assoc_key, 0) + 1

        print(f"  Association strengthened ({n_reps} repetitions)")
        print(f"  Stored associations: {list(self.associations.keys())}")

        return True

    def recall(self, trigger_pattern, n_steps=50):
        """
        Present a pattern and see what the network evolves toward.

        If we learned A->B, presenting A should cause the network
        to drift toward B's representation.
        """
        self.reset()
        self.inject_text(trigger_pattern, 0.9)

        # Track overlaps over time
        overlap_history = {name: [] for name in self.pattern_traces}

        for step in range(n_steps):
            self.step(0.1)

            # Measure current overlap with all stored patterns
            for name, pattern in self.pattern_traces.items():
                dot = np.dot(self.A, pattern)
                norm = np.linalg.norm(self.A) * np.linalg.norm(pattern) + 1e-8
                overlap_history[name].append(dot / norm)

        # Analyze: which pattern has highest overlap at the end?
        final_overlaps = {name: hist[-1] for name, hist in overlap_history.items()}

        # Also check: which pattern INCREASED the most?
        delta_overlaps = {}
        for name, hist in overlap_history.items():
            if len(hist) > 10:
                early = np.mean(hist[:10])
                late = np.mean(hist[-10:])
                delta_overlaps[name] = late - early

        return {
            'trigger': trigger_pattern,
            'final_overlaps': final_overlaps,
            'delta_overlaps': delta_overlaps,
            'history': overlap_history
        }

    def dream(self, n_steps=200, verbose=True):
        """
        Dream mode: run the network with only noise, allowing temporal
        Hebbian learning to consolidate associations.

        This is like sleep consolidation:
        - No external input (just spontaneous noise)
        - Whatever patterns naturally co-activate get strengthened
        - Chains like cat→meow→purr reinforce themselves

        Returns statistics about what happened during dreaming.
        """
        if verbose:
            print(f"Dreaming for {n_steps} steps...")
            print("  (Temporal Hebbian learning active, no external input)")

        # Track which patterns activate during dreaming
        pattern_activations = {name: [] for name in self.pattern_traces}
        avalanches_before = len(self.avalanche_sizes)

        # Start from low activity state
        self.A = np.random.uniform(0, 0.1, self.n)

        for step in range(n_steps):
            # Add small noise to trigger spontaneous activity
            noise_units = np.random.choice(self.n, size=5, replace=False)
            self.A[noise_units] += np.random.uniform(0.1, 0.3, len(noise_units))
            self.A = np.clip(self.A, 0, 1)

            # Step with dream_mode=True
            self.step(0.1, dream_mode=True)

            # Track pattern overlaps
            for name, pattern in self.pattern_traces.items():
                dot = np.dot(self.A, pattern)
                norm = np.linalg.norm(self.A) * np.linalg.norm(pattern) + 1e-8
                pattern_activations[name].append(dot / norm)

        # Analyze what patterns were active during dreaming
        avg_activations = {name: np.mean(hist) for name, hist in pattern_activations.items()}
        peak_activations = {name: np.max(hist) for name, hist in pattern_activations.items()}
        avalanches_during = len(self.avalanche_sizes) - avalanches_before

        if verbose:
            print(f"\nDream summary:")
            print(f"  Avalanches: {avalanches_during}")
            if avg_activations:
                print(f"  Pattern activations (avg):")
                for name, avg in sorted(avg_activations.items(), key=lambda x: -x[1]):
                    peak = peak_activations[name]
                    print(f"    {name}: avg={avg:.3f}, peak={peak:.3f}")
            print(f"  Temporal trace max: {np.max(self.temporal_trace):.4f}")
            print("Done dreaming.")

        return {
            'n_steps': n_steps,
            'avalanches': avalanches_during,
            'avg_activations': avg_activations,
            'peak_activations': peak_activations,
            'pattern_history': pattern_activations
        }

    def get_avalanche_stats(self):
        """Get statistics about avalanche sizes (for SOC analysis)."""
        if len(self.avalanche_sizes) < 10:
            return {'n_avalanches': len(self.avalanche_sizes), 'status': 'not enough data'}

        sizes = np.array(self.avalanche_sizes)
        return {
            'n_avalanches': len(sizes),
            'mean_size': np.mean(sizes),
            'max_size': np.max(sizes),
            'median_size': np.median(sizes),
            'std_size': np.std(sizes),
            # Power law indicator: ratio of mean to median
            # For power law: mean >> median
            'mean_median_ratio': np.mean(sizes) / (np.median(sizes) + 1e-8)
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
        self.is_dreaming = True  # Start in dream mode

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
        """Process command queue. Returns True if input was processed."""
        had_input = False
        try:
            while True:
                text = self.input_queue.get_nowait().strip()
                if not text:
                    continue

                had_input = True
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
                elif text.lower().startswith('assoc '):
                    # assoc cat meow -> learn that 'cat' triggers 'meow'
                    parts = text[6:].split()
                    if len(parts) >= 2:
                        pattern_a, pattern_b = parts[0], parts[1]
                        self.mind.learn_association(pattern_a, pattern_b)
                        sys.stdout.flush()
                        self.last_input = f"[ASSOC: {pattern_a}->{pattern_b}]"
                    else:
                        print("Usage: assoc <pattern_a> <pattern_b>")
                        sys.stdout.flush()
                elif text.lower().startswith('recall '):
                    # recall cat -> show what patterns are triggered
                    word = text[7:].strip()
                    result = self.mind.recall(word)

                    print(f"\n[RECALL '{word}']")
                    print(f"  Final overlaps (after evolution):")
                    for name, overlap in sorted(result['final_overlaps'].items(),
                                                 key=lambda x: -x[1]):
                        marker = " <-- trigger" if name == word else ""
                        print(f"    {name}: {overlap:.3f}{marker}")

                    print(f"\n  Delta (change over time):")
                    for name, delta in sorted(result['delta_overlaps'].items(),
                                               key=lambda x: -x[1]):
                        direction = "+" if delta > 0 else ""
                        marker = " <-- RECALLED!" if delta > 0.05 and name != word else ""
                        print(f"    {name}: {direction}{delta:.3f}{marker}")

                    sys.stdout.flush()
                    self.last_input = f"[RECALL: {word}]"
                elif text.lower().startswith('dream'):
                    # dream or dream 500 -> run dream mode for N steps
                    parts = text.split()
                    n_steps = int(parts[1]) if len(parts) > 1 else 200
                    result = self.mind.dream(n_steps=n_steps)
                    sys.stdout.flush()
                    self.last_input = f"[DREAM: {n_steps} steps]"
                elif text.lower() == 'soc':
                    # Show SOC statistics
                    stats = self.mind.get_avalanche_stats()
                    print(f"\n[SOC Statistics]")
                    if 'status' in stats:
                        print(f"  {stats['status']}")
                    else:
                        print(f"  Avalanches recorded: {stats['n_avalanches']}")
                        print(f"  Mean size: {stats['mean_size']:.2f}")
                        print(f"  Median size: {stats['median_size']:.2f}")
                        print(f"  Max size: {stats['max_size']}")
                        print(f"  Mean/Median ratio: {stats['mean_median_ratio']:.2f}")
                        print(f"  (Ratio > 2 suggests power-law / criticality)")
                    print(f"\n  Synaptic scale range: [{np.min(self.mind.synaptic_scale):.3f}, {np.max(self.mind.synaptic_scale):.3f}]")
                    print(f"  Activity trace range: [{np.min(self.mind.activity_trace):.4f}, {np.max(self.mind.activity_trace):.4f}]")
                    sys.stdout.flush()
                    self.last_input = "[SOC]"
                else:
                    self.mind.inject_text(text, 0.7)
                    self.last_input = text

        except queue.Empty:
            pass
        return had_input

    def update(self, frame):
        """Animation update."""
        had_input = self._process_input()
        self.is_dreaming = not had_input  # Track mode for display

        # Run steps - dream mode when idle, awake mode when input received
        for _ in range(5):
            if had_input:
                # Awake mode: just processed real input, run normal dynamics
                self.mind.step(0.1, dream_mode=False)
            else:
                # Dream/idle mode: no input, consolidate associations
                # Add noise EVERY step to trigger spontaneous activity (match explicit dream)
                noise_units = np.random.choice(self.mind.n, size=5, replace=False)
                self.mind.A[noise_units] += np.random.uniform(0.1, 0.3, len(noise_units))
                self.mind.A = np.clip(self.mind.A, 0, 1)
                self.mind.step(0.1, dream_mode=True)

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

        mode = "DREAMING" if self.is_dreaming else "AWAKE"
        n_avalanches = len(self.mind.avalanche_sizes)

        stats = f"""Mode: {mode}
Time: {self.mind.t:.1f}
Threshold: {self.mind.threshold:.3f}
Active: {np.sum(self.mind.A > self.mind.threshold)}
Avalanches: {n_avalanches}
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
        print("DEFAULT: Network is DREAMING (consolidating associations)")
        print("         Switches to AWAKE mode briefly when you input.")
        print()
        print("Commands:")
        print("  <text>         - Inject pattern (just activates, no learning)")
        print("  learn a b c    - Learn patterns with Hebbian strengthening")
        print("  assoc a b      - Learn association: a triggers b")
        print("  recall <word>  - See what patterns are triggered by <word>")
        print("  test <word>    - Test familiarity (overlap with learned patterns)")
        print("  dream [N]      - Dream for N steps (consolidate associations)")
        print("  soc            - Show SOC statistics (avalanches, synaptic scaling)")
        print("  train a b c    - Train on sequence (prediction learning)")
        print("  reset          - Reset state (keep weights)")
        print("  quit           - Exit")
        print()
        print("Try: assoc cat meow  -> then: recall cat  (should trigger meow)")
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
