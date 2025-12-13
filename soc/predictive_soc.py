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

        # === Hierarchical Predictive Learning ===
        # Each unit predicts the input it will receive from other units
        # Slow units (high τ) predict patterns in fast units (low τ)
        # This creates a predictive hierarchy where higher levels predict lower levels
        #
        # W_hier: prediction weights (connection-level)
        # Unit i predicts what input it will receive via: pred_input = Σ W_hier_ij * A_i
        self.W_hier_data = np.zeros(self._n_connections)  # Hierarchical prediction weights
        self.input_pred = np.zeros(n_units)      # Predicted input for each unit
        self.input_pred_error = np.zeros(n_units)  # Error: actual - predicted
        self.lr_hier = 0.02                       # Learning rate for hierarchical prediction

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

        # === Continuous Contrastive Learning ===
        # A_target: what external input wants the network state to be (sparse)
        # This enables real prediction: network flows toward A_target
        # Error = A_target - A is local and drives three-factor learning
        self.A_target = np.zeros(n_units)      # Target activation (sparse)
        self.target_strength = 0.0              # β: how strongly to pull toward target
        self.target_decay = 0.9                 # How fast target decays without new input
        self.lr_contrastive = 0.1               # Learning rate for contrastive rule (increased for faster learning)

        # === Adaptive Timescales (for emergent hierarchy) ===
        # Each unit has its own time constant τ
        # Small τ → fast response → low-level features
        # Large τ → slow integration → high-level patterns
        # τ adapts based on input variance and prediction error
        self.tau = np.exp(np.random.randn(n_units) * 0.5)  # Log-normal, centered ~1
        self.tau = np.clip(self.tau, 0.1, 10.0)  # Keep in reasonable range

        # Track input variance per unit (for tau adaptation)
        self.input_mean = np.zeros(n_units)      # Running mean of input
        self.input_var = np.ones(n_units) * 0.1  # Running variance of input
        self.var_decay = 0.95                     # Decay for variance tracking

        # === Fast Hebbian Weights (Working Memory) ===
        # When input arrives, immediately update fast weights (Hebbian)
        # These weights persist even as activation decays
        # At prediction time, fast weights encode "what was recently seen"
        #
        # W_fast: temporary weight changes on same sparse structure as W_dyn
        # Updated when input is injected, decays slowly over time
        self.W_fast_data = np.zeros(self._n_connections)
        self.fast_decay = 0.92  # How fast the fast weights decay each step
        self.fast_lr = 0.5  # Learning rate for fast weight updates
        self.fast_weight_scale = 0.3  # How much fast weights contribute to dynamics

        # Also keep a simple input trace (leaky integrator of inputs)
        # This directly tracks what patterns have been seen recently
        self.input_trace = np.zeros(n_units)
        self.trace_decay = 0.97  # Decay per step (slow decay to persist during learning)
        self.trace_weight = 0.5  # How much trace contributes to dynamics

        # Tau adaptation parameters
        self.lr_tau = 0.001                       # Learning rate for tau adaptation
        self.tau_min = 0.1                        # Minimum timescale
        self.tau_max = 10.0                       # Maximum timescale

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

        # 5. Learn from prediction error (internal model)
        self._learn_from_error(dt)

        # 6. === Continuous Contrastive Learning ===
        # Three-factor: pre * post * local_error
        # This is the EXTERNAL prediction learning
        self._contrastive_learning(dt)

        # 7. === Adaptive Timescales ===
        # Units learn their own τ based on input variance and prediction error
        # This enables emergent hierarchy
        self._adapt_timescales(dt)

        # 8. === Hierarchical Predictive Learning ===
        # Units learn to predict their inputs (slow predicts fast)
        self._hierarchical_learning(dt)

        # 9. Homeostatic threshold adjustment (maintains criticality)
        self._homeostatic_adjustment()

        # 8. === SOC: Synaptic Scaling ===
        self._synaptic_scaling(dt)

        # 9. === Avalanche Tracking ===
        self._track_avalanche(A_next)

        # 10. Update state
        self.A = A_next
        self._update_phases(dt)
        self._update_energy(dt)

        # 11. Decay target (external signal fades without reinforcement)
        self.A_target *= self.target_decay
        self.target_strength *= self.target_decay

        # 12. Dream mode: temporal Hebbian learning during free evolution
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

        # === Hierarchical Prediction ===
        # Each unit predicts the input it will receive based on its own state
        # Slow units (high τ) have smooth activation that carries temporal context
        # They use this context to predict what input they'll receive from fast units
        #
        # For connection j→i: unit i predicts j's contribution as W_hier_ij * A_i
        # This means slow units learn to predict patterns in their inputs
        hier_pred_input = self.W_hier_data * self.A[rows]  # Use receiving unit's state
        predicted_input = np.zeros(self.n)
        np.add.at(predicted_input, rows, hier_pred_input)
        predicted_input = predicted_input / in_degree

        # Store prediction and error for learning
        self.input_pred = predicted_input
        self.input_pred_error = total_input - predicted_input  # Actual - Predicted

        # === Track input statistics for tau adaptation ===
        # Update running mean and variance of input per unit
        self.input_mean = self.var_decay * self.input_mean + (1 - self.var_decay) * total_input
        input_diff = total_input - self.input_mean
        self.input_var = self.var_decay * self.input_var + (1 - self.var_decay) * (input_diff ** 2)

        # === Continuous Contrastive: Pull toward target ===
        # local_error = A_target - A (what should be minus what is)
        # This term pulls the network toward the target state
        target_pull = self.target_strength * (self.A_target - self.A)

        # === Fast Weights: Memory of recent inputs ===
        # W_fast encodes what patterns were recently injected (Hebbian traces)
        # These add to the dynamics, allowing past inputs to influence current state

        # Compute input from fast weights (same structure as W_dyn)
        fast_exc = np.maximum(self.W_fast_data, 0) * firing[cols] * self.E[cols]
        fast_sum = np.zeros(self.n)
        np.add.at(fast_sum, rows, fast_exc)
        fast_input = self.fast_weight_scale * fast_sum / in_degree

        # Input trace contribution: directly excite units that were recently input
        trace_input = self.trace_weight * self.input_trace

        # === Decay fast weights and input trace ===
        self.W_fast_data *= self.fast_decay
        self.input_trace *= self.trace_decay

        # === Adaptive Timescale Dynamics ===
        # dA/dt = (1/τ) * [input - decay*A + noise + target_pull + fast_input + trace_input]
        # Small τ → fast response (low-level features)
        # Large τ → slow integration (high-level patterns)
        # Fast weights and input trace add historical context
        raw_dA = (total_input
                  + fast_input
                  + trace_input
                  - self.decay * self.A
                  + np.random.normal(0, self.noise, self.n)
                  + target_pull)
        dA = (1.0 / self.tau) * raw_dA

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

    def _contrastive_learning(self, dt):
        """
        Three-factor learning for continuous contrastive prediction.

        The unified equation:
            dW_ij/dt = η * pre_i * post_j * local_error_j

        Where:
            - pre_i = activation of source unit i
            - post_j = activation of target unit j
            - local_error_j = A_target_j - A_j (how wrong is unit j?)

        This is THREE-FACTOR because learning requires:
            1. Pre-synaptic activity (source is active)
            2. Post-synaptic activity (target is active)
            3. Error signal (target knows it's wrong)

        The error is LOCAL - each unit knows its own target and its own state.
        No backprop needed. Credit assignment is implicit in the sparse activation.

        For sparse patterns:
            - Most local_error_j ≈ 0 (both target and actual are 0 or both are 1)
            - Learning only happens where there's mismatch
            - And only for active pre-synaptic units
        """
        if self.target_strength < 0.01:
            # No target active, skip contrastive learning
            return

        rows, cols = self._conn_rows, self._conn_cols

        # Local error at each unit: what it should be minus what it is
        local_error = self.A_target - self.A  # [n_units]

        # Delta rule for contrastive learning:
        # dW_ij = η * pre_i * error_j
        #
        # This is simpler than three-factor but handles both cases:
        # - error > 0 (should be more active): strengthen from active pre
        # - error < 0 (should be less active): weaken from active pre
        #
        # KEY INSIGHT: Use input_trace as pre, not current A!
        # input_trace contains "what was seen" without target contamination.
        # This learns: "when these inputs were seen → predict target"
        # Rather than: "when target is already active → strengthen target"
        pre = self.input_trace[cols]  # What was INPUT, not what is currently active
        error_post = local_error[rows]

        dW_forward = self.lr_contrastive * pre * error_post

        # === BIDIRECTIONAL LEARNING ===
        # Also learn the REVERSE association: target → input
        # This allows recalling what LED TO the target, not just what follows
        # When target is active AND input_trace shows what was seen,
        # strengthen connections from target back to input context.
        # This creates backward flow: slow → fast, target → context
        target_pre = self.A_target[cols]  # Target as source
        trace_post = self.input_trace[rows]  # Input trace as destination
        dW_backward = self.lr_contrastive * 0.5 * target_pre * trace_post

        # Combine forward and backward learning
        dW = dW_forward + dW_backward

        # Apply to dynamics weights (this IS the real learning)
        self.W_dyn_data += dt * dW
        self.W_dyn_data = np.clip(self.W_dyn_data, -1.0, 1.0)

    def _adapt_timescales(self, dt):
        """
        Adapt per-unit timescales based on input variance and prediction error.

        The differential equation for τ:
            dτ/dt = lr * prediction_error * (target_τ - τ)

        Where target_τ is derived from input variance:
            - High variance → small target_τ (need to track fast changes)
            - Low variance → large target_τ (can integrate longer)

        The prediction_error gates the learning:
            - High error → τ needs to change (current timescale not working)
            - Low error → τ is good, keep it stable

        This creates emergent hierarchy:
            - Units receiving variable input become fast (low-level)
            - Units receiving stable input become slow (high-level)
            - Units with high error explore different timescales
        """
        # Compute target tau from input variance
        # target_tau = k / (variance + epsilon)
        # High variance → low target (fast), Low variance → high target (slow)
        variance_scale = 0.5  # Scaling factor
        target_tau = variance_scale / (self.input_var + 0.01)
        target_tau = np.clip(target_tau, self.tau_min, self.tau_max)

        # Prediction error gates the adaptation
        # Only adapt tau when prediction is wrong
        pred_error = np.abs(self.prediction_error)

        # Combined adaptation rule:
        # dτ/dt = lr * error * (target - current) + small_noise_for_exploration
        d_tau = self.lr_tau * pred_error * (target_tau - self.tau)

        # Add small exploration noise to prevent getting stuck
        d_tau += 0.0001 * np.random.randn(self.n)

        # Update tau
        self.tau += dt * d_tau
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

    def _hierarchical_learning(self, dt):
        """
        Learn hierarchical predictions: units learn to predict their inputs.

        The key insight: slow units (high τ) have smooth activation that
        carries temporal context. They can use this to predict patterns
        in the input they receive from fast units.

        Learning rule:
            dW_hier_ij/dt = lr * input_pred_error_i * A_i

        Where:
            - input_pred_error_i = actual_input_i - predicted_input_i
            - A_i = activation of the receiving unit (the predictor)

        If error > 0 (actual > predicted): increase W_hier (predict more)
        If error < 0 (actual < predicted): decrease W_hier (predict less)

        This creates a predictive hierarchy:
            - Slow units learn to predict their inputs
            - They become "higher level" by modeling patterns in fast units
            - The hierarchy emerges from the timescale gradient
        """
        rows, cols = self._conn_rows, self._conn_cols

        # Error at each receiving unit
        error_i = self.input_pred_error[rows]

        # Activation of receiving unit (the predictor)
        A_predictor = self.A[rows]

        # Learning rule: adjust prediction weights based on error
        # Scale by tau - slow units (high tau) should learn to predict more
        tau_factor = np.sqrt(self.tau[rows])  # Slow units learn predictions more

        dW_hier = self.lr_hier * error_i * A_predictor * tau_factor

        # Update hierarchical prediction weights
        self.W_hier_data += dt * dW_hier
        self.W_hier_data = np.clip(self.W_hier_data, -2.0, 2.0)

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

        # === Fast Hebbian Weight Update ===
        # When input arrives, strengthen connections between co-active input units
        # This stores the input pattern in the weight structure
        rows, cols = self._conn_rows, self._conn_cols

        # Create input pattern vector
        input_pattern = np.zeros(self.n)
        input_pattern[units] = strength

        # Hebbian: dW_ij = pre_j * post_i (strengthen connections within input pattern)
        dW_fast = self.fast_lr * input_pattern[rows] * input_pattern[cols]
        self.W_fast_data += dW_fast

        # Also update input trace (simple leaky integrator)
        self.input_trace[units] += strength

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

    def set_target(self, text_or_id, strength=1.0):
        """
        Set the target activation (what the network should predict).

        This is used for contrastive learning:
        1. Network sees input A, evolves
        2. set_target(B) tells network "you should have predicted B"
        3. Error = B_pattern - current_state drives learning

        The target decays over time, so learning is strongest right after setting.
        """
        if isinstance(text_or_id, str):
            pattern_id = hash(text_or_id) % 10000
        else:
            pattern_id = text_or_id

        # Generate the same sparse pattern as inject would
        np.random.seed(pattern_id)
        n_active = np.random.randint(20, 50)
        units = np.random.choice(self.n, n_active, replace=False)
        np.random.seed(None)

        # Set target as sparse activation
        self.A_target = np.zeros(self.n)
        self.A_target[units] = 1.0

        # Set target strength (how hard to pull toward target)
        self.target_strength = strength

        return {'units': units, 'n_active': n_active}

    def get_prediction_error(self):
        """
        Get the current prediction error: how far is network from target?
        Returns mean absolute error for active target units.
        """
        if self.target_strength < 0.01:
            return 0.0
        active_targets = self.A_target > 0.5
        if not np.any(active_targets):
            return 0.0
        return np.mean(np.abs(self.A_target[active_targets] - self.A[active_targets]))

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
        """Reset network state (but keep learned weights and tau)."""
        self.A = np.zeros(self.n)
        self.theta = np.random.uniform(0, 2*np.pi, self.n)
        self.E = np.ones(self.n)
        self.prediction_error = np.zeros(self.n)
        # Reset target but keep tau (tau is learned structure)
        self.A_target = np.zeros(self.n)
        self.target_strength = 0.0

    def reset_fast_weights(self):
        """Reset fast weights and input trace (for fresh sequence learning)."""
        self.W_fast_data = np.zeros(self._n_connections)
        self.input_trace = np.zeros(self.n)

    def get_tau_stats(self):
        """Get statistics about timescale distribution (for analyzing hierarchy)."""
        return {
            'mean': np.mean(self.tau),
            'std': np.std(self.tau),
            'min': np.min(self.tau),
            'max': np.max(self.tau),
            'median': np.median(self.tau),
            'fast_units': np.sum(self.tau < 0.5),    # Units with τ < 0.5
            'medium_units': np.sum((self.tau >= 0.5) & (self.tau < 2.0)),
            'slow_units': np.sum(self.tau >= 2.0),   # Units with τ >= 2.0
            'tau_range': np.max(self.tau) / (np.min(self.tau) + 1e-8),  # Dynamic range
        }

    def get_hier_stats(self):
        """Get statistics about hierarchical prediction learning."""
        return {
            'W_hier_mean': np.mean(np.abs(self.W_hier_data)),
            'W_hier_max': np.max(np.abs(self.W_hier_data)),
            'W_hier_nonzero': np.sum(np.abs(self.W_hier_data) > 0.01),
            'input_pred_error_mean': np.mean(np.abs(self.input_pred_error)),
        }


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


def benchmark_sequence_prediction():
    """
    Benchmark: Can the model learn and recall sequences?

    Tests:
    1. Learn multiple distinct sequences (A→B→C, D→E→F, etc.)
    2. Test if presenting first item recalls the sequence
    3. Measure accuracy: does the correct next item have highest activation?
    4. Test interference: do sequences stay separate?
    5. Test chaining: A→B→C, does A eventually reach C?
    """
    print("=" * 70)
    print("SEQUENCE PREDICTION BENCHMARK")
    print("=" * 70)

    # Define test sequences - using distinct words to avoid overlap
    sequences = [
        ['cat', 'meow', 'purr'],
        ['dog', 'bark', 'woof'],
        ['bird', 'chirp', 'fly'],
        ['fish', 'swim', 'splash'],
        ['sun', 'bright', 'warm'],
    ]

    print(f"\nTest sequences:")
    for seq in sequences:
        print(f"  {' -> '.join(seq)}")

    # Create model
    mind = PredictiveSOCMind(n_units=1024)

    # Warmup
    print("\nWarming up network...")
    for _ in range(100):
        mind.step(0.1)

    # === TRAINING PHASE ===
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    for seq in sequences:
        print(f"\nLearning sequence: {' -> '.join(seq)}")
        # Learn each transition in the sequence
        for i in range(len(seq) - 1):
            mind.learn_association(seq[i], seq[i+1])

    # Dream to consolidate
    print("\nConsolidating with dreaming (2000 steps)...")
    mind.dream(n_steps=2000, verbose=True)

    # === TESTING PHASE ===
    print("\n" + "=" * 70)
    print("TESTING PHASE")
    print("=" * 70)

    results = {
        'direct_recall': [],      # A→B (direct association)
        'chain_recall': [],       # A→C (through B)
        'interference': [],       # Does dog trigger meow? (should NOT)
    }

    # Test 1: Direct recall (A->B)
    print("\n--- Test 1: Direct Recall (A -> B) ---")
    print(f"{'Trigger':<10} {'Expected':<10} {'Top Recall':<10} {'Correct?':<10} {'Delta':<10}")
    print("-" * 55)

    for seq in sequences:
        trigger = seq[0]
        expected = seq[1]

        recall_result = mind.recall(trigger, n_steps=50)

        # Find which pattern had highest delta (most recalled)
        deltas = recall_result['delta_overlaps']
        if deltas:
            top_pattern = max(deltas.keys(), key=lambda k: deltas[k])
            top_delta = deltas[top_pattern]
            correct = (top_pattern == expected)
            results['direct_recall'].append(correct)

            print(f"{trigger:<10} {expected:<10} {top_pattern:<10} {'Y' if correct else 'N':<10} {top_delta:+.3f}")
        else:
            print(f"{trigger:<10} {expected:<10} {'N/A':<10} {'N':<10}")
            results['direct_recall'].append(False)

    direct_accuracy = sum(results['direct_recall']) / len(results['direct_recall']) * 100
    print(f"\nDirect recall accuracy: {direct_accuracy:.1f}%")

    # Test 2: Chain recall (A->C through B)
    print("\n--- Test 2: Chain Recall (A -> C through B) ---")
    print(f"{'Trigger':<10} {'Expected':<10} {'In Top 3?':<10} {'Delta':<10}")
    print("-" * 45)

    for seq in sequences:
        if len(seq) < 3:
            continue

        trigger = seq[0]
        expected_chain = seq[2]  # The end of the chain

        # Run longer to let chain propagate
        recall_result = mind.recall(trigger, n_steps=100)

        deltas = recall_result['delta_overlaps']
        if deltas and expected_chain in deltas:
            # Check if chain target is in top 3
            sorted_deltas = sorted(deltas.items(), key=lambda x: -x[1])
            top_3 = [x[0] for x in sorted_deltas[:3]]
            in_top_3 = expected_chain in top_3
            delta = deltas[expected_chain]
            results['chain_recall'].append(in_top_3)

            print(f"{trigger:<10} {expected_chain:<10} {'Y' if in_top_3 else 'N':<10} {delta:+.3f}")
        else:
            print(f"{trigger:<10} {expected_chain:<10} {'N/A':<10}")
            results['chain_recall'].append(False)

    chain_accuracy = sum(results['chain_recall']) / len(results['chain_recall']) * 100 if results['chain_recall'] else 0
    print(f"\nChain recall accuracy (in top 3): {chain_accuracy:.1f}%")

    # Test 3: Interference (cross-sequence errors)
    print("\n--- Test 3: Interference Check ---")
    print("Testing if sequences stay separate (trigger should NOT recall other sequences)")
    print(f"{'Trigger':<10} {'Seq':<6} {'Wrong Recalls':<30}")
    print("-" * 50)

    for i, seq in enumerate(sequences):
        trigger = seq[0]
        own_items = set(seq)

        recall_result = mind.recall(trigger, n_steps=50)
        deltas = recall_result['delta_overlaps']

        # Find items from OTHER sequences that were strongly recalled
        wrong_recalls = []
        for pattern, delta in deltas.items():
            if pattern not in own_items and delta > 0.1:  # Threshold for "recalled"
                wrong_recalls.append(f"{pattern}({delta:+.2f})")

        no_interference = len(wrong_recalls) == 0
        results['interference'].append(no_interference)

        wrong_str = ', '.join(wrong_recalls[:3]) if wrong_recalls else 'None'
        print(f"{trigger:<10} {i:<6} {wrong_str:<30}")

    interference_score = sum(results['interference']) / len(results['interference']) * 100
    print(f"\nInterference resistance: {interference_score:.1f}% (higher = better separation)")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Direct recall (A->B):    {direct_accuracy:5.1f}%")
    print(f"  Chain recall (A->C):     {chain_accuracy:5.1f}%")
    print(f"  Interference resistance: {interference_score:5.1f}%")

    overall = (direct_accuracy + chain_accuracy + interference_score) / 3
    print(f"\n  Overall score:           {overall:5.1f}%")
    print("=" * 70)

    # SOC statistics
    print("\nSOC Statistics:")
    soc_stats = mind.get_avalanche_stats()
    if 'mean_size' in soc_stats:
        print(f"  Avalanches: {soc_stats['n_avalanches']}")
        print(f"  Mean/Median ratio: {soc_stats['mean_median_ratio']:.2f} (>2 suggests criticality)")

    return mind, results


def benchmark_contrastive_prediction():
    """
    Test the continuous contrastive learning mechanism.

    This is a REAL prediction test - no pre-stored vocabulary.

    Training:
        1. Show input A
        2. Let network evolve (it makes implicit prediction)
        3. Set target B (this is what SHOULD come next)
        4. Network learns from error: A -> B

    Testing:
        1. Show input A
        2. Let network evolve WITHOUT setting target
        3. Measure how close network gets to B pattern
        4. Compare trained pairs vs untrained pairs
    """
    print("=" * 70)
    print("CONTRASTIVE PREDICTION BENCHMARK")
    print("=" * 70)
    print("\nThis tests REAL prediction (no vocabulary lookup)")
    print("Network learns: input A should predict state B")

    # Training pairs
    train_pairs = [
        ('alpha', 'beta'),
        ('one', 'two'),
        ('up', 'down'),
        ('hot', 'cold'),
        ('left', 'right'),
    ]

    # Untrained pairs (for comparison)
    untrained_pairs = [
        ('alpha', 'cold'),   # Wrong pairing
        ('one', 'down'),     # Wrong pairing
        ('up', 'two'),       # Wrong pairing
    ]

    print(f"\nTraining pairs:")
    for a, b in train_pairs:
        print(f"  {a} -> {b}")

    # Create model
    mind = PredictiveSOCMind(n_units=1024)

    # Warmup
    print("\nWarming up network...")
    for _ in range(100):
        mind.step(0.1)

    # === TRAINING PHASE ===
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    n_epochs = 20
    steps_per_exposure = 10

    for epoch in range(n_epochs):
        epoch_error = 0.0
        for input_word, target_word in train_pairs:
            # 1. Inject input
            mind.inject_text(input_word, strength=0.8)

            # 2. Let network evolve a few steps
            for _ in range(steps_per_exposure):
                mind.step(0.1)

            # 3. Set target (this is what should come next)
            mind.set_target(target_word, strength=1.5)

            # 4. Run more steps with target active (learning happens)
            for _ in range(steps_per_exposure):
                mind.step(0.1)
                epoch_error += mind.get_prediction_error()

            # 5. Clear for next pair
            mind.reset()

        avg_error = epoch_error / (len(train_pairs) * steps_per_exposure)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{n_epochs}: avg prediction error = {avg_error:.4f}")

    # === TESTING PHASE ===
    print("\n" + "=" * 70)
    print("TESTING PHASE")
    print("=" * 70)

    def measure_prediction_overlap(mind, input_word, target_word, n_steps=30):
        """
        Measure how well the network predicts target after seeing input.
        Returns overlap between network state and target pattern.
        """
        mind.reset()

        # Inject input
        mind.inject_text(input_word, strength=0.8)

        # Let network evolve WITHOUT target (pure prediction)
        for _ in range(n_steps):
            mind.step(0.1)

        # Generate target pattern to compare against
        target_id = hash(target_word) % 10000
        np.random.seed(target_id)
        n_active = np.random.randint(20, 50)
        target_units = np.random.choice(mind.n, n_active, replace=False)
        np.random.seed(None)

        target_pattern = np.zeros(mind.n)
        target_pattern[target_units] = 1.0

        # Measure overlap (cosine similarity)
        dot = np.dot(mind.A, target_pattern)
        norm = np.linalg.norm(mind.A) * np.linalg.norm(target_pattern) + 1e-8
        overlap = dot / norm

        return overlap

    # Test trained pairs
    print("\n--- Trained Pairs ---")
    print(f"{'Input':<10} {'Target':<10} {'Overlap':<10}")
    print("-" * 35)

    trained_overlaps = []
    for input_word, target_word in train_pairs:
        overlap = measure_prediction_overlap(mind, input_word, target_word)
        trained_overlaps.append(overlap)
        print(f"{input_word:<10} {target_word:<10} {overlap:.4f}")

    avg_trained = np.mean(trained_overlaps)
    print(f"\nAverage trained overlap: {avg_trained:.4f}")

    # Test untrained pairs
    print("\n--- Untrained Pairs (should be lower) ---")
    print(f"{'Input':<10} {'Target':<10} {'Overlap':<10}")
    print("-" * 35)

    untrained_overlaps = []
    for input_word, target_word in untrained_pairs:
        overlap = measure_prediction_overlap(mind, input_word, target_word)
        untrained_overlaps.append(overlap)
        print(f"{input_word:<10} {target_word:<10} {overlap:.4f}")

    avg_untrained = np.mean(untrained_overlaps)
    print(f"\nAverage untrained overlap: {avg_untrained:.4f}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    ratio = avg_trained / (avg_untrained + 1e-8)
    print(f"  Trained pairs overlap:   {avg_trained:.4f}")
    print(f"  Untrained pairs overlap: {avg_untrained:.4f}")
    print(f"  Signal/Noise ratio:      {ratio:.2f}x")

    if ratio > 1.5:
        print("\n  PASS: Network learned to predict trained pairs better!")
    elif ratio > 1.1:
        print("\n  PARTIAL: Some learning detected, but weak.")
    else:
        print("\n  FAIL: No significant learning detected.")

    print("=" * 70)

    return mind, {'trained': trained_overlaps, 'untrained': untrained_overlaps}


def benchmark_fibonacci():
    """
    Can the network learn the Fibonacci rule?

    f(n) = f(n-1) + f(n-2)

    This is harder than simple A->B because:
    1. It requires combining TWO previous inputs
    2. It's a mathematical pattern, not arbitrary associations
    3. We can test generalization to unseen numbers

    Training approach:
    - Show two consecutive Fibonacci numbers
    - Set target to the next number
    - Network state after seeing (a, b) should predict (a+b)
    """
    print("=" * 70)
    print("FIBONACCI PREDICTION BENCHMARK")
    print("=" * 70)
    print("\nCan the network learn: f(n) = f(n-1) + f(n-2)?")

    # Generate Fibonacci sequence
    fib = [1, 1]
    for _ in range(15):
        fib.append(fib[-1] + fib[-2])
    print(f"\nFibonacci sequence: {fib[:12]}...")

    # Training triplets (a, b) -> c where c = a + b
    train_triplets = []
    for i in range(8):  # Train on first 8 triplets
        train_triplets.append((fib[i], fib[i+1], fib[i+2]))

    # Test triplets (including some not in training)
    test_triplets = []
    for i in range(10):  # Test on more, including unseen
        test_triplets.append((fib[i], fib[i+1], fib[i+2]))

    print(f"\nTraining triplets: {len(train_triplets)}")
    for a, b, c in train_triplets:
        print(f"  ({a}, {b}) -> {c}")

    # Create model
    mind = PredictiveSOCMind(n_units=1024)

    # Warmup
    print("\nWarming up network...")
    for _ in range(100):
        mind.step(0.1)

    # === TRAINING PHASE ===
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)

    n_epochs = 30
    steps_between = 8  # Steps between injections
    steps_with_target = 15  # Steps with target active

    for epoch in range(n_epochs):
        epoch_error = 0.0

        for a, b, c in train_triplets:
            mind.reset()
            mind.reset_fast_weights()  # Fresh memory for each sequence

            # 1. Inject first number
            mind.inject_text(f"fib_{a}", strength=0.8)
            for _ in range(steps_between):
                mind.step(0.1)

            # 2. Inject second number
            mind.inject_text(f"fib_{b}", strength=0.8)
            for _ in range(steps_between):
                mind.step(0.1)

            # 3. Set target (what should come next)
            mind.set_target(f"fib_{c}", strength=1.5)

            # 4. Learn with target active
            for _ in range(steps_with_target):
                mind.step(0.1)
                epoch_error += mind.get_prediction_error()

        avg_error = epoch_error / (len(train_triplets) * steps_with_target)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{n_epochs}: avg prediction error = {avg_error:.4f}")

    # === TESTING PHASE ===
    print("\n" + "=" * 70)
    print("TESTING PHASE")
    print("=" * 70)

    # All candidate answers the network could predict
    all_candidates = list(set(fib[:12]))  # All Fibonacci numbers we know

    def get_pattern_for_number(n, n_units):
        """Generate the sparse pattern for a Fibonacci number."""
        pattern_id = hash(f"fib_{n}") % 10000
        np.random.seed(pattern_id)
        n_active = np.random.randint(20, 50)
        units = np.random.choice(n_units, n_active, replace=False)
        np.random.seed(None)
        pattern = np.zeros(n_units)
        pattern[units] = 1.0
        return pattern

    # === DIAGNOSTIC: Test fast weights memory ===
    print("\n" + "=" * 70)
    print("FAST WEIGHTS DIAGNOSTIC")
    print("=" * 70)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # Test with (1, 2) -> 3
    mind.reset()
    mind.reset_fast_weights()

    pattern_1 = get_pattern_for_number(1, mind.n)
    pattern_2 = get_pattern_for_number(2, mind.n)
    pattern_3 = get_pattern_for_number(3, mind.n)

    print(f"\nTest sequence: (1, 2) -> 3")
    print(f"Pattern overlap 1 vs 2: {cosine_sim(pattern_1, pattern_2):.3f}")
    print(f"Pattern overlap 1 vs 3: {cosine_sim(pattern_1, pattern_3):.3f}")
    print(f"Pattern overlap 2 vs 3: {cosine_sim(pattern_2, pattern_3):.3f}")

    # Inject first number
    print(f"\n1. Inject fib_1:")
    mind.inject_text("fib_1", strength=0.8)
    print(f"   A overlap with pattern_1: {cosine_sim(mind.A, pattern_1):.3f}")
    print(f"   W_fast sum: {mind.W_fast_data.sum():.3f}")
    print(f"   Input trace sum: {mind.input_trace.sum():.3f}")

    for step in range(steps_between):
        mind.step(0.1)

    print(f"\n2. After {steps_between} steps (before inject 2):")
    print(f"   A overlap with pattern_1: {cosine_sim(mind.A, pattern_1):.3f}")
    print(f"   A overlap with pattern_2: {cosine_sim(mind.A, pattern_2):.3f}")
    print(f"   W_fast sum: {mind.W_fast_data.sum():.3f}")
    print(f"   Input trace overlap with pattern_1: {cosine_sim(mind.input_trace, pattern_1):.3f}")

    # Inject second number
    print(f"\n3. Inject fib_2:")
    mind.inject_text("fib_2", strength=0.8)
    print(f"   A overlap with pattern_1: {cosine_sim(mind.A, pattern_1):.3f}")
    print(f"   A overlap with pattern_2: {cosine_sim(mind.A, pattern_2):.3f}")
    print(f"   W_fast sum: {mind.W_fast_data.sum():.3f}")
    print(f"   Input trace sum: {mind.input_trace.sum():.3f}")

    for step in range(steps_between):
        mind.step(0.1)

    print(f"\n4. After {steps_between} more steps (prediction time):")
    print(f"   Current activation (A):")
    print(f"      overlap with pattern_1: {cosine_sim(mind.A, pattern_1):.3f}")
    print(f"      overlap with pattern_2: {cosine_sim(mind.A, pattern_2):.3f}")
    print(f"      overlap with pattern_3: {cosine_sim(mind.A, pattern_3):.3f}")
    print(f"   W_fast sum: {mind.W_fast_data.sum():.3f}")
    print(f"   Input trace:")
    print(f"      overlap with pattern_1: {cosine_sim(mind.input_trace, pattern_1):.3f}")
    print(f"      overlap with pattern_2: {cosine_sim(mind.input_trace, pattern_2):.3f}")
    print(f"      sum: {mind.input_trace.sum():.3f}")

    print("\n" + "=" * 70)

    def predict_fib(mind, a, b, n_steps=40):
        """
        Show (a, b), return which candidate has highest overlap.
        Returns: (predicted_number, correct_number, all_overlaps_dict)
        """
        mind.reset()
        mind.reset_fast_weights()  # Fresh memory for each sequence

        # Inject first number
        mind.inject_text(f"fib_{a}", strength=0.8)
        for _ in range(steps_between):
            mind.step(0.1)

        # Inject second number
        mind.inject_text(f"fib_{b}", strength=0.8)
        for _ in range(n_steps):
            mind.step(0.1)

        # Compute overlap with ALL candidates
        overlaps = {}
        for candidate in all_candidates:
            pattern = get_pattern_for_number(candidate, mind.n)
            dot = np.dot(mind.A, pattern)
            norm = np.linalg.norm(mind.A) * np.linalg.norm(pattern) + 1e-8
            overlaps[candidate] = dot / norm

        # Find winner
        predicted = max(overlaps, key=overlaps.get)
        correct = a + b  # Fibonacci rule

        return predicted, correct, overlaps

    # Test on all triplets
    print("\n--- Fibonacci Predictions (Top-1 Accuracy) ---")
    print(f"{'Input':<12} {'Correct':<10} {'Predicted':<10} {'Match?':<8} {'Confidence':<10}")
    print("-" * 55)

    results = []
    trained_correct = 0
    trained_total = 0
    untrained_correct = 0
    untrained_total = 0

    for i, (a, b, c) in enumerate(test_triplets):
        predicted, correct, overlaps = predict_fib(mind, a, b)
        is_correct = (predicted == correct)
        trained = i < len(train_triplets)

        # Confidence = gap between top prediction and second best
        sorted_overlaps = sorted(overlaps.values(), reverse=True)
        confidence = sorted_overlaps[0] - sorted_overlaps[1] if len(sorted_overlaps) > 1 else 0

        results.append({
            'a': a, 'b': b, 'correct': correct, 'predicted': predicted,
            'is_correct': is_correct, 'trained': trained,
            'overlaps': overlaps, 'confidence': confidence
        })

        if trained:
            trained_total += 1
            if is_correct:
                trained_correct += 1
        else:
            untrained_total += 1
            if is_correct:
                untrained_correct += 1

        match_str = "YES" if is_correct else "NO"
        train_str = "" if trained else "(new)"
        print(f"({a},{b}){train_str:<6} {correct:<10} {predicted:<10} {match_str:<8} {confidence:.3f}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    trained_acc = trained_correct / trained_total * 100 if trained_total > 0 else 0
    untrained_acc = untrained_correct / untrained_total * 100 if untrained_total > 0 else 0
    total_acc = (trained_correct + untrained_correct) / (trained_total + untrained_total) * 100

    print(f"\n  ACCURACY:")
    print(f"    Trained triplets:   {trained_correct}/{trained_total} = {trained_acc:.1f}%")
    print(f"    Untrained triplets: {untrained_correct}/{untrained_total} = {untrained_acc:.1f}%")
    print(f"    Overall:            {trained_correct + untrained_correct}/{trained_total + untrained_total} = {total_acc:.1f}%")

    if untrained_acc > 0:
        print(f"\n  Generalization: Network correctly predicted {untrained_correct} unseen triplets!")

    if trained_acc >= 80 and untrained_acc >= 50:
        print("\n  PASS: Strong learning with generalization!")
    elif trained_acc >= 50:
        print("\n  PARTIAL: Some learning, but weak generalization.")
    else:
        print("\n  FAIL: Network did not learn the pattern.")

    # === Emergent Hierarchy Analysis ===
    print("\n" + "-" * 70)
    print("EMERGENT HIERARCHY (Timescale Distribution)")
    print("-" * 70)

    tau_stats = mind.get_tau_stats()
    print(f"  Tau range:    {tau_stats['min']:.3f} - {tau_stats['max']:.3f}")
    print(f"  Tau mean:     {tau_stats['mean']:.3f} (std: {tau_stats['std']:.3f})")
    print(f"  Dynamic range: {tau_stats['tau_range']:.1f}x")
    print(f"\n  Fast units (tau < 0.5):   {tau_stats['fast_units']}")
    print(f"  Medium units (0.5-2.0):   {tau_stats['medium_units']}")
    print(f"  Slow units (tau >= 2.0):  {tau_stats['slow_units']}")

    if tau_stats['tau_range'] > 10:
        print("\n  Strong timescale separation detected (potential hierarchy)!")
    elif tau_stats['tau_range'] > 3:
        print("\n  Moderate timescale separation detected.")
    else:
        print("\n  Weak timescale separation (no clear hierarchy).")

    # Hierarchical prediction stats
    print("\n" + "-" * 70)
    print("HIERARCHICAL PREDICTION")
    print("-" * 70)

    hier_stats = mind.get_hier_stats()
    print(f"  W_hier mean magnitude:   {hier_stats['W_hier_mean']:.4f}")
    print(f"  W_hier max magnitude:    {hier_stats['W_hier_max']:.4f}")
    print(f"  Active predictions:      {hier_stats['W_hier_nonzero']}")
    print(f"  Input pred error:        {hier_stats['input_pred_error_mean']:.4f}")

    print("=" * 70)

    return mind, results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_prediction_learning()
        elif sys.argv[1] == '--sequence':
            demo_sequence_learning()
        elif sys.argv[1] == '--benchmark':
            benchmark_sequence_prediction()
        elif sys.argv[1] == '--contrastive':
            benchmark_contrastive_prediction()
        elif sys.argv[1] == '--fibonacci':
            benchmark_fibonacci()
    else:
        print("Starting interactive mode...")
        mind = PredictiveSOCMind(n_units=1024)

        # Brief warmup
        for _ in range(100):
            mind.step(0.1)

        viz = InteractiveVisualizer(mind)
        viz.run()
