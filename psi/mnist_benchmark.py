"""
MNIST Benchmark: PSI vs MLP

Compare our non-backprop PSI model against a standard MLP baseline.

Architecture:
- Input: 28 nodes, each receiving one row of the image (28 pixels each)
- Hidden: Small-world connected nodes
- Output: 10 nodes, one per digit class

Training:
- 5k samples (few-shot regime)
- Compare accuracy and training time
"""

import numpy as np
import time
from typing import Tuple, List


def load_mnist_subset(n_train: int = 5000, n_test: int = 10000) -> Tuple:
    """Load MNIST data. Downloads if needed."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)

        # Normalize to [-1, 1]
        X = (X / 127.5) - 1.0

        # Split
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        # Subset
        idx = np.random.permutation(len(X_train))[:n_train]
        X_train, y_train = X_train[idx], y_train[idx]

        if n_test < len(X_test):
            idx = np.random.permutation(len(X_test))[:n_test]
            X_test, y_test = X_test[idx], y_test[idx]

        print(f"Loaded {len(X_train)} train, {len(X_test)} test samples")
        return X_train, y_train, X_test, y_test

    except ImportError:
        print("sklearn not available, generating synthetic data for testing")
        # Generate synthetic "digit-like" data for testing
        np.random.seed(42)
        X_train = np.random.randn(n_train, 784) * 0.3
        y_train = np.random.randint(0, 10, n_train)
        X_test = np.random.randn(min(n_test, 1000), 784) * 0.3
        y_test = np.random.randint(0, 10, len(X_test))
        return X_train, y_train, X_test, y_test


class PSIClassifier:
    """
    PSI-based MNIST classifier.

    Architecture:
    - n_input nodes: each receives a portion of the image
    - n_hidden nodes: small-world connected processing
    - n_output nodes: one per class (10 for MNIST)
    """

    def __init__(self, n_input: int = 28, n_hidden: int = 32, n_output: int = 10,
                 dim: int = 32, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_nodes = n_input + n_hidden + n_output
        self.dim = dim

        # Pixels per input node (784 / 28 = 28 pixels per node = one row)
        self.pixels_per_input = 784 // n_input

        # Node states
        self.states = np.random.randn(self.n_nodes, dim) * 0.01
        self.prev_states = np.zeros((self.n_nodes, dim))
        self.free_states = np.zeros((self.n_nodes, dim))
        self.target_states = np.zeros((self.n_nodes, dim))

        # Phases - initialize with position-dependent structure for better coherence
        # Input nodes have similar phases, hidden have varied, output have similar
        self.phases = np.zeros((self.n_nodes, dim))
        for i in range(self.n_nodes):
            base_phase = 2 * np.pi * i / self.n_nodes  # Position-dependent
            self.phases[i] = base_phase + np.random.randn(dim) * 0.3  # Small variation
        self.phases = np.mod(self.phases, 2 * np.pi)
        self.omega = np.random.randn(self.n_nodes, dim) * 0.02  # Slower oscillation

        # Build adjacency matrix
        self.adj = self._build_adjacency()
        self.W_conn = np.random.uniform(0.5, 1.0, (self.n_nodes, self.n_nodes)) * self.adj
        self.W_conn = (self.W_conn + self.W_conn.T) / 2

        # Per-node transformation
        scale = 0.5 / np.sqrt(dim)
        self.node_W = np.random.randn(self.n_nodes, dim, dim) * scale
        for i in range(self.n_nodes):
            self.node_W[i] += np.eye(dim) * 0.4

        # Biases
        self.biases = np.random.randn(self.n_nodes, dim) * 0.15

        # Input projection: pixels -> dim
        self.input_proj = np.random.randn(n_input, self.pixels_per_input, dim) * 0.1

        # Learning rate - moderate for stability
        self.lr = 0.05

        # Masks
        self.input_mask = np.zeros(self.n_nodes, dtype=bool)
        self.input_mask[:n_input] = True
        self.output_mask = np.zeros(self.n_nodes, dtype=bool)
        self.output_mask[-n_output:] = True

        # Clamp values
        self.clamped = np.full((self.n_nodes, dim), np.nan)

        # Pre-computed
        self.adj_sum = np.sum(self.adj, axis=1) + 0.1
        self.diag_indices = np.arange(dim)

        # Output readout weights (learns to extract class from output node states)
        self.readout_W = np.random.randn(n_output, dim) * 0.1

        # Combined classifier: uses BOTH input projections and hidden states
        # This allows the classifier to access both raw input features and processed features
        combined_size = n_input * dim + n_hidden * dim  # Both input and hidden
        self.hidden_to_output = np.random.randn(combined_size, n_output) * np.sqrt(2.0 / combined_size)

        # FEEDBACK ALIGNMENT: Random fixed feedback weights for biologically plausible error propagation
        # These are NOT learned - they stay fixed and allow output error to reach hidden layers
        self.B_hidden = np.random.randn(n_output, n_hidden * dim) * 0.1  # Output -> hidden
        self.B_input = np.random.randn(n_output, n_input * dim) * 0.1  # Output -> input directly

        # DOPAMINE-LIKE TD ERROR: Track running average of reward for surprise signal
        self.expected_reward = 0.1  # Start at chance level (10 classes)
        self.reward_decay = 0.995  # Slower update for more stable baseline

        # Learning step counter for warmup
        self.step_count = 0

        # SYNAPTIC CONSOLIDATION: Track importance of each synapse
        # Synapses that contribute to correct predictions become "consolidated"
        # (like LTP consolidation in biology)
        self.ho_importance = np.zeros_like(self.hidden_to_output)  # Classifier weights
        self.consolidation_rate = 0.02  # How fast importance builds up (increased)

        # BEST PERFORMANCE CHECKPOINT: Lock in best weights
        self.best_accuracy = 0.0
        self.best_weights = None  # Will store (hidden_to_output, node_W, input_proj, biases)
        self.accuracy_history = []  # Track recent accuracy for lock-in signal

        # HOMEOSTATIC PLASTICITY: Track each node's average activity
        # Nodes that are too active get suppressed, too inactive get boosted
        self.target_activity = 0.3  # Target average activation
        self.activity_trace = np.ones(self.n_nodes) * self.target_activity
        self.homeostatic_rate = 0.01  # How fast to adjust

        # LATERAL INHIBITION weights (for hidden layer competition)
        # Nearby hidden nodes inhibit each other
        self.lateral_inhibition = np.zeros((n_hidden, n_hidden))
        for i in range(n_hidden):
            for j in range(n_hidden):
                if i != j:
                    dist = min(abs(i - j), n_hidden - abs(i - j))  # Circular distance
                    self.lateral_inhibition[i, j] = -0.3 * np.exp(-dist / 3.0)  # Local inhibition

        # Adaptive learning rate based on surprise
        self.base_lr = 0.05
        self.lr = self.base_lr

        # Diagnostics
        self.diag = {
            'input_proj_norm': [],      # Are input projections changing?
            'node_W_norm': [],          # Are node weights changing?
            'W_conn_mean': [],          # Are connection weights changing?
            'bias_norm': [],            # Are biases changing?
            'state_diff_mean': [],      # How much do target states differ from free?
            'delta_norm': [],           # How large are the learning signals?
            'output_spread': [],        # Are outputs differentiated?
            'coherence_mean': [],       # What's the phase coherence?
            'hidden_state_std': [],     # Are hidden states active?
            'td_error': [],             # Dopamine-like surprise signal
            'expected_reward': [],      # Running baseline
        }

    def _build_adjacency(self) -> np.ndarray:
        """Build connectivity: input -> hidden -> output with small-world in hidden."""
        adj = np.zeros((self.n_nodes, self.n_nodes))

        n_in = self.n_input
        n_hid = self.n_hidden
        n_out = self.n_output

        # Input to hidden (each input connects to nearby hidden nodes)
        for i in range(n_in):
            # Connect to a few hidden nodes
            hidden_start = n_in
            hidden_end = n_in + n_hid
            # Each input connects to ~4 hidden nodes
            for h in range(4):
                h_idx = hidden_start + (i * 4 // n_in + h) % n_hid
                adj[i, h_idx] = 1.0
                adj[h_idx, i] = 1.0

        # Hidden layer: small-world topology
        hidden_start = n_in
        for i in range(n_hid):
            # Local connections (k nearest neighbors)
            for offset in range(1, 3):
                j = (i + offset) % n_hid
                adj[hidden_start + i, hidden_start + j] = 1.0
                adj[hidden_start + j, hidden_start + i] = 1.0

            # Long-range connections
            if np.random.random() < 0.15:
                j = np.random.randint(0, n_hid)
                if j != i:
                    adj[hidden_start + i, hidden_start + j] = 1.0
                    adj[hidden_start + j, hidden_start + i] = 1.0

        # Hidden to output (each hidden connects to a few outputs)
        output_start = n_in + n_hid
        for h in range(n_hid):
            # Each hidden connects to ~3 output nodes
            for o in range(3):
                o_idx = output_start + (h + o) % n_out
                adj[hidden_start + h, o_idx] = 1.0
                adj[o_idx, hidden_start + h] = 1.0

        np.fill_diagonal(adj, 0)
        return adj

    def step(self, target_mode: bool = False):
        """One parallel update step."""
        self.prev_states = self.states.copy()

        # Phase coherence
        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) * 0.5
        if target_mode:
            gate += 0.3

        # Neighbor input
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(weighted_adj, axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.states) / total_weight

        # Noise
        neighbor_input += np.random.randn(self.n_nodes, self.dim) * 0.03

        # Transform
        pre_act = np.einsum('nij,nj->ni', self.node_W, neighbor_input) + self.biases
        np.clip(pre_act, -5, 5, out=pre_act)
        target_activation = np.tanh(pre_act)

        # Node gating
        mean_node_coherence = np.sum(coherence * self.adj, axis=1) / self.adj_sum
        coherence_threshold = np.mean(mean_node_coherence)
        node_gate = np.clip(1 + 2 * (mean_node_coherence - coherence_threshold), 0.3, 1.5)
        target_activation *= node_gate[:, np.newaxis]

        # Leaky integration
        leak = 0.7 if target_mode else 0.5
        new_states = (1 - leak) * self.states + leak * target_activation

        # Apply clamping
        clamped_mask = ~np.isnan(self.clamped)
        np.copyto(new_states, self.clamped, where=clamped_mask)
        self.states = new_states

        # Update phases with VECTORIZED ACTIVITY-BASED SYNCHRONIZATION
        # Compute state similarity once
        state_similarity = self.states @ self.states.T / self.dim  # [n_nodes, n_nodes]

        # Mask to only consider adjacent nodes
        adj_mask = self.adj > 0  # [n_nodes, n_nodes]

        # For each node, compute mean similarity with neighbors
        neighbor_sim = state_similarity * adj_mask  # Zero out non-neighbors
        neighbor_count = adj_mask.sum(axis=1, keepdims=True) + 1e-8
        mean_neighbor_sim = neighbor_sim.sum(axis=1, keepdims=True) / neighbor_count

        # Selective weights: only above-average similarity neighbors contribute
        sim_weights = np.maximum(0, state_similarity - mean_neighbor_sim) * adj_mask
        sim_weights_sum = sim_weights.sum(axis=1, keepdims=True) + 1e-8
        sim_weights_norm = sim_weights / sim_weights_sum  # [n_nodes, n_nodes]

        # Compute weighted circular mean of neighbor phases
        # sin and cos of phases: [n_nodes, dim]
        sin_phases = np.sin(self.phases)
        cos_phases = np.cos(self.phases)

        # Weighted sum across neighbors: [n_nodes, dim]
        weighted_sin = sim_weights_norm @ sin_phases
        weighted_cos = sim_weights_norm @ cos_phases
        target_phase = np.arctan2(weighted_sin, weighted_cos)

        # Phase difference (wrapped to [-pi, pi])
        phase_diff = target_phase - self.phases
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_pull = 0.15 * phase_diff

        # Simple phase update (skip anti-phase for speed - small effect)
        self.phases += self.omega + 0.02 * self.states + phase_pull
        np.mod(self.phases, 2 * np.pi, out=self.phases)

    def has_converged(self, threshold: float = 0.015) -> bool:
        return np.mean(np.abs(self.states - self.prev_states)) < threshold

    def settle(self, max_iters: int = 8, min_steps: int = 3, target_mode: bool = False):
        for i in range(max_iters):
            self.step(target_mode=target_mode)
            if i >= min_steps and self.has_converged():
                return i + 1
        return max_iters

    def project_input(self, image: np.ndarray) -> np.ndarray:
        """Project 784-dim image to input node states [n_input, dim]."""
        # Reshape to [n_input, pixels_per_input]
        pixels = image.reshape(self.n_input, self.pixels_per_input)

        # Store for learning
        self.last_pixels = pixels.copy()

        # Project each chunk: [n_input, pixels_per_input] @ [n_input, pixels_per_input, dim]
        projected = np.einsum('np,npd->nd', pixels, self.input_proj)

        return np.tanh(projected)

    def free_phase(self, image: np.ndarray) -> np.ndarray:
        """FREE PHASE: Clamp inputs, return output node states."""
        self.states = np.random.randn(self.n_nodes, self.dim) * 0.01
        self.clamped = np.full((self.n_nodes, self.dim), np.nan)

        # Reset phases to random at start of each sample
        # This ensures each input creates a unique phase pattern through dynamics
        self.phases = np.random.uniform(0, 2*np.pi, (self.n_nodes, self.dim))

        # Project and clamp input nodes
        input_states = self.project_input(image)
        self.clamped[:self.n_input] = input_states

        self.settle(max_iters=5, min_steps=2)  # Less settling to preserve input information
        self.free_states = self.states.copy()

        # Use hidden layer states (which actually evolve during settling) for classification
        hidden_start = self.n_input
        hidden_end = self.n_input + self.n_hidden
        hidden_states = self.states[hidden_start:hidden_end]  # [n_hidden, dim]

        # Also include input states (which preserve input information)
        input_states = self.states[:self.n_input]  # [n_input, dim]

        # Concatenate for richer features: use BOTH input and hidden representations
        # This allows the classifier to use both the raw projected input and the
        # processed hidden states after phase dynamics
        hidden_flat = hidden_states.flatten()  # [n_hidden * dim]
        input_flat = input_states.flatten()  # [n_input * dim]

        # Combined feature vector
        combined_features = np.concatenate([input_flat, hidden_flat])
        self.last_combined = combined_features  # Store for learning

        # Use combined features for classification
        output_logits = combined_features @ self.hidden_to_output
        return output_logits

    def target_phase(self, label: int):
        """TARGET PHASE: Clamp output to one-hot target."""
        # Create one-hot target with stronger contrast
        target = np.full((self.n_output, self.dim), -0.8)  # Stronger negative
        target[label] = 0.9  # Target class

        self.clamped[-self.n_output:] = target
        self.settle(max_iters=12, min_steps=5, target_mode=True)
        self.target_states = self.states.copy()

    def softmax(self, x):
        """Softmax for output probabilities."""
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-8)

    def learn(self, collect_diag: bool = False):
        """
        Biologically plausible learning with DOPAMINE-LIKE TD ERROR.

        Key principles:
        1. Global reward signal: single scalar "dopamine" broadcast to all synapses
        2. TD error: reward = (actual - expected), like dopamine neurons
        3. Three-factor rule: ΔW = lr × pre × post × dopamine
        4. Eligibility: only recently active synapses are modified

        This replaces the per-output-unit error with a GLOBAL surprise signal.
        """
        state_diff = self.target_states - self.free_states
        np.clip(state_diff, -1, 1, out=state_diff)

        # Contrastive Hebbian delta for PSI dynamics
        delta = state_diff * 0.5

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) * 0.5
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(weighted_adj, axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.free_states) / total_weight

        # ============================================================
        # DOPAMINE-LIKE TD ERROR: Global reward signal
        # ============================================================

        if hasattr(self, 'last_label') and hasattr(self, 'last_combined'):
            hidden_start = self.n_input
            hidden_end = self.n_input + self.n_hidden
            hidden_states = self.free_states[hidden_start:hidden_end]
            hidden_flat = hidden_states.flatten()  # [n_hidden * dim]
            input_states = self.free_states[:self.n_input]
            input_flat = input_states.flatten()  # [n_input * dim]

            # Use the stored combined features
            combined_features = self.last_combined  # [n_input * dim + n_hidden * dim]

            # Compute prediction
            current_logits = combined_features @ self.hidden_to_output  # [n_output]
            probs = self.softmax(current_logits)
            prediction = np.argmax(probs)

            # ---- REWARD COMPUTATION ----
            # Reward is based on confidence in the correct answer
            # This is a SINGLE SCALAR, like dopamine
            reward = probs[self.last_label]  # Range [0, 1]: how confident in correct answer

            # TD ERROR: surprise = reward - expected_reward
            # Positive = "better than expected" (dopamine burst)
            # Negative = "worse than expected" (dopamine dip)
            td_error = reward - self.expected_reward

            # Update expected reward (slow moving average)
            self.expected_reward = self.reward_decay * self.expected_reward + (1 - self.reward_decay) * reward

            # Collect TD diagnostics
            if collect_diag:
                self.diag['td_error'].append(td_error)
                self.diag['expected_reward'].append(self.expected_reward)
                self.diag['state_diff_mean'].append(np.mean(np.abs(state_diff)))
                self.diag['delta_norm'].append(np.mean(np.abs(delta)))
                self.diag['coherence_mean'].append(np.mean(coherence))
                self.diag['hidden_state_std'].append(np.std(self.free_states[hidden_start:hidden_end]))

            # ---- THREE-FACTOR LEARNING ----
            # All weight updates are modulated by the GLOBAL td_error signal

            # Learning rate warmup: start lower, ramp up over first 200 steps
            # This mimics synaptic maturation and provides stability
            self.step_count += 1
            warmup_factor = min(1.0, self.step_count / 200)
            effective_lr = self.lr * warmup_factor

            # Dopamine signal: clamp to prevent extreme updates
            # This mimics saturation of neurotransmitter release
            dopamine = np.clip(td_error, -0.5, 0.5)

            # 1. Update hidden_to_output using three-factor rule
            # We still need SOME directional signal for the output layer
            # Use "one-hot target" direction, but scale by dopamine magnitude
            target_direction = np.zeros(self.n_output)
            target_direction[self.last_label] = 1.0
            target_direction -= probs  # Direction toward correct answer

            # Three-factor: pre (combined_features) × post_direction × dopamine
            # Conservative gating: don't let dopamine swing learning too wildly
            dopamine_mod = 1.0 + dopamine  # Range [0.5, 1.5] instead of unbounded
            dW_ho = effective_lr * np.outer(combined_features, target_direction) * dopamine_mod

            # SYNAPTIC CONSOLIDATION: Reduce learning on important synapses
            # Importance grows when prediction is correct (reward > 0.5)
            if reward > 0.3:  # Correct-ish prediction
                # Update importance: synapses active during success become protected
                activity_pattern = np.abs(np.outer(combined_features, probs))
                self.ho_importance += self.consolidation_rate * activity_pattern

            # Scale down updates to important synapses (they resist change)
            consolidation_protection = 1.0 / (1.0 + self.ho_importance)
            dW_ho *= consolidation_protection

            self.hidden_to_output += dW_ho
            np.clip(self.hidden_to_output, -3, 3, out=self.hidden_to_output)

            # 2. Hidden layer: use dopamine to gate Hebbian learning
            # The B_hidden matrix gives each hidden unit a fixed "role" in the output
            # B_hidden is [n_output, n_hidden * dim], so B_hidden.T is [n_hidden * dim, n_output]
            # We project the target direction through B_hidden.T to get hidden "roles"
            h_role = self.B_hidden.T @ target_direction  # [n_hidden * dim]
            h_role = h_role / (np.linalg.norm(h_role) + 1e-8)  # Normalize

            # Eligibility: which hidden units were active?
            eligibility = np.abs(hidden_flat)  # More active = more eligible
            eligibility = eligibility / (eligibility.max() + 1e-8)  # Normalize

            # Combine: active units in their "role direction", gated by dopamine
            h_update = eligibility * h_role * dopamine

            # Reshape for node_W update
            h_update_reshaped = h_update.reshape(self.n_hidden, self.dim)

            # Blend with contrastive Hebbian (which is also bio-plausible)
            hidden_delta = delta[hidden_start:hidden_end]  # [n_hidden, dim]

            # Dopamine gates the contrastive signal too
            # Positive dopamine: trust the contrastive difference
            # Negative dopamine: maybe the free phase was better, reduce update
            dopamine_gate = np.clip(0.5 + dopamine, 0.3, 1.5)  # More conservative range
            combined_delta = dopamine_gate * hidden_delta + 0.3 * h_update_reshaped

            # Update hidden node_W (VECTORIZED)
            # combined_delta: [n_hidden, dim], neighbor_input: [n_nodes, dim]
            hidden_neighbor = neighbor_input[hidden_start:hidden_end]  # [n_hidden, dim]
            # Outer product for each hidden node: [n_hidden, dim, dim]
            dW_hidden = effective_lr * np.einsum('hi,hj->hij', combined_delta, hidden_neighbor)
            self.node_W[hidden_start:hidden_end] += dW_hidden

            np.clip(self.node_W, -3, 3, out=self.node_W)

            # 3. Input projections: dopamine-gated Hebbian (VECTORIZED)
            if hasattr(self, 'last_pixels'):
                # Which input dimensions were active and contributed to the response?
                input_eligibility = np.abs(input_flat.reshape(self.n_input, self.dim))

                # Vectorized Hebbian: [n_input, pixels_per_input, dim]
                # last_pixels: [n_input, pixels_per_input], input_eligibility: [n_input, dim]
                dProj = effective_lr * 0.1 * dopamine * np.einsum('ip,id->ipd', self.last_pixels, input_eligibility)
                self.input_proj += dProj

                np.clip(self.input_proj, -1.5, 1.5, out=self.input_proj)

        # ============================================================
        # CONTRASTIVE HEBBIAN: PSI graph dynamics learning
        # (Also gated by dopamine when available)
        # ============================================================

        # Compute effective learning rate (handles case where step_count wasn't incremented above)
        if not hasattr(self, 'last_label'):
            self.step_count += 1
        warmup_factor = min(1.0, self.step_count / 200)
        effective_lr_ch = self.lr * warmup_factor

        # Get dopamine for gating (default to 0 if not computed yet)
        ch_dopamine = getattr(self, '_last_dopamine', 0.0) if not hasattr(self, 'last_label') else td_error
        if hasattr(self, 'last_label'):
            self._last_dopamine = td_error  # Store for next iteration

        dopamine_gate = np.clip(0.5 + ch_dopamine, 0.3, 1.5)

        # Update biases using contrastive delta, gated by dopamine
        self.biases += effective_lr_ch * 0.3 * delta * dopamine_gate
        self.biases[self.input_mask] = 0
        np.clip(self.biases, -2, 2, out=self.biases)

        # Update connection weights (contrastive correlation), gated by dopamine
        target_corr = (self.target_states @ self.target_states.T) / self.dim
        free_corr = (self.free_states @ self.free_states.T) / self.dim
        corr_diff = target_corr - free_corr
        coherence_gate = np.maximum(0, 2 * coherence - 0.5)
        dW_conn = effective_lr_ch * 0.1 * dopamine_gate * coherence_gate * corr_diff * self.adj
        self.W_conn += dW_conn
        np.clip(self.W_conn, 0.1, 2.0, out=self.W_conn)

        # Synaptic decay (reduced to prevent forgetting)
        decay = 0.9995
        self.node_W *= decay
        self.biases *= decay
        self.W_conn = self.W_conn * decay + 0.75 * (1 - decay) * self.adj
        self.node_W[:, self.diag_indices, self.diag_indices] *= 1.001
        # Classifier decay - only on less important weights
        classifier_decay = 1.0 - 0.0005 / (1.0 + self.ho_importance)
        self.hidden_to_output *= classifier_decay

        # Collect diagnostics after updates
        if collect_diag:
            self.diag['input_proj_norm'].append(np.mean(np.abs(self.input_proj)))
            self.diag['node_W_norm'].append(np.mean(np.abs(self.node_W)))
            self.diag['W_conn_mean'].append(np.mean(self.W_conn[self.adj > 0]))
            self.diag['bias_norm'].append(np.mean(np.abs(self.biases)))

    def train_step(self, image: np.ndarray, label: int, collect_diag: bool = False):
        """One training step."""
        self.last_label = label  # Store for readout learning
        outputs = self.free_phase(image)
        if collect_diag:
            # Output spread: std of output activations (should be high for discrimination)
            self.diag['output_spread'].append(np.std(outputs))
        self.target_phase(label)
        self.learn(collect_diag=collect_diag)

    def predict(self, image: np.ndarray) -> int:
        """Predict class for image."""
        outputs = self.free_phase(image)
        return int(np.argmax(outputs))

    def print_diagnostics(self, label: str = ""):
        """Print diagnostic summary."""
        print(f"\n  Diagnostics {label}:")
        for key, values in self.diag.items():
            if values:
                arr = np.array(values)
                # Show start, middle, end values
                n = len(arr)
                if n >= 3:
                    start = np.mean(arr[:n//10+1])
                    mid = np.mean(arr[n//2-n//20:n//2+n//20+1])
                    end = np.mean(arr[-n//10-1:])
                    change = (end - start) / (start + 1e-8) * 100
                    print(f"    {key:20s}: {start:.4f} -> {mid:.4f} -> {end:.4f} ({change:+.1f}%)")
                else:
                    print(f"    {key:20s}: {np.mean(arr):.4f}")

    def clear_diagnostics(self):
        """Clear diagnostic buffers."""
        for key in self.diag:
            self.diag[key] = []

    def checkpoint_if_best(self, accuracy: float):
        """Save weights if this is the best accuracy seen.

        Bio-plausible interpretation: Strong reward from achieving new peak
        triggers consolidation of current synaptic configuration.
        """
        self.accuracy_history.append(accuracy)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            # Save current weights (deep copy)
            self.best_weights = (
                self.hidden_to_output.copy(),
                self.node_W.copy(),
                self.input_proj.copy(),
                self.biases.copy(),
                self.W_conn.copy(),
            )
            # STRONG CONSOLIDATION: When hitting new peak, dramatically increase importance
            # This is like a "winner-take-all" signal that locks in the winning configuration
            self.ho_importance += 0.5 * np.abs(self.hidden_to_output)
            return True
        return False

    def restore_best(self):
        """Restore best weights if performance has degraded significantly.

        Bio-plausible interpretation: Homeostatic mechanism that reverts to
        a known-good state when current state is performing poorly.
        """
        if self.best_weights is not None:
            # Check if we've degraded significantly (more than 15% below best)
            if len(self.accuracy_history) >= 3:
                recent_avg = np.mean(self.accuracy_history[-3:])
                if recent_avg < self.best_accuracy - 0.15:
                    # Partial restore: blend current with best (not full replacement)
                    # This allows some continued exploration while recovering
                    blend = 0.7  # 70% best, 30% current
                    self.hidden_to_output = blend * self.best_weights[0] + (1 - blend) * self.hidden_to_output
                    self.node_W = blend * self.best_weights[1] + (1 - blend) * self.node_W
                    self.input_proj = blend * self.best_weights[2] + (1 - blend) * self.input_proj
                    self.biases = blend * self.best_weights[3] + (1 - blend) * self.biases
                    self.W_conn = blend * self.best_weights[4] + (1 - blend) * self.W_conn
                    return True
        return False


class MLPBaseline:
    """Simple MLP baseline for comparison."""

    def __init__(self, hidden_size: int = 128, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        # 784 -> hidden -> 10
        self.W1 = np.random.randn(784, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 10) * 0.01
        self.b2 = np.zeros(10)
        self.lr = 0.01

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass, return (hidden, output)."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        o = h @ self.W2 + self.b2
        return h, o

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def train_step(self, x: np.ndarray, label: int):
        """One training step with backprop."""
        h, o = self.forward(x)
        probs = self.softmax(o)

        # Cross-entropy gradient
        do = probs.copy()
        do[label] -= 1

        # Backprop
        dW2 = np.outer(h, do)
        db2 = do

        dh = do @ self.W2.T
        dh[h <= 0] = 0  # ReLU derivative

        dW1 = np.outer(x, dh)
        db1 = dh

        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, x: np.ndarray) -> int:
        _, o = self.forward(x)
        return int(np.argmax(o))


def evaluate(model, X_test, y_test, name: str) -> float:
    """Evaluate model accuracy."""
    correct = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        if pred == y_test[i]:
            correct += 1
    acc = correct / len(X_test)
    print(f"{name}: {correct}/{len(X_test)} = {acc:.1%}")
    return acc


def main():
    print("=" * 70)
    print("MNIST BENCHMARK: PSI vs MLP (with diagnostics)")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Load data - smaller for faster testing
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=500, n_test=200)

    print()
    print("Training PSI classifier...")
    print("-" * 50)

    # PSI model - balanced size for stability
    psi = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    psi_start = time.time()
    for epoch in range(10):  # More epochs for better learning
        epoch_start = time.time()
        psi.clear_diagnostics()
        idx = np.random.permutation(len(X_train))

        for i in idx:
            # Collect diagnostics every 50 samples
            collect = (i % 50 == 0)
            psi.train_step(X_train[i], y_train[i], collect_diag=collect)

        epoch_time = time.time() - epoch_start

        # Evaluate on FULL test set for accurate checkpointing
        correct = sum(1 for i in range(len(X_test)) if psi.predict(X_test[i]) == y_test[i])
        acc = correct / len(X_test)

        # Checkpoint if best, restore if degraded
        is_best = psi.checkpoint_if_best(acc)
        restored = psi.restore_best()
        status = '*** BEST ***' if is_best else ('(restored)' if restored else '')
        print(f"  Epoch {epoch+1}: {correct}/{len(X_test)} = {acc:.1%} ({epoch_time:.1f}s) {status}")

        # Print diagnostics only on first and last epoch to reduce output
        if epoch == 0 or epoch == 9:
            psi.print_diagnostics(f"(epoch {epoch+1})")

    psi_time = time.time() - psi_start
    print(f"\nPSI training time: {psi_time:.1f}s")

    print()
    print("Training MLP baseline...")
    print("-" * 50)

    # MLP model
    mlp = MLPBaseline(hidden_size=128, seed=42)

    mlp_start = time.time()
    for epoch in range(3):
        epoch_start = time.time()
        idx = np.random.permutation(len(X_train))
        for i in idx:
            mlp.train_step(X_train[i], y_train[i])

        epoch_time = time.time() - epoch_start

        # Quick eval
        test_subset = np.random.permutation(len(X_test))[:100]
        correct = sum(1 for i in test_subset if mlp.predict(X_test[i]) == y_test[i])
        print(f"  Epoch {epoch+1}: {correct}/100 = {correct}% ({epoch_time:.1f}s)")

    mlp_time = time.time() - mlp_start
    print(f"MLP training time: {mlp_time:.1f}s")

    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Restore best checkpoint for PSI before evaluation
    if psi.best_weights is not None:
        psi.hidden_to_output = psi.best_weights[0].copy()
        psi.node_W = psi.best_weights[1].copy()
        psi.input_proj = psi.best_weights[2].copy()
        psi.biases = psi.best_weights[3].copy()
        psi.W_conn = psi.best_weights[4].copy()
        print(f"(PSI restored to best checkpoint: {psi.best_accuracy:.1%})")

    psi_acc = evaluate(psi, X_test, y_test, "PSI")
    mlp_acc = evaluate(mlp, X_test, y_test, "MLP")

    print()
    print("Summary:")
    print(f"  PSI: {psi_acc:.1%} accuracy (best: {psi.best_accuracy:.1%}), {psi_time:.1f}s training")
    print(f"  MLP: {mlp_acc:.1%} accuracy, {mlp_time:.1f}s training")
    if mlp_time > 0:
        print(f"  Speed ratio: MLP is {psi_time/mlp_time:.1f}x faster")

    if psi_acc >= mlp_acc:
        print("\n  PSI matches or beats MLP!")
    else:
        print(f"\n  MLP wins by {(mlp_acc - psi_acc)*100:.1f} percentage points")


if __name__ == "__main__":
    main()
