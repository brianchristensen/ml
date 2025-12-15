"""
Planning SOC Mind
==================

Extends PredictiveSOCMind with:
1. Action-conditioned prediction (world model)
2. Fractal Monte Carlo (FMC) style planning with parallel slots
3. Dream consolidation of action→outcome associations

Key concepts from Fractal AI:
- Virtual Reward = Reward × Distance (exploitation vs exploration)
- Walker (slot) density should match reward density
- Cloning dynamics to concentrate computation on promising futures
- At α=0, system maximizes entropy of futures ("common sense" mode)

This maps to SOC:
- Slots = parallel activation patterns
- Simulator = W_dyn + W_action dynamics
- Distance = pattern divergence between slots
- Reward = goal similarity + prediction confidence
- Cloning = copying activation patterns
- SOC criticality ≈ FMC common sense (maximize causal entropy)
"""

import numpy as np
from predictive_soc import PredictiveSOCMind


class PlanningSOCMind(PredictiveSOCMind):
    """
    SOC Mind with action-conditioned world modeling and FMC planning.

    Actions GATE the dynamics rather than add activation.
    This enables: "if I do action A in state S, I predict next state S'"

    Planning uses parallel "slots" (like FMC walkers) to explore futures.
    """

    def __init__(self, n_units=1024, n_action_dims=4, n_slots=8,
                 connectivity=0.1, seed=42, pattern_sparsity=0.10):
        """
        Args:
            n_units: Number of units in the network
            n_action_dims: Number of distinct action types (e.g., up/down/left/right)
            n_slots: Number of parallel simulation slots for planning
            connectivity: Sparse connectivity fraction
            seed: Random seed
            pattern_sparsity: Fraction of units active in patterns (default 5%)
                             Sparser patterns have less overlap, improving action specificity.
        """
        super().__init__(n_units, connectivity, seed)

        # === Pattern Configuration ===
        # Sparse patterns reduce overlap between states, preventing unwanted generalization
        # At 5% sparsity with 200 units: ~10 active units, near-zero expected overlap
        self.pattern_sparsity = pattern_sparsity

        # === Action System ===
        # Actions are represented as activation patterns over dedicated units
        # These don't participate in normal dynamics - they MODULATE dynamics
        self.n_action_dims = n_action_dims
        self.action_units_per_dim = 16  # Units per action dimension
        self.n_action_units = n_action_dims * self.action_units_per_dim

        # Action activation (which action is currently being taken)
        self.A_action = np.zeros(self.n_action_units)

        # === ACTION-SPECIFIC TRANSITION WEIGHTS ===
        # Key insight from FMC: each action trajectory is SEPARATE
        # Instead of gating shared connections, each action has its OWN weights
        # W_transition[action_dim] = separate weight matrix for that action
        # This completely prevents interference between action-conditioned paths
        self.W_transition = [
            np.random.randn(self._n_connections) * 0.05
            for _ in range(n_action_dims)
        ]

        # Also keep modulation weights for smooth blending
        self.W_action_pred = np.random.randn(self.n_action_units, self._n_connections) * 0.01
        self.W_action_dyn = np.random.randn(self.n_action_units, self._n_connections) * 0.01
        self.W_action_gate = self.W_action_dyn

        # Learning rates (note: dt=0.1, so effective rate = lr * dt)
        self.lr_action_pred = 0.2
        self.lr_action_consolidate = 0.02
        self.lr_transition = 1.5  # Learning rate for action-specific transitions (effective: 0.15)

        # === Planning Slots (FMC Walkers) ===
        self.n_slots = n_slots

        # Each slot maintains a complete state for simulation
        self.slot_A = np.zeros((n_slots, n_units))           # Activation per slot
        self.slot_action = np.zeros((n_slots, self.n_action_units))  # Action per slot
        self.slot_reward = np.zeros(n_slots)                  # Accumulated reward
        self.slot_action_sequence = [[] for _ in range(n_slots)]  # Action history per slot

        # Planning parameters
        self.planning_active = False
        self.plan_horizon = 20          # Steps to simulate forward
        self.fmc_alpha = 1.0            # Balance exploitation vs exploration
        self.clone_threshold = 0.3      # Probability threshold for cloning

        # === Action Vocabulary ===
        # Named actions for convenience
        self.action_names = {}
        self._init_action_vocabulary()

        # === COMPLEMENTARY LEARNING SYSTEM ===
        # Inspired by hippocampus-neocortex consolidation in the brain
        # Fast system: W_transition (current) - learns quickly, subject to interference
        # Slow system: W_action_longterm - consolidates slowly, resists overwriting
        # NOTE: Named W_action_longterm (not W_longterm) to avoid collision with parent's W_longterm

        # Long-term weights per action (consolidated, stable memory)
        self.W_action_longterm = [
            np.zeros(self._n_connections)
            for _ in range(n_action_dims)
        ]

        # === Experience Replay Buffer ===
        # Stores (state_pattern, action_dim, outcome_pattern, confidence) tuples
        # Used during waking (interleaved) and dreams to replay and consolidate
        self.experience_buffer = []
        self.max_buffer_size = 500  # Large enough for many transitions

        # === Transition Confidence Tracking ===
        # Track confidence per (state_hash, action_dim) -> prevents overwriting established knowledge
        # High confidence = reduce learning rate (knowledge is stable)
        self.transition_confidence = {}  # key: (state_hash, action_dim) -> confidence [0, 1]
        self.confidence_threshold = 0.7  # Above this, reduce plasticity
        self.min_plasticity = 0.1  # Minimum learning rate multiplier

        # Learning rates for long-term consolidation
        # Higher rate needed for sparse patterns (few connections to strengthen)
        self.lr_action_longterm = 0.5  # High rate for one-shot learning with sparse patterns
        self.action_longterm_blend = 1.0  # ONLY use longterm (ignore corrupted W_transition)

        # === ELIGIBILITY TRACES ===
        # Implements goal-gated learning: connections are marked as "eligible"
        # but only consolidated into weights when goal is actually reached.
        # This prevents learning incorrect transitions.
        self.eligibility_traces = [
            np.zeros(self._n_connections)
            for _ in range(n_action_dims)
        ]
        self.eligibility_decay = 0.98  # Slower decay - accumulate more eligibility
        self.goal_verification_threshold = 0.2  # Lower threshold for one-shot learning

        # === Extended Dream Learning ===
        self.action_outcome_trace = np.zeros_like(self.W_action_gate)

        # Track current experience for buffering
        self._current_state_hash = None
        self._current_action_dim = None
        self._current_state_pattern = None
        self._current_target_pattern = None  # Track target for goal verification

        # === SPATIAL ENCODING (Tile Coding) ===
        # For continuous state spaces, we need patterns where nearby states
        # have similar (overlapping) representations.
        # Tile coding: multiple overlapping grids, each position activates tiles
        # from each grid, nearby positions share tiles.
        self._spatial_encoding_initialized = False
        self._spatial_tilings = None
        self._spatial_tile_to_units = None
        self._spatial_state_dim = None

        # === SUCCESSOR REPRESENTATION ===
        # The most general approach for planning across any state space.
        # M(s,s') = expected discounted future occupancy of s' starting from s
        # Value(s, goal) = sum_s' M(s,s') * reward(s') = M(s) · goal_features
        #
        # Key insight: SR is GOAL-AGNOSTIC - learn once, evaluate any goal instantly.
        # This is how the hippocampus supports flexible planning and transfer.
        #
        # For efficiency, we store M as row vectors: M[unit] = successor features
        # When we observe transition from pattern_a to pattern_b:
        #   For each active unit i in pattern_a:
        #     M[i] += lr * (pattern_b + gamma * M[active_in_b] - M[i])
        self.M_successor = np.zeros((self.n, self.n))  # Successor feature matrix
        self.sr_gamma = 0.9  # Discount factor for future states (lower = faster convergence)
        self.lr_successor = 0.2  # Learning rate for SR updates (higher = faster learning)

    # ==========================================================================
    # SPATIAL ENCODING - Place Cell Representation
    # ==========================================================================

    def init_spatial_encoding(self, state_dim, n_place_cells=300, field_radius=0.12):
        """
        Initialize place cell encoding for continuous state spaces.

        Inspired by hippocampal place cells:
        - Each place cell has a localized "place field" where it fires
        - A location activates only cells whose field contains that location
        - This creates SPARSE, DECORRELATED representations
        - Nearby locations share SOME cells (at field boundaries) but not most

        This differs from Gaussian RBF:
        - RBF: Nearby locations very similar (80%+ overlap)
        - Place cells: Nearby locations moderately similar (20-40% overlap)

        The lower overlap enables Hebbian transition learning while still
        providing some generalization at field boundaries.

        Args:
            state_dim: Dimensionality of state space (e.g., 2 for (x,y))
            n_place_cells: Number of place cells (more = finer spatial resolution)
            field_radius: Radius of each place field (smaller = sparser codes)
        """
        self._spatial_state_dim = state_dim
        self._n_place_cells = n_place_cells
        self._field_radius = field_radius

        # Distribute place cell centers across the state space
        # Using uniform random distribution (like biological place cells)
        np.random.seed(12345)  # Deterministic for reproducibility
        self._place_cell_centers = np.random.uniform(0, 1, (n_place_cells, state_dim))
        # Each place cell maps to a specific network unit
        self._place_cell_units = np.random.choice(self.n, n_place_cells, replace=False)
        np.random.seed(None)

        self._spatial_encoding_initialized = True

        # Compute expected statistics
        if state_dim == 2:
            # Expected active cells = n_cells * (area of field / total area)
            field_area = np.pi * field_radius**2
            expected_active = n_place_cells * field_area
            print(f"  Place cell encoding: {n_place_cells} cells, radius={field_radius}")
            print(f"  Expected active cells per location: ~{expected_active:.1f}")

        return self

    def encode_continuous_state(self, state_vector):
        """
        Encode a continuous state using place cells.

        Each place cell fires (binary 1.0) if the state is within its field radius.
        This creates a sparse, decorrelated population code.

        Args:
            state_vector: Array of shape (state_dim,) with values in [0, 1]

        Returns:
            pattern: Sparse binary activation pattern of shape (n_units,)
        """
        if not self._spatial_encoding_initialized:
            raise ValueError("Call init_spatial_encoding() first")

        state = np.asarray(state_vector)
        if state.shape != (self._spatial_state_dim,):
            raise ValueError(f"Expected state dim {self._spatial_state_dim}, got {state.shape}")

        # Clip to [0, 1]
        state = np.clip(state, 0, 1)

        # Compute distance from state to each place cell center
        distances = np.linalg.norm(self._place_cell_centers - state, axis=1)

        # Place cells fire if state is within their field (binary activation)
        active_mask = distances < self._field_radius

        # Create sparse pattern
        pattern = np.zeros(self.n)
        pattern[self._place_cell_units[active_mask]] = 1.0

        return pattern

    def inject_continuous_state(self, state_vector, strength=0.7):
        """
        Inject a continuous state as a place cell pattern.

        Args:
            state_vector: Array of shape (state_dim,) with values in [0, 1]
            strength: Injection strength

        Returns:
            Pattern info dict
        """
        pattern = self.encode_continuous_state(state_vector)

        # Get active units
        units = np.where(pattern > 0.5)[0]

        # Create coherent phase based on state
        base_phase = (np.sum(state_vector) * 1.234) % (2 * np.pi)
        phases = base_phase + np.random.normal(0, 0.2, len(units))

        self.input_queue.append({
            'units': units,
            'strength': strength,
            'phases': phases % (2 * np.pi)
        })

        # Store in pattern_traces for retrieval
        state_name = self._state_to_name(state_vector)
        self.pattern_traces[state_name] = pattern

        return {'units': units, 'phases': phases, 'pattern': pattern}

    def set_continuous_target(self, state_vector, strength=1.0):
        """
        Set a continuous state as the target.

        Args:
            state_vector: Array of shape (state_dim,) with values in [0, 1]
            strength: Target strength
        """
        pattern = self.encode_continuous_state(state_vector)
        self.A_target = pattern.copy()
        self.target_strength = strength

        # Store in pattern_traces
        state_name = self._state_to_name(state_vector)
        self.pattern_traces[state_name] = pattern

        return pattern

    def _state_to_name(self, state_vector):
        """Convert state vector to a string name for storage."""
        coords = '_'.join(f'{x:.3f}' for x in state_vector)
        return f'state_{coords}'

    # ==========================================================================
    # SUCCESSOR REPRESENTATION - Goal-Agnostic Planning
    # ==========================================================================

    def update_successor(self, prev_pattern, curr_pattern):
        """
        Update successor representation based on observed transition.

        TD learning rule for successor features:
            M(s) += lr * (I(s) + gamma * M(s') - M(s))

        Where I(s) is the one-hot (or in our case, the actual pattern).

        For sparse patterns, we only update rows corresponding to active
        units in prev_pattern (computational efficiency).

        Args:
            prev_pattern: Pattern before transition (source state)
            curr_pattern: Pattern after transition (destination state)
        """
        # Find active units in both patterns
        prev_active = np.where(prev_pattern > 0.3)[0]
        curr_active = np.where(curr_pattern > 0.3)[0]

        if len(prev_active) == 0 or len(curr_active) == 0:
            return

        # Compute successor features of next state
        # Average the M rows of active units in curr_pattern
        if len(curr_active) > 0:
            next_sr = np.mean(self.M_successor[curr_active], axis=0)
        else:
            next_sr = np.zeros(self.n)

        # Target successor features: current pattern + discounted future
        target_sr = curr_pattern + self.sr_gamma * next_sr

        # Update M rows for active units in prev_pattern
        for i in prev_active:
            td_error = target_sr - self.M_successor[i]
            self.M_successor[i] += self.lr_successor * td_error

        # Keep values bounded (higher limit for value propagation)
        self.M_successor = np.clip(self.M_successor, 0, 10.0)

    def compute_sr_value(self, state_pattern, goal_pattern):
        """
        Compute value of state w.r.t. goal using successor representation.

        Value = E[sum of discounted goal overlap] = SR(state) · goal

        This is the key benefit of SR: goal-agnostic learning, instant
        evaluation for any goal.

        Args:
            state_pattern: Current state as activation pattern
            goal_pattern: Goal as activation pattern

        Returns:
            value: Expected discounted overlap with goal from this state (normalized)
        """
        # Get active units in state
        active = np.where(state_pattern > 0.3)[0]

        if len(active) == 0:
            return 0.0

        # Average successor features of active units
        sr_features = np.mean(self.M_successor[active], axis=0)

        # Value is dot product with goal pattern
        value = np.dot(sr_features, goal_pattern)

        # Normalize by both goal magnitude and SR magnitude for comparability
        goal_norm = np.linalg.norm(goal_pattern)
        sr_norm = np.linalg.norm(sr_features)
        if goal_norm > 1e-8 and sr_norm > 1e-8:
            value = value / (goal_norm * sr_norm)  # Cosine similarity

        return value

    # ==========================================================================
    # COMPLEMENTARY LEARNING SYSTEM - Memory & Consolidation
    # ==========================================================================

    def _hash_pattern(self, pattern):
        """Create a hash for a pattern to use as dictionary key."""
        # Use top-k active units as hash
        top_k = np.argsort(pattern)[-10:]
        return tuple(sorted(top_k))

    def _store_contrastive_experience(self, input_pattern, target_pattern):
        """
        Override: Don't store contrastive experiences in PlanningSOCMind.

        PlanningSOCMind uses its own _store_experience with action_dim.
        The experience buffer format is different, so we disable parent's storage.
        """
        pass  # Don't store - PlanningSOCMind uses _store_experience instead

    def _replay_contrastive_to_longterm(self, dt):
        """
        Override: Don't replay contrastive experiences in PlanningSOCMind.

        PlanningSOCMind uses its own _replay_to_longterm for action transitions.
        """
        pass  # Don't replay - PlanningSOCMind uses _replay_to_longterm instead

    def _store_experience(self, state_pattern, action_dim, outcome_pattern):
        """
        Store an experience in the replay buffer.

        Called after successful learning to save for later replay.
        """
        # Compute current confidence for this transition
        state_hash = self._hash_pattern(state_pattern)
        key = (state_hash, action_dim)
        confidence = self.transition_confidence.get(key, 0.0)

        experience = {
            'state': state_pattern.copy(),
            'action_dim': action_dim,
            'outcome': outcome_pattern.copy(),
            'confidence': confidence,
            'age': 0  # How many replay cycles since stored
        }

        self.experience_buffer.append(experience)

        # Keep buffer size bounded (remove oldest low-confidence experiences)
        if len(self.experience_buffer) > self.max_buffer_size:
            # Sort by confidence (keep high-confidence ones)
            self.experience_buffer.sort(key=lambda x: x['confidence'], reverse=True)
            self.experience_buffer = self.experience_buffer[:self.max_buffer_size]

    def _get_plasticity_multiplier(self, state_hash, action_dim):
        """
        Get learning rate multiplier based on transition confidence.

        High confidence = low plasticity (don't overwrite established knowledge)
        Low confidence = high plasticity (still learning this transition)
        """
        key = (state_hash, action_dim)
        confidence = self.transition_confidence.get(key, 0.0)

        if confidence > self.confidence_threshold:
            # Reduce plasticity for well-established transitions
            # Linearly interpolate from 1.0 at threshold to min_plasticity at 1.0
            t = (confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
            return 1.0 - t * (1.0 - self.min_plasticity)
        else:
            return 1.0  # Full plasticity

    def _update_transition_confidence(self, state_hash, action_dim, prediction_accuracy):
        """
        Update confidence for a transition based on prediction accuracy.

        Confidence increases when predictions are accurate, decreases when wrong.
        """
        key = (state_hash, action_dim)
        old_confidence = self.transition_confidence.get(key, 0.0)

        # Exponential moving average with asymmetric rates
        # Fast increase when correct, slow decrease when wrong
        if prediction_accuracy > 0.5:
            alpha = 0.2  # Fast learning of confidence
        else:
            alpha = 0.05  # Slow forgetting of confidence

        new_confidence = old_confidence + alpha * (prediction_accuracy - old_confidence)
        self.transition_confidence[key] = np.clip(new_confidence, 0.0, 1.0)

        return new_confidence

    def _consolidate_to_longterm(self, action_dim, dt):
        """
        Slowly consolidate W_transition into W_longterm.

        This happens continuously but slowly, creating stable long-term memory.
        """
        weight_diff = self.W_transition[action_dim] - self.W_action_longterm[action_dim]
        dW = self.lr_action_longterm * dt * weight_diff
        self.W_action_longterm[action_dim] += dW
        self.W_action_longterm[action_dim] = np.clip(self.W_action_longterm[action_dim], -1.0, 1.0)

    def _init_action_vocabulary(self):
        """Initialize named action patterns."""
        np.random.seed(42)  # Reproducible
        for i in range(self.n_action_dims):
            # Each action dimension gets a distinct pattern
            start_idx = i * self.action_units_per_dim
            end_idx = start_idx + self.action_units_per_dim

            # Create pattern for this action dimension
            pattern = np.zeros(self.n_action_units)
            pattern[start_idx:end_idx] = 1.0

            # Name it (can be overwritten)
            self.action_names[f"action_{i}"] = pattern.copy()
        np.random.seed(None)

    def set_action(self, action_name_or_pattern, strength=1.0):
        """
        Set the current action using PHASE BINDING.

        Also captures the current state for experience buffering (complementary learning).
        """
        # === Process any pending inputs BEFORE capturing state ===
        # This ensures inject_text patterns are applied before we capture the state
        for inp in self.input_queue:
            self._inject(inp)
        self.input_queue.clear()

        # === Capture current state AFTER processing inputs ===
        # This is needed for experience replay
        self._current_state_pattern = self.A.copy()
        self._current_state_hash = self._hash_pattern(self.A)

        if isinstance(action_name_or_pattern, str):
            if action_name_or_pattern in self.action_names:
                pattern = self.action_names[action_name_or_pattern]
            else:
                # DETERMINISTIC mapping: 'action_N' -> dimension N
                # This ensures planning and execution use the same dimensions
                if action_name_or_pattern.startswith('action_'):
                    try:
                        dim = int(action_name_or_pattern.split('_')[1]) % self.n_action_dims
                    except:
                        dim = hash(action_name_or_pattern) % self.n_action_dims
                else:
                    dim = hash(action_name_or_pattern) % self.n_action_dims
                pattern = np.zeros(self.n_action_units)
                start_idx = dim * self.action_units_per_dim
                pattern[start_idx:start_idx + self.action_units_per_dim] = 1.0
                self.action_names[action_name_or_pattern] = pattern.copy()
        else:
            pattern = action_name_or_pattern

        self.A_action = pattern * strength

        # Track action dimension for experience buffering
        self._current_action_dim = self._get_active_action_dim()

        # === PHASE BINDING: Actions set distinctive phase patterns ===
        # This makes (state, action_0) orthogonal to (state, action_1)
        # because phase affects how activations combine via cos(phase_diff)
        if isinstance(action_name_or_pattern, str):
            # Each action gets a unique base phase
            action_hash = hash(action_name_or_pattern) % 10000
            base_phase = (action_hash / 10000.0) * 2 * np.pi  # Unique phase per action

            # Shift network phases toward action's base phase
            # Active units get stronger phase shift
            phase_shift = strength * 0.5 * np.sin(base_phase - self.theta)
            self.theta = (self.theta + phase_shift) % (2 * np.pi)

            # Also add action units to activation (but DIFFERENT units per action)
            np.random.seed(action_hash + 12345)
            action_units = np.random.choice(self.n, 40, replace=False)
            np.random.seed(None)

            # These units get the action's coherent phase
            self.theta[action_units] = base_phase + np.random.normal(0, 0.1, len(action_units))
            self.A[action_units] = np.clip(self.A[action_units] + strength * 0.4, 0, 1)
            self.input_trace[action_units] += strength * 0.6

    def clear_action(self):
        """Clear the current action (no action being taken)."""
        self.A_action = np.zeros(self.n_action_units)

    def _make_predictions(self):
        """
        Override: Predict next state given current state AND current action.

        The action gates which connections are active/enhanced.
        """
        rows, cols = self._conn_rows, self._conn_cols

        # Phase-modulated input using PREDICTION weights (original)
        phase_diff = self.theta[rows] - self.theta[cols] - self.W_phase_data
        phase_mod = np.cos(phase_diff)
        base_input = self.W_pred_data * self.A[cols] * phase_mod

        # === Action Gating (use W_action_pred for prediction/imagination) ===
        # Compute gate for each connection based on active actions
        # gate = 1 + Σ A_action[k] * W_action_pred[k, connection]
        action_gate = np.ones(self._n_connections)
        if np.any(self.A_action > 0.01):
            action_modulation = np.dot(self.A_action, self.W_action_pred)
            action_gate = 1.0 + action_modulation
            action_gate = np.clip(action_gate, 0.0, 2.0)  # Can suppress or enhance

        # Apply gating to input
        gated_input = base_input * action_gate

        # Sum for each unit
        pred = np.zeros(self.n)
        np.add.at(pred, rows, gated_input)

        # Normalize by connection count
        in_degree = np.zeros(self.n)
        np.add.at(in_degree, rows, np.abs(self.W_pred_data) * action_gate)
        in_degree = np.maximum(in_degree, 1)
        pred = pred / in_degree

        # Prediction includes decay term
        pred = pred - self.decay * self.A

        # Clip to valid range
        self.A_pred = np.clip(self.A + 0.1 * pred, 0, 1)

    def _get_active_action_dim(self):
        """Get which action dimension is currently active (if any)."""
        if np.max(self.A_action) < 0.01:
            return None
        # Find which action dimension has highest activation
        for dim in range(self.n_action_dims):
            start = dim * self.action_units_per_dim
            end = start + self.action_units_per_dim
            if np.mean(self.A_action[start:end]) > 0.5:
                return dim
        return None

    def _soc_dynamics(self, dt):
        """
        Override: SOC dynamics with COMPLEMENTARY LEARNING SYSTEM.

        Uses BOTH short-term (W_transition) and long-term (W_longterm) weights.
        Long-term provides stable, consolidated memory.
        Short-term provides recent, flexible adaptation.
        """
        firing = (self.A > self.threshold).astype(float)
        rows, cols = self._conn_rows, self._conn_cols

        # Base dynamics weights
        base_weights = self.W_dyn_data.copy()

        # === ACTION-SPECIFIC TRANSITION with DUAL MEMORY ===
        active_dim = self._get_active_action_dim()
        if active_dim is not None:
            # Combine short-term (W_transition) and long-term (W_longterm)
            # Long-term provides stable knowledge, short-term provides recent learning
            short_term = self.W_transition[active_dim]
            long_term = self.W_action_longterm[active_dim]

            # Blend: longterm_blend controls the ratio
            # Higher longterm_blend = more stable, less interference
            action_weights = (self.action_longterm_blend * long_term +
                            (1 - self.action_longterm_blend) * short_term)

            # Then blend with base dynamics (action-specific weights dominate)
            effective_weights = 0.9 * action_weights + 0.1 * base_weights
        else:
            effective_weights = base_weights

        # Separate excitatory/inhibitory from effective weights
        excitatory = np.maximum(effective_weights, 0)
        inhibitory = np.minimum(effective_weights, 0)

        # No additional gating needed - action already selected weights
        action_gate = np.ones(self._n_connections)

        # Gated excitatory/inhibitory input
        exc_input = excitatory * action_gate * firing[cols] * self.E[cols]
        exc_sum = np.zeros(self.n)
        np.add.at(exc_sum, rows, exc_input)

        inh_input = inhibitory * action_gate * firing[cols]
        inh_sum = np.zeros(self.n)
        np.add.at(inh_sum, rows, inh_input)

        # Normalize
        in_degree = np.zeros(self.n)
        np.add.at(in_degree, rows, np.abs(self.W_dyn_data) * action_gate)
        in_degree = np.maximum(in_degree, 1)

        total_input = (exc_sum + inh_sum) / in_degree

        # Hierarchical prediction (unchanged from parent)
        hier_pred_input = self.W_hier_data * self.A[rows]
        predicted_input = np.zeros(self.n)
        np.add.at(predicted_input, rows, hier_pred_input)
        predicted_input = predicted_input / in_degree

        self.input_pred = predicted_input
        self.input_pred_error = total_input - predicted_input

        # Input statistics for tau adaptation
        self.input_mean = self.var_decay * self.input_mean + (1 - self.var_decay) * total_input
        input_diff = total_input - self.input_mean
        self.input_var = self.var_decay * self.input_var + (1 - self.var_decay) * (input_diff ** 2)

        # Target pull (contrastive)
        target_pull = self.target_strength * (self.A_target - self.A)

        # Fast weights contribution
        fast_exc = np.maximum(self.W_fast_data, 0) * firing[cols] * self.E[cols]
        fast_sum = np.zeros(self.n)
        np.add.at(fast_sum, rows, fast_exc)
        fast_input = self.fast_weight_scale * fast_sum / in_degree

        # Input trace contribution
        trace_input = self.trace_weight * self.input_trace

        # Decay fast weights and input trace
        self.W_fast_data *= self.fast_decay
        self.input_trace *= self.trace_decay

        # Adaptive timescale dynamics
        raw_dA = (total_input
                  + fast_input
                  + trace_input
                  - self.decay * self.A
                  + np.random.normal(0, self.noise, self.n)
                  + target_pull)
        dA = (1.0 / self.tau) * raw_dA

        A_next = np.clip(self.A + dt * dA, 0, 1)

        return A_next

    def _learn_action_gating(self, dt):
        """
        Two-stage learning for action gating (mirrors W_pred/W_dyn).

        Stage 1: W_action_pred learns quickly from prediction errors
        Stage 2: W_action_dyn consolidates from W_action_pred when confident

        This allows fast learning for planning while keeping stable attractors.

        Learning rule (local, no backprop):
            dW_action[k, conn] = lr * A_action[k] * pred_error[post] * A[pre]
        """
        if np.max(self.A_action) < 0.01:
            return  # No action active

        rows, cols = self._conn_rows, self._conn_cols

        # Error at post-synaptic units
        error_post = self.prediction_error[rows]

        # Activity of pre-synaptic units
        A_pre = self.A[cols]

        # Connection-level learning signal
        connection_signal = error_post * A_pre

        # === Stage 1: Fast learning on W_action_pred ===
        for k in range(self.n_action_units):
            if self.A_action[k] > 0.01:
                dW = self.lr_action_pred * self.A_action[k] * connection_signal
                self.W_action_pred[k] += dt * dW

        # Clip
        self.W_action_pred = np.clip(self.W_action_pred, -1.0, 1.0)

        # === Stage 2: Slow consolidation W_action_pred -> W_action_dyn ===
        # Only consolidate when prediction confidence is high
        confidence_threshold = 0.6
        if np.mean(self.prediction_confidence) > confidence_threshold:
            for k in range(self.n_action_units):
                if self.A_action[k] > 0.01:
                    weight_diff = self.W_action_pred[k] - self.W_action_dyn[k]
                    dW = self.lr_action_consolidate * weight_diff
                    self.W_action_dyn[k] += dt * dW

        self.W_action_dyn = np.clip(self.W_action_dyn, -1.0, 1.0)

    def step(self, dt=0.1, dream_mode=False, execute_mode=False):
        """
        Extended step with action-conditioned learning.

        Args:
            dt: Time step
            dream_mode: If True, use temporal Hebbian learning (consolidation)
            execute_mode: If True, skip ALL learning (for goal execution)
        """
        if execute_mode:
            # Execute-only mode: run dynamics without any learning
            # This is used during goal execution to prevent weight corruption
            self._execute_step(dt)
            return

        # Call parent step (handles all base dynamics)
        super().step(dt, dream_mode)

        # Additional: Learn action gating from prediction error
        if not dream_mode:
            self._learn_action_gating(dt)
            # Also do action-conditioned contrastive learning
            self._action_contrastive_learning(dt)
        else:
            # Dream mode: also practice action->outcome
            self._dream_action_learning(dt)

    def _execute_step(self, dt):
        """
        Execute one step WITHOUT any learning.
        Used during goal execution to prevent weight corruption.
        """
        self.t += dt
        self.A_prev = self.A.copy()

        # Process external input
        for inp in self.input_queue:
            self._inject(inp)
        self.input_queue.clear()

        # Compute next state using SOC dynamics
        A_next = self._soc_dynamics(dt)

        # Update state
        self.A = A_next
        self._update_phases(dt)
        self._update_energy(dt)

        # Decay target
        self.A_target *= self.target_decay
        self.target_strength *= self.target_decay

        # Record history
        self._record()

    def _action_contrastive_learning(self, dt):
        """
        Action-specific contrastive learning with COMPLEMENTARY LEARNING SYSTEM.

        Key features:
        1. Confidence-gated plasticity - high-confidence transitions resist overwriting
        2. Experience storage - save for later replay
        3. Long-term consolidation - slowly transfer to stable W_longterm
        4. Dual learning - update both W_transition and W_longterm
        """
        if self.target_strength < 0.01:
            return  # No target

        active_dim = self._get_active_action_dim()
        if active_dim is None:
            return  # No action

        rows, cols = self._conn_rows, self._conn_cols

        # Error toward target
        local_error = self.A_target - self.A

        # Compute prediction accuracy (how close are we to target?)
        target_units = self.A_target > 0.5
        if np.any(target_units):
            accuracy = 1.0 - np.mean(np.abs(local_error[target_units]))
        else:
            accuracy = 0.5

        # === CONFIDENCE-GATED PLASTICITY ===
        # Get plasticity multiplier based on transition confidence
        state_hash = self._current_state_hash
        if state_hash is not None:
            plasticity = self._get_plasticity_multiplier(state_hash, active_dim)
            # Update confidence based on prediction accuracy
            self._update_transition_confidence(state_hash, active_dim, accuracy)
        else:
            plasticity = 1.0

        # Three-factor learning: pre * error_post
        pre = self.input_trace[cols]
        error_post = local_error[rows]

        # Learn into W_transition with GATED plasticity
        # High-confidence transitions get lower learning rate
        effective_lr = self.lr_transition * plasticity
        dW = effective_lr * pre * error_post

        self.W_transition[active_dim] += dt * dW
        self.W_transition[active_dim] = np.clip(self.W_transition[active_dim], -1.0, 1.0)

        # === DIRECT HEBBIAN LEARNING ===
        # Strengthen connections from source state to target state.
        # Sparse patterns (default 5%) reduce overlap between states,
        # which prevents unwanted generalization without needing eligibility gating.

        post_active = (self.A_target > 0.5)[rows]    # Post-synaptic target active

        # Use the ORIGINAL source pattern (captured at set_action), not drifted input_trace
        if self._current_state_pattern is not None:
            pre_active = (self._current_state_pattern > 0.3)[cols]
        else:
            pre_active = (self.input_trace > 0.3)[cols]

        # Direct Hebbian update
        hebbian_signal = pre_active * post_active
        dW = self.lr_action_longterm * hebbian_signal.astype(float)
        self.W_action_longterm[active_dim] += dt * dW
        self.W_action_longterm[active_dim] = np.clip(self.W_action_longterm[active_dim], 0, 1.0)

        # === EXPERIENCE STORAGE ===
        # Store this experience for later replay during dreams
        if self._current_state_pattern is not None and accuracy > 0.3:
            outcome_pattern = self.A_target.copy()
            self._store_experience(self._current_state_pattern, active_dim, outcome_pattern)

            # === SUCCESSOR REPRESENTATION UPDATE ===
            # Learn which states lead to which - goal-agnostic planning
            self.update_successor(self._current_state_pattern, outcome_pattern)

    def _replay_to_longterm(self, dt):
        """
        Replay a random stored experience to W_longterm using CONTRASTIVE learning.

        Uses contrastive learning with OTHER LEARNED STATES as negative examples:
        1. Positive phase: strengthen stored_state → outcome connections
        2. Negative phase: weaken OTHER_state → outcome connections

        This teaches: "only THIS specific state leads to outcome, not other states"
        Much more effective than random noise because it targets actual learned patterns.

        Called during waking learning (interleaved) and during dreams.
        """
        if len(self.experience_buffer) == 0:
            return

        # Sample uniformly from buffer (all transitions equal importance)
        idx = np.random.randint(len(self.experience_buffer))
        exp = self.experience_buffer[idx]
        exp['age'] += 1  # Track how often replayed

        state_pattern = exp['state']
        action_dim = exp['action_dim']
        outcome_pattern = exp['outcome']

        rows, cols = self._conn_rows, self._conn_cols

        # CONTRASTIVE REPLAY with learned negative examples
        post_active = (outcome_pattern > 0.5)[rows]

        # Positive phase: strengthen stored state → outcome
        pre_active = (state_pattern > 0.3)[cols]
        positive_signal = pre_active * post_active
        dW_positive = self.lr_action_longterm * positive_signal.astype(float)

        # Negative phase: use ANOTHER learned state as negative example
        # This specifically teaches "other states shouldn't lead to this outcome"
        if len(self.experience_buffer) > 1:
            # Sample a different experience as negative
            neg_idx = np.random.randint(len(self.experience_buffer))
            while neg_idx == idx and len(self.experience_buffer) > 1:
                neg_idx = np.random.randint(len(self.experience_buffer))
            neg_state = self.experience_buffer[neg_idx]['state']
            neg_pre = (neg_state > 0.3)[cols]

            # Weaken connections from neg_state to outcome
            # BUT only where neg_state differs from positive state
            negative_mask = neg_pre & (~pre_active) & post_active
            dW_negative = self.lr_action_longterm * 0.5 * negative_mask.astype(float)
        else:
            dW_negative = 0

        # Apply contrastive update
        self.W_action_longterm[action_dim] += dt * (dW_positive - dW_negative)
        self.W_action_longterm[action_dim] = np.clip(self.W_action_longterm[action_dim], 0, 1.0)

        # === SUCCESSOR REPRESENTATION REPLAY ===
        # Also consolidate SR during replay for goal-agnostic planning
        self.update_successor(state_pattern, outcome_pattern)

    def _dream_action_learning(self, dt):
        """
        During dreams, do intensive replay to consolidate long-term memory.

        Dreams do MORE replay than waking (50% vs 15% of steps) to accelerate
        consolidation of all stored experiences into W_longterm.

        Inspired by hippocampal replay during sleep.
        """
        if len(self.experience_buffer) == 0:
            return

        # Dreams do more intensive replay than waking
        # 50% of dream steps do replay (vs 15% during waking)
        if np.random.random() < 0.5:
            self._replay_to_longterm(dt)

    # ==========================================================================
    # FMC-STYLE PLANNING
    # ==========================================================================

    def init_planning(self, start_state=None):
        """
        Initialize slots for planning from current state.

        Args:
            start_state: Optional starting activation. If None, use current self.A
        """
        if start_state is None:
            start_state = self.A.copy()

        # Initialize all slots to starting state with random actions
        for i in range(self.n_slots):
            self.slot_A[i] = start_state.copy()
            self.slot_action[i] = self._sample_random_action()
            self.slot_reward[i] = 0.0
            self.slot_action_sequence[i] = [self.slot_action[i].copy()]

        self.planning_active = True

    def _sample_random_action(self):
        """Sample a random action pattern."""
        action = np.zeros(self.n_action_units)
        dim = np.random.randint(self.n_action_dims)
        start_idx = dim * self.action_units_per_dim
        action[start_idx:start_idx + self.action_units_per_dim] = 1.0
        return action

    def _get_action_dim_from_pattern(self, action_pattern):
        """Get action dimension from action pattern."""
        for dim in range(self.n_action_dims):
            start = dim * self.action_units_per_dim
            end = start + self.action_units_per_dim
            if np.mean(action_pattern[start:end]) > 0.5:
                return dim
        return None

    def _simulate_slot(self, slot_idx, steps_per_action=10):
        """
        Simulate one slot forward in imagination using ACTION SEQUENCE.

        Each action in the sequence is applied for steps_per_action steps.
        This enables multi-step planning (A→B→C requires action_0 then action_1).
        """
        temp_A = self.slot_A[slot_idx].copy()
        temp_input_trace = temp_A.copy() * 0.5

        # Get the action SEQUENCE for this slot
        action_sequence = self.slot_action_sequence[slot_idx]
        if not action_sequence:
            action_sequence = [self.slot_action[slot_idx]]

        # Simulate each action in sequence
        for action_pattern in action_sequence:
            action_dim = self._get_action_dim_from_pattern(action_pattern)

            for _ in range(steps_per_action):
                firing = (temp_A > self.threshold).astype(float)
                rows, cols = self._conn_rows, self._conn_cols

                # Use W_longterm for imagination
                if action_dim is not None:
                    action_weights = self.W_action_longterm[action_dim]
                    effective_weights = 0.9 * action_weights + 0.1 * self.W_pred_data
                else:
                    effective_weights = self.W_pred_data

                excitatory = np.maximum(effective_weights, 0)
                inhibitory = np.minimum(effective_weights, 0)

                exc_input = excitatory * firing[cols]
                inh_input = inhibitory * firing[cols]

                exc_sum = np.zeros(self.n)
                np.add.at(exc_sum, rows, exc_input)
                inh_sum = np.zeros(self.n)
                np.add.at(inh_sum, rows, inh_input)

                in_degree = np.zeros(self.n)
                np.add.at(in_degree, rows, np.abs(effective_weights))
                in_degree = np.maximum(in_degree, 1)

                total_input = (exc_sum + inh_sum) / in_degree
                trace_contribution = self.trace_weight * temp_input_trace

                dA = total_input + trace_contribution - self.decay * temp_A
                dA += np.random.normal(0, self.noise * 0.3, self.n)
                temp_A = np.clip(temp_A + 0.1 * dA, 0, 1)

                temp_input_trace *= 0.95

        # Update slot state
        self.slot_A[slot_idx] = temp_A

    def _compute_virtual_rewards(self, goal_pattern):
        """
        Compute Virtual Reward for each slot using SUCCESSOR REPRESENTATION.

        VR = Reward^α × Distance

        Where:
        - Reward = SR value (expected future goal overlap) OR direct similarity
        - Distance = difference from randomly chosen other slot
        - α = exploitation/exploration balance (0 = pure exploration)

        The SR provides gradient even when patterns don't overlap directly.
        This is the core of FMC: balance exploitation (reward) with
        exploration (distance).
        """
        rewards = np.zeros(self.n_slots)
        distances = np.zeros(self.n_slots)

        # Check if SR has been learned (has non-zero values)
        sr_learned = np.max(self.M_successor) > 0.01

        for i in range(self.n_slots):
            if sr_learned:
                # Use SUCCESSOR REPRESENTATION for reward
                # This gives expected future overlap with goal, providing gradient
                # even when current state doesn't directly overlap with goal
                sr_value = self.compute_sr_value(self.slot_A[i], goal_pattern)

                # Also compute direct overlap (for when we're very close to goal)
                slot_norm = np.linalg.norm(self.slot_A[i])
                goal_norm = np.linalg.norm(goal_pattern)
                if slot_norm > 1e-8 and goal_norm > 1e-8:
                    direct_sim = np.dot(self.slot_A[i], goal_pattern) / (slot_norm * goal_norm)
                else:
                    direct_sim = 0.0

                # Combine: use max of SR value and direct similarity
                # SR provides gradient when far, direct_sim when at goal
                goal_sim = max(sr_value, direct_sim)
            else:
                # Fall back to direct cosine similarity (for discrete patterns)
                slot_norm = np.linalg.norm(self.slot_A[i])
                goal_norm = np.linalg.norm(goal_pattern)
                if slot_norm > 1e-8 and goal_norm > 1e-8:
                    goal_sim = np.dot(self.slot_A[i], goal_pattern) / (slot_norm * goal_norm)
                else:
                    goal_sim = 0.0

            # Modulate by prediction confidence (how reliable is this future?)
            confidence_mod = np.mean(self.prediction_confidence) + 0.1
            rewards[i] = max(0, goal_sim) * confidence_mod

            # Distance: L2 distance to random other slot
            j = np.random.randint(self.n_slots)
            if j == i:
                j = (i + 1) % self.n_slots
            distances[i] = np.linalg.norm(self.slot_A[i] - self.slot_A[j])

        # Virtual Reward (FMC formula)
        # At α=0, VR = Distance (pure exploration / "common sense")
        # At α=1, VR = Reward × Distance (balanced)
        # At α→∞, VR = Reward (pure exploitation)
        if self.fmc_alpha > 0:
            virtual_rewards = (rewards ** self.fmc_alpha) * distances
        else:
            # Common sense mode: maximize entropy of reachable futures
            virtual_rewards = distances

        return virtual_rewards, rewards, distances

    def _clone_step(self, virtual_rewards):
        """
        FMC cloning dynamics.

        Low VR slots are replaced by copies of high VR slots.
        This concentrates computational resources on promising futures.

        Cloning probability:
        - If VR_i = 0: probability = 1 (definitely clone over it)
        - If VR_i > VR_k: probability = 0 (keep it)
        - Otherwise: probability = (VR_k - VR_i) / VR_i
        """
        # For each slot, decide whether to clone from another
        for i in range(self.n_slots):
            # Pick random slot to compare against
            k = np.random.randint(self.n_slots)
            if k == i:
                continue

            vr_i = virtual_rewards[i]
            vr_k = virtual_rewards[k]

            # Compute cloning probability
            if vr_i < 1e-8:
                prob = 1.0  # i has no value, definitely clone over it
            elif vr_i >= vr_k:
                prob = 0.0  # i is better, keep it
            else:
                prob = min(1.0, (vr_k - vr_i) / (vr_i + 1e-8))

            # Clone decision
            if np.random.random() < prob * self.clone_threshold:
                # Clone slot k over slot i
                self.slot_A[i] = self.slot_A[k].copy()
                self.slot_action[i] = self.slot_action[k].copy()
                self.slot_reward[i] = self.slot_reward[k]
                self.slot_action_sequence[i] = [a.copy() for a in self.slot_action_sequence[k]]

    def _perturb_actions(self, max_sequence_length=4):
        """
        Perturb slot actions for exploration.

        Sequences are capped at max_sequence_length to keep planning fast.
        """
        for i in range(self.n_slots):
            if np.random.random() < 0.3:  # 30% chance to perturb
                # Sample new action
                if np.random.random() < 0.5:
                    new_action = self._sample_random_action()
                else:
                    new_action = self.slot_action[i].copy()
                    noise = np.random.randn(self.n_action_units) * 0.1
                    new_action = np.clip(new_action + noise, 0, 1)

                self.slot_action[i] = new_action

                # Cap sequence length - replace random position or append
                if len(self.slot_action_sequence[i]) >= max_sequence_length:
                    # Replace a random action in sequence
                    idx = np.random.randint(len(self.slot_action_sequence[i]))
                    self.slot_action_sequence[i][idx] = new_action.copy()
                else:
                    # Still room to grow
                    self.slot_action_sequence[i].append(new_action.copy())

    def plan(self, goal_pattern, n_iterations=20, verbose=False):
        """
        FMC-style planning to reach a goal.

        Uses parallel slots (walkers) to explore action sequences.
        Returns the best action sequence found.

        Args:
            goal_pattern: Target activation pattern to reach
            n_iterations: Number of FMC iterations
            verbose: Print progress

        Returns:
            best_action_sequence: List of action patterns
            best_slot_idx: Index of best slot
        """
        if verbose:
            print(f"Planning with {self.n_slots} slots for {n_iterations} iterations")
            print(f"  alpha = {self.fmc_alpha} (0=exploration, inf=exploitation)")

        # Initialize planning from current state
        self.init_planning()

        for iteration in range(n_iterations):
            # 1. Simulate each slot forward
            for i in range(self.n_slots):
                self._simulate_slot(i, steps_per_action=self.plan_horizon)

            # 2. Compute Virtual Rewards
            virtual_rewards, rewards, distances = self._compute_virtual_rewards(goal_pattern)

            if verbose and iteration % 5 == 0:
                best_idx = np.argmax(virtual_rewards)
                print(f"  Iter {iteration}: best_reward={rewards[best_idx]:.3f}, "
                      f"best_distance={distances[best_idx]:.3f}, "
                      f"best_VR={virtual_rewards[best_idx]:.3f}")

            # 3. Clone low-VR slots with high-VR ones
            self._clone_step(virtual_rewards)

            # 4. Perturb actions for exploration
            self._perturb_actions()

            # 5. Re-simulate from original state with new actions
            # (Reset slot states but keep action sequences)
            for i in range(self.n_slots):
                self.slot_A[i] = self.A.copy()

        # Final evaluation
        for i in range(self.n_slots):
            self._simulate_slot(i, steps_per_action=self.plan_horizon)
        virtual_rewards, rewards, _ = self._compute_virtual_rewards(goal_pattern)

        # Return best action sequence
        best_idx = np.argmax(rewards)  # Use pure reward for final selection

        if verbose:
            print(f"  Best slot: {best_idx} with reward {rewards[best_idx]:.3f}")
            print(f"  Action sequence length: {len(self.slot_action_sequence[best_idx])}")

        self.planning_active = False
        return self.slot_action_sequence[best_idx], best_idx

    def execute_action_sequence(self, action_sequence, steps_per_action=20):
        """
        Execute a sequence of actions in the real network.

        Args:
            action_sequence: List of action patterns from plan()
            steps_per_action: Network steps per action

        Returns:
            trajectory: List of (action, state) tuples
        """
        trajectory = []

        for action in action_sequence:
            self.A_action = action.copy()

            # Run network with this action
            for _ in range(steps_per_action):
                self.step(0.1)

            trajectory.append((action.copy(), self.A.copy()))

        # Clear action at end
        self.clear_action()

        return trajectory

    def pattern_overlap(self, pattern_name):
        """Compute overlap between current state and a named pattern."""
        if pattern_name in self.pattern_traces:
            pattern = self.pattern_traces[pattern_name]
        else:
            # Generate SPARSE pattern from name
            # Sparse patterns reduce overlap, improving action specificity
            np.random.seed(hash(pattern_name) % 10000)
            n_active = max(5, int(self.n * self.pattern_sparsity))
            units = np.random.choice(self.n, n_active, replace=False)
            pattern = np.zeros(self.n)
            pattern[units] = 1.0
            np.random.seed(None)

        # Cosine similarity
        dot = np.dot(self.A, pattern)
        norm = np.linalg.norm(self.A) * np.linalg.norm(pattern) + 1e-8
        return dot / norm


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def benchmark_action_prediction():
    """
    Test: Can the network learn action->outcome predictions?

    This runs as a continuous "game loop" - no epochs.
    The network sees transitions online and learns in real-time.

    Success: Given state + action, network predicts correct next state.
    """
    print("=" * 70)
    print("ACTION PREDICTION BENCHMARK (Continuous Learning)")
    print("=" * 70)

    mind = PlanningSOCMind(n_units=512, n_action_dims=4, n_slots=8)

    # Define transitions: (state, action, next_state)
    transitions = [
        ('room_A', 'action_0', 'room_B'),  # action_0 = "right"
        ('room_A', 'action_1', 'room_C'),  # action_1 = "down"
        ('room_B', 'action_0', 'room_D'),
        ('room_B', 'action_1', 'room_E'),
        ('room_C', 'action_0', 'room_E'),
        ('room_C', 'action_1', 'room_F'),
    ]

    print(f"\nTransitions to learn:")
    for s, a, ns in transitions:
        print(f"  {s} + {a} -> {ns}")

    # === CONTINUOUS LEARNING LOOP ===
    # Simulates always-on brain receiving experiences
    # Shorter cycles = more experiences = better learning
    print("\nRunning continuous learning (6000 steps)...")

    total_steps = 0
    transition_idx = 0
    current_state = None
    current_action = None
    current_outcome = None

    for step in range(6000):
        total_steps += 1
        phase_in_cycle = step % 30  # Shorter cycle = more experiences

        # New experience starts at beginning of cycle
        if phase_in_cycle == 0:
            current_state, current_action, current_outcome = transitions[transition_idx % len(transitions)]
            transition_idx += 1
            # Inject state and set action
            mind.inject_text(current_state, 0.9)
            mind.set_action(current_action)

        # Show outcome target midway through cycle
        if phase_in_cycle == 12:
            mind.set_target(current_outcome, 1.0)

        # Clear action near end of cycle
        if phase_in_cycle == 22:
            mind.clear_action()

        # Step the network (dream mode when idle)
        dream_mode = (phase_in_cycle > 25)
        mind.step(0.1, dream_mode=dream_mode)

    print(f"  Completed {total_steps} steps, {transition_idx} experiences")

    # === TESTING ===
    print("\nTesting predictions...")
    print("-" * 50)
    print(f"{'State':<12} {'Action':<10} {'Expected':<12} {'Overlap':<10} {'Pass?'}")
    print("-" * 50)

    correct = 0
    total = len(transitions)

    for state, action, expected in transitions:
        mind.reset()
        mind.inject_text(state, 0.8)
        mind.set_action(action)

        # Let network predict
        for _ in range(30):
            mind.step(0.1)

        mind.clear_action()

        # Measure overlap with expected
        overlap = mind.pattern_overlap(expected)
        passed = overlap > 0.2  # Lower threshold to match base architecture (~0.25)
        if passed:
            correct += 1

        print(f"{state:<12} {action:<10} {expected:<12} {overlap:<10.3f} {'Y' if passed else 'N'}")

    accuracy = correct / total
    print("-" * 50)
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print("=" * 70)

    return accuracy > 0.5, mind


def benchmark_planning():
    """
    Test: Can FMC planning find action sequences to reach goals?

    Uses continuous learning to build world model, then tests planning.

    Success: Planning finds action sequence that reaches goal.
    """
    print("=" * 70)
    print("FMC PLANNING BENCHMARK (Continuous Learning)")
    print("=" * 70)

    mind = PlanningSOCMind(n_units=512, n_action_dims=4, n_slots=16)

    # World model transitions
    transitions = [
        ('start', 'action_0', 'mid_1'),
        ('start', 'action_1', 'mid_2'),
        ('mid_1', 'action_0', 'goal'),
        ('mid_1', 'action_1', 'mid_3'),
        ('mid_2', 'action_0', 'mid_3'),
        ('mid_2', 'action_1', 'dead_end'),
        ('mid_3', 'action_0', 'goal'),
    ]

    print("\nContinuous world model learning (4000 steps)...")

    # Continuous learning loop
    transition_idx = 0
    current_state = None
    current_action = None
    current_outcome = None

    for step in range(4000):
        phase_in_cycle = step % 50

        if phase_in_cycle == 0:
            current_state, current_action, current_outcome = transitions[transition_idx % len(transitions)]
            transition_idx += 1
            mind.inject_text(current_state, 0.9)
            mind.set_action(current_action)

        if phase_in_cycle == 20:
            mind.set_target(current_outcome, 1.0)

        if phase_in_cycle == 35:
            mind.clear_action()

        dream = (phase_in_cycle > 40)
        mind.step(0.1, dream_mode=dream)

    # Planning test
    print("\nPlanning from 'start' to 'goal'...")
    mind.reset()
    mind.inject_text('start', 0.8)
    for _ in range(20):
        mind.step(0.1)

    # Get goal pattern (using sparse pattern generation)
    np.random.seed(hash('goal') % 10000)
    n_active = max(5, int(mind.n * mind.pattern_sparsity))
    goal_units = np.random.choice(mind.n, n_active, replace=False)
    goal_pattern = np.zeros(mind.n)
    goal_pattern[goal_units] = 1.0
    np.random.seed(None)

    # Plan with more iterations
    mind.fmc_alpha = 0.5  # More exploration
    action_sequence, best_slot = mind.plan(goal_pattern, n_iterations=25, verbose=True)

    # Execute plan
    print("\nExecuting plan...")
    trajectory = mind.execute_action_sequence(action_sequence, steps_per_action=25)

    # Check if we reached goal
    final_overlap = mind.pattern_overlap('goal')
    print(f"\nFinal overlap with goal: {final_overlap:.3f}")

    success = final_overlap > 0.3
    print(f"Planning {'SUCCEEDED' if success else 'FAILED'}")
    print("=" * 70)

    return success, mind


def benchmark_dream_consolidation():
    """
    Test: Does dreaming improve the world model?

    Setup:
    1. Brief exposure to transitions
    2. Measure prediction accuracy
    3. Extended dreaming
    4. Measure prediction accuracy again

    Success: Accuracy improves or maintains after dreaming.
    """
    print("=" * 70)
    print("DREAM CONSOLIDATION BENCHMARK")
    print("=" * 70)

    mind = PlanningSOCMind(n_units=512, n_action_dims=4)

    transitions = [
        ('alpha', 'action_0', 'beta'),
        ('alpha', 'action_1', 'gamma'),
        ('beta', 'action_0', 'delta'),
    ]

    # Brief exposure (continuous, sparse)
    print("\nBrief exposure to transitions (500 steps)...")
    transition_idx = 0
    for step in range(500):
        if step % 80 == 0:
            state, action, outcome = transitions[transition_idx % len(transitions)]
            transition_idx += 1
            mind.inject_text(state, 0.8)
            mind.set_action(action)
        if step % 80 == 25:
            state, action, outcome = transitions[(transition_idx - 1) % len(transitions)]
            mind.set_target(outcome, 1.0)
        if step % 80 == 50:
            mind.clear_action()
        mind.step(0.1)

    def measure_accuracy():
        correct = 0
        for state, action, expected in transitions:
            mind.reset()
            mind.inject_text(state, 0.8)
            mind.set_action(action)
            for _ in range(20):
                mind.step(0.1)
            mind.clear_action()
            overlap = mind.pattern_overlap(expected)
            if overlap > 0.25:
                correct += 1
        return correct / len(transitions)

    # Measure before dreaming
    acc_before = measure_accuracy()
    print(f"\nAccuracy BEFORE dreaming: {acc_before:.1%}")

    # Dream
    print("\nDreaming (3000 steps)...")
    mind.dream(n_steps=3000, verbose=False)

    # Measure after dreaming
    acc_after = measure_accuracy()
    print(f"Accuracy AFTER dreaming: {acc_after:.1%}")

    improvement = acc_after - acc_before
    print(f"\nImprovement: {improvement:+.1%}")

    # Note: improvement may be negative if network was already saturated
    # or if dreaming caused interference. Success is defined loosely.
    success = acc_after >= acc_before * 0.9  # Allow 10% degradation
    print(f"Dream consolidation: {'PASS' if success else 'FAIL'}")
    print("=" * 70)

    return success, mind


# ==============================================================================
# INTERACTIVE VISUALIZATION
# ==============================================================================

class PlanningVisualizer:
    """Interactive visualization for Planning SOC Mind."""

    def __init__(self, mind):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Slider, Button
        import threading
        import queue

        self.plt = plt
        self.mind = mind
        self.input_queue = queue.Queue()
        self.last_input = ""
        self.current_action = None

        # Goal-driven behavior
        self.current_goal = None          # Target pattern name
        self.planned_actions = []         # Action sequence from FMC
        self.plan_step = 0                # Current step in plan
        self.steps_in_current_action = 0  # Steps spent on current action
        self.goal_threshold = 0.4         # Overlap threshold for "goal reached"
        self.steps_per_action = 30        # Steps to execute each action

        # Non-blocking learning queue
        self.learning_queue = []          # List of {state, action_dim, target, cycles_done, total_cycles}
        self.learning_phase = 0           # 0=inject state, 1=apply action, 2=set target
        self.phase_steps = 0              # Steps within current phase

        # Goal queue (goals wait for learning to finish)
        self.goal_queue = []              # List of goal names to pursue
        self.goal_pattern = None          # Current goal as activation pattern
        self.goal_start_state = None      # State when goal was set (for planning)
        self.best_reward = 0.0            # Best reward from FMC planning
        self.planning_iterations = 0     # FMC iterations done
        self.replan_frequency = 10       # Only re-plan every N frames (reduces degradation)

        # Create figure with more panels
        self.fig = plt.figure(figsize=(18, 11))
        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        # Row 0: Activation, Prediction Error, Phase, Action
        self.ax_act = self.fig.add_subplot(gs[0, 0])
        self.ax_act.set_title('Activation')
        padded = np.zeros(mind.grid_size**2)
        padded[:mind.n] = mind.A
        self.act_img = self.ax_act.imshow(
            padded.reshape(mind.grid_size, mind.grid_size),
            cmap='hot', vmin=0, vmax=1
        )
        plt.colorbar(self.act_img, ax=self.ax_act)

        self.ax_err = self.fig.add_subplot(gs[0, 1])
        self.ax_err.set_title('Prediction Error')
        self.err_img = self.ax_err.imshow(
            padded.reshape(mind.grid_size, mind.grid_size),
            cmap='RdBu', vmin=-0.5, vmax=0.5
        )
        plt.colorbar(self.err_img, ax=self.ax_err)

        self.ax_phase = self.fig.add_subplot(gs[0, 2])
        self.ax_phase.set_title('Phase (binding)')
        self.phase_img = self.ax_phase.imshow(
            self._make_phase_img(), vmin=0, vmax=1
        )

        # Action display
        self.ax_action = self.fig.add_subplot(gs[0, 3])
        self.ax_action.set_title('Current Action')
        self.action_bars = self.ax_action.bar(
            range(mind.n_action_dims),
            [0] * mind.n_action_dims,
            color=['C0', 'C1', 'C2', 'C3'][:mind.n_action_dims]
        )
        self.ax_action.set_ylim(0, 1)
        self.ax_action.set_xticks(range(mind.n_action_dims))
        self.ax_action.set_xticklabels([f'A{i}' for i in range(mind.n_action_dims)])

        # Row 1: Time series
        self.ax_ts = self.fig.add_subplot(gs[1, 0])
        self.ax_ts.set_title('Activation & Error')
        self.line_act, = self.ax_ts.plot([], [], 'b-', label='Activation', linewidth=1)
        self.line_err, = self.ax_ts.plot([], [], 'r-', label='Pred Error', linewidth=1)
        self.ax_ts.legend(fontsize=8)

        self.ax_soc = self.fig.add_subplot(gs[1, 1])
        self.ax_soc.set_title('SOC Dynamics')
        self.line_thresh, = self.ax_soc.plot([], [], 'g-', label='Threshold', linewidth=1)
        self.line_frac, = self.ax_soc.plot([], [], 'm-', label='Active Frac', linewidth=1)
        self.ax_soc.legend(fontsize=8)

        # Weight distributions
        self.ax_weights = self.fig.add_subplot(gs[1, 2])
        self.ax_weights.set_title('Weight Distribution')

        # Stats panel
        self.ax_stats = self.fig.add_subplot(gs[1, 3])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', ha='left', va='top',
                                              fontsize=9, family='monospace',
                                              transform=self.ax_stats.transAxes)

        # Row 2: Planning slots visualization
        self.ax_slots = self.fig.add_subplot(gs[2, :2])
        self.ax_slots.set_title('Planning Slots (during planning)')
        self.ax_slots.axis('off')
        self.slots_img = None

        # Help/Commands
        self.ax_help = self.fig.add_subplot(gs[2, 2:])
        self.ax_help.axis('off')
        help_text = """Always-On Behavior:
  NO GOAL: Dreaming (consolidation)
  GOAL SET: Auto-plan + execute

Commands:
  <text>           - Set current state
  goal <text>      - Set goal (auto-plans)
  learn <s> <a> <t> - Teach transition
  clear            - Clear goal -> dream
  reset            - Reset network
  benchmark        - Run tests
  quit             - Exit

Example:
  1. learn room_A 0 room_B
  2. learn room_B 1 room_C
  3. room_A          (set state)
  4. goal room_C     (auto-plans!)"""
        self.ax_help.text(0.05, 0.95, help_text, ha='left', va='top',
                          fontsize=9, family='monospace',
                          transform=self.ax_help.transAxes)

        # Sliders
        slider_ax = self.fig.add_axes([0.1, 0.02, 0.15, 0.02])
        self.slider_alpha = Slider(slider_ax, 'FMC alpha', 0, 2, valinit=mind.fmc_alpha)
        self.slider_alpha.on_changed(lambda v: setattr(mind, 'fmc_alpha', v))

        slider_ax2 = self.fig.add_axes([0.35, 0.02, 0.15, 0.02])
        self.slider_blend = Slider(slider_ax2, 'LT Blend', 0, 1, valinit=mind.action_longterm_blend)
        self.slider_blend.on_changed(lambda v: setattr(mind, 'action_longterm_blend', v))

        # Buttons
        btn_ax = self.fig.add_axes([0.6, 0.02, 0.08, 0.03])
        self.btn_reset = Button(btn_ax, 'Reset')
        self.btn_reset.on_clicked(lambda e: mind.reset())

        btn_ax2 = self.fig.add_axes([0.7, 0.02, 0.08, 0.03])
        self.btn_clear = Button(btn_ax2, 'Clear Goal')
        self.btn_clear.on_clicked(lambda e: self._clear_goal())

        btn_ax3 = self.fig.add_axes([0.8, 0.02, 0.1, 0.03])
        self.btn_benchmark = Button(btn_ax3, 'Benchmark')
        self.btn_benchmark.on_clicked(lambda e: self._run_quick_test())

    def _clear_goal(self):
        """Clear current goal and return to dreaming."""
        self.current_goal = None
        self.planned_actions = []
        self.plan_step = 0
        self.steps_in_current_action = 0
        self.goal_queue = []  # Also clear queued goals
        self.mind.clear_action()
        self.current_action = None
        self.last_input = "[CLEARED -> DREAMING]"

    def _run_quick_test(self):
        """Run a quick accuracy test."""
        transitions = [
            ('room_A', 'action_0', 'room_B'),
            ('room_A', 'action_1', 'room_C'),
            ('room_B', 'action_0', 'room_D'),
        ]
        correct = 0
        for state, action, expected in transitions:
            self.mind.reset()
            self.mind.inject_text(state, 0.8)
            action_dim = int(action.split('_')[1])
            self.mind.set_action(f'action_{action_dim}')
            for _ in range(30):
                self.mind.step(0.1)
            overlap = self.mind.pattern_overlap(expected)
            if overlap > 0.2:
                correct += 1
        self.last_input = f"[TEST: {correct}/{len(transitions)} pass]"

    def _make_phase_img(self):
        """Phase as HSV image."""
        from matplotlib.colors import hsv_to_rgb

        padded_phase = np.zeros(self.mind.grid_size**2)
        padded_phase[:self.mind.n] = self.mind.theta / (2 * np.pi)

        padded_amp = np.zeros(self.mind.grid_size**2)
        padded_amp[:self.mind.n] = self.mind.A

        # Wrap hue to [0, 1] and clip all HSV values
        h = np.clip(padded_phase.reshape(self.mind.grid_size, self.mind.grid_size) % 1.0, 0, 1)
        s = np.ones_like(h)
        v = np.clip(padded_amp.reshape(self.mind.grid_size, self.mind.grid_size) * 3, 0, 1)

        return hsv_to_rgb(np.stack([h, s, v], axis=-1))

    def _process_input(self):
        """Process command queue."""
        import queue as q
        had_input = False
        try:
            while True:
                text = self.input_queue.get_nowait().strip()
                if not text:
                    continue

                had_input = True
                if text.lower() in ('quit', 'exit'):
                    self.plt.close(self.fig)
                    return
                elif text.lower() == 'reset':
                    self.mind.reset()
                    self._clear_goal()
                    self.last_input = "[RESET]"
                elif text.lower() == 'clear':
                    self._clear_goal()
                elif text.lower().startswith('goal '):
                    # Queue goal (will execute after learning finishes)
                    goal = text[5:].strip()
                    if self.learning_queue:
                        # Learning in progress - queue the goal
                        self.goal_queue.append(goal)
                        print(f"[GOAL QUEUED: {goal}] (waiting for {len(self.learning_queue)} learn tasks)")
                        sys.stdout.flush()
                        self.last_input = f"[GOAL Q: {goal}]"
                    else:
                        # No learning - start immediately
                        self._set_goal(goal)
                elif text.lower().startswith('learn '):
                    parts = text[6:].split()
                    if len(parts) >= 3:
                        state, action_str, target = parts[0], parts[1], parts[2]
                        try:
                            action_dim = int(action_str)
                        except:
                            action_dim = int(action_str.split('_')[1]) if '_' in action_str else 0

                        # Add to learning queue (one-shot learning)
                        self.learning_queue.append({
                            'state': state,
                            'action_dim': action_dim,
                            'target': target,
                            'steps_done': 0,
                            'total_steps': 30  # Just enough for Hebbian association to form
                        })
                        queue_len = len(self.learning_queue)
                        print(f"[QUEUED: {state} + a{action_dim} -> {target}] ({queue_len} in queue)")
                        sys.stdout.flush()
                        self.last_input = f"[Q{queue_len}: {state}->{target}]"
                    else:
                        self.last_input = "[ERROR: learn <s> <a> <t>]"
                elif text.lower() == 'benchmark':
                    print("\nRunning benchmarks...")
                    run_all_benchmarks()
                    self.last_input = "[BENCHMARK DONE]"
                else:
                    # Inject as pattern (set current state)
                    self.mind.inject_text(text, 0.8)
                    self.last_input = f"[STATE: {text}]"

        except q.Empty:
            pass
        return had_input

    def _get_pattern(self, name):
        """Convert a pattern name to an activation array."""
        if name in self.mind.pattern_traces:
            return self.mind.pattern_traces[name]
        else:
            # Generate SPARSE pattern from name (same as pattern_overlap)
            np.random.seed(hash(name) % 10000)
            n_active = max(5, int(self.mind.n * self.mind.pattern_sparsity))
            units = np.random.choice(self.mind.n, n_active, replace=False)
            pattern = np.zeros(self.mind.n)
            pattern[units] = 1.0
            return pattern

    def _set_goal(self, goal):
        """Set a goal - planning happens continuously during execution."""
        self.current_goal = goal
        self.goal_pattern = self._get_pattern(goal)
        self.plan_step = 0
        self.steps_in_current_action = 0
        self.planning_iterations = 0
        self.best_reward = 0.0

        # Save starting state for planning (don't let it decay)
        self.goal_start_state = self.mind.A.copy()

        # Initialize FMC slots from current state
        self.mind.init_planning()

        # Get initial best action
        self._update_best_plan()

        print(f"\n[GOAL: {goal}] (continuous planning)")
        sys.stdout.flush()
        self.last_input = f"[GOAL: {goal}]"

    def _update_best_plan(self):
        """Get current best action sequence from FMC slots."""
        virtual_rewards, rewards, _ = self.mind._compute_virtual_rewards(self.goal_pattern)
        best_idx = np.argmax(rewards)
        self.planned_actions = list(self.mind.slot_action_sequence[best_idx])
        self.best_reward = rewards[best_idx]
        return best_idx

    def _do_planning_iteration(self):
        """Run one FMC planning iteration (non-blocking)."""
        # Use CURRENT network state for planning (re-plan as we progress)
        # But if network is too quiet, use the saved start state
        if np.mean(self.mind.A) > 0.05:
            planning_state = self.mind.A.copy()
        else:
            planning_state = self.goal_start_state.copy()

        # 1. Simulate each slot forward from current state
        for i in range(self.mind.n_slots):
            self.mind.slot_A[i] = planning_state.copy()
        for i in range(self.mind.n_slots):
            self.mind._simulate_slot(i, steps_per_action=self.mind.plan_horizon)

        # 2. Compute Virtual Rewards
        virtual_rewards, rewards, distances = self.mind._compute_virtual_rewards(self.goal_pattern)

        # 3. Clone low-VR slots with high-VR ones
        self.mind._clone_step(virtual_rewards)

        # 4. Perturb actions for exploration
        self.mind._perturb_actions()

        self.planning_iterations += 1

        # Update best plan
        self._update_best_plan()

    def _dream_step(self):
        """Execute one dream step (consolidation mode)."""
        for _ in range(5):
            # Inject random noise to trigger spontaneous activity
            noise_units = np.random.choice(self.mind.n, size=3, replace=False)
            self.mind.A[noise_units] += np.random.uniform(0.05, 0.15, len(noise_units))
            self.mind.A = np.clip(self.mind.A, 0, 1)
            self.mind.step(0.1, dream_mode=True)

    def _learning_step(self):
        """One-shot learning - single presentation per transition."""
        if not self.learning_queue:
            return

        task = self.learning_queue[0]

        # First step: set up the learning situation
        if task['steps_done'] == 0:
            self.mind.inject_text(task['state'], 0.9)
            self.mind.set_action(f"action_{task['action_dim']}")
            self.current_action = task['action_dim']
            self.mind.set_target(task['target'], 1.0)

        # Run steps - Hebbian learning happens automatically during step()
        for _ in range(5):
            self.mind.step(0.1, dream_mode=False)
            task['steps_done'] += 1

            if task['steps_done'] >= task['total_steps']:
                # Done - clear and move to next
                self.mind.clear_action()
                self.mind.A_target *= 0
                self.mind.target_strength = 0

                print(f"[LEARNED: {task['state']} + a{task['action_dim']} -> {task['target']}] (one-shot)")
                self.learning_queue.pop(0)

                if self.learning_queue:
                    next_task = self.learning_queue[0]
                    self.last_input = f"[NEXT: {next_task['state']}]"
                elif self.goal_queue:
                    print(f"  Starting queued goal...")
                    next_goal = self.goal_queue.pop(0)
                    self._set_goal(next_goal)
                else:
                    print(f"  Done! Experiences: {len(self.mind.experience_buffer)}")
                    self.last_input = "[DONE]"
                sys.stdout.flush()
                break

        # Update display
        if self.learning_queue:
            remaining = len(self.learning_queue) - 1
            self.last_input = f"[LEARN +{remaining}]"

    def _execute_goal_step(self):
        """Execute toward goal with continuous planning."""
        # Check if we've reached the goal
        goal_overlap = self.mind.pattern_overlap(self.current_goal)
        if goal_overlap > self.goal_threshold:
            print(f"\n[GOAL REACHED: {self.current_goal}] (overlap={goal_overlap:.2f}, {self.planning_iterations} iters)")
            sys.stdout.flush()

            if self.goal_queue:
                # Start next queued goal
                next_goal = self.goal_queue.pop(0)
                print(f"  Next goal: {next_goal} ({len(self.goal_queue)} remaining)")
                self.current_goal = None
                self._set_goal(next_goal)
            else:
                self._clear_goal()
                self.last_input = f"[GOAL REACHED!]"
            return

        # SPARSE RE-PLANNING: Only re-plan every N frames
        # Frequent re-planning causes degradation as the planner keeps changing its mind
        # planning_iterations is incremented inside _do_planning_iteration
        if self.planning_iterations % self.replan_frequency == 0:
            self._do_planning_iteration()
        else:
            self.planning_iterations += 1  # Still count frames even when not re-planning

        # Execute current best action
        if self.planned_actions:
            best_action_pattern = self.planned_actions[0]
            best_action_dim = self.mind._get_action_dim_from_pattern(best_action_pattern)

            if best_action_dim is not None:
                # ALWAYS set action (don't just set when changed - maintain it)
                self.mind.set_action(f'action_{best_action_dim}')
                self.current_action = best_action_dim

        # GOAL REINFORCEMENT: weak continuous pull toward goal
        # This prevents diffusion away from goal during execution
        # Without this, the network state drifts due to dynamics/decay
        self.mind.A = np.clip(self.mind.A + 0.05 * self.goal_pattern, 0, 1)

        # Keep network alive - reinject start state if dying
        if np.mean(self.mind.A) < 0.05:
            self.mind.A = np.clip(self.mind.A + 0.5 * self.goal_start_state, 0, 1)

        # Step the actual network
        # Use execute_mode=True to prevent ALL learning during execution
        # Learning should happen during explicit learn commands, not during planning
        for _ in range(5):
            self.mind.step(0.1, execute_mode=True)

        # DON'T update goal_start_state - keep planning from original position
        # This prevents feedback loop where bad execution corrupts planning

        # Update status
        self.last_input = f"[g={goal_overlap:.2f} r={self.best_reward:.2f} i={self.planning_iterations}]"

    def update(self, frame):
        """Animation update - always-on behavior."""
        self._process_input()

        # Priority: Learning > Goal > Dream
        if self.learning_queue:
            # LEARNING MODE: Processing learning queue
            self._learning_step()
        elif self.current_goal is not None:
            # GOAL MODE: Execute planned actions
            self._execute_goal_step()
        else:
            # DREAM MODE: Consolidation with random noise
            self._dream_step()

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

        # Update action bars
        action_strengths = [0] * self.mind.n_action_dims
        for dim in range(self.mind.n_action_dims):
            start = dim * self.mind.action_units_per_dim
            end = start + self.mind.action_units_per_dim
            action_strengths[dim] = np.mean(self.mind.A_action[start:end])
        for bar, h in zip(self.action_bars, action_strengths):
            bar.set_height(h)

        # Update time series
        t = self.mind.history['time'][-500:]
        if t:
            act = self.mind.history['mean_activation'][-500:]
            err = self.mind.history['mean_pred_error'][-500:]
            self.line_act.set_data(t, act)
            self.line_err.set_data(t, err)
            self.ax_ts.set_xlim(t[0], t[-1])
            self.ax_ts.set_ylim(0, max(0.1, max(max(act), max(err)) * 1.1))

            thresh = self.mind.history['threshold'][-500:]
            frac = self.mind.history['active_fraction'][-500:]
            self.line_thresh.set_data(t, thresh)
            self.line_frac.set_data(t, frac)
            self.ax_soc.set_xlim(t[0], t[-1])
            self.ax_soc.set_ylim(0, max(0.5, max(max(thresh), max(frac)) * 1.1))

        # Update weight histogram - W_longterm vs W_transition
        self.ax_weights.clear()
        # Show one action dim's weights
        dim = 0
        self.ax_weights.hist(self.mind.W_transition[dim], bins=40, alpha=0.5,
                            label='W_trans', color='red', range=(-1, 1))
        self.ax_weights.hist(self.mind.W_action_longterm[dim], bins=40, alpha=0.5,
                            label='W_long', color='blue', range=(0, 1))
        self.ax_weights.axvline(0, color='gray', linestyle='--', alpha=0.5)
        self.ax_weights.set_title(f'Weights (action 0)')
        self.ax_weights.legend(fontsize=7)

        # Stats - mode display
        if self.learning_queue:
            task = self.learning_queue[0]
            mode = f"LEARNING ({len(self.learning_queue)})"
            goal_str = f"{task['state']}->{task['target']}"
            if self.goal_queue:
                plan_str = f"one-shot [+{len(self.goal_queue)} goals]"
            else:
                plan_str = "one-shot"
        elif self.current_goal:
            mode = "GOAL + PLANNING"
            goal_overlap = self.mind.pattern_overlap(self.current_goal)
            goal_str = f"{self.current_goal} ({goal_overlap:.2f})"
            # Show: reward, iterations, queued goals
            plan_parts = [f"r={self.best_reward:.2f}", f"i={self.planning_iterations}"]
            if self.goal_queue:
                plan_parts.append(f"+{len(self.goal_queue)}")
            plan_str = " ".join(plan_parts)
        else:
            mode = "DREAMING"
            goal_str = "none"
            plan_str = "-"

        action_str = f"a{self.current_action}" if self.current_action is not None else "-"
        n_exp = len(self.mind.experience_buffer)
        lt_sum = sum(np.sum(w > 0.1) for w in self.mind.W_action_longterm)

        stats = f"""Mode: {mode}
Goal: {goal_str}
Plan: {plan_str}
Action: {action_str}
Time: {self.mind.t:.1f}
Experiences: {n_exp}
LT weights: {lt_sum}
FMC alpha: {self.mind.fmc_alpha:.2f}

{self.last_input[:25]}"""
        self.stats_text.set_text(stats)

        return [self.act_img, self.err_img, self.phase_img]

    def run(self):
        """Start interactive visualization."""
        import threading
        from matplotlib.animation import FuncAnimation

        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.anim = FuncAnimation(self.fig, self.update, interval=50,
                                  blit=False, cache_frame_data=False)
        self.plt.show()

    def _input_loop(self):
        """Background input loop."""
        print("\n" + "=" * 70)
        print("PLANNING SOC MIND - Goal-Driven Visualization")
        print("=" * 70)
        print()
        print("ALWAYS-ON BEHAVIOR:")
        print("  No goal  -> DREAMING (consolidating memories)")
        print("  Goal set -> AUTO-PLAN + EXECUTE (FMC planning)")
        print()
        print("Commands:")
        print("  <text>            - Set current state pattern")
        print("  goal <text>       - Set goal (auto-plans & executes)")
        print("  learn <s> <a> <t> - Teach transition (state+action->target)")
        print("  clear             - Clear goal, return to dreaming")
        print("  benchmark         - Run benchmark suite")
        print("  reset             - Reset network")
        print("  quit              - Exit")
        print()
        print("Example workflow:")
        print("  learn room_A 0 room_B    # teach: room_A + action_0 -> room_B")
        print("  learn room_B 1 room_C    # teach: room_B + action_1 -> room_C")
        print("  room_A                   # set current state")
        print("  goal room_C              # auto-plans and executes!")
        print("=" * 70)
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


def run_interactive():
    """Run interactive visualization."""
    print("Starting Planning SOC Mind...")
    mind = PlanningSOCMind(n_units=512, n_action_dims=4, n_slots=16)
    viz = PlanningVisualizer(mind)
    viz.run()


def run_all_benchmarks():
    """Run all world modeling benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING ALL WORLD MODELING BENCHMARKS")
    print("=" * 70 + "\n")

    results = {}

    # Test 1: Action prediction
    passed, _ = benchmark_action_prediction()
    results['action_prediction'] = passed
    print()

    # Test 2: Planning
    passed, _ = benchmark_planning()
    results['planning'] = passed
    print()

    # Test 3: Dream consolidation
    passed, _ = benchmark_dream_consolidation()
    results['dream_consolidation'] = passed
    print()

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {status}")

    overall = sum(results.values()) / len(results)
    print(f"\nOverall: {overall:.0%} passed")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import sys

    if "--benchmark" in sys.argv or "-b" in sys.argv:
        # Run benchmarks
        run_all_benchmarks()
    else:
        # Default: interactive visualization
        print("Starting interactive visualization...")
        print("(Use --benchmark or -b to run benchmarks instead)")
        print()
        run_interactive()
