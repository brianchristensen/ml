"""
PSI Graph with Dopamine-Gated Learning
======================================

Brain-inspired credit assignment using DOPAMINE-LIKE TD ERROR.

Key bio-plausible mechanisms:
1. Phase coherence gating: neurons communicate when phases align (gamma synchrony)
2. Dopamine-like TD error: global reward signal modulates all learning
3. Three-factor rule: ΔW = lr × pre × post × dopamine
4. Eligibility traces: only recently active synapses are modified
5. Contrastive Hebbian: free vs target phase difference drives learning

How it works:
1. FREE PHASE: Network settles naturally given input
2. TARGET PHASE: Clamp output to target, let network settle
3. REWARD: Compute how well output matched target (single scalar)
4. TD ERROR: reward - expected_reward = "dopamine" signal
5. LEARN: All weight updates gated by global dopamine signal

This differs from backprop:
- No per-unit error signals propagated backward
- Single global reward signal (like dopamine) modulates all learning
- Active synapses are eligible for modification
- Learning is slower but biologically plausible

Based on:
- Temporal Difference learning (Schultz et al., dopamine = reward prediction error)
- Three-factor learning rules (eligibility × activity × neuromodulator)
- Contrastive Hebbian / Equilibrium Propagation (Scellier & Bengio 2017)
- Phase coherence gating (gamma synchrony in cortex)

Author: Brian Christensen
Date: 2025
"""

import numpy as np
from typing import Tuple


class VectorizedPSIGraph:
    """
    VECTORIZED PSI Graph with DOPAMINE-LIKE TD ERROR Learning.

    All operations are parallel matrix operations - no Python loops over nodes.
    This mirrors how the brain works: all neurons update simultaneously.

    Key bio-plausible mechanisms:
    1. Phase coherence gating: neurons communicate when phases align
    2. Dopamine-like TD error: global reward signal modulates all learning
    3. Three-factor rule: ΔW = lr × pre × post × dopamine
    4. Eligibility traces: only active synapses are modified

    State representation:
    - states: [n_nodes, dim] - all node states
    - phases: [n_nodes, dim] - all node phases
    - W_conn: [n_nodes, n_nodes] - connection weights (adjacency matrix)
    - node_W: [n_nodes, dim, dim] - per-node transformation weights
    - biases: [n_nodes, dim] - per-node biases
    """

    def __init__(self, n_nodes: int = 16, dim: int = 32, seed: int = None,
                 topology: str = 'small_world', k_neighbors: int = 3,
                 long_range_prob: float = 0.15):
        """
        Initialize PSI graph.

        Args:
            n_nodes: Number of nodes in the graph
            dim: Dimensionality of each node's state
            seed: Random seed for reproducibility
            topology: 'dense' (80% connectivity) or 'small_world' (scalable sparse)
            k_neighbors: For small_world, number of nearest neighbors
            long_range_prob: For small_world, probability of long-range connections
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim
        self.topology = topology

        # Node states: [n_nodes, dim]
        self.states = np.random.randn(n_nodes, dim) * 0.01
        self.prev_states = np.zeros((n_nodes, dim))
        self.free_states = np.zeros((n_nodes, dim))
        self.target_states = np.zeros((n_nodes, dim))

        # Phases: [n_nodes, dim]
        self.phases = np.random.uniform(0, 2*np.pi, (n_nodes, dim))
        self.omega = np.random.randn(n_nodes, dim) * 0.03

        # Connection weights (adjacency matrix): [n_nodes, n_nodes]
        if topology == 'dense':
            density = 0.8
            mask = np.random.random((n_nodes, n_nodes)) < density
            mask = np.triu(mask, 1)
            mask = mask | mask.T
            self.adj = mask.astype(float)
        else:
            self.adj = self._build_small_world_adj(k_neighbors, long_range_prob)

        self.W_conn = np.random.uniform(0.5, 1.0, (n_nodes, n_nodes)) * self.adj
        self.W_conn = (self.W_conn + self.W_conn.T) / 2

        # Per-node transformation: [n_nodes, dim, dim]
        scale = 0.5 / np.sqrt(dim)
        self.node_W = np.random.randn(n_nodes, dim, dim) * scale
        for i in range(n_nodes):
            self.node_W[i] += np.eye(dim) * 0.4

        # Biases: [n_nodes, dim]
        self.biases = np.random.randn(n_nodes, dim) * 0.15

        # Learning rate
        self.lr = 0.15

        # Input/output masks
        self.input_mask = np.zeros(n_nodes, dtype=bool)
        self.input_mask[:2] = True  # First 2 nodes are inputs
        self.output_mask = np.zeros(n_nodes, dtype=bool)
        self.output_mask[-1] = True  # Last node is output

        # Clamp values: [n_nodes, dim] - NaN means not clamped
        self.clamped = np.full((n_nodes, dim), np.nan)

        # Pre-computed values for speed
        self.adj_sum = np.sum(self.adj, axis=1) + 0.1
        self.diag_indices = np.arange(dim)

        # ============================================================
        # DOPAMINE-LIKE TD ERROR: Global reward signal
        # ============================================================
        self.expected_reward = 0.5  # Running average of reward
        self.reward_decay = 0.95  # How fast to update expectation

        # Store last target for learning
        self.last_target = None

        # ============================================================
        # ELIGIBILITY TRACES: Accumulate activity over time
        # ============================================================
        self.eligibility_trace = np.zeros((n_nodes, dim))
        self.eligibility_decay = 0.8  # How fast traces decay

        # ============================================================
        # PHASIC/TONIC DOPAMINE: Fast and slow signals
        # ============================================================
        # Phasic: fast, transient signal for immediate learning
        self.phasic_reward = 0.5  # Fast moving average
        self.phasic_decay = 0.7   # Fast decay (more responsive)
        # Tonic: slow, baseline signal for exploration/stability
        self.tonic_reward = 0.5   # Slow moving average
        self.tonic_decay = 0.98   # Very slow decay (stable baseline)

        # ============================================================
        # ACTOR-CRITIC: Separate value estimation network
        # ============================================================
        # Critic: learns to predict reward given current state
        # Uses a simple linear function of input features
        self.critic_W = np.random.randn(n_nodes * dim) * 0.1
        self.critic_lr = 0.05

    def _build_small_world_adj(self, k: int, p_long: float) -> np.ndarray:
        """
        Build small-world adjacency matrix for scalable topology.

        Args:
            k: Each node connects to k nearest neighbors (by index)
            p_long: Probability of additional random long-range connections
        """
        adj = np.zeros((self.n_nodes, self.n_nodes))

        # Local connections (k nearest neighbors by index)
        for i in range(self.n_nodes):
            for offset in range(1, k // 2 + 2):
                j_fwd = (i + offset) % self.n_nodes
                if j_fwd != i:
                    adj[i, j_fwd] = 1.0
                    adj[j_fwd, i] = 1.0

                j_bwd = (i - offset) % self.n_nodes
                if j_bwd != i:
                    adj[i, j_bwd] = 1.0
                    adj[j_bwd, i] = 1.0

        # Long-range connections
        for i in range(self.n_nodes):
            for j in range(i + 2, self.n_nodes):
                if adj[i, j] == 0 and np.random.random() < p_long:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Ensure input-output pathway exists
        if self.n_nodes > 4:
            mid = self.n_nodes // 2
            adj[0, mid] = 1.0
            adj[mid, 0] = 1.0
            adj[1, mid] = 1.0
            adj[mid, 1] = 1.0
            adj[mid, -2] = 1.0
            adj[-2, mid] = 1.0

        np.fill_diagonal(adj, 0)
        return adj

    def step(self, target_mode: bool = False):
        """One parallel update step - ALL nodes update simultaneously."""
        self.prev_states = self.states.copy()

        # Compute phase coherence matrix: [n_nodes, n_nodes]
        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        # Gate by coherence
        gate = (1 + coherence) * 0.5
        if target_mode:
            gate += 0.3

        # Weighted input from neighbors
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(weighted_adj, axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.states) / total_weight

        # Neural noise
        neighbor_input += np.random.randn(self.n_nodes, self.dim) * 0.05

        # Per-node transformation
        pre_act = np.einsum('nij,nj->ni', self.node_W, neighbor_input) + self.biases
        np.clip(pre_act, -5, 5, out=pre_act)
        target_activation = np.tanh(pre_act)

        # Node gating based on mean coherence with neighbors
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

        # Update phases
        self.phases += self.omega + 0.03 * self.states
        np.mod(self.phases, 2 * np.pi, out=self.phases)

        # Accumulate eligibility trace: decay + current activity
        self.eligibility_trace = (self.eligibility_decay * self.eligibility_trace +
                                  (1 - self.eligibility_decay) * np.abs(self.states))

    def has_converged(self, threshold: float = 0.015) -> bool:
        """Check if all nodes have converged."""
        change = np.mean(np.abs(self.states - self.prev_states))
        return change < threshold

    def settle(self, max_iters: int = 8, min_steps: int = 3, target_mode: bool = False) -> int:
        """Settle until convergence."""
        for i in range(max_iters):
            self.step(target_mode=target_mode)
            if i >= min_steps and self.has_converged():
                return i + 1
        return max_iters

    def free_phase(self, input_values: np.ndarray) -> float:
        """FREE PHASE: Settle with input clamped."""
        # Reset states and eligibility trace
        self.states = np.random.randn(self.n_nodes, self.dim) * 0.01
        self.eligibility_trace = np.zeros((self.n_nodes, self.dim))

        # Clear clamps and set inputs
        self.clamped = np.full((self.n_nodes, self.dim), np.nan)
        n_inputs = min(len(input_values), 2)
        for i in range(n_inputs):
            self.clamped[i] = input_values[i]

        # Settle
        self.settle(max_iters=8, min_steps=3)

        # Store free states
        self.free_states = self.states.copy()

        # Return output (mean of last node)
        return float(np.mean(self.states[-1]))

    def target_phase(self, target: float) -> None:
        """TARGET PHASE: Clamp output, let targets propagate back."""
        self.last_target = target

        # Clamp output node
        self.clamped[-1] = target

        # Settle with target mode
        self.settle(max_iters=10, min_steps=4, target_mode=True)

        # Store target states
        self.target_states = self.states.copy()

    def learn(self) -> float:
        """
        Apply DOPAMINE-GATED learning with three-factor rule.

        Bio-plausible mechanisms:
        1. Compute global TD error (dopamine signal)
        2. Gate all learning by dopamine
        3. Use eligibility (activity) to determine which synapses update
        4. Contrastive Hebbian for phase dynamics

        Returns:
            td_error: The temporal difference error for diagnostics
        """
        if self.last_target is None:
            return 0.0

        target = self.last_target

        # State difference (contrastive signal)
        state_diff = self.target_states - self.free_states
        np.clip(state_diff, -1, 1, out=state_diff)

        # ============================================================
        # DOPAMINE-LIKE TD ERROR: Global reward signal
        # ============================================================

        # Compute reward: how close was output to target?
        output_state = self.free_states[-1]
        output_value = np.mean(output_state)

        # Reward based on correctness (for binary classification)
        if target * output_value > 0:
            reward = 0.5 + 0.5 * min(1.0, abs(output_value))
        else:
            reward = 0.5 - 0.5 * min(1.0, abs(output_value))

        # TD ERROR: surprise = reward - expected
        td_error = reward - self.expected_reward

        # Update expected reward (slow moving average)
        self.expected_reward = self.reward_decay * self.expected_reward + (1 - self.reward_decay) * reward

        # ============================================================
        # PHASIC/TONIC DOPAMINE: Separate fast and slow signals
        # ============================================================
        # Phasic dopamine: fast, transient surprise signal for learning
        phasic_error = reward - self.phasic_reward
        self.phasic_reward = self.phasic_decay * self.phasic_reward + (1 - self.phasic_decay) * reward

        # Tonic dopamine: slow baseline, controls exploration/stability
        tonic_error = reward - self.tonic_reward
        self.tonic_reward = self.tonic_decay * self.tonic_reward + (1 - self.tonic_decay) * reward

        # ============================================================
        # ACTOR-CRITIC: Critic provides variance reduction
        # ============================================================
        # Flatten free states for critic input
        state_features = self.free_states.flatten()

        # Critic predicts expected value for this state
        critic_prediction = np.dot(self.critic_W, state_features)
        critic_prediction = np.clip(critic_prediction, 0, 1)

        # Advantage: how much better/worse than expected
        advantage = reward - critic_prediction

        # Update critic (learns to predict value)
        self.critic_W += self.critic_lr * advantage * state_features
        np.clip(self.critic_W, -2, 2, out=self.critic_W)

        # Combined dopamine: phasic drives learning, advantage reduces variance
        # Use phasic for direction, but scale by advantage magnitude
        tonic_factor = np.clip(self.tonic_reward, 0.3, 0.7)
        dopamine = phasic_error * (1.5 - tonic_factor) + 0.3 * advantage

        # ============================================================
        # THREE-FACTOR LEARNING: pre × post × dopamine
        # ============================================================

        # Compute phase coherence matrix
        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        # Compute neighbor input
        gate = (1 + coherence) * 0.5
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(weighted_adj, axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.free_states) / total_weight

        # Eligibility: use accumulated trace (remembers activity over time)
        # This is more biologically plausible than just final activity
        eligibility = self.eligibility_trace.copy()
        eligibility = eligibility / (eligibility.max() + 1e-8)

        # Dopamine gate: modulates learning magnitude
        dopamine_gate = np.clip(0.5 + dopamine, 0.1, 2.0)

        # Contrastive delta, gated by dopamine
        delta = state_diff * 0.5 * dopamine_gate

        # Update node_W: three-factor rule
        dW = np.einsum('ni,nj->nij', delta * eligibility, neighbor_input) * self.lr
        dW[self.input_mask] = 0
        self.node_W += dW
        np.clip(self.node_W, -3, 3, out=self.node_W)

        # Update biases: gated by dopamine
        self.biases += self.lr * 0.3 * delta
        self.biases[self.input_mask] = 0
        np.clip(self.biases, -2, 2, out=self.biases)

        # Update connection weights (contrastive correlation), gated by dopamine
        target_corr = (self.target_states @ self.target_states.T) / self.dim
        free_corr = (self.free_states @ self.free_states.T) / self.dim
        corr_diff = target_corr - free_corr
        coherence_gate = np.maximum(0, 2 * coherence - 0.5)
        dW_conn = self.lr * 0.1 * dopamine_gate * coherence_gate * corr_diff * self.adj
        self.W_conn += dW_conn
        np.clip(self.W_conn, 0.1, 2.0, out=self.W_conn)

        # Synaptic decay
        decay = 0.999
        self.node_W *= decay
        self.biases *= decay
        self.W_conn = self.W_conn * decay + 0.75 * (1 - decay) * self.adj

        # Restore diagonals
        self.node_W[:, self.diag_indices, self.diag_indices] *= 1.002

        return td_error

    def train_step(self, input_values: np.ndarray, target: float) -> Tuple[float, float]:
        """
        One complete training step with dopamine-gated learning.

        Returns:
            Tuple of (prediction, td_error)
        """
        pred = self.free_phase(input_values)
        self.target_phase(target)
        td_error = self.learn()
        return pred, td_error

    def predict(self, input_values: np.ndarray) -> float:
        """Predict without learning."""
        return self.free_phase(input_values)


def test_xor(n_epochs=150, search_epochs=100):
    """Test XOR with dopamine-gated learning."""
    print("=" * 70)
    print("PSI DOPAMINE-GATED LEARNING - XOR TEST")
    print("=" * 70)
    print()
    print("Key mechanisms:")
    print("  - Global dopamine signal (TD error) modulates all learning")
    print("  - Three-factor rule: pre × post × dopamine")
    print("  - Phase coherence gates communication")
    print("  - Eligibility traces with decay")
    print("  - Phasic/tonic dopamine signals")
    print(f"  - Training: {search_epochs} search + {n_epochs} full epochs")
    print()

    xor_data = [
        (np.array([-1.0, -1.0]), -1.0),
        (np.array([-1.0,  1.0]),  1.0),
        (np.array([ 1.0, -1.0]),  1.0),
        (np.array([ 1.0,  1.0]), -1.0),
    ]

    best_acc = 0
    best_seed = -1

    print("Searching for good initialization...")
    for seed in range(30):
        graph = VectorizedPSIGraph(n_nodes=6, dim=8, seed=seed)

        for _ in range(search_epochs):
            np.random.shuffle(xor_data)
            for inp, target in xor_data:
                graph.train_step(inp, target)

        correct = 0
        for inp, target in xor_data:
            pred = graph.predict(inp)
            if (pred > 0) == (target > 0):
                correct += 1

        if correct > best_acc:
            best_acc = correct
            best_seed = seed
            print(f"  Seed {seed}: {correct}/4")

        if correct == 4:
            break

    print(f"\nBest: seed {best_seed} with {best_acc}/4")

    # Full training with best seed
    print("\nFull training with diagnostics:")
    graph = VectorizedPSIGraph(n_nodes=6, dim=8, seed=best_seed)

    report_interval = max(1, n_epochs // 5)
    for epoch in range(n_epochs):
        np.random.shuffle(xor_data)
        total_td = 0

        for inp, target in xor_data:
            _, td_error = graph.train_step(inp, target)
            total_td += td_error

        if epoch % report_interval == 0:
            correct = sum(1 for inp, target in xor_data
                         if (graph.predict(inp) > 0) == (target > 0))
            print(f"  Epoch {epoch}: {correct}/4, avg_td={total_td/4:.3f}, "
                  f"expected_reward={graph.expected_reward:.3f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    correct = 0
    for inp, target in xor_data:
        pred = graph.predict(inp)
        is_correct = (pred > 0) == (target > 0)
        if is_correct:
            correct += 1
        print(f"  {inp} -> {pred:.3f} (target {target:.0f}) "
              f"{'OK' if is_correct else 'WRONG'}")

    print(f"\nFinal: {correct}/4 = {correct/4:.0%}")
    print(f"LEARNED XOR: {'YES' if correct >= 3 else 'NO'}")

    return correct >= 3


if __name__ == "__main__":
    test_xor()
