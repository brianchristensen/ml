"""
Test fix for scaling problem.

Root cause: Fixed 80% density means more nodes = more neighbors to average over,
diluting both forward signal and backward learning signal.

Fix: Use CONSTANT neighborhood size regardless of total node count.
- Small-world topology: each node connects to k nearest neighbors + random long-range
- This keeps information flow focused rather than diffuse
"""

import numpy as np
import time


class ScalablePSIGraph:
    """
    PSI Graph with topology that scales properly.

    Key change: Instead of density-based connectivity, use k-nearest neighbors
    with sparse long-range connections (small-world network).
    """

    def __init__(self, n_nodes: int, dim: int, seed: int = None,
                 k_neighbors: int = 3, long_range_prob: float = 0.1):
        """
        k_neighbors: Each node connects to k nearest neighbors (by index)
        long_range_prob: Probability of additional random long-range connections
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim
        self.k_neighbors = k_neighbors

        # Node states
        self.states = np.random.randn(n_nodes, dim) * 0.01
        self.prev_states = np.zeros((n_nodes, dim))
        self.free_states = np.zeros((n_nodes, dim))
        self.target_states = np.zeros((n_nodes, dim))

        # Phases
        self.phases = np.random.uniform(0, 2*np.pi, (n_nodes, dim))
        self.omega = np.random.randn(n_nodes, dim) * 0.03

        # Build SPARSE adjacency - small world topology
        self.adj = self._build_small_world_adj(k_neighbors, long_range_prob)

        # Connection weights
        self.W_conn = np.random.uniform(0.5, 1.0, (n_nodes, n_nodes)) * self.adj
        self.W_conn = (self.W_conn + self.W_conn.T) / 2

        # Per-node transformation
        scale = 0.5 / np.sqrt(dim)
        self.node_W = np.random.randn(n_nodes, dim, dim) * scale
        for i in range(n_nodes):
            self.node_W[i] += np.eye(dim) * 0.4

        # Biases
        self.biases = np.random.randn(n_nodes, dim) * 0.15

        # Learning rate
        self.lr = 0.25

        # Input/output masks
        self.input_mask = np.zeros(n_nodes, dtype=bool)
        self.input_mask[:2] = True
        self.output_mask = np.zeros(n_nodes, dtype=bool)
        self.output_mask[-1] = True

        # Clamp values
        self.clamped = np.full((n_nodes, dim), np.nan)

        # Holographic memory
        self.memory_real = np.zeros((n_nodes, dim))
        self.memory_imag = np.zeros((n_nodes, dim))
        self.memory_strength = np.zeros(n_nodes)

    def _build_small_world_adj(self, k: int, p_long: float) -> np.ndarray:
        """
        Build small-world adjacency matrix.

        - Each node i connects to nodes i-k/2 to i+k/2 (wraparound)
        - Plus random long-range connections with probability p_long
        """
        adj = np.zeros((self.n_nodes, self.n_nodes))

        # Local connections (k nearest neighbors by index)
        for i in range(self.n_nodes):
            for offset in range(1, k // 2 + 2):
                # Forward neighbor
                j_fwd = (i + offset) % self.n_nodes
                if j_fwd != i:
                    adj[i, j_fwd] = 1.0
                    adj[j_fwd, i] = 1.0

                # Backward neighbor
                j_bwd = (i - offset) % self.n_nodes
                if j_bwd != i:
                    adj[i, j_bwd] = 1.0
                    adj[j_bwd, i] = 1.0

        # Long-range connections (skip connections for faster propagation)
        for i in range(self.n_nodes):
            for j in range(i + 2, self.n_nodes):
                if adj[i, j] == 0 and np.random.random() < p_long:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Ensure input-output pathway exists (critical for learning)
        # Add direct connections from input nodes to some middle nodes
        # and from middle nodes to output
        if self.n_nodes > 4:
            mid = self.n_nodes // 2
            # Input nodes (0, 1) connect to middle region
            adj[0, mid] = 1.0
            adj[mid, 0] = 1.0
            adj[1, mid] = 1.0
            adj[mid, 1] = 1.0
            # Middle connects to near-output
            adj[mid, -2] = 1.0
            adj[-2, mid] = 1.0

        np.fill_diagonal(adj, 0)  # No self-connections
        return adj

    def step(self, target_mode: bool = False):
        self.prev_states = self.states.copy()

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) / 2
        if target_mode:
            gate = gate + 0.3

        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(np.abs(weighted_adj), axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.states) / total_weight

        noise = np.random.randn(self.n_nodes, self.dim) * 0.05
        neighbor_input = neighbor_input + noise

        pre_act = np.einsum('nij,nj->ni', self.node_W, neighbor_input) + self.biases
        pre_act = np.clip(pre_act, -5, 5)
        target_activation = np.tanh(pre_act)

        mean_node_coherence = np.mean(coherence * self.adj, axis=1) / (np.sum(self.adj, axis=1) + 0.1)
        coherence_threshold = np.mean(mean_node_coherence)
        node_gate = np.clip(1 + 2 * (mean_node_coherence - coherence_threshold), 0.3, 1.5)
        target_activation = target_activation * node_gate[:, np.newaxis]

        leak = 0.7 if target_mode else 0.5
        new_states = (1 - leak) * self.states + leak * target_activation

        clamped_mask = ~np.isnan(self.clamped)
        new_states = np.where(clamped_mask, self.clamped, new_states)

        self.states = new_states
        self.phases += self.omega + 0.03 * self.states
        self.phases = np.mod(self.phases, 2 * np.pi)

    def has_converged(self, threshold: float = 0.015) -> bool:
        change = np.mean(np.abs(self.states - self.prev_states))
        return change < threshold

    def settle(self, max_iters: int = 12, min_steps: int = 3, target_mode: bool = False) -> int:
        for i in range(max_iters):
            self.step(target_mode=target_mode)
            if i >= min_steps and self.has_converged():
                return i + 1
        return max_iters

    def free_phase(self, input_values: np.ndarray) -> float:
        self.states = np.random.randn(self.n_nodes, self.dim) * 0.01
        self.clamped = np.full((self.n_nodes, self.dim), np.nan)

        for i, val in enumerate(input_values):
            if i < 2:
                self.clamped[i] = np.ones(self.dim) * val

        self.settle()
        self.free_states = self.states.copy()
        return float(np.mean(self.states[-1]))

    def target_phase(self, target: float) -> None:
        self.clamped[-1] = np.ones(self.dim) * target
        self.settle(max_iters=18, min_steps=6, target_mode=True)
        self.target_states = self.states.copy()

    def learn(self):
        state_diff = self.target_states - self.free_states
        state_diff = np.clip(state_diff, -1, 1)

        tanh_deriv = 1 - self.free_states ** 2
        delta = state_diff * tanh_deriv

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) / 2
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(np.abs(weighted_adj), axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.free_states) / total_weight

        dW = np.einsum('ni,nj->nij', delta, neighbor_input) * self.lr
        dW[self.input_mask] = 0
        self.node_W += dW
        self.node_W = np.clip(self.node_W, -3, 3)

        self.biases += self.lr * 0.5 * delta
        self.biases[self.input_mask] = 0
        self.biases = np.clip(self.biases, -2, 2)

        target_corr = (self.target_states @ self.target_states.T) / self.dim
        free_corr = (self.free_states @ self.free_states.T) / self.dim
        corr_diff = target_corr - free_corr

        coherence_gate = np.maximum(0, 2 * coherence - 0.5)
        dW_conn = self.lr * 0.15 * coherence_gate * corr_diff * self.adj
        self.W_conn = np.clip(self.W_conn + dW_conn, 0.1, 2.0)

        decay_rate = 0.002
        baseline_scale = 0.5 / np.sqrt(self.dim)
        self.node_W = self.node_W * (1 - decay_rate) + baseline_scale * decay_rate * np.random.randn(self.n_nodes, self.dim, self.dim)

        for i in range(self.n_nodes):
            self.node_W[i, np.arange(self.dim), np.arange(self.dim)] *= (1 + decay_rate * 2)

        self.biases = self.biases * (1 - decay_rate * 0.5)

        baseline_conn = 0.75
        self.W_conn = self.W_conn * (1 - decay_rate) + baseline_conn * decay_rate * self.adj

        self.memory_strength = self.memory_strength * 0.995
        self.memory_real = self.memory_real * 0.995
        self.memory_imag = self.memory_imag * 0.995

    def train_step(self, input_values: np.ndarray, target: float):
        self.free_phase(input_values)
        self.target_phase(target)
        self.learn()

    def predict(self, input_values: np.ndarray) -> float:
        return self.free_phase(input_values)


def target_func(x1, x2):
    return 1.0 if x1 * x2 > 0 else -1.0


def test_configuration(graph_class, n_nodes: int, n_train: int = 20,
                       n_seeds: int = 10, n_epochs: int = 100, **kwargs):
    """Test a specific configuration and return OOD generalization rate."""
    successes = 0

    for seed in range(n_seeds):
        np.random.seed(seed)

        graph = graph_class(n_nodes=n_nodes, dim=8, seed=seed, **kwargs)

        # Generate training data from 3 quadrants (exclude quadrant 4)
        train_data = []
        for _ in range(n_train):
            quadrant = np.random.choice([1, 2, 3])
            if quadrant == 1:
                x1 = np.random.uniform(0.3, 1.0)
                x2 = np.random.uniform(0.3, 1.0)
            elif quadrant == 2:
                x1 = np.random.uniform(-1.0, -0.3)
                x2 = np.random.uniform(0.3, 1.0)
            else:
                x1 = np.random.uniform(-1.0, -0.3)
                x2 = np.random.uniform(-1.0, -0.3)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        # Train
        for epoch in range(n_epochs):
            np.random.shuffle(train_data)
            for inp, target in train_data:
                graph.train_step(inp, target)

        # Test on held-out quadrant 4 (x1 > 0, x2 < 0)
        correct = 0
        for _ in range(20):
            x1 = np.random.uniform(0.3, 1.0)
            x2 = np.random.uniform(-1.0, -0.3)
            pred = graph.predict(np.array([x1, x2]))
            target = target_func(x1, x2)
            if (pred > 0) == (target > 0):
                correct += 1

        if correct >= 14:  # 70% on unseen quadrant
            successes += 1

    return successes


# Also include the original dense graph for comparison
class DensePSIGraph:
    """Original dense connectivity (80% density) - for comparison."""

    def __init__(self, n_nodes: int, dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim

        self.states = np.random.randn(n_nodes, dim) * 0.01
        self.prev_states = np.zeros((n_nodes, dim))
        self.free_states = np.zeros((n_nodes, dim))
        self.target_states = np.zeros((n_nodes, dim))

        self.phases = np.random.uniform(0, 2*np.pi, (n_nodes, dim))
        self.omega = np.random.randn(n_nodes, dim) * 0.03

        # DENSE connectivity - the problematic approach
        density = 0.8
        mask = np.random.random((n_nodes, n_nodes)) < density
        mask = np.triu(mask, 1)
        mask = mask | mask.T
        self.adj = mask.astype(float)
        self.W_conn = np.random.uniform(0.5, 1.0, (n_nodes, n_nodes)) * self.adj
        self.W_conn = (self.W_conn + self.W_conn.T) / 2

        scale = 0.5 / np.sqrt(dim)
        self.node_W = np.random.randn(n_nodes, dim, dim) * scale
        for i in range(n_nodes):
            self.node_W[i] += np.eye(dim) * 0.4

        self.biases = np.random.randn(n_nodes, dim) * 0.15
        self.lr = 0.25

        self.input_mask = np.zeros(n_nodes, dtype=bool)
        self.input_mask[:2] = True
        self.output_mask = np.zeros(n_nodes, dtype=bool)
        self.output_mask[-1] = True

        self.clamped = np.full((n_nodes, dim), np.nan)

        self.memory_real = np.zeros((n_nodes, dim))
        self.memory_imag = np.zeros((n_nodes, dim))
        self.memory_strength = np.zeros(n_nodes)

    def step(self, target_mode: bool = False):
        self.prev_states = self.states.copy()

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) / 2
        if target_mode:
            gate = gate + 0.3

        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(np.abs(weighted_adj), axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.states) / total_weight

        noise = np.random.randn(self.n_nodes, self.dim) * 0.05
        neighbor_input = neighbor_input + noise

        pre_act = np.einsum('nij,nj->ni', self.node_W, neighbor_input) + self.biases
        pre_act = np.clip(pre_act, -5, 5)
        target_activation = np.tanh(pre_act)

        mean_node_coherence = np.mean(coherence * self.adj, axis=1) / (np.sum(self.adj, axis=1) + 0.1)
        coherence_threshold = np.mean(mean_node_coherence)
        node_gate = np.clip(1 + 2 * (mean_node_coherence - coherence_threshold), 0.3, 1.5)
        target_activation = target_activation * node_gate[:, np.newaxis]

        leak = 0.7 if target_mode else 0.5
        new_states = (1 - leak) * self.states + leak * target_activation

        clamped_mask = ~np.isnan(self.clamped)
        new_states = np.where(clamped_mask, self.clamped, new_states)

        self.states = new_states
        self.phases += self.omega + 0.03 * self.states
        self.phases = np.mod(self.phases, 2 * np.pi)

    def has_converged(self, threshold: float = 0.015) -> bool:
        change = np.mean(np.abs(self.states - self.prev_states))
        return change < threshold

    def settle(self, max_iters: int = 12, min_steps: int = 3, target_mode: bool = False) -> int:
        for i in range(max_iters):
            self.step(target_mode=target_mode)
            if i >= min_steps and self.has_converged():
                return i + 1
        return max_iters

    def free_phase(self, input_values: np.ndarray) -> float:
        self.states = np.random.randn(self.n_nodes, self.dim) * 0.01
        self.clamped = np.full((self.n_nodes, self.dim), np.nan)

        for i, val in enumerate(input_values):
            if i < 2:
                self.clamped[i] = np.ones(self.dim) * val

        self.settle()
        self.free_states = self.states.copy()
        return float(np.mean(self.states[-1]))

    def target_phase(self, target: float) -> None:
        self.clamped[-1] = np.ones(self.dim) * target
        self.settle(max_iters=18, min_steps=6, target_mode=True)
        self.target_states = self.states.copy()

    def learn(self):
        state_diff = self.target_states - self.free_states
        state_diff = np.clip(state_diff, -1, 1)

        tanh_deriv = 1 - self.free_states ** 2
        delta = state_diff * tanh_deriv

        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) / 2
        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(np.abs(weighted_adj), axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.free_states) / total_weight

        dW = np.einsum('ni,nj->nij', delta, neighbor_input) * self.lr
        dW[self.input_mask] = 0
        self.node_W += dW
        self.node_W = np.clip(self.node_W, -3, 3)

        self.biases += self.lr * 0.5 * delta
        self.biases[self.input_mask] = 0
        self.biases = np.clip(self.biases, -2, 2)

        target_corr = (self.target_states @ self.target_states.T) / self.dim
        free_corr = (self.free_states @ self.free_states.T) / self.dim
        corr_diff = target_corr - free_corr

        coherence_gate = np.maximum(0, 2 * coherence - 0.5)
        dW_conn = self.lr * 0.15 * coherence_gate * corr_diff * self.adj
        self.W_conn = np.clip(self.W_conn + dW_conn, 0.1, 2.0)

        decay_rate = 0.002
        baseline_scale = 0.5 / np.sqrt(self.dim)
        self.node_W = self.node_W * (1 - decay_rate) + baseline_scale * decay_rate * np.random.randn(self.n_nodes, self.dim, self.dim)

        for i in range(self.n_nodes):
            self.node_W[i, np.arange(self.dim), np.arange(self.dim)] *= (1 + decay_rate * 2)

        self.biases = self.biases * (1 - decay_rate * 0.5)

        baseline_conn = 0.75
        self.W_conn = self.W_conn * (1 - decay_rate) + baseline_conn * decay_rate * self.adj

        self.memory_strength = self.memory_strength * 0.995
        self.memory_real = self.memory_real * 0.995
        self.memory_imag = self.memory_imag * 0.995

    def train_step(self, input_values: np.ndarray, target: float):
        self.free_phase(input_values)
        self.target_phase(target)
        self.learn()

    def predict(self, input_values: np.ndarray) -> float:
        return self.free_phase(input_values)


def main():
    print("=" * 70)
    print("SCALING FIX VERIFICATION")
    print("=" * 70)
    print()
    print("Comparing DENSE (80% density) vs SPARSE (small-world) topology")
    print("Testing OOD generalization (train 3 quadrants, test held-out 4th)")
    print()

    node_counts = [4, 6, 10, 16, 24]

    print("-" * 70)
    print("ORIGINAL (Dense 80% connectivity) - SHOULD DEGRADE")
    print("-" * 70)

    dense_results = {}
    for n_nodes in node_counts:
        start = time.time()
        successes = test_configuration(DensePSIGraph, n_nodes=n_nodes,
                                        n_train=20, n_seeds=10)
        elapsed = time.time() - start
        dense_results[n_nodes] = successes
        print(f"  {n_nodes:2d} nodes: {successes}/10 generalize ({elapsed:.1f}s)")

    print()
    print("-" * 70)
    print("NEW (Small-world sparse connectivity) - SHOULD NOT DEGRADE")
    print("-" * 70)

    sparse_results = {}
    for n_nodes in node_counts:
        start = time.time()
        successes = test_configuration(ScalablePSIGraph, n_nodes=n_nodes,
                                        n_train=20, n_seeds=10,
                                        k_neighbors=3, long_range_prob=0.15)
        elapsed = time.time() - start
        sparse_results[n_nodes] = successes
        print(f"  {n_nodes:2d} nodes: {successes}/10 generalize ({elapsed:.1f}s)")

    print()
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"{'Nodes':>6} | {'Dense':>6} | {'Sparse':>6} | {'Change':>8}")
    print("-" * 70)

    for n_nodes in node_counts:
        dense = dense_results[n_nodes]
        sparse = sparse_results[n_nodes]
        change = sparse - dense
        sign = "+" if change >= 0 else ""
        print(f"{n_nodes:>6} | {dense:>6} | {sparse:>6} | {sign}{change:>7}")

    print()

    # Check if fix worked
    dense_degrades = dense_results[24] < dense_results[6]
    sparse_maintains = sparse_results[24] >= sparse_results[6] - 1  # Allow small variance

    if dense_degrades and sparse_maintains:
        print("SUCCESS: Sparse topology maintains performance at scale!")
    elif not dense_degrades:
        print("NOTE: Dense didn't degrade this run (variance)")
    else:
        print("PARTIAL: Sparse still degrades - need to refine topology")

    # Show neighbor counts for verification
    print()
    print("-" * 70)
    print("NEIGHBOR COUNT VERIFICATION")
    print("-" * 70)

    for n_nodes in node_counts:
        dense_g = DensePSIGraph(n_nodes=n_nodes, dim=8, seed=0)
        sparse_g = ScalablePSIGraph(n_nodes=n_nodes, dim=8, seed=0,
                                     k_neighbors=3, long_range_prob=0.15)

        dense_neighbors = np.mean(np.sum(dense_g.adj, axis=1))
        sparse_neighbors = np.mean(np.sum(sparse_g.adj, axis=1))

        print(f"  {n_nodes:2d} nodes: Dense={dense_neighbors:.1f} neighbors, Sparse={sparse_neighbors:.1f} neighbors")


if __name__ == "__main__":
    main()
