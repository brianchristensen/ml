"""
Diagnose why adding more nodes degrades performance.

Hypothesis 1: Signal Dilution
- With 80% density, more nodes = exponentially more connections
- Each node averages over MORE neighbors, diluting the signal
- 6 nodes: ~4.8 neighbors avg, 24 nodes: ~19.2 neighbors avg

Hypothesis 2: Input-to-Output Path Length
- With more nodes, the "distance" from input to output increases
- Signal must propagate through more hops, losing strength

Hypothesis 3: Learning Signal Dilution
- Target signal from output must propagate backward
- More intermediate nodes = weaker learning signal at input-adjacent nodes

Hypothesis 4: Phase Coherence Averaging
- With more nodes, mean coherence becomes more stable but less informative
- Routing becomes less discriminative
"""

import numpy as np


class DiagnosticPSIGraph:
    """PSI Graph with diagnostic instrumentation."""

    def __init__(self, n_nodes: int, dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim

        # Node states
        self.states = np.random.randn(n_nodes, dim) * 0.01
        self.prev_states = np.zeros((n_nodes, dim))
        self.free_states = np.zeros((n_nodes, dim))
        self.target_states = np.zeros((n_nodes, dim))

        # Phases
        self.phases = np.random.uniform(0, 2*np.pi, (n_nodes, dim))
        self.omega = np.random.randn(n_nodes, dim) * 0.03

        # Connection weights (adjacency matrix)
        density = 0.8
        mask = np.random.random((n_nodes, n_nodes)) < density
        mask = np.triu(mask, 1)
        mask = mask | mask.T
        self.adj = mask.astype(float)
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

        # Diagnostics storage
        self.diag_neighbor_counts = []
        self.diag_signal_strengths = []
        self.diag_coherence_means = []
        self.diag_learning_signals = []

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

        # DIAGNOSTIC: Track signal strength before transformation
        signal_mag = np.mean(np.abs(neighbor_input))
        self.diag_signal_strengths.append(signal_mag)

        # DIAGNOSTIC: Track mean coherence
        self.diag_coherence_means.append(np.mean(coherence))

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

        # DIAGNOSTIC: Learning signal at each node
        learning_signal = np.mean(np.abs(state_diff), axis=1)
        self.diag_learning_signals.append(learning_signal.copy())

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

    def train_step(self, input_values: np.ndarray, target: float):
        self.free_phase(input_values)
        self.target_phase(target)
        self.learn()

    def predict(self, input_values: np.ndarray) -> float:
        return self.free_phase(input_values)

    def get_diagnostics(self):
        """Return diagnostic summary."""
        avg_neighbors = np.mean(np.sum(self.adj, axis=1))

        # Learning signal distribution across node indices (normalized by position)
        if self.diag_learning_signals:
            learning_signals = np.array(self.diag_learning_signals)
            mean_per_node = np.mean(learning_signals, axis=0)
        else:
            mean_per_node = np.zeros(self.n_nodes)

        return {
            'avg_neighbors': avg_neighbors,
            'avg_signal_strength': np.mean(self.diag_signal_strengths) if self.diag_signal_strengths else 0,
            'avg_coherence': np.mean(self.diag_coherence_means) if self.diag_coherence_means else 0,
            'learning_signal_per_node': mean_per_node,
            'learning_signal_at_input_neighbors': mean_per_node[2] if len(mean_per_node) > 2 else 0,  # Node 2 is first non-input
            'learning_signal_at_output_neighbors': mean_per_node[-2] if len(mean_per_node) > 1 else 0,  # Second to last
        }


def target_func(x1, x2):
    return 1.0 if x1 * x2 > 0 else -1.0


def diagnose_node_count(n_nodes: int, seed: int = 42):
    """Run diagnosis on a specific node count."""
    np.random.seed(seed)

    graph = DiagnosticPSIGraph(n_nodes=n_nodes, dim=8, seed=seed)

    # Generate training data
    train_data = []
    for _ in range(20):
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
    for epoch in range(50):
        np.random.shuffle(train_data)
        for inp, target in train_data:
            graph.train_step(inp, target)

    return graph.get_diagnostics()


def main():
    print("=" * 70)
    print("SCALING DIAGNOSTICS")
    print("=" * 70)
    print()

    node_counts = [4, 6, 10, 16, 24]

    print("HYPOTHESIS 1: Signal Dilution (more neighbors = diluted average)")
    print("-" * 70)
    print(f"{'Nodes':>6} | {'Avg Neighbors':>14} | {'Signal Strength':>15} | {'Coherence':>10}")
    print("-" * 70)

    results = {}
    for n_nodes in node_counts:
        diag = diagnose_node_count(n_nodes)
        results[n_nodes] = diag
        print(f"{n_nodes:>6} | {diag['avg_neighbors']:>14.1f} | {diag['avg_signal_strength']:>15.4f} | {diag['avg_coherence']:>10.4f}")

    print()
    print("HYPOTHESIS 2: Learning Signal Distribution")
    print("-" * 70)
    print("Learning signal strength at different positions in graph:")
    print()

    for n_nodes in node_counts:
        diag = results[n_nodes]
        learning = diag['learning_signal_per_node']

        # Show input neighbors (node 2), middle, and output neighbors (node -2)
        input_neighbor = learning[2] if len(learning) > 2 else 0
        middle_idx = len(learning) // 2
        middle = learning[middle_idx] if len(learning) > middle_idx else 0
        output_neighbor = learning[-2] if len(learning) > 1 else 0

        print(f"  {n_nodes} nodes:")
        print(f"    Near-input (node 2):  {input_neighbor:.4f}")
        print(f"    Middle (node {middle_idx}):      {middle:.4f}")
        print(f"    Near-output (node -2): {output_neighbor:.4f}")

        # Show ratio of output-neighbor to input-neighbor learning signal
        if input_neighbor > 0.001:
            ratio = output_neighbor / input_neighbor
            print(f"    Output/Input ratio:    {ratio:.2f}x")
        print()

    print()
    print("ANALYSIS:")
    print("-" * 70)

    # Compute correlations
    nodes = np.array(node_counts)
    neighbors = np.array([results[n]['avg_neighbors'] for n in node_counts])
    signals = np.array([results[n]['avg_signal_strength'] for n in node_counts])

    # Neighbor count grows linearly with node count for fixed density
    print(f"Neighbor count growth: {neighbors[0]:.1f} -> {neighbors[-1]:.1f} ({neighbors[-1]/neighbors[0]:.1f}x)")
    print(f"Signal strength change: {signals[0]:.4f} -> {signals[-1]:.4f} ({signals[-1]/signals[0]:.2f}x)")

    print()
    print("KEY INSIGHT:")
    print("  With fixed 80% density, more nodes = more neighbors to average over")
    print("  This DILUTES the information from any single neighbor")
    print("  The input signal gets lost in the noise of many connections")


if __name__ == "__main__":
    main()
