"""
Analyze what's limiting OOD generalization.

Questions:
1. Is the network memorizing training points or learning a decision boundary?
2. How does the learned representation look in the hidden nodes?
3. What happens with more capacity (nodes, dimensions)?
4. Does the network benefit from curriculum learning?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AnalysisPSIGraph:
    """PSI Graph with analysis hooks."""

    def __init__(self, n_nodes: int, dim: int, seed: int = None,
                 k_neighbors: int = 3, long_range_prob: float = 0.15):
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

        self.adj = self._build_small_world_adj(k_neighbors, long_range_prob)
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

    def _build_small_world_adj(self, k: int, p_long: float) -> np.ndarray:
        adj = np.zeros((self.n_nodes, self.n_nodes))
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

        for i in range(self.n_nodes):
            for j in range(i + 2, self.n_nodes):
                if adj[i, j] == 0 and np.random.random() < p_long:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

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
        return np.mean(np.abs(self.states - self.prev_states)) < threshold

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

    def get_hidden_representation(self, input_values: np.ndarray) -> np.ndarray:
        """Get the hidden layer representation for analysis."""
        self.free_phase(input_values)
        # Return middle nodes (not input or output)
        return self.free_states[2:-1].flatten()


def target_func(x1, x2):
    return 1.0 if x1 * x2 > 0 else -1.0


def analyze_decision_boundary(graph, title="Decision Boundary"):
    """Visualize what decision boundary the network learned."""
    # Create grid
    x1_range = np.linspace(-1, 1, 30)
    x2_range = np.linspace(-1, 1, 30)
    predictions = np.zeros((30, 30))

    for i, x1 in enumerate(x1_range):
        for j, x2 in enumerate(x2_range):
            pred = graph.predict(np.array([x1, x2]))
            predictions[j, i] = pred

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Prediction map
    im = axes[0].imshow(predictions, extent=[-1, 1, -1, 1], origin='lower',
                        cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_title(f'{title} - Predictions')
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    plt.colorbar(im, ax=axes[0])

    # True function
    true_vals = np.zeros((30, 30))
    for i, x1 in enumerate(x1_range):
        for j, x2 in enumerate(x2_range):
            true_vals[j, i] = target_func(x1, x2)

    im2 = axes[1].imshow(true_vals, extent=[-1, 1, -1, 1], origin='lower',
                         cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    axes[1].set_title('True Function')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.replace(" ", "_")}.png', dpi=100)
    plt.close()
    print(f"  Saved: decision_boundary_{title.replace(' ', '_')}.png")


def test_capacity_scaling():
    """Test if more capacity helps OOD generalization."""
    print("\n" + "=" * 70)
    print("CAPACITY SCALING ANALYSIS")
    print("=" * 70)

    configs = [
        (6, 8, "6 nodes, dim=8"),
        (12, 8, "12 nodes, dim=8"),
        (24, 8, "24 nodes, dim=8"),
        (6, 16, "6 nodes, dim=16"),
        (12, 16, "12 nodes, dim=16"),
        (6, 32, "6 nodes, dim=32"),
    ]

    results = {}

    for n_nodes, dim, label in configs:
        successes = 0
        for seed in range(10):
            np.random.seed(seed)
            graph = AnalysisPSIGraph(n_nodes=n_nodes, dim=dim, seed=seed)

            # Train on 3 quadrants
            train_data = []
            for _ in range(50):  # More training data
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

            for epoch in range(100):
                np.random.shuffle(train_data)
                for inp, target in train_data:
                    graph.train_step(inp, target)

            # Test on held-out quadrant 4
            correct = 0
            for _ in range(20):
                x1 = np.random.uniform(0.3, 1.0)
                x2 = np.random.uniform(-1.0, -0.3)
                pred = graph.predict(np.array([x1, x2]))
                if (pred > 0) == (target_func(x1, x2) > 0):
                    correct += 1

            if correct >= 14:
                successes += 1

        results[label] = successes
        print(f"  {label}: {successes}/10 generalize")

    return results


def test_training_variations():
    """Test different training approaches."""
    print("\n" + "=" * 70)
    print("TRAINING VARIATION ANALYSIS")
    print("=" * 70)

    def run_experiment(name, train_fn, n_seeds=10):
        successes = 0
        for seed in range(n_seeds):
            np.random.seed(seed)
            graph = AnalysisPSIGraph(n_nodes=12, dim=16, seed=seed)

            train_fn(graph, seed)

            # Test on held-out quadrant
            correct = 0
            for _ in range(20):
                x1 = np.random.uniform(0.3, 1.0)
                x2 = np.random.uniform(-1.0, -0.3)
                pred = graph.predict(np.array([x1, x2]))
                if (pred > 0) == (target_func(x1, x2) > 0):
                    correct += 1

            if correct >= 14:
                successes += 1

        print(f"  {name}: {successes}/{n_seeds} generalize")
        return successes

    # Baseline
    def train_baseline(graph, seed):
        np.random.seed(seed)
        train_data = []
        for _ in range(50):
            quadrant = np.random.choice([1, 2, 3])
            if quadrant == 1:
                x1, x2 = np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0)
            elif quadrant == 2:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(0.3, 1.0)
            else:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(-1.0, -0.3)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        for epoch in range(100):
            np.random.shuffle(train_data)
            for inp, target in train_data:
                graph.train_step(inp, target)

    run_experiment("Baseline (50 samples, 100 epochs)", train_baseline)

    # More epochs
    def train_more_epochs(graph, seed):
        np.random.seed(seed)
        train_data = []
        for _ in range(50):
            quadrant = np.random.choice([1, 2, 3])
            if quadrant == 1:
                x1, x2 = np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0)
            elif quadrant == 2:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(0.3, 1.0)
            else:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(-1.0, -0.3)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        for epoch in range(200):  # 2x epochs
            np.random.shuffle(train_data)
            for inp, target in train_data:
                graph.train_step(inp, target)

    run_experiment("More epochs (200)", train_more_epochs)

    # Curriculum: start near boundary, expand
    def train_curriculum(graph, seed):
        np.random.seed(seed)

        for phase in range(4):
            margin = 0.1 + phase * 0.2  # Start near boundary, expand

            train_data = []
            for _ in range(30):
                quadrant = np.random.choice([1, 2, 3])
                if quadrant == 1:
                    x1 = np.random.uniform(margin, 1.0)
                    x2 = np.random.uniform(margin, 1.0)
                elif quadrant == 2:
                    x1 = np.random.uniform(-1.0, -margin)
                    x2 = np.random.uniform(margin, 1.0)
                else:
                    x1 = np.random.uniform(-1.0, -margin)
                    x2 = np.random.uniform(-1.0, -margin)
                train_data.append((np.array([x1, x2]), target_func(x1, x2)))

            for epoch in range(25):
                np.random.shuffle(train_data)
                for inp, target in train_data:
                    graph.train_step(inp, target)

    run_experiment("Curriculum (boundary -> edges)", train_curriculum)

    # Balanced quadrants
    def train_balanced(graph, seed):
        np.random.seed(seed)

        for epoch in range(100):
            # Each epoch: one sample from each quadrant
            for quadrant in [1, 2, 3]:
                if quadrant == 1:
                    x1 = np.random.uniform(0.3, 1.0)
                    x2 = np.random.uniform(0.3, 1.0)
                elif quadrant == 2:
                    x1 = np.random.uniform(-1.0, -0.3)
                    x2 = np.random.uniform(0.3, 1.0)
                else:
                    x1 = np.random.uniform(-1.0, -0.3)
                    x2 = np.random.uniform(-1.0, -0.3)
                graph.train_step(np.array([x1, x2]), target_func(x1, x2))

    run_experiment("Balanced (1 per quadrant per epoch)", train_balanced)

    # Include near-boundary examples
    def train_with_boundary(graph, seed):
        np.random.seed(seed)
        train_data = []

        # Regular samples from 3 quadrants
        for _ in range(30):
            quadrant = np.random.choice([1, 2, 3])
            if quadrant == 1:
                x1, x2 = np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0)
            elif quadrant == 2:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(0.3, 1.0)
            else:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(-1.0, -0.3)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        # Add boundary examples (near x1=0 or x2=0)
        for _ in range(20):
            if np.random.random() < 0.5:
                # Near x1=0
                x1 = np.random.uniform(-0.15, 0.15)
                x2 = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
            else:
                # Near x2=0
                x1 = np.random.choice([-1, 1]) * np.random.uniform(0.3, 1.0)
                x2 = np.random.uniform(-0.15, 0.15)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        for epoch in range(100):
            np.random.shuffle(train_data)
            for inp, target in train_data:
                graph.train_step(inp, target)

    run_experiment("With boundary examples", train_with_boundary)


def visualize_best_and_worst():
    """Train and visualize decision boundaries for analysis."""
    print("\n" + "=" * 70)
    print("DECISION BOUNDARY VISUALIZATION")
    print("=" * 70)

    # Find a seed that generalizes and one that doesn't
    good_seed = None
    bad_seed = None

    for seed in range(20):
        np.random.seed(seed)
        graph = AnalysisPSIGraph(n_nodes=12, dim=16, seed=seed)

        train_data = []
        for _ in range(50):
            quadrant = np.random.choice([1, 2, 3])
            if quadrant == 1:
                x1, x2 = np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0)
            elif quadrant == 2:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(0.3, 1.0)
            else:
                x1, x2 = np.random.uniform(-1.0, -0.3), np.random.uniform(-1.0, -0.3)
            train_data.append((np.array([x1, x2]), target_func(x1, x2)))

        for epoch in range(100):
            np.random.shuffle(train_data)
            for inp, target in train_data:
                graph.train_step(inp, target)

        # Test
        correct = 0
        for _ in range(20):
            x1 = np.random.uniform(0.3, 1.0)
            x2 = np.random.uniform(-1.0, -0.3)
            pred = graph.predict(np.array([x1, x2]))
            if (pred > 0) == (target_func(x1, x2) > 0):
                correct += 1

        if correct >= 16 and good_seed is None:
            good_seed = seed
            analyze_decision_boundary(graph, f"Good_seed{seed}")
        elif correct <= 6 and bad_seed is None:
            bad_seed = seed
            analyze_decision_boundary(graph, f"Bad_seed{seed}")

        if good_seed is not None and bad_seed is not None:
            break

    print(f"  Good seed: {good_seed}, Bad seed: {bad_seed}")


def main():
    print("=" * 70)
    print("OOD GENERALIZATION ANALYSIS")
    print("=" * 70)

    # Test capacity scaling
    test_capacity_scaling()

    # Test training variations
    test_training_variations()

    # Visualize decision boundaries
    visualize_best_and_worst()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey questions:")
    print("1. Does more capacity help? -> Check capacity scaling results")
    print("2. Does training approach matter? -> Check training variations")
    print("3. What does the learned boundary look like? -> Check saved plots")


if __name__ == "__main__":
    main()
