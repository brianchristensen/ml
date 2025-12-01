"""
Test different node counts and phase initialization strategies.

Questions to answer:
1. Do more nodes help OOD generalization (better routing paths)?
2. Does structured phase initialization outperform random?
"""

import numpy as np
import time


class VectorizedPSIGraph:
    """
    VECTORIZED PSI Target Learning Graph.
    Copy of the main class with configurable phase initialization.
    """

    def __init__(self, n_nodes: int, dim: int, seed: int = None,
                 phase_init: str = 'random'):
        """
        phase_init options:
        - 'random': uniform random phases [0, 2pi)
        - 'spread': evenly spread phases around circle
        - 'layered': phases grouped by layer distance from input
        - 'hierarchical': phase increases with node index (input->output gradient)
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim
        self.phase_init = phase_init

        # Node states: [n_nodes, dim]
        self.states = np.random.randn(n_nodes, dim) * 0.01
        self.prev_states = np.zeros((n_nodes, dim))
        self.free_states = np.zeros((n_nodes, dim))
        self.target_states = np.zeros((n_nodes, dim))

        # Initialize phases based on strategy
        self.phases = self._init_phases()
        self.omega = np.random.randn(n_nodes, dim) * 0.03

        # Connection weights (adjacency matrix): [n_nodes, n_nodes]
        density = 0.8
        mask = np.random.random((n_nodes, n_nodes)) < density
        mask = np.triu(mask, 1)  # Upper triangle only
        mask = mask | mask.T  # Make symmetric
        self.adj = mask.astype(float)
        self.W_conn = np.random.uniform(0.5, 1.0, (n_nodes, n_nodes)) * self.adj
        self.W_conn = (self.W_conn + self.W_conn.T) / 2  # Symmetric

        # Per-node transformation: [n_nodes, dim, dim]
        scale = 0.5 / np.sqrt(dim)
        self.node_W = np.random.randn(n_nodes, dim, dim) * scale
        for i in range(n_nodes):
            self.node_W[i] += np.eye(dim) * 0.4  # Self-connection

        # Biases: [n_nodes, dim]
        self.biases = np.random.randn(n_nodes, dim) * 0.15

        # Learning rate
        self.lr = 0.25

        # Input/output masks - always use first 2 as input, last 1 as output
        self.input_mask = np.zeros(n_nodes, dtype=bool)
        self.input_mask[:2] = True  # First 2 nodes are inputs
        self.output_mask = np.zeros(n_nodes, dtype=bool)
        self.output_mask[-1] = True  # Last node is output

        # Clamp values: [n_nodes, dim] - NaN means not clamped
        self.clamped = np.full((n_nodes, dim), np.nan)

        # Holographic memory
        self.memory_real = np.zeros((n_nodes, dim))
        self.memory_imag = np.zeros((n_nodes, dim))
        self.memory_strength = np.zeros(n_nodes)
        self.avg_confidence = np.ones(n_nodes) * 0.3

    def _init_phases(self) -> np.ndarray:
        """Initialize phases based on strategy."""
        if self.phase_init == 'random':
            # Uniform random - original approach
            return np.random.uniform(0, 2*np.pi, (self.n_nodes, self.dim))

        elif self.phase_init == 'spread':
            # Evenly spread phases around the circle
            # Each node gets a different base phase, dimensions vary slightly
            base_phases = np.linspace(0, 2*np.pi, self.n_nodes, endpoint=False)
            phases = base_phases[:, np.newaxis] + np.random.randn(self.n_nodes, self.dim) * 0.2
            return np.mod(phases, 2*np.pi)

        elif self.phase_init == 'layered':
            # Group nodes into "layers" with similar phases
            # Input nodes = 0, middle nodes = pi/2 to 3pi/2, output = pi
            n_middle = self.n_nodes - 3  # 2 input + 1 output
            phases = np.zeros((self.n_nodes, self.dim))

            # Input nodes: phase 0
            phases[:2] = 0 + np.random.randn(2, self.dim) * 0.1

            # Output node: phase pi (opposite to inputs)
            phases[-1] = np.pi + np.random.randn(self.dim) * 0.1

            # Middle nodes: spread between input and output
            for i in range(2, self.n_nodes - 1):
                progress = (i - 2 + 1) / (n_middle + 1)  # 0 to 1
                phases[i] = progress * np.pi + np.random.randn(self.dim) * 0.2

            return np.mod(phases, 2*np.pi)

        elif self.phase_init == 'hierarchical':
            # Phase increases monotonically from input to output
            # Creates an implicit "forward" direction via phase gradient
            node_indices = np.arange(self.n_nodes)
            base_phases = (node_indices / self.n_nodes) * np.pi  # 0 to pi
            phases = base_phases[:, np.newaxis] + np.random.randn(self.n_nodes, self.dim) * 0.15
            return np.mod(phases, 2*np.pi)

        else:
            raise ValueError(f"Unknown phase_init: {self.phase_init}")

    def step(self, target_mode: bool = False):
        """One parallel update step."""
        self.prev_states = self.states.copy()

        # Phase coherence matrix
        phase_diff = self.phases[:, np.newaxis, :] - self.phases[np.newaxis, :, :]
        coherence = np.mean(np.cos(phase_diff), axis=2)

        gate = (1 + coherence) / 2
        if target_mode:
            gate = gate + 0.3

        weighted_adj = self.W_conn * gate * self.adj
        total_weight = np.sum(np.abs(weighted_adj), axis=1, keepdims=True) + 0.1
        neighbor_input = (weighted_adj @ self.states) / total_weight

        # Neural noise
        noise = np.random.randn(self.n_nodes, self.dim) * 0.05
        neighbor_input = neighbor_input + noise

        # Per-node transformation
        pre_act = np.einsum('nij,nj->ni', self.node_W, neighbor_input) + self.biases
        pre_act = np.clip(pre_act, -5, 5)
        target_activation = np.tanh(pre_act)

        # Phase-based node routing
        mean_node_coherence = np.mean(coherence * self.adj, axis=1) / (np.sum(self.adj, axis=1) + 0.1)
        coherence_threshold = np.mean(mean_node_coherence)
        node_gate = np.clip(1 + 2 * (mean_node_coherence - coherence_threshold), 0.3, 1.5)
        target_activation = target_activation * node_gate[:, np.newaxis]

        # Leaky integration
        leak = 0.7 if target_mode else 0.5
        new_states = (1 - leak) * self.states + leak * target_activation

        # Apply clamping
        clamped_mask = ~np.isnan(self.clamped)
        new_states = np.where(clamped_mask, self.clamped, new_states)

        self.states = new_states

        # Update phases
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

        # Synaptic decay
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
    """Target: sign of product (quadrant function)."""
    return 1.0 if x1 * x2 > 0 else -1.0


def test_configuration(n_nodes: int, phase_init: str, n_train: int = 20,
                       n_seeds: int = 10, n_epochs: int = 100):
    """Test a specific configuration and return OOD generalization rate."""
    successes = 0

    for seed in range(n_seeds):
        np.random.seed(seed)

        graph = VectorizedPSIGraph(n_nodes=n_nodes, dim=8, seed=seed,
                                    phase_init=phase_init)

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


def main():
    print("=" * 70)
    print("NODE COUNT & PHASE INITIALIZATION EXPERIMENT")
    print("=" * 70)
    print()
    print("Testing OOD generalization (train on 3 quadrants, test on held-out 4th)")
    print("Success = 70%+ accuracy on never-seen quadrant")
    print()

    # Test different node counts with random phase init
    print("-" * 70)
    print("EXPERIMENT 1: Node Count (random phase init)")
    print("-" * 70)

    node_counts = [4, 6, 10, 16, 24]

    for n_nodes in node_counts:
        start = time.time()
        successes = test_configuration(n_nodes=n_nodes, phase_init='random',
                                        n_train=20, n_seeds=10)
        elapsed = time.time() - start
        print(f"  {n_nodes:2d} nodes: {successes}/10 generalize ({elapsed:.1f}s)")

    print()

    # Test different phase initialization strategies
    print("-" * 70)
    print("EXPERIMENT 2: Phase Initialization (12 nodes)")
    print("-" * 70)

    phase_inits = ['random', 'spread', 'layered', 'hierarchical']

    for phase_init in phase_inits:
        start = time.time()
        successes = test_configuration(n_nodes=12, phase_init=phase_init,
                                        n_train=20, n_seeds=10)
        elapsed = time.time() - start
        print(f"  {phase_init:15s}: {successes}/10 generalize ({elapsed:.1f}s)")

    print()

    # Find best combination
    print("-" * 70)
    print("EXPERIMENT 3: Best combinations")
    print("-" * 70)

    best_configs = [
        (12, 'hierarchical'),
        (16, 'hierarchical'),
        (12, 'layered'),
        (16, 'layered'),
    ]

    for n_nodes, phase_init in best_configs:
        start = time.time()
        successes = test_configuration(n_nodes=n_nodes, phase_init=phase_init,
                                        n_train=20, n_seeds=15)  # More seeds
        elapsed = time.time() - start
        print(f"  {n_nodes} nodes + {phase_init:15s}: {successes}/15 generalize ({elapsed:.1f}s)")

    print()

    # Test sample efficiency with best config
    print("-" * 70)
    print("EXPERIMENT 4: Sample efficiency (best config vs baseline)")
    print("-" * 70)

    sample_sizes = [10, 20, 50, 100]

    print("  Baseline (6 nodes, random):")
    for n_train in sample_sizes:
        successes = test_configuration(n_nodes=6, phase_init='random',
                                        n_train=n_train, n_seeds=10)
        print(f"    {n_train:3d} samples: {successes}/10")

    print()
    print("  Best config (16 nodes, hierarchical):")
    for n_train in sample_sizes:
        successes = test_configuration(n_nodes=16, phase_init='hierarchical',
                                        n_train=n_train, n_seeds=10)
        print(f"    {n_train:3d} samples: {successes}/10")

    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()
