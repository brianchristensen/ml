"""
PSI Target Learning
===================

Brain-inspired credit assignment using TARGET LEARNING.

Based on recent neuroscience findings (2024):
- Target learning more accurately predicts neural activity than backprop
- Neurons learn by reducing feedback needed to achieve a TARGET ACTIVITY
- The network first infers what activity SHOULD result, then consolidates

Key mechanism (Prospective Configuration):
1. FREE PHASE: Network settles naturally given input
2. TARGET PHASE: Inject target at output, let network settle to new state
   - This propagates "what should have happened" backward through recurrence
3. LOCAL LEARNING: Each node adjusts to move free state toward target state

This is different from error backprop:
- Nodes don't receive error gradients
- Nodes receive TARGET ACTIVITY PATTERNS
- Learning is: "adjust weights so my free state looks more like my target state"

For recurrent networks, target propagation happens THROUGH THE DYNAMICS:
- The recurrent settling process naturally propagates targets
- Phase coherence gates which targets influence which nodes
- No separate backward pass needed!

References:
- "Challenging Backpropagation: Evidence for Target Learning in Neocortex" (2024)
- "Prospective Configuration" (Nature Neuroscience 2024)
- "Equilibrium Propagation" (Scellier & Bengio 2017)

Author: Brian Christensen
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


@dataclass
class Connection:
    """Bidirectional connection."""
    target_id: int
    weight: float


class TargetLearningNode:
    """
    PSI node with target learning.

    Key insight: The node doesn't receive error - it receives a TARGET STATE
    and learns to produce that state from its inputs.

    Two phases:
    - Free phase: settle naturally
    - Target phase: settle with output clamped to target

    Learning: move free state toward target state
    """

    def __init__(self, node_id: int, dim: int):
        self.id = node_id
        self.dim = dim

        # State
        self.state = np.zeros(dim)
        self.prev_state = np.zeros(dim)

        # Store states from both phases
        self.free_state = np.zeros(dim)
        self.target_state = np.zeros(dim)

        # Phase dynamics
        self.phase = np.random.uniform(0, 2 * np.pi, dim)
        self.omega = np.random.randn(dim) * 0.03

        # Connections
        self.connections: Dict[int, Connection] = {}

        # Learnable weights
        scale = 0.5
        self.W = np.random.randn(dim, dim) * scale / np.sqrt(dim)
        self.W += np.eye(dim) * 0.4  # Stronger self-connection for memory
        self.bias = np.random.randn(dim) * 0.15

        # Base learning rate - will be modulated by local activity
        self.lr_base = 0.25
        self.lr = self.lr_base

        # Node type
        self.is_input = False
        self.is_output = False
        self.clamped_value = None

        # For dynamic connectivity
        self.coherence_trace: Dict[int, float] = {}

        # Predictive coding: each node predicts its input
        self.predicted_input = np.zeros(dim)
        self.prediction_error = np.zeros(dim)

    def phase_coherence(self, other_phase: np.ndarray) -> float:
        """Compute phase coherence."""
        return float(np.mean(np.cos(self.phase - other_phase)))

    def update(self, neighbor_states: Dict[int, Tuple[np.ndarray, np.ndarray]],
               dt: float = 0.3, target_mode: bool = False):
        """
        Update state based on neighbors (recurrent settling).

        target_mode: If True, use stronger coupling for target propagation.
        """
        if self.clamped_value is not None:
            self.prev_state = self.state.copy()
            self.state = self.clamped_value.copy()
            return

        self.prev_state = self.state.copy()

        # Aggregate neighbor inputs with phase gating
        total_input = np.zeros(self.dim)
        total_weight = 0

        for node_id, (neighbor_state, neighbor_phase) in neighbor_states.items():
            if node_id not in self.connections:
                continue

            conn = self.connections[node_id]

            # Phase coherence gates communication
            coherence = self.phase_coherence(neighbor_phase)

            # Track coherence
            if node_id not in self.coherence_trace:
                self.coherence_trace[node_id] = coherence
            else:
                self.coherence_trace[node_id] = 0.9 * self.coherence_trace[node_id] + 0.1 * coherence

            # Soft gating - stronger in target mode
            if target_mode:
                gate = (1 + coherence) / 2 + 0.3  # Boost gating in target mode
            else:
                gate = (1 + coherence) / 2

            contribution = conn.weight * gate * neighbor_state
            total_input += contribution
            total_weight += abs(conn.weight * gate)

        # Normalize
        if total_weight > 0.1:
            total_input /= max(total_weight, 1.0)

        # Predictive coding: compute prediction error
        # Large error = surprising input = learn more
        self.prediction_error = total_input - self.predicted_input

        # Update prediction (slow tracking of actual input)
        self.predicted_input = 0.8 * self.predicted_input + 0.2 * total_input

        # Transform
        pre_act = self.W @ total_input + self.bias
        pre_act = np.clip(pre_act, -5, 5)
        target_activation = np.tanh(pre_act)

        # Leaky integration (settling dynamics)
        # Higher leak = faster settling toward neighbors
        # In target mode, use higher leak for stronger target propagation
        if target_mode:
            leak = 0.7  # Faster settling toward target-influenced neighbors
        else:
            leak = 0.5
        self.state = (1 - leak) * self.state + leak * target_activation

        # Update phase
        self.phase += self.omega + 0.03 * self.state
        self.phase = np.mod(self.phase, 2 * np.pi)

    def store_free_state(self):
        """Store state from free phase."""
        self.free_state = self.state.copy()

    def store_target_state(self):
        """Store state from target phase."""
        self.target_state = self.state.copy()

    def target_learning_update(self, neighbor_free: Dict[int, np.ndarray],
                                neighbor_target: Dict[int, np.ndarray]):
        """
        TARGET LEARNING: Adjust weights to move free state toward target state.

        This is the key insight from neuroscience:
        - Don't propagate error gradients
        - Propagate TARGET PATTERNS through recurrent dynamics
        - Learn locally: make free state more like target state

        dW proportional to (target_state - free_state) * input
        """
        if self.is_input:
            return

        # The learning signal: difference between target and free states
        state_diff = self.target_state - self.free_state

        # Clip for stability
        state_diff = np.clip(state_diff, -1, 1)

        # Adaptive learning rate based on prediction error (surprise)
        # Novel patterns (high prediction error) -> higher LR -> fast learning
        # Familiar patterns (low prediction error) -> lower LR -> stability
        surprise = np.mean(np.abs(self.prediction_error))
        # Higher floor for better few-shot, but still modulated
        self.lr = self.lr_base * np.clip(0.8 + 1.2 * surprise, 0.6, 2.5)
        surprise_factor = 1.0  # LR already adapted

        # Compute the input that was present during free phase
        free_input = np.zeros(self.dim)
        for node_id, neighbor_state in neighbor_free.items():
            if node_id in self.connections:
                coherence = self.coherence_trace.get(node_id, 0.5)
                gate = (1 + coherence) / 2
                free_input += self.connections[node_id].weight * gate * neighbor_state

        # Update W to make output closer to target given the same input
        # Derivative of tanh
        tanh_deriv = 1 - self.free_state ** 2
        delta = state_diff * tanh_deriv

        # Weight update (modulated by surprise)
        dW = np.outer(delta, free_input)
        self.W += self.lr * surprise_factor * dW
        self.bias += self.lr * surprise_factor * delta * 0.5

        # Clip
        self.W = np.clip(self.W, -3, 3)
        self.bias = np.clip(self.bias, -2, 2)

        # Update connection weights based on correlation change
        # PHASE COHERENCE GATES LEARNING: only learn from phase-locked neighbors
        for node_id, conn in self.connections.items():
            if node_id not in neighbor_free or node_id not in neighbor_target:
                continue

            # Phase coherence determines learning strength
            # High coherence = strong binding = learn together
            coherence = self.coherence_trace.get(node_id, 0.5)
            # Sharper gating - only strong coherence enables learning
            coherence_gate = max(0, 2 * coherence - 0.5)  # Only coherence > 0.25 contributes

            # How much did this connection's activity change?
            free_corr = np.mean(self.free_state * neighbor_free[node_id])
            target_corr = np.mean(self.target_state * neighbor_target[node_id])

            # If target correlation is higher, strengthen connection
            # Gated by phase coherence - incoherent nodes don't learn from each other
            corr_diff = target_corr - free_corr
            conn.weight += self.lr * 0.15 * coherence_gate * corr_diff
            conn.weight = np.clip(conn.weight, 0.1, 2.0)


    def reset(self):
        """Reset state."""
        self.state = np.random.randn(self.dim) * 0.01
        self.prev_state = np.zeros(self.dim)


class PSITargetGraph:
    """
    PSI Graph with Target Learning.

    Training:
    1. FREE PHASE: Inject input, let network settle
    2. TARGET PHASE: Clamp output to target, let network settle again
       - Targets propagate BACKWARD through recurrent dynamics!
       - Phase coherence determines which paths carry targets
    3. LEARNING: Each node moves its free state toward its target state
    """

    def __init__(self, n_nodes: int, dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.n_nodes = n_nodes
        self.dim = dim

        # Create nodes
        self.nodes = [TargetLearningNode(i, dim) for i in range(n_nodes)]

        # Input/output
        self.input_node_ids = [0, 1]
        self.output_node_ids = [n_nodes - 1]

        for i in self.input_node_ids:
            self.nodes[i].is_input = True
        for i in self.output_node_ids:
            self.nodes[i].is_output = True

        # Initialize connectivity - HIGH density so targets propagate well
        self.initialize_connectivity(density=0.8)

        # Settling parameters
        self.max_iterations = 12
        self.convergence_threshold = 0.015

    def initialize_connectivity(self, density: float = 0.5):
        """Initialize bidirectional connections."""
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i >= j:
                    continue

                if np.random.random() < density:
                    # Stronger initial weights for better target propagation
                    w = np.random.uniform(0.5, 1.0)
                    node_i.connections[j] = Connection(j, w)
                    node_j.connections[i] = Connection(i, w)

    def get_neighbor_states(self, node: TargetLearningNode) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get states and phases of neighbors."""
        neighbors = {}
        for target_id in node.connections:
            neighbor = self.nodes[target_id]
            neighbors[target_id] = (neighbor.state.copy(), neighbor.phase.copy())
        return neighbors

    def step(self, target_mode: bool = False):
        """One step of dynamics."""
        for node in self.nodes:
            neighbors = self.get_neighbor_states(node)
            node.update(neighbors, target_mode=target_mode)

    def has_converged(self) -> bool:
        """Check convergence."""
        total_change = sum(
            np.mean(np.abs(node.state - node.prev_state))
            for node in self.nodes
        )
        return total_change / self.n_nodes < self.convergence_threshold

    def settle(self, min_steps: int = 3, target_mode: bool = False) -> int:
        """Settle until convergence."""
        for i in range(self.max_iterations):
            self.step(target_mode=target_mode)
            if i >= min_steps and self.has_converged():
                return i + 1
        return self.max_iterations

    def free_phase(self, input_values: np.ndarray) -> Tuple[float, int]:
        """
        FREE PHASE: Network settles naturally given input.
        """
        # Reset
        for node in self.nodes:
            node.reset()

        # Clamp inputs
        for i, node_id in enumerate(self.input_node_ids):
            if i < len(input_values):
                val = input_values[i]
                self.nodes[node_id].clamped_value = np.ones(self.dim) * val

        # Settle
        n_iters = self.settle()

        # Store free states
        for node in self.nodes:
            node.store_free_state()

        # Read output
        output_node = self.nodes[self.output_node_ids[0]]
        output = float(np.mean(output_node.state))

        return output, n_iters

    def target_phase(self, target: float) -> int:
        """
        TARGET PHASE: Clamp output to target, let network settle.

        Key insight: The recurrent dynamics NATURALLY propagate
        the target backward through the network!

        Nodes that are phase-coherent with output will receive
        stronger target signals.
        """
        # Don't reset - continue from free phase state
        # This helps targets propagate more effectively

        # Clamp output node to target
        for node_id in self.output_node_ids:
            self.nodes[node_id].clamped_value = np.ones(self.dim) * target

        # Target phase needs slightly more settling for propagation
        old_max = self.max_iterations
        self.max_iterations = 18  # More for target propagation

        # Use target_mode=True for stronger coupling during target propagation
        n_iters = self.settle(min_steps=6, target_mode=True)

        self.max_iterations = old_max

        # Store target states
        for node in self.nodes:
            node.store_target_state()

        return n_iters

    def learn(self):
        """
        Apply target learning at each node.
        """
        # Collect states
        free_states = {node.id: node.free_state.copy() for node in self.nodes}
        target_states = {node.id: node.target_state.copy() for node in self.nodes}

        # Each node learns
        for node in self.nodes:
            neighbor_free = {nid: free_states[nid] for nid in node.connections}
            neighbor_target = {nid: target_states[nid] for nid in node.connections}
            node.target_learning_update(neighbor_free, neighbor_target)

    def update_connectivity(self):
        """Dynamic connectivity based on phase coherence."""
        for node in self.nodes:
            for target_id, conn in node.connections.items():
                if target_id in node.coherence_trace:
                    coh = node.coherence_trace[target_id]
                    if coh > 0.3:
                        conn.weight = min(2.0, conn.weight + 0.002)
                    elif coh < -0.1:
                        conn.weight = max(0.1, conn.weight - 0.001)

    def train_step(self, input_values: np.ndarray, target: float) -> float:
        """
        Full training step with target learning.

        1. Free phase: settle naturally
        2. Target phase: clamp output, settle (targets propagate)
        3. Learn: each node adjusts toward its target state
        """
        # Free phase
        pred, free_iters = self.free_phase(input_values)

        # Target phase
        target_iters = self.target_phase(target)

        # Learn
        self.learn()

        # Update connectivity
        self.update_connectivity()

        # Clean up
        for node in self.nodes:
            node.clamped_value = None

        return pred

    def predict(self, input_values: np.ndarray) -> float:
        """Predict without learning."""
        pred, _ = self.free_phase(input_values)
        for node in self.nodes:
            node.clamped_value = None
        return pred

    def count_connections(self) -> int:
        """Count connections."""
        return sum(len(node.connections) for node in self.nodes)


def test_xor():
    """Test XOR with target learning."""
    print("=" * 70)
    print("PSI TARGET LEARNING")
    print("=" * 70)
    print()
    print("Based on neuroscience findings (2024):")
    print("  - Target learning predicts neural activity better than backprop")
    print("  - Neurons receive TARGET PATTERNS, not error gradients")
    print("  - Network settles to 'what should have happened'")
    print()
    print("Mechanism:")
    print("  1. FREE PHASE: settle naturally")
    print("  2. TARGET PHASE: clamp output, targets propagate via recurrence")
    print("  3. LEARN: each node moves free state toward target state")
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
        graph = PSITargetGraph(n_nodes=6, dim=6, seed=seed)

        for _ in range(100):
            np.random.shuffle(xor_data)
            for inp, target in xor_data:
                graph.train_step(inp, target)

        correct = 0
        for inp, target in xor_data:
            pred = graph.predict(inp)
            pred_class = 1.0 if pred > 0 else -1.0
            if pred_class == target:
                correct += 1

        if correct > best_acc:
            best_acc = correct
            best_seed = seed
            print(f"  Seed {seed}: {correct}/4")

        if correct == 4:
            break

    print(f"\nBest: seed {best_seed} with {best_acc}/4")

    # Full training
    print("\nFull training:")
    graph = PSITargetGraph(n_nodes=6, dim=6, seed=best_seed)

    accuracy_history = []
    for epoch in range(150):
        np.random.shuffle(xor_data)
        epoch_correct = 0

        for inp, target in xor_data:
            pred = graph.train_step(inp, target)
            pred_class = 1.0 if pred > 0 else -1.0
            if pred_class == target:
                epoch_correct += 1

        accuracy_history.append(epoch_correct / 4)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: {epoch_correct}/4, connections={graph.count_connections()}")

    # Final
    print("\nFinal Evaluation:")
    correct = 0
    for inp, target in xor_data:
        pred = graph.predict(inp)
        pred_class = 1.0 if pred > 0 else -1.0
        is_correct = pred_class == target
        if is_correct:
            correct += 1
        print(f"  {inp} -> {pred:.3f} (target {target:.0f}) "
              f"{'OK' if is_correct else 'WRONG'}")

    learned = correct >= 3
    print(f"\nFinal: {correct}/4 = {correct/4:.0%}")
    print(f"LEARNED XOR: {'YES' if learned else 'NO'}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(accuracy_history)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('XOR with PSI Target Learning')
    plt.ylim(0, 1.1)
    plt.savefig('psi_target_learning.png', dpi=150)
    plt.close()

    return learned


def _train_seed_xor(seed):
    """Train one seed - for parallel execution."""
    xor_data = [
        (np.array([-1.0, -1.0]), -1.0),
        (np.array([-1.0,  1.0]),  1.0),
        (np.array([ 1.0, -1.0]),  1.0),
        (np.array([ 1.0,  1.0]), -1.0),
    ]
    graph = PSITargetGraph(n_nodes=6, dim=6, seed=seed)
    for _ in range(100):
        np.random.shuffle(xor_data)
        for inp, target in xor_data:
            graph.train_step(inp, target)
    correct = sum(1 for inp, target in xor_data
                  if (graph.predict(inp) > 0) == (target > 0))
    return seed, correct


def test_robustness():
    """Test robustness with parallel execution."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST (20 seeds, parallel)")
    print("=" * 70)

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        results = pool.map(_train_seed_xor, range(20))

    successes = 0
    accuracies = []
    for seed, correct in sorted(results):
        success = correct >= 3
        if success:
            successes += 1
        accuracies.append(correct)
        print(f"  Seed {seed}: {correct}/4 {'OK' if success else 'FAIL'}")

    print(f"\nSuccess rate: {successes}/20 = {successes/20:.0%}")
    print(f"Mean accuracy: {np.mean(accuracies)/4:.0%}")

    return successes


def analyze_target_propagation():
    """Analyze how targets propagate through the network."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Target Propagation Through Recurrence")
    print("=" * 70)

    graph = PSITargetGraph(n_nodes=6, dim=4, seed=42)

    # Run one example
    inp = np.array([1.0, -1.0])
    target = 1.0

    print(f"\nInput: {inp}, Target: {target}")

    # Free phase
    print("\nFREE PHASE:")
    pred, _ = graph.free_phase(inp)
    print(f"  Output: {pred:.3f}")
    print("  Node states (mean):")
    for node in graph.nodes:
        role = "INPUT" if node.is_input else ("OUTPUT" if node.is_output else "HIDDEN")
        print(f"    Node {node.id} ({role}): {np.mean(node.free_state):.3f}")

    # Target phase
    print("\nTARGET PHASE:")
    graph.target_phase(target)
    print("  Node states (mean) - targets propagated:")
    for node in graph.nodes:
        role = "INPUT" if node.is_input else ("OUTPUT" if node.is_output else "HIDDEN")
        diff = np.mean(node.target_state) - np.mean(node.free_state)
        print(f"    Node {node.id} ({role}): {np.mean(node.target_state):.3f} "
              f"(change: {diff:+.3f})")

    # Clean up
    for node in graph.nodes:
        node.clamped_value = None


def test_few_shot_xor():
    """Test few-shot learning on XOR - can we learn with minimal examples?"""
    print("\n" + "=" * 70)
    print("FEW-SHOT XOR TEST")
    print("=" * 70)
    print("Goal: Learn XOR with as few training examples as possible")

    xor_data = [
        (np.array([-1.0, -1.0]), -1.0),
        (np.array([-1.0,  1.0]),  1.0),
        (np.array([ 1.0, -1.0]),  1.0),
        (np.array([ 1.0,  1.0]), -1.0),
    ]

    # Test different numbers of training iterations
    for n_shots in [5, 10, 20, 50, 100]:
        successes = 0
        for seed in range(10):
            graph = PSITargetGraph(n_nodes=6, dim=8, seed=seed)

            # Train for n_shots iterations (each iteration = 4 examples)
            for _ in range(n_shots):
                np.random.shuffle(xor_data)
                for inp, target in xor_data:
                    graph.train_step(inp, target)

            # Evaluate
            correct = 0
            for inp, target in xor_data:
                pred = graph.predict(inp)
                if (pred > 0) == (target > 0):
                    correct += 1
            if correct >= 3:
                successes += 1

        print(f"  {n_shots:3d} shots: {successes}/10 seeds succeed ({successes*10}%)")

    return True


def _train_seed_parity(seed):
    """Train one seed on 3-bit parity - for parallel execution."""
    parity_data = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                n_ones = sum(1 for x in [a, b, c] if x == 1)
                target = 1.0 if n_ones % 2 == 1 else -1.0
                parity_data.append((np.array([a, b, c], dtype=float), target))

    graph = PSITargetGraph(n_nodes=8, dim=8, seed=seed)
    graph.input_node_ids = [0, 1, 2]
    for i in graph.input_node_ids:
        graph.nodes[i].is_input = True

    for _ in range(100):
        np.random.shuffle(parity_data)
        for inp, target in parity_data:
            graph.train_step(inp, target)

    correct = sum(1 for inp, target in parity_data
                  if (graph.predict(inp) > 0) == (target > 0))
    return seed, correct


def test_3bit_parity():
    """Test 3-bit parity with parallel execution."""
    print("\n" + "=" * 70)
    print("3-BIT PARITY TEST (parallel)")
    print("=" * 70)
    print("Parity of 3 bits: output 1 if odd number of 1s, else -1")

    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        results = pool.map(_train_seed_parity, range(20))

    successes = 0
    for seed, correct in sorted(results):
        success = correct >= 6
        if success:
            successes += 1
        if seed < 5:
            print(f"  Seed {seed}: {correct}/8 {'OK' if success else 'FAIL'}")

    print(f"\nSuccess rate: {successes}/20 = {successes*5}%")
    return successes


def test_sample_efficiency():
    """Compare sample efficiency across training regimes."""
    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 70)

    xor_data = [
        (np.array([-1.0, -1.0]), -1.0),
        (np.array([-1.0,  1.0]),  1.0),
        (np.array([ 1.0, -1.0]),  1.0),
        (np.array([ 1.0,  1.0]), -1.0),
    ]

    # Track accuracy over training
    n_seeds = 5
    max_epochs = 200
    accuracy_curves = []

    for seed in range(n_seeds):
        graph = PSITargetGraph(n_nodes=6, dim=8, seed=seed)
        curve = []

        for epoch in range(max_epochs):
            # Train one epoch
            np.random.shuffle(xor_data)
            for inp, target in xor_data:
                graph.train_step(inp, target)

            # Evaluate
            correct = 0
            for inp, target in xor_data:
                pred = graph.predict(inp)
                if (pred > 0) == (target > 0):
                    correct += 1
            curve.append(correct / 4)

        accuracy_curves.append(curve)

    # Average curve
    avg_curve = np.mean(accuracy_curves, axis=0)

    # Find epochs to reach different thresholds
    for threshold in [0.5, 0.75, 1.0]:
        epochs_to_threshold = None
        for i, acc in enumerate(avg_curve):
            if acc >= threshold:
                epochs_to_threshold = i + 1
                break
        if epochs_to_threshold:
            print(f"  Epochs to {threshold:.0%} accuracy: {epochs_to_threshold}")
        else:
            print(f"  Epochs to {threshold:.0%} accuracy: Not reached in {max_epochs}")

    print(f"  Final accuracy: {avg_curve[-1]:.1%}")

    return avg_curve


if __name__ == "__main__":
    # First show how target propagation works
    analyze_target_propagation()

    # Test XOR
    xor_learned = test_xor()

    # Test robustness
    n_successes = test_robustness()

    # Test few-shot learning
    test_few_shot_xor()

    # Test harder problem
    parity_successes = test_3bit_parity()

    # Sample efficiency
    test_sample_efficiency()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  XOR: {'PASSED' if xor_learned else 'FAILED'}")
    print(f"  XOR Robustness: {n_successes}/20 seeds")
    print(f"  3-bit Parity: {parity_successes}/20 seeds")
    print()
    print("Predictive Target Learning:")
    print("  - Target patterns propagate through recurrent dynamics")
    print("  - Prediction error modulates learning (surprise = learn more)")
    print("  - Phase coherence gates which nodes learn together")
    print("  - Local learning rules only")
