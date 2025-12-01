"""
PSI-Graph Prototype
===================

A directed cyclic graph of PSI nodes with:
- Local prediction error learning (no backprop)
- Phase coherence-based connectivity (dynamic)
- Homeostatic stability
- Global signal injection
- Voting readout

Author: Brian Christensen
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class Message:
    """Message passed between nodes"""
    from_id: int
    phase: np.ndarray
    magnitude: float
    content: np.ndarray


@dataclass
class GlobalSignal:
    """Global modulatory signal"""
    type: str  # "reward", "attention", "novelty"
    intensity: float = 1.0


class PSINode:
    """
    A single PSI node in the graph.
    
    Implements:
    - Phase-based state evolution
    - Local prediction error learning
    - Homeostatic regulation
    - Dynamic connectivity based on phase coherence
    """
    
    def __init__(self, node_id: int, dim: int):
        self.id = node_id
        self.dim = dim
        self.is_hub = False
        
        # Phase state
        self.phase = np.random.uniform(0, 2 * np.pi, dim)
        self.omega = np.random.randn(dim) * 0.1  # Frequency
        
        # Activation state
        self.state = np.zeros(dim)
        self.magnitude = np.ones(dim) * 0.1
        
        # Homeostasis
        self.magnitude_ema = np.ones(dim) * 0.1
        self.target_magnitude = 1.0
        self.damping = 0.1
        
        # Connectivity (dynamic)
        self.neighbors: Dict[int, 'PSINode'] = {}
        self.weights: Dict[int, np.ndarray] = {}
        self.disconnect_count: Dict[int, int] = {}
        
        # Inbox for messages
        self.inbox: List[Message] = []
        
        # Learning
        self.lr = 0.01
        self.predictions: Dict[int, np.ndarray] = {}
    
    def receive(self, msg: Message):
        """Receive a message from a neighbor"""
        self.inbox.append(msg)
    
    def phase_coherence(self, other_phase: np.ndarray) -> float:
        """Compute phase coherence with another node's phase"""
        return float(np.mean(np.cos(self.phase - other_phase)))
    
    def predict_neighbor(self, neighbor_id: int) -> np.ndarray:
        """Get or initialize prediction for a neighbor"""
        if neighbor_id not in self.predictions:
            self.predictions[neighbor_id] = np.zeros(self.dim)
        return self.predictions[neighbor_id]
    
    def update(self, global_signal: Optional[GlobalSignal] = None):
        """
        Main update step:
        1. Process inbox (prediction error learning)
        2. Update phase
        3. Update state
        4. Homeostasis
        5. Apply global signal if present
        """
        # Process inbox
        total_input = np.zeros(self.dim)
        
        for msg in self.inbox:
            coherence = self.phase_coherence(msg.phase)
            weight = self.weights.get(msg.from_id, np.ones(self.dim))
            
            # Local prediction error learning
            prediction = self.predict_neighbor(msg.from_id)
            error = msg.content - prediction
            
            # Update prediction (local learning rule)
            self.predictions[msg.from_id] += self.lr * error * coherence
            
            # Weighted input (coherence-gated)
            total_input += coherence * weight * msg.content * msg.magnitude
        
        # Clear inbox
        self.inbox = []
        
        # Phase update (oscillation)
        self.phase += self.omega
        self.phase = np.mod(self.phase, 2 * np.pi)
        
        # State update (leaky integration)
        self.state = self.state * (1 - self.damping) + total_input * self.damping
        
        # Magnitude update
        self.magnitude = np.sqrt(self.state ** 2 + 0.01)  # Soft absolute value
        self.magnitude_ema = 0.95 * self.magnitude_ema + 0.05 * self.magnitude
        
        # Homeostasis
        self.apply_homeostasis()
        
        # Global signal modulation
        if global_signal:
            self.apply_global_signal(global_signal)
    
    def apply_homeostasis(self):
        """Maintain oscillation within target range"""
        delta = 0.5
        mean_mag = np.mean(self.magnitude_ema)
        
        if mean_mag > self.target_magnitude + delta:
            self.damping = min(0.99, self.damping + 0.005)
        elif mean_mag < self.target_magnitude - delta:
            self.damping = max(0.01, self.damping - 0.005)
        
        # Also normalize state if it explodes
        state_norm = np.linalg.norm(self.state)
        if state_norm > 10:
            self.state = self.state / state_norm * 10
    
    def apply_global_signal(self, signal: GlobalSignal):
        """Respond to global modulatory signal"""
        if signal.type == "reward":
            # Increase learning rate temporarily
            self.lr = min(0.1, self.lr * (1 + 0.1 * signal.intensity))
        elif signal.type == "attention":
            # Boost active nodes
            if np.mean(self.magnitude) > 0.5:
                self.state *= (1 + 0.1 * signal.intensity)
        elif signal.type == "novelty":
            # Reset predictions, increase learning
            self.predictions = {}
            self.lr = min(0.1, self.lr * (1 + 0.1 * signal.intensity))
        elif signal.type == "decay_lr":
            # Decay learning rate back to baseline
            self.lr = max(0.01, self.lr * 0.99)
    
    def broadcast(self) -> Message:
        """Create message to send to neighbors"""
        return Message(
            from_id=self.id,
            phase=self.phase.copy(),
            magnitude=float(np.mean(self.magnitude)),
            content=self.state.copy()
        )


class PSIGraph:
    """
    A graph of PSI nodes with:
    - Scale-free topology with rich club hubs
    - Dynamic connectivity based on phase coherence
    - Local learning only
    - Global signal broadcast
    - Voting readout
    """
    
    def __init__(self, n_nodes: int, n_hubs: int, dim: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.n_nodes = n_nodes
        self.n_hubs = n_hubs
        self.dim = dim
        
        # Create nodes
        self.nodes = [PSINode(i, dim) for i in range(n_nodes)]
        
        # Designate hubs
        self.hubs = self.nodes[:n_hubs]
        for hub in self.hubs:
            hub.is_hub = True
            hub.target_magnitude = 1.5  # Hubs more active
        
        # Initialize connectivity
        self.initialize_connectivity()
        
        # Input/output nodes (random subsets from non-hubs)
        non_hubs = self.nodes[n_hubs:]
        n_io = max(2, len(non_hubs) // 10)
        
        indices = np.random.permutation(len(non_hubs))
        self.input_nodes = [non_hubs[i] for i in indices[:n_io]]
        self.output_nodes = [non_hubs[i] for i in indices[n_io:2*n_io]]
        
        # Ensure input and output don't overlap
        input_ids = {n.id for n in self.input_nodes}
        self.output_nodes = [n for n in self.output_nodes if n.id not in input_ids]
        if len(self.output_nodes) < 2:
            remaining = [n for n in non_hubs if n.id not in input_ids and n not in self.output_nodes]
            self.output_nodes.extend(remaining[:2])
        
        # Statistics tracking
        self.step_count = 0
        self.edge_count_history = []
        self.energy_history = []
    
    def initialize_connectivity(self):
        """
        Initialize scale-free network with rich club.
        - Hubs fully connected to each other
        - Other nodes connect preferentially to high-degree nodes
        """
        # Rich club: fully connect hubs
        for i, hub1 in enumerate(self.hubs):
            for j, hub2 in enumerate(self.hubs):
                if i != j:
                    self.connect(hub1, hub2)
        
        # Scale-free attachment for other nodes
        for node in self.nodes[self.n_hubs:]:
            # Connect to 1-2 random hubs
            n_hub_connections = np.random.randint(1, min(3, self.n_hubs + 1))
            hub_targets = np.random.choice(self.hubs, size=n_hub_connections, replace=False)
            for hub in hub_targets:
                self.connect(node, hub)
                self.connect(hub, node)  # Bidirectional
            
            # Connect to a few random other nodes (preferential attachment approximation)
            other_nodes = [n for n in self.nodes if n.id != node.id and n.id not in [h.id for h in hub_targets]]
            if other_nodes:
                n_other = np.random.randint(1, min(4, len(other_nodes) + 1))
                # Prefer nodes with more connections
                degrees = np.array([len(n.neighbors) + 1 for n in other_nodes])
                probs = degrees / degrees.sum()
                targets = np.random.choice(other_nodes, size=min(n_other, len(other_nodes)), replace=False, p=probs)
                for target in targets:
                    if np.random.random() < 0.5:
                        self.connect(node, target)
                    if np.random.random() < 0.5:
                        self.connect(target, node)
    
    def connect(self, from_node: PSINode, to_node: PSINode):
        """Create directed connection"""
        if to_node.id not in from_node.neighbors:
            from_node.neighbors[to_node.id] = to_node
            from_node.weights[to_node.id] = np.ones(self.dim) * 0.5
            from_node.disconnect_count[to_node.id] = 0
    
    def disconnect(self, from_node: PSINode, to_node_id: int):
        """Remove directed connection"""
        if to_node_id in from_node.neighbors:
            del from_node.neighbors[to_node_id]
            if to_node_id in from_node.weights:
                del from_node.weights[to_node_id]
            if to_node_id in from_node.disconnect_count:
                del from_node.disconnect_count[to_node_id]
            if to_node_id in from_node.predictions:
                del from_node.predictions[to_node_id]
    
    def count_edges(self) -> int:
        """Count total directed edges"""
        return sum(len(n.neighbors) for n in self.nodes)
    
    def step(self, global_signal: Optional[GlobalSignal] = None):
        """
        One step of graph dynamics:
        1. All nodes broadcast
        2. Deliver messages
        3. All nodes update
        4. Maybe update connectivity
        """
        # 1. All nodes broadcast
        messages = {node.id: node.broadcast() for node in self.nodes}
        
        # 2. Deliver messages
        for node in self.nodes:
            for neighbor_id in node.neighbors:
                node.receive(messages[neighbor_id])
        
        # 3. All nodes update (parallel in concept)
        for node in self.nodes:
            node.update(global_signal)
        
        # 4. Update connectivity (less frequent)
        if self.step_count % 10 == 0:
            self.update_connectivity()
        
        # Track statistics
        self.step_count += 1
        if self.step_count % 5 == 0:
            self.edge_count_history.append(self.count_edges())
            self.energy_history.append(self.total_energy())
    
    def update_connectivity(self):
        """
        Update connections based on phase coherence.
        - Disconnect persistently incoherent pairs
        - Connect coherent nearby pairs
        """
        connect_threshold = 0.6
        disconnect_threshold = 0.2
        patience = 5
        
        for node in self.nodes:
            # Check existing connections
            to_disconnect = []
            for neighbor_id, neighbor in list(node.neighbors.items()):
                coherence = node.phase_coherence(neighbor.phase)
                
                if coherence < disconnect_threshold:
                    node.disconnect_count[neighbor_id] = node.disconnect_count.get(neighbor_id, 0) + 1
                    if node.disconnect_count[neighbor_id] > patience:
                        # Don't disconnect from hubs easily
                        if not neighbor.is_hub or np.random.random() < 0.1:
                            to_disconnect.append(neighbor_id)
                else:
                    node.disconnect_count[neighbor_id] = 0
                    # Strengthen coherent connections
                    node.weights[neighbor_id] = np.clip(
                        node.weights[neighbor_id] * (1 + 0.01 * coherence),
                        0.1, 2.0
                    )
            
            for neighbor_id in to_disconnect:
                self.disconnect(node, neighbor_id)
            
            # Check potential new connections (2-hop neighbors)
            current_neighbor_ids = set(node.neighbors.keys())
            candidates = set()
            
            for neighbor in node.neighbors.values():
                for second_hop_id in neighbor.neighbors:
                    if second_hop_id != node.id and second_hop_id not in current_neighbor_ids:
                        candidates.add(second_hop_id)
            
            # Limit new connections per step
            candidates = list(candidates)[:5]
            
            for candidate_id in candidates:
                candidate = self.nodes[candidate_id]
                coherence = node.phase_coherence(candidate.phase)
                if coherence > connect_threshold:
                    # Probabilistic connection
                    if np.random.random() < coherence:
                        self.connect(node, candidate)
    
    def inject_input(self, input_values: np.ndarray):
        """Inject input into input nodes"""
        for i, node in enumerate(self.input_nodes):
            if i < len(input_values):
                # Add to state, boosting magnitude
                node.state += input_values[i]
                node.magnitude = np.sqrt(node.state ** 2 + 0.01)
    
    def readout(self) -> np.ndarray:
        """Read output via weighted voting"""
        if not self.output_nodes:
            return np.zeros(self.dim)
        
        votes = []
        weights = []
        
        for node in self.output_nodes:
            votes.append(node.state)
            weights.append(np.mean(node.magnitude) + 0.01)
        
        votes = np.array(votes)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(votes, axis=0, weights=weights)
    
    def total_energy(self) -> float:
        """Total energy (sum of magnitudes) in the graph"""
        return sum(np.mean(n.magnitude) for n in self.nodes)
    
    def is_settled(self, window: int = 20, threshold: float = 0.1) -> bool:
        """Check if graph has settled to stable state"""
        if len(self.energy_history) < window:
            return False
        recent = self.energy_history[-window:]
        return np.std(recent) < threshold
    
    def compute(self, input_values: np.ndarray, max_steps: int = 100) -> np.ndarray:
        """
        Inject input and let graph settle, then readout.
        """
        self.inject_input(input_values)
        
        for step in range(max_steps):
            self.step()
            if self.is_settled():
                break
        
        return self.readout()
    
    def get_phase_coherence_matrix(self) -> np.ndarray:
        """Compute pairwise phase coherence matrix"""
        n = len(self.nodes)
        coherence = np.zeros((n, n))
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    coherence[i, j] = node_i.phase_coherence(node_j.phase)
        return coherence


# =============================================================================
# TESTS
# =============================================================================

def test_settling(n_nodes: int = 50, n_hubs: int = 5, dim: int = 16, 
                  max_steps: int = 500, seed: int = 42):
    """
    Test 1: Does the graph settle to a stable state?
    """
    print("=" * 60)
    print("TEST 1: SETTLING")
    print("=" * 60)
    
    graph = PSIGraph(n_nodes=n_nodes, n_hubs=n_hubs, dim=dim, seed=seed)
    
    print(f"Graph created: {n_nodes} nodes, {n_hubs} hubs, dim={dim}")
    print(f"Input nodes: {len(graph.input_nodes)}")
    print(f"Output nodes: {len(graph.output_nodes)}")
    print(f"Initial edges: {graph.count_edges()}")
    
    # Inject random input
    input_vals = np.random.randn(len(graph.input_nodes), dim) * 2
    graph.inject_input(input_vals)
    
    # Run and track
    energies = []
    for step in range(max_steps):
        graph.step()
        energy = graph.total_energy()
        energies.append(energy)
        
        if step % 100 == 0:
            print(f"  Step {step}: energy = {energy:.4f}")
    
    # Analyze
    final_energy = np.mean(energies[-50:])
    final_std = np.std(energies[-50:])
    initial_energy = np.mean(energies[:50])
    
    print(f"\nResults:")
    print(f"  Initial energy: {initial_energy:.4f}")
    print(f"  Final energy: {final_energy:.4f}")
    print(f"  Final std: {final_std:.6f}")
    print(f"  Final edges: {graph.count_edges()}")
    
    settled = final_std < 0.5
    print(f"\n  SETTLED: {'YES' if settled else 'NO'}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(energies)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Total Energy')
    axes[0].set_title('Energy Over Time')
    axes[0].axhline(y=final_energy, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_energy:.2f}')
    axes[0].legend()
    
    if graph.edge_count_history:
        axes[1].plot(graph.edge_count_history)
        axes[1].set_xlabel('Step (÷5)')
        axes[1].set_ylabel('Edge Count')
        axes[1].set_title('Connectivity Over Time')
    
    plt.tight_layout()
    plt.savefig('test1_settling.png', dpi=150)
    plt.close()
    
    return graph, settled


def test_connectivity_evolution(n_nodes: int = 50, n_hubs: int = 5, dim: int = 16,
                                 max_steps: int = 500, seed: int = 42):
    """
    Test 2: Does connectivity evolve based on phase coherence?
    """
    print("\n" + "=" * 60)
    print("TEST 2: CONNECTIVITY EVOLUTION")
    print("=" * 60)
    
    graph = PSIGraph(n_nodes=n_nodes, n_hubs=n_hubs, dim=dim, seed=seed)
    
    initial_edges = graph.count_edges()
    initial_coherence = graph.get_phase_coherence_matrix()
    
    print(f"Initial edges: {initial_edges}")
    print(f"Initial mean coherence: {np.mean(initial_coherence):.4f}")
    
    # Run with periodic input
    for step in range(max_steps):
        # Inject input periodically
        if step % 50 == 0:
            input_vals = np.random.randn(len(graph.input_nodes), dim)
            graph.inject_input(input_vals)
        
        graph.step()
    
    final_edges = graph.count_edges()
    final_coherence = graph.get_phase_coherence_matrix()
    
    print(f"\nFinal edges: {final_edges}")
    print(f"Final mean coherence: {np.mean(final_coherence):.4f}")
    print(f"Edge change: {final_edges - initial_edges} ({100*(final_edges-initial_edges)/initial_edges:+.1f}%)")
    
    # Check if connected pairs have higher coherence
    connected_coherences = []
    unconnected_coherences = []
    
    for i, node in enumerate(graph.nodes):
        for j, other in enumerate(graph.nodes):
            if i != j:
                coh = final_coherence[i, j]
                if other.id in node.neighbors:
                    connected_coherences.append(coh)
                else:
                    unconnected_coherences.append(coh)
    
    mean_connected = np.mean(connected_coherences) if connected_coherences else 0
    mean_unconnected = np.mean(unconnected_coherences) if unconnected_coherences else 0
    
    print(f"\nMean coherence (connected pairs): {mean_connected:.4f}")
    print(f"Mean coherence (unconnected pairs): {mean_unconnected:.4f}")
    
    coherence_diff = mean_connected - mean_unconnected
    evolved = coherence_diff > 0.05
    print(f"\n  CONNECTIVITY EVOLVED MEANINGFULLY: {'YES' if evolved else 'NO'}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(initial_coherence, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Initial Phase Coherence')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    
    axes[1].imshow(final_coherence, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Final Phase Coherence')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    
    if graph.edge_count_history:
        axes[2].plot(graph.edge_count_history)
        axes[2].set_xlabel('Step (÷5)')
        axes[2].set_ylabel('Edge Count')
        axes[2].set_title('Edge Count Over Time')
    
    plt.tight_layout()
    plt.savefig('test2_connectivity.png', dpi=150)
    plt.close()
    
    return graph, evolved


def test_xor_learning(n_nodes: int = 50, n_hubs: int = 5, dim: int = 8,
                      n_epochs: int = 100, seed: int = 42):
    """
    Test 3: Can it learn XOR with local learning + global reward?
    """
    print("\n" + "=" * 60)
    print("TEST 3: XOR LEARNING")
    print("=" * 60)
    
    graph = PSIGraph(n_nodes=n_nodes, n_hubs=n_hubs, dim=dim, seed=seed)
    
    # XOR dataset
    xor_data = [
        (np.array([0.0, 0.0]), 0),
        (np.array([0.0, 1.0]), 1),
        (np.array([1.0, 0.0]), 1),
        (np.array([1.0, 1.0]), 0),
    ]
    
    print(f"Training XOR with {n_epochs} epochs")
    print(f"Graph: {n_nodes} nodes, {n_hubs} hubs")
    
    accuracy_history = []
    
    for epoch in range(n_epochs):
        correct = 0
        total = 0
        
        for inp, target in xor_data:
            # Encode input: spread across input nodes
            encoded = np.zeros((len(graph.input_nodes), dim))
            encoded[0, :2] = inp * 2 - 1  # Scale to [-1, 1]
            
            # Let graph settle
            output = graph.compute(encoded, max_steps=30)
            
            # Decode output: simple threshold on mean
            prediction = 1 if np.mean(output) > 0 else 0
            
            # Check and reward
            if prediction == target:
                correct += 1
                graph.step(GlobalSignal("reward", 1.0))
            else:
                graph.step(GlobalSignal("novelty", 0.5))
            
            total += 1
            
            # Decay learning rate
            graph.step(GlobalSignal("decay_lr", 0.1))
        
        accuracy = correct / total
        accuracy_history.append(accuracy)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: {correct}/{total} = {accuracy:.1%}")
    
    final_accuracy = np.mean(accuracy_history[-10:])
    learned = final_accuracy > 0.6
    
    print(f"\nFinal accuracy (last 10 epochs): {final_accuracy:.1%}")
    print(f"\n  LEARNED XOR: {'YES' if learned else 'NO'}")
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('XOR Learning Progress')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig('test3_xor.png', dpi=150)
    plt.close()
    
    return graph, learned


def test_cluster_formation(n_nodes: int = 50, n_hubs: int = 5, dim: int = 16,
                           max_steps: int = 300, seed: int = 42):
    """
    Test 4: Do phase-coherent clusters form?
    """
    print("\n" + "=" * 60)
    print("TEST 4: CLUSTER FORMATION")
    print("=" * 60)
    
    graph = PSIGraph(n_nodes=n_nodes, n_hubs=n_hubs, dim=dim, seed=seed)
    
    # Inject different inputs to different input nodes to encourage clustering
    for step in range(max_steps):
        if step % 30 == 0:
            # Different frequencies for different input nodes
            for i, node in enumerate(graph.input_nodes):
                freq = (i + 1) * 0.5
                input_val = np.sin(np.arange(dim) * freq + step * 0.1)
                node.state += input_val
        
        graph.step()
    
    # Analyze clusters via phase coherence
    coherence_matrix = graph.get_phase_coherence_matrix()
    
    # Simple clustering: threshold coherence matrix
    threshold = 0.5
    adjacency = (coherence_matrix > threshold).astype(int)
    
    # Count cluster sizes via connected components (simple BFS)
    visited = set()
    clusters = []
    
    for start in range(n_nodes):
        if start in visited:
            continue
        
        cluster = []
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            
            for neighbor in range(n_nodes):
                if neighbor not in visited and adjacency[node, neighbor]:
                    queue.append(neighbor)
        
        if cluster:
            clusters.append(cluster)
    
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    
    print(f"Found {len(clusters)} clusters")
    print(f"Cluster sizes: {cluster_sizes[:10]}...")  # Top 10
    print(f"Largest cluster: {cluster_sizes[0]} nodes ({100*cluster_sizes[0]/n_nodes:.1f}%)")
    
    # Good clustering = multiple clusters, not one giant one
    good_clustering = len(clusters) >= 3 and cluster_sizes[0] < 0.8 * n_nodes
    print(f"\n  CLUSTERS FORMED: {'YES' if good_clustering else 'NO'}")
    
    # Plot coherence matrix sorted by cluster
    sorted_indices = []
    for cluster in sorted(clusters, key=len, reverse=True):
        sorted_indices.extend(cluster)
    
    sorted_coherence = coherence_matrix[np.ix_(sorted_indices, sorted_indices)]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(sorted_coherence, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Phase Coherence')
    plt.title(f'Phase Coherence Matrix (sorted by cluster)\n{len(clusters)} clusters found')
    plt.xlabel('Node (sorted)')
    plt.ylabel('Node (sorted)')
    plt.savefig('test4_clusters.png', dpi=150)
    plt.close()
    
    return graph, good_clustering


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "=" * 70)
    print("PSI-GRAPH PROTOTYPE TESTS")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Settling
    _, results['settling'] = test_settling()
    
    # Test 2: Connectivity Evolution
    _, results['connectivity'] = test_connectivity_evolution()
    
    # Test 3: XOR Learning
    _, results['xor'] = test_xor_learning()
    
    # Test 4: Cluster Formation
    _, results['clusters'] = test_cluster_formation()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name:20s}: {status}")
    
    total_passed = sum(results.values())
    print(f"\n  Total: {total_passed}/{len(results)} tests passed")
    
    return results


if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()
    
    # Or run individual tests:
    # test_settling()
    # test_connectivity_evolution()
    # test_xor_learning()
    # test_cluster_formation()