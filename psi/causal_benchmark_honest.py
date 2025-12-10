"""
HONEST Causal Benchmark: What does Clifford memory ACTUALLY offer?

Previous benchmark was misleading - it used standard MLPs with hardcoded
causal structure, not Clifford memory.

The REAL question: What can Clifford phasor memory do that standard
approaches can't?

Hypothesis: Clifford memory's strength is in BINDING and RETRIEVAL with
minimal interference. For causal reasoning, this could help with:

1. Storing multiple causal relationships without interference
2. Compositional retrieval (query: "effect of X on Y given Z")
3. Scaling to many variables without O(n²) pairwise mechanisms

Let's test this HONESTLY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# TRUE Clifford Memory for Causal Reasoning
# =============================================================================

class CliffordCausalMemory(nn.Module):
    """
    Clifford phasor memory applied to causal reasoning.

    Key idea: Store causal relationships as phasor-bound vectors.
    - Each variable gets a unique phase address
    - Causal relationships are stored as: cause_phasor * effect_phasor * mechanism
    - Retrieval uses conjugate unbinding

    This is the ACTUAL Clifford approach, not just MLPs.
    """
    def __init__(self, dim: int, n_phases: int = 16, n_vars: int = 10):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.n_vars = n_vars

        # Each variable gets a learnable phase signature
        self.var_phases = nn.Parameter(torch.randn(n_vars, n_phases) * math.pi)

        # Mechanism encoder: transforms cause value into effect contribution
        self.mechanism_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Value encoder/decoder for memory
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def get_phasor(self, var_idx: int) -> torch.Tensor:
        """Get complex phasor for a variable."""
        return torch.exp(1j * self.var_phases[var_idx])

    def bind(self, phasor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Bind value to phasor address. Returns [batch, n_phases, dim]."""
        value_enc = self.encoder(value).to(torch.complex64)
        # phasor: [n_phases], value_enc: [batch, dim]
        return phasor.unsqueeze(0).unsqueeze(-1) * value_enc.unsqueeze(1)

    def retrieve(self, memory: torch.Tensor, query_phasor: torch.Tensor) -> torch.Tensor:
        """Retrieve from memory using conjugate query."""
        # memory: [batch, n_phases, dim], query_phasor: [n_phases]
        retrieved = memory * query_phasor.conj().unsqueeze(0).unsqueeze(-1)
        # Sum across phases, take real part
        return self.decoder(retrieved.sum(dim=1).real)

    def store_causal_link(self, memory: torch.Tensor,
                          cause_idx: int, effect_idx: int,
                          cause_value: torch.Tensor) -> torch.Tensor:
        """
        Store a causal relationship: cause -> effect.

        The key insight: we bind the MECHANISM OUTPUT to the joint address
        of cause and effect. This allows retrieval of "what effect does
        cause have on effect variable?"
        """
        cause_phasor = self.get_phasor(cause_idx)
        effect_phasor = self.get_phasor(effect_idx)

        # Joint address: product of cause and effect phasors
        joint_phasor = cause_phasor * effect_phasor

        # Compute mechanism output
        mechanism_output = self.mechanism_net(cause_value)

        # Bind to joint address and add to memory
        binding = self.bind(joint_phasor, mechanism_output)
        return memory + binding

    def query_causal_effect(self, memory: torch.Tensor,
                            cause_idx: int, effect_idx: int) -> torch.Tensor:
        """
        Query: What is the causal effect of cause on effect?

        This retrieves the mechanism output stored at the joint address.
        """
        cause_phasor = self.get_phasor(cause_idx)
        effect_phasor = self.get_phasor(effect_idx)
        joint_phasor = cause_phasor * effect_phasor

        return self.retrieve(memory, joint_phasor)

    def init_memory(self, batch_size: int) -> torch.Tensor:
        """Initialize empty memory."""
        return torch.zeros(batch_size, self.n_phases, self.dim,
                          dtype=torch.complex64, device=device)


class StandardCausalModel(nn.Module):
    """
    Standard approach: One MLP per causal relationship.

    This is the baseline - explicit mechanism for each edge in the graph.
    Scales as O(E) where E is number of edges.
    """
    def __init__(self, dim: int, n_vars: int = 10):
        super().__init__()
        self.dim = dim
        self.n_vars = n_vars

        # One mechanism per possible edge (O(n²) parameters!)
        self.mechanisms = nn.ModuleDict()
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    self.mechanisms[f"{i}_to_{j}"] = nn.Sequential(
                        nn.Linear(dim, dim * 2),
                        nn.GELU(),
                        nn.Linear(dim * 2, dim)
                    )

    def get_effect(self, cause_idx: int, effect_idx: int,
                   cause_value: torch.Tensor) -> torch.Tensor:
        """Get effect of cause on effect variable."""
        key = f"{cause_idx}_to_{effect_idx}"
        if key in self.mechanisms:
            return self.mechanisms[key](cause_value)
        return torch.zeros_like(cause_value)


# =============================================================================
# Test 1: Memory Interference
# =============================================================================

def test_memory_interference():
    """
    Test: Can Clifford memory store multiple causal relationships without
    interference, while standard memory suffers from crosstalk?

    This tests the core claim of phasor memory: orthogonal storage.
    """
    print("=" * 70)
    print("TEST 1: Memory Interference with Multiple Causal Relationships")
    print("=" * 70)

    dim = 32
    n_phases = 16
    n_vars = 8
    batch_size = 64

    clifford = CliffordCausalMemory(dim, n_phases, n_vars).to(device)

    # Store multiple causal relationships
    memory = clifford.init_memory(batch_size)

    # Create distinct cause values for each relationship
    cause_values = {}
    for i in range(n_vars - 1):
        cause_values[(i, i+1)] = torch.randn(batch_size, dim, device=device)
        # Store: variable i causes variable i+1
        memory = clifford.store_causal_link(memory, i, i+1, cause_values[(i, i+1)])

    # Test retrieval accuracy
    print(f"\nStored {n_vars - 1} causal relationships")
    print("\nRetrieval test (should retrieve what was stored):")

    total_error = 0
    for i in range(n_vars - 1):
        # What we stored
        original = clifford.mechanism_net(cause_values[(i, i+1)])

        # What we retrieve
        retrieved = clifford.query_causal_effect(memory, i, i+1)

        # Error
        error = F.mse_loss(retrieved, original).item()
        total_error += error
        print(f"  Edge {i}->{i+1}: MSE = {error:.4f}")

    avg_error = total_error / (n_vars - 1)
    print(f"\nAverage retrieval error: {avg_error:.4f}")

    # Test interference: query non-existent relationships
    print("\nInterference test (querying relationships that DON'T exist):")
    interference_total = 0
    n_queries = 0

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and (i, j) not in cause_values:
                retrieved = clifford.query_causal_effect(memory, i, j)
                magnitude = retrieved.abs().mean().item()
                interference_total += magnitude
                n_queries += 1

    avg_interference = interference_total / n_queries if n_queries > 0 else 0
    print(f"  Average magnitude of non-existent queries: {avg_interference:.4f}")
    print(f"  (Should be ~0 if no interference, >0 means crosstalk)")

    # Signal-to-noise ratio
    if avg_interference > 0:
        snr = avg_error / avg_interference
        print(f"\n  Signal/Noise ratio: {snr:.2f}")

    return avg_error, avg_interference


# =============================================================================
# Test 2: Scaling with Number of Variables
# =============================================================================

def test_scaling():
    """
    Test: How does performance scale as we add more variables/relationships?

    Clifford memory should maintain retrieval accuracy due to orthogonal phases.
    Standard approaches would need O(n²) parameters.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Scaling with Number of Variables")
    print("=" * 70)

    dim = 32
    n_phases = 32  # More phases for more variables
    batch_size = 32

    var_counts = [4, 8, 16, 32]

    print(f"\n{'N vars':<10} {'Relationships':<15} {'Retrieval MSE':<15} {'Interference':<15}")
    print("-" * 55)

    for n_vars in var_counts:
        clifford = CliffordCausalMemory(dim, n_phases, n_vars).to(device)
        memory = clifford.init_memory(batch_size)

        # Store chain of causal relationships: 0->1->2->...->n
        cause_values = {}
        for i in range(n_vars - 1):
            cause_values[(i, i+1)] = torch.randn(batch_size, dim, device=device)
            memory = clifford.store_causal_link(memory, i, i+1, cause_values[(i, i+1)])

        # Measure retrieval accuracy
        total_error = 0
        for i in range(n_vars - 1):
            original = clifford.mechanism_net(cause_values[(i, i+1)])
            retrieved = clifford.query_causal_effect(memory, i, i+1)
            total_error += F.mse_loss(retrieved, original).item()
        avg_error = total_error / (n_vars - 1)

        # Measure interference
        interference_total = 0
        n_queries = 0
        for i in range(min(n_vars, 10)):  # Sample to save time
            for j in range(min(n_vars, 10)):
                if i != j and (i, j) not in cause_values:
                    retrieved = clifford.query_causal_effect(memory, i, j)
                    interference_total += retrieved.abs().mean().item()
                    n_queries += 1
        avg_interference = interference_total / n_queries if n_queries > 0 else 0

        n_relationships = n_vars - 1
        print(f"{n_vars:<10} {n_relationships:<15} {avg_error:<15.4f} {avg_interference:<15.4f}")


# =============================================================================
# Test 3: Actual Causal Learning Task
# =============================================================================

def test_causal_learning():
    """
    Test: Can Clifford memory learn causal relationships from data?

    Setup:
    - Multiple variables with known causal structure
    - Train to predict effects from causes
    - Test generalization to new cause values
    """
    print("\n" + "=" * 70)
    print("TEST 3: Learning Causal Relationships from Data")
    print("=" * 70)

    dim = 32
    n_phases = 16
    n_vars = 4

    # True causal structure: X0 -> X1 -> X2 -> X3 (chain)
    # True mechanisms: X_{i+1} = W_i @ X_i + noise
    true_weights = [torch.randn(dim, dim, device=device) * 0.3 for _ in range(n_vars - 1)]

    def generate_data(n_samples):
        X = [torch.randn(n_samples, dim, device=device)]
        for i in range(n_vars - 1):
            X.append(X[-1] @ true_weights[i].T + torch.randn(n_samples, dim, device=device) * 0.1)
        return X

    # Models
    clifford = CliffordCausalMemory(dim, n_phases, n_vars).to(device)
    standard = StandardCausalModel(dim, n_vars).to(device)

    opt_cliff = torch.optim.Adam(clifford.parameters(), lr=1e-3)
    opt_std = torch.optim.Adam(standard.parameters(), lr=1e-3)

    print("\nTraining both models on causal chain X0 -> X1 -> X2 -> X3")

    for epoch in range(500):
        X = generate_data(64)

        # Clifford: store relationships and predict
        memory = clifford.init_memory(64)
        loss_cliff = 0
        for i in range(n_vars - 1):
            memory = clifford.store_causal_link(memory, i, i+1, X[i])
            predicted = clifford.query_causal_effect(memory, i, i+1)
            # Should predict the mechanism output that leads to X[i+1]
            target = X[i+1] - X[i] @ true_weights[i].T  # Residual after linear part
            loss_cliff += F.mse_loss(predicted, X[i+1])  # Simplified: predict next var

        opt_cliff.zero_grad()
        loss_cliff.backward()
        opt_cliff.step()

        # Standard: use dedicated mechanisms
        loss_std = 0
        for i in range(n_vars - 1):
            predicted = standard.get_effect(i, i+1, X[i])
            loss_std += F.mse_loss(predicted, X[i+1])

        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Clifford loss = {loss_cliff.item():.4f}, "
                  f"Standard loss = {loss_std.item():.4f}")

    # Test on new data
    print("\nTesting on held-out data:")
    with torch.no_grad():
        X_test = generate_data(200)

        # Clifford
        memory = clifford.init_memory(200)
        for i in range(n_vars - 1):
            memory = clifford.store_causal_link(memory, i, i+1, X_test[i])

        cliff_errors = []
        std_errors = []

        for i in range(n_vars - 1):
            # Clifford prediction
            cliff_pred = clifford.query_causal_effect(memory, i, i+1)
            cliff_err = F.mse_loss(cliff_pred, X_test[i+1]).item()
            cliff_errors.append(cliff_err)

            # Standard prediction
            std_pred = standard.get_effect(i, i+1, X_test[i])
            std_err = F.mse_loss(std_pred, X_test[i+1]).item()
            std_errors.append(std_err)

            print(f"  Edge {i}->{i+1}: Clifford MSE = {cliff_err:.4f}, Standard MSE = {std_err:.4f}")

        print(f"\n  Average - Clifford: {np.mean(cliff_errors):.4f}, Standard: {np.mean(std_errors):.4f}")

    # Parameter comparison
    cliff_params = sum(p.numel() for p in clifford.parameters())
    std_params = sum(p.numel() for p in standard.parameters())
    print(f"\n  Parameters - Clifford: {cliff_params:,}, Standard: {std_params:,}")
    print(f"  Ratio: Standard has {std_params/cliff_params:.1f}x more parameters")


# =============================================================================
# Test 4: The REAL Advantage - Compositional Queries
# =============================================================================

def test_compositional_queries():
    """
    Test: Can Clifford memory support compositional causal queries?

    This is where phasor binding should shine: combining multiple
    queries through phasor algebra.

    Example: Query "effect of X on Z through Y" should be computable
    by composing X->Y and Y->Z retrievals.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Compositional Causal Queries")
    print("=" * 70)

    dim = 32
    n_phases = 16
    batch_size = 64

    clifford = CliffordCausalMemory(dim, n_phases, n_vars=4).to(device)

    # Store: X(0) -> Y(1) -> Z(2)
    memory = clifford.init_memory(batch_size)

    X_val = torch.randn(batch_size, dim, device=device)
    Y_val = torch.randn(batch_size, dim, device=device)

    memory = clifford.store_causal_link(memory, 0, 1, X_val)  # X -> Y
    memory = clifford.store_causal_link(memory, 1, 2, Y_val)  # Y -> Z

    # Direct queries
    xy_effect = clifford.query_causal_effect(memory, 0, 1)
    yz_effect = clifford.query_causal_effect(memory, 1, 2)

    print("\nDirect queries:")
    print(f"  X->Y effect magnitude: {xy_effect.abs().mean().item():.4f}")
    print(f"  Y->Z effect magnitude: {yz_effect.abs().mean().item():.4f}")

    # Compositional query: X -> Z (transitive)
    # In phasor algebra: X_phasor * Z_phasor should retrieve nothing (no direct edge)
    # But X_phasor * Y_phasor * Y_phasor * Z_phasor = X_phasor * Z_phasor (Y cancels!)
    # This is the theoretical advantage of holographic/phasor representations

    xz_direct = clifford.query_causal_effect(memory, 0, 2)
    print(f"\n  X->Z direct query (no edge stored): {xz_direct.abs().mean().item():.4f}")
    print(f"  (Should be ~0 since we didn't store X->Z)")

    # Can we compose? This requires more sophisticated algebra...
    print("\n  Note: True compositional queries require additional mechanisms")
    print("  (binding algebra for causal chains is an open research question)")


if __name__ == "__main__":
    print("HONEST ASSESSMENT: What does Clifford Memory offer for Causal Reasoning?")
    print("=" * 70)
    print()

    test_memory_interference()
    test_scaling()
    test_causal_learning()
    test_compositional_queries()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. MEMORY INTERFERENCE: Clifford phasor binding provides some isolation
   between stored relationships, but interference grows with density.

2. SCALING: Clifford memory uses O(n_phases * dim) parameters regardless
   of number of variables, while standard approaches need O(n² * dim²).

3. LEARNING: Both approaches can learn causal mechanisms, but standard
   MLPs may be more efficient for small numbers of variables.

4. COMPOSITIONAL QUERIES: The theoretical advantage of phasor algebra
   for compositional reasoning is not fully realized in this implementation.

HONEST ASSESSMENT:
- Clifford memory's main advantage is PARAMETER EFFICIENCY for many variables
- For small causal graphs (<10 variables), standard MLPs may be better
- The "magic" of phasor binding for causal reasoning needs more work
- Previous benchmark results were NOT due to Clifford memory!
""")
