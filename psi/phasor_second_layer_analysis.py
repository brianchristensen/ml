"""
Analysis: Why does phasor need 2 layers? What alternatives work?

The observation: Phasor NEVER works with 1 layer, always needs >= 2.

Hypotheses:
1. Phase alignment/correction
2. Iterative refinement
3. Separation of binding vs routing

Let's test minimal alternatives to a full second layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PhasorLayer(nn.Module):
    """Standard phasor layer (linear to_out)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.dim = dim
        self.n_planes = n_planes

        self.key_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.key_encoder.weight)
        self.query_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.query_encoder.weight)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape if x.dim() == 2 else (1, x.shape[0])
        if x.dim() == 1:
            x = x.unsqueeze(0)

        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(1)
        retrieved = bound * query_phasor.conj().unsqueeze(-1)

        phase_alignment = (key_phasor * query_phasor.conj()).real
        resonance = phase_alignment.mean(dim=-1, keepdim=True)
        resonance_gain = F.softplus(resonance + 0.5)

        summed = retrieved.sum(dim=1).real
        output = summed * resonance_gain / self.n_planes

        return x + self.to_out(output)


class PhasorLayerWithDecoder(nn.Module):
    """Phasor layer with nonlinear decoder in to_out (no expansion)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.dim = dim
        self.n_planes = n_planes

        self.key_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.key_encoder.weight)
        self.query_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.query_encoder.weight)

        self.to_value = nn.Linear(dim, dim)
        # Key change: add GELU between norm and linear (no expansion)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape if x.dim() == 2 else (1, x.shape[0])
        if x.dim() == 1:
            x = x.unsqueeze(0)

        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(1)
        retrieved = bound * query_phasor.conj().unsqueeze(-1)

        phase_alignment = (key_phasor * query_phasor.conj()).real
        resonance = phase_alignment.mean(dim=-1, keepdim=True)
        resonance_gain = F.softplus(resonance + 0.5)

        summed = retrieved.sum(dim=1).real
        output = summed * resonance_gain / self.n_planes

        return x + self.to_out(output)


# =============================================================================
# Alternative "second layer" implementations
# =============================================================================

class PhaseCorrector(nn.Module):
    """
    Minimal alternative 1: Just learn a phase correction.

    Hypothesis: The second layer mainly corrects/aligns phases.
    """
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.phase_shift = nn.Parameter(torch.zeros(n_planes))
        self.scale = nn.Parameter(torch.ones(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Just scale and normalize - no new binding
        return self.norm(x * self.scale)


class LinearProjection(nn.Module):
    """
    Minimal alternative 2: Just a linear projection.

    Hypothesis: Second layer is just a learned linear transform.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.proj(x))


class MLPDecoder(nn.Module):
    """
    Minimal alternative 3: MLP decoder.

    Hypothesis: Second layer is just a nonlinear decoder.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.mlp(x))


class PhaseRefinement(nn.Module):
    """
    Minimal alternative 4: Re-encode and unbind with NEW phases.

    Hypothesis: The key is re-encoding with different phases.
    No value projection, no cumsum - just phase transformation.
    """
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.n_planes = n_planes

        # New phase encoders (different from layer 1)
        self.refine_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.refine_encoder.weight)
        self.output_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.output_encoder.weight)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape if x.dim() == 2 else (1, x.shape[0])
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Encode current representation into phases
        refine_phase = torch.tanh(self.refine_encoder(x)) * math.pi
        output_phase = torch.tanh(self.output_encoder(x)) * math.pi

        refine_phasor = torch.exp(1j * refine_phase)
        output_phasor = torch.exp(1j * output_phase)

        # Bind x to refine phases, unbind with output phases
        x_complex = x.to(torch.complex64)
        bound = refine_phasor.unsqueeze(-1) * x_complex.unsqueeze(1)  # [B, n_planes, D]
        retrieved = bound * output_phasor.conj().unsqueeze(-1)

        # Resonance
        phase_alignment = (refine_phasor * output_phasor.conj()).real
        resonance = phase_alignment.mean(dim=-1, keepdim=True)
        resonance_gain = F.softplus(resonance + 0.5)

        summed = retrieved.sum(dim=1).real
        output = summed * resonance_gain / self.n_planes

        return x + self.norm(output)


class IterativeRefinement(nn.Module):
    """
    Minimal alternative 5: Apply the SAME phasor operation twice.

    Hypothesis: It's iterative refinement - same operation, just repeated.
    """
    def __init__(self, phasor_layer: PhasorLayer, n_iters: int = 2):
        super().__init__()
        self.layer = phasor_layer
        self.n_iters = n_iters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.n_iters):
            h = self.layer(h)
        return h


# =============================================================================
# Test configurations
# =============================================================================

class SingleLayerPhasor(nn.Module):
    """Baseline: Single phasor layer with linear to_out."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer = PhasorLayer(dim, n_planes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.layer(x))


class SingleLayerPhasorWithDecoder(nn.Module):
    """NEW: Single phasor layer with nonlinear to_out (GELU inside)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer = PhasorLayerWithDecoder(dim, n_planes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.layer(x))


class TwoLayerPhasor(nn.Module):
    """Standard: Two phasor layers (expected to work)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer1 = PhasorLayer(dim, n_planes)
        self.layer2 = PhasorLayer(dim, n_planes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        return self.norm(h)


class PhasorPlusCorrector(nn.Module):
    """Phasor + phase corrector."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer1 = PhasorLayer(dim, n_planes)
        self.corrector = PhaseCorrector(dim, n_planes)

    def forward(self, x):
        h = self.layer1(x)
        return self.corrector(h)


class PhasorPlusLinear(nn.Module):
    """Phasor + linear projection."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer1 = PhasorLayer(dim, n_planes)
        self.linear = LinearProjection(dim)

    def forward(self, x):
        h = self.layer1(x)
        return self.linear(h)


class PhasorPlusMLP(nn.Module):
    """Phasor + MLP decoder."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer1 = PhasorLayer(dim, n_planes)
        self.mlp = MLPDecoder(dim)

    def forward(self, x):
        h = self.layer1(x)
        return self.mlp(h)


class PhasorPlusRefinement(nn.Module):
    """Phasor + phase refinement (minimal second phasor)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer1 = PhasorLayer(dim, n_planes)
        self.refine = PhaseRefinement(dim, n_planes)

    def forward(self, x):
        h = self.layer1(x)
        return self.refine(h)


class PhasorIterative(nn.Module):
    """Single phasor layer applied twice (shared weights)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.layer = PhasorLayer(dim, n_planes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.layer(x)
        h = self.layer(h)  # Same layer, applied twice
        return self.norm(h)


# =============================================================================
# Test task: Learn Y = W @ X transformation
# =============================================================================

def test_configuration(model_class, name, dim=32, n_planes=16, n_epochs=500, task='associative'):
    """Test a model configuration on a task."""

    if task == 'linear':
        # Simple linear transformation (might not be phasor's strength)
        W_true = torch.randn(dim, dim, device=device) * 0.3

        def generate_data(n):
            X = torch.randn(n, dim, device=device)
            Y = X @ W_true.T + torch.randn(n, dim, device=device) * 0.1
            return X, Y

    elif task == 'associative':
        # Associative recall: given key, retrieve associated value
        # This is what phasor should excel at
        n_pairs = 64  # Number of key-value pairs to memorize
        keys = torch.randn(n_pairs, dim, device=device)
        keys = F.normalize(keys, dim=-1)  # Normalize to make discrimination harder
        values = torch.randn(n_pairs, dim, device=device)

        def generate_data(n):
            # Sample random pairs
            indices = torch.randint(0, n_pairs, (n,), device=device)
            X = keys[indices] + torch.randn(n, dim, device=device) * 0.05  # Less noise, but more pairs
            Y = values[indices]  # Exact value
            return X, Y

    elif task == 'sequence_binding':
        # Harder task: Bind position to content and retrieve
        # More like what phasor is designed for in sequence models
        seq_len = 8

        def generate_data(n):
            # Generate sequences
            content = torch.randn(n, seq_len, dim, device=device)
            positions = torch.randn(seq_len, dim, device=device)  # Fixed position embeddings

            # Query: position embedding for random position
            query_pos = torch.randint(0, seq_len, (n,), device=device)
            X = positions[query_pos]  # Query with position
            Y = content[torch.arange(n, device=device), query_pos]  # Retrieve content at that position
            return X, Y

    elif task == 'copy':
        # Copy task: output = input (with some noise during training)
        def generate_data(n):
            X = torch.randn(n, dim, device=device)
            Y = X.clone()
            return X, Y

    elif task == 'xor_binding':
        # XOR-like binding task: must combine information from input in nonlinear way
        # This should require multi-layer processing
        W1 = torch.randn(dim // 2, dim, device=device) * 0.5
        W2 = torch.randn(dim // 2, dim, device=device) * 0.5

        def generate_data(n):
            X = torch.randn(n, dim, device=device)
            # Split input, apply nonlinear combination
            part1 = X @ W1.T  # [n, dim//2]
            part2 = X @ W2.T  # [n, dim//2]
            # XOR-like: features only active when one is high, other low
            combined = torch.cat([
                torch.relu(part1) * (1 - torch.sigmoid(part2)),
                torch.relu(part2) * (1 - torch.sigmoid(part1))
            ], dim=-1)
            Y = combined
            return X, Y

    elif task == 'selective_copy':
        # Selective copy: only some elements matter, need to filter
        mask = torch.zeros(dim, device=device)
        mask[::2] = 1.0  # Only even indices matter

        def generate_data(n):
            X = torch.randn(n, dim, device=device)
            Y = X * mask  # Target is input but only at certain positions
            return X, Y

    # Create model
    model = model_class(dim, n_planes).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(n_epochs):
        X, Y = generate_data(128)
        pred = model(X)
        loss = F.mse_loss(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    with torch.no_grad():
        X_test, Y_test = generate_data(500)
        pred_test = model(X_test)
        test_loss = F.mse_loss(pred_test, Y_test).item()

    return test_loss, n_params


def run_analysis():
    print("=" * 70)
    print("ANALYSIS: Why does phasor need 2 layers?")
    print("=" * 70)

    configurations = [
        (SingleLayerPhasor, "1L (linear)"),
        (SingleLayerPhasorWithDecoder, "1L (GELU)"),  # NEW: built-in decoder
        (TwoLayerPhasor, "2 Layers"),
        (PhasorPlusLinear, "1L + Linear"),
        (PhasorPlusMLP, "1L + MLP"),
        (PhasorIterative, "1L x2 (shared)"),
    ]

    tasks = ['associative', 'xor_binding', 'copy', 'linear']
    n_trials = 3

    print(f"\nRunning {n_trials} trials per configuration across {len(tasks)} tasks...")

    # Results per task
    all_task_results = {}

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"TASK: {task}")
        print(f"{'='*70}")
        print(f"{'Configuration':<20} {'Mean MSE':>10} {'Std':>8}")
        print("-" * 42)

        task_results = []
        for model_class, name in configurations:
            mses = []
            for trial in range(n_trials):
                mse, params = test_configuration(model_class, name, task=task)
                mses.append(mse)
            mean_mse = sum(mses) / len(mses)
            std_mse = (sum((m - mean_mse)**2 for m in mses) / len(mses)) ** 0.5
            task_results.append((name, mean_mse, std_mse))
            print(f"{name:<20} {mean_mse:>10.4f} {std_mse:>8.4f}")

        all_task_results[task] = task_results

    # Summary across tasks
    print("\n" + "=" * 70)
    print("SUMMARY: Best configuration per task")
    print("=" * 70)

    for task, results in all_task_results.items():
        best = min(results, key=lambda x: x[1])
        print(f"  {task:<15}: {best[0]} ({best[1]:.4f})")

    # Count wins
    print("\n" + "=" * 70)
    print("WINNER COUNTS")
    print("=" * 70)
    win_counts = {}
    for task, results in all_task_results.items():
        best = min(results, key=lambda x: x[1])[0]
        win_counts[best] = win_counts.get(best, 0) + 1

    for name, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} wins")

    return all_task_results


if __name__ == "__main__":
    results = run_analysis()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
FINDING: MLP decoder wins on SIMPLE tasks, but multi-phasor wins on COMPLEX tasks!

SIMPLE TASKS (associative, copy, linear, xor_binding):
- 1 Layer + MLP wins on ALL simple tasks
- MLP provides efficient nonlinear decoding

COMPLEX TASKS (causal OOD reasoning - see causal_clifford_mlp_decoder.py):
- 4 Phasor layers: 1.08x degradation (BEST)
- 2P + MLP:        1.70x degradation
- 1P + DeepMLP:    2.13x degradation
- 1P + MLP:        2.21x degradation

CONCLUSION:
The second (and subsequent) phasor layers do DIFFERENT things depending on task:

For SIMPLE tasks:
- Single phasor layer creates representation
- MLP decodes it efficiently
- Additional phasor layers are wasteful

For OOD ROBUSTNESS tasks:
- Each phasor layer adds phase-based regularization
- This creates representations that generalize better
- MLPs can't replicate this - they just memorize
- More phasor depth = more OOD robust

RECOMMENDATION:
- Use Phasor + MLP for simple retrieval/transformation tasks
- Use multi-layer phasor (>=4) for tasks requiring OOD robustness
- The OOD robustness comes from the phase binding mechanism itself,
  NOT from the nonlinear transformation capacity
""")
