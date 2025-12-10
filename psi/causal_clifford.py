"""
Causal Reasoning with ACTUAL Clifford/Phasor Memory

Previous attempt cheated by using standard MLPs. This version properly uses
phasor binding and retrieval from clifford_memory.py for causal mechanisms.

Key insight: Each causal variable gets a learned phase signature.
Causal relationships are stored by binding cause_phasor * effect_phasor * value.
Retrieval uses resonance-based matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PhasorCausalLayer(nn.Module):
    """Single phasor layer for causal transformation."""
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
        """Apply phasor transformation."""
        B, D = x.shape

        # Encode into phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        # Bind and store (no cumsum - each sample independent)
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(1)  # [B, n_planes, D]

        # Retrieve
        retrieved = bound * query_phasor.conj().unsqueeze(-1)  # [B, n_planes, D]

        # Resonance
        phase_alignment = (key_phasor * query_phasor.conj()).real
        resonance = phase_alignment.mean(dim=-1, keepdim=True)
        resonance_gain = F.softplus(resonance + 0.5)

        # Sum and normalize
        summed = retrieved.sum(dim=1).real
        output = summed * resonance_gain / self.n_planes

        return x + self.to_out(output)


class PhasorCausalMechanism(nn.Module):
    """
    Multi-layer phasor-based causal mechanism.

    Uses the same "2 layers" insight from OptimalPhasorModel:
    "Layer 1 retrieves, Layer 2 decodes"
    """
    def __init__(self, dim: int, n_planes: int = 32, n_layers: int = 4):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            PhasorCausalLayer(dim, n_planes) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, cause_value: torch.Tensor) -> torch.Tensor:
        if cause_value.dim() == 1:
            cause_value = cause_value.unsqueeze(0)

        h = cause_value
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class PhasorCausalMemory(nn.Module):
    """
    Full causal memory system using phasor binding.

    Stores observations and causal relationships in a unified phasor memory.
    Supports:
    - Storing variable observations
    - Storing causal links (mechanisms)
    - Querying effects given causes
    - Interventional queries (blocking confounders)
    """
    def __init__(self, dim: int, n_phases: int = 16, max_vars: int = 16):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.max_vars = max_vars

        # Each variable gets a learned phase signature
        self.var_phases = nn.Parameter(torch.randn(max_vars, n_phases) * math.pi)

        # Value encoder/decoder
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        # Mechanism network: transforms cause representation to effect
        self.mechanism = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def get_var_phasor(self, var_idx: int) -> torch.Tensor:
        """Get phasor for a variable."""
        return torch.exp(1j * self.var_phases[var_idx])

    def init_memory(self, batch_size: int) -> torch.Tensor:
        """Initialize empty memory."""
        return torch.zeros(batch_size, self.n_phases, self.dim,
                          dtype=torch.complex64, device=device)

    def store_observation(self, memory: torch.Tensor, var_idx: int,
                         value: torch.Tensor) -> torch.Tensor:
        """Store an observation: variable = value."""
        phasor = self.get_var_phasor(var_idx)  # [n_phases]
        value_enc = self.encoder(value).to(torch.complex64)  # [B, dim]

        # Bind: phasor * value
        binding = phasor.unsqueeze(0).unsqueeze(-1) * value_enc.unsqueeze(1)  # [B, n_phases, dim]

        return memory + binding

    def store_causal_link(self, memory: torch.Tensor,
                          cause_idx: int, effect_idx: int,
                          cause_value: torch.Tensor) -> torch.Tensor:
        """
        Store a causal relationship: cause -> effect.

        Uses joint address (cause * effect phasors) to store the mechanism output.
        """
        cause_phasor = self.get_var_phasor(cause_idx)
        effect_phasor = self.get_var_phasor(effect_idx)

        # Joint address
        joint_phasor = cause_phasor * effect_phasor  # [n_phases]

        # Compute mechanism output
        mechanism_out = self.mechanism(cause_value).to(torch.complex64)  # [B, dim]

        # Bind to joint address
        binding = joint_phasor.unsqueeze(0).unsqueeze(-1) * mechanism_out.unsqueeze(1)

        return memory + binding

    def retrieve(self, memory: torch.Tensor, query_phasor: torch.Tensor) -> torch.Tensor:
        """Retrieve from memory using query phasor."""
        # Unbind with conjugate
        retrieved = memory * query_phasor.conj().unsqueeze(0).unsqueeze(-1)
        # Sum across phases
        return self.decoder(retrieved.sum(dim=1).real)

    def query_observation(self, memory: torch.Tensor, var_idx: int) -> torch.Tensor:
        """Query the value of a variable."""
        phasor = self.get_var_phasor(var_idx)
        return self.retrieve(memory, phasor)

    def query_causal_effect(self, memory: torch.Tensor,
                            cause_idx: int, effect_idx: int) -> torch.Tensor:
        """Query the causal effect of cause on effect."""
        cause_phasor = self.get_var_phasor(cause_idx)
        effect_phasor = self.get_var_phasor(effect_idx)
        joint_phasor = cause_phasor * effect_phasor
        return self.retrieve(memory, joint_phasor)


class PhasorCausalGraph(nn.Module):
    """
    A causal graph where each mechanism is a PhasorCausalMechanism.

    This replaces the standard MLP-based CausalGraph with phasor-based mechanisms.
    """
    def __init__(self, var_names: List[str], edges: List[Tuple[str, str]],
                 dim: int = 32, n_phases: int = 16):
        super().__init__()
        self.var_names = var_names
        self.var_to_idx = {name: i for i, name in enumerate(var_names)}
        self.edges = edges
        self.dim = dim

        # Memory system
        self.memory = PhasorCausalMemory(dim, n_phases, max_vars=len(var_names))

        # Phasor-based mechanisms for each edge
        self.mechanisms = nn.ModuleDict()
        for cause, effect in edges:
            key = f"{cause}_to_{effect}"
            self.mechanisms[key] = PhasorCausalMechanism(dim, n_phases)

        # Parents lookup
        self.parents = {name: [] for name in var_names}
        for cause, effect in edges:
            self.parents[effect].append(cause)

    def forward_causal(self, exogenous: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through causal graph."""
        batch_size = list(exogenous.values())[0].shape[0]

        values = {}
        memory = self.memory.init_memory(batch_size)

        # Topological order
        for var in self.var_names:
            if var in exogenous:
                values[var] = exogenous[var]
            else:
                # Compute from parents using phasor mechanisms
                var_value = torch.zeros(batch_size, self.dim, device=device)
                for parent in self.parents[var]:
                    key = f"{parent}_to_{var}"
                    contribution = self.mechanisms[key](values[parent])
                    var_value = var_value + contribution
                values[var] = var_value

            # Store in memory
            var_idx = self.var_to_idx[var]
            memory = self.memory.store_observation(memory, var_idx, values[var])

            # Store causal links
            for parent in self.parents[var]:
                parent_idx = self.var_to_idx[parent]
                memory = self.memory.store_causal_link(
                    memory, parent_idx, var_idx, values[parent]
                )

        return values, memory

    def get_mechanism_effect(self, cause_var: str, effect_var: str,
                             cause_value: torch.Tensor) -> torch.Tensor:
        """Get the direct causal effect using the phasor mechanism."""
        key = f"{cause_var}_to_{effect_var}"
        if key in self.mechanisms:
            return self.mechanisms[key](cause_value)
        return torch.zeros_like(cause_value)


# =============================================================================
# Data Generator (same as before)
# =============================================================================

class CausalDataGenerator:
    """Generates data from known SCM: Z -> X -> Y, Z -> Y"""
    def __init__(self, dim=32, noise_scale=0.1):
        self.dim = dim
        self.noise_scale = noise_scale
        self.W_zx = torch.randn(dim, dim, device=device) * 0.5
        self.W_xy = torch.randn(dim, dim, device=device) * 0.3
        self.W_zy = torch.randn(dim, dim, device=device) * 0.3

    def sample_observational(self, n: int) -> Dict[str, torch.Tensor]:
        Z = torch.randn(n, self.dim, device=device)
        X = Z @ self.W_zx.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        Y = X @ self.W_xy.T + Z @ self.W_zy.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        return {'Z': Z, 'X': X, 'Y': Y}

    def sample_interventional(self, n: int) -> Dict[str, torch.Tensor]:
        Z = torch.randn(n, self.dim, device=device)
        X = torch.randn(n, self.dim, device=device) * 2  # Different distribution!
        Y = X @ self.W_xy.T + Z @ self.W_zy.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        return {'Z': Z, 'X': X, 'Y': Y}

    def true_causal_effect(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_xy.T


# =============================================================================
# Baselines
# =============================================================================

class MLPBaseline(nn.Module):
    """Standard MLP baseline for comparison."""
    def __init__(self, dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


class MLPCausalModel(nn.Module):
    """MLP-based causal model (the one I cheated with before)."""
    def __init__(self, dim=32):
        super().__init__()
        self.mech_zx = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
        self.mech_zy = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
        self.mech_xy = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))

    def forward(self, x, z):
        return self.mech_xy(x) + self.mech_zy(z)

    def get_causal_effect(self, x):
        return self.mech_xy(x)


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark():
    print("=" * 70)
    print("PHASOR CAUSAL MODEL vs MLP CAUSAL MODEL vs CORRELATIONAL")
    print("=" * 70)

    dim = 32
    n_phases = 16
    generator = CausalDataGenerator(dim=dim)

    # Models
    phasor_model = PhasorCausalGraph(
        var_names=['Z', 'X', 'Y'],
        edges=[('Z', 'X'), ('Z', 'Y'), ('X', 'Y')],
        dim=dim,
        n_phases=n_phases
    ).to(device)

    mlp_causal = MLPCausalModel(dim=dim).to(device)
    correlational = MLPBaseline(dim=dim).to(device)

    opt_phasor = torch.optim.Adam(phasor_model.parameters(), lr=1e-3)
    opt_mlp = torch.optim.Adam(mlp_causal.parameters(), lr=1e-3)
    opt_corr = torch.optim.Adam(correlational.parameters(), lr=1e-3)

    # Count parameters
    phasor_params = sum(p.numel() for p in phasor_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_causal.parameters())
    corr_params = sum(p.numel() for p in correlational.parameters())

    print(f"\nParameters:")
    print(f"  Phasor Causal: {phasor_params:,}")
    print(f"  MLP Causal:    {mlp_params:,}")
    print(f"  Correlational: {corr_params:,}")

    print("\n--- Training on MIXED data (80% obs, 20% interventional) ---")

    for epoch in range(1000):
        # Mixed data
        data_obs = generator.sample_observational(102)
        data_int = generator.sample_interventional(26)

        X = torch.cat([data_obs['X'], data_int['X']])
        Y = torch.cat([data_obs['Y'], data_int['Y']])
        Z = torch.cat([data_obs['Z'], data_int['Z']])

        # Phasor causal model
        values, memory = phasor_model.forward_causal({'Z': Z})
        loss_phasor = F.mse_loss(values['X'], X) + F.mse_loss(values['Y'], Y)
        opt_phasor.zero_grad()
        loss_phasor.backward()
        opt_phasor.step()

        # MLP causal model
        pred_mlp = mlp_causal(X, Z)
        loss_mlp = F.mse_loss(pred_mlp, Y)
        opt_mlp.zero_grad()
        loss_mlp.backward()
        opt_mlp.step()

        # Correlational
        pred_corr = correlational(X)
        loss_corr = F.mse_loss(pred_corr, Y)
        opt_corr.zero_grad()
        loss_corr.backward()
        opt_corr.step()

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Phasor={loss_phasor.item():.4f}, "
                  f"MLP={loss_mlp.item():.4f}, Corr={loss_corr.item():.4f}")

    # Evaluation
    print("\n--- Evaluation ---")

    with torch.no_grad():
        # In-distribution
        test_obs = generator.sample_observational(500)
        X_obs, Y_obs, Z_obs = test_obs['X'], test_obs['Y'], test_obs['Z']

        values_obs, _ = phasor_model.forward_causal({'Z': Z_obs})
        phasor_obs_mse = F.mse_loss(values_obs['Y'], Y_obs).item()
        mlp_obs_mse = F.mse_loss(mlp_causal(X_obs, Z_obs), Y_obs).item()
        corr_obs_mse = F.mse_loss(correlational(X_obs), Y_obs).item()

        print(f"\n1. In-Distribution (Observational):")
        print(f"   Phasor MSE:       {phasor_obs_mse:.4f}")
        print(f"   MLP Causal MSE:   {mlp_obs_mse:.4f}")
        print(f"   Correlational MSE: {corr_obs_mse:.4f}")

        # Out-of-distribution
        test_int = generator.sample_interventional(500)
        X_int, Y_int, Z_int = test_int['X'], test_int['Y'], test_int['Z']

        # For phasor: need to handle intervention properly
        # Store Z, then query Y given intervened X
        values_int, _ = phasor_model.forward_causal({'Z': Z_int})
        # Add the X->Y effect for intervened X
        y_phasor_int = phasor_model.get_mechanism_effect('X', 'Y', X_int) + \
                       phasor_model.get_mechanism_effect('Z', 'Y', Z_int)

        phasor_int_mse = F.mse_loss(y_phasor_int, Y_int).item()
        mlp_int_mse = F.mse_loss(mlp_causal(X_int, Z_int), Y_int).item()
        corr_int_mse = F.mse_loss(correlational(X_int), Y_int).item()

        print(f"\n2. Out-of-Distribution (Interventional):")
        print(f"   Phasor MSE:       {phasor_int_mse:.4f}")
        print(f"   MLP Causal MSE:   {mlp_int_mse:.4f}")
        print(f"   Correlational MSE: {corr_int_mse:.4f}")

        # Degradation
        print(f"\n3. Degradation (OOD/ID ratio):")
        print(f"   Phasor:       {phasor_int_mse/phasor_obs_mse:.2f}x")
        print(f"   MLP Causal:   {mlp_int_mse/mlp_obs_mse:.2f}x")
        print(f"   Correlational: {corr_int_mse/corr_obs_mse:.2f}x")

        # THE KEY: Causal effect estimation
        print(f"\n4. CAUSAL EFFECT ESTIMATION:")
        x_high = torch.ones(100, dim, device=device) * 2
        x_low = torch.ones(100, dim, device=device) * -2

        true_effect = generator.true_causal_effect(x_high - x_low).mean().item()

        phasor_high = phasor_model.get_mechanism_effect('X', 'Y', x_high).mean().item()
        phasor_low = phasor_model.get_mechanism_effect('X', 'Y', x_low).mean().item()
        phasor_effect = phasor_high - phasor_low

        mlp_high = mlp_causal.get_causal_effect(x_high).mean().item()
        mlp_low = mlp_causal.get_causal_effect(x_low).mean().item()
        mlp_effect = mlp_high - mlp_low

        corr_high = correlational(x_high).mean().item()
        corr_low = correlational(x_low).mean().item()
        corr_effect = corr_high - corr_low

        print(f"   True causal effect: {true_effect:.4f}")
        print(f"   Phasor estimate:    {phasor_effect:.4f} (error: {abs(phasor_effect - true_effect):.4f})")
        print(f"   MLP estimate:       {mlp_effect:.4f} (error: {abs(mlp_effect - true_effect):.4f})")
        print(f"   Corr estimate:      {corr_effect:.4f} (error: {abs(corr_effect - true_effect):.4f})")


def run_multiple_trials(n_trials=5):
    """Run multiple trials to get stable results."""
    print("=" * 70)
    print(f"RUNNING {n_trials} TRIALS FOR STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    results = {
        'phasor_deg': [], 'mlp_deg': [], 'corr_deg': [],
        'phasor_effect_err': [], 'mlp_effect_err': [], 'corr_effect_err': []
    }

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        torch.manual_seed(trial * 42)

        dim = 32
        generator = CausalDataGenerator(dim=dim)

        phasor_model = PhasorCausalGraph(
            var_names=['Z', 'X', 'Y'],
            edges=[('Z', 'X'), ('Z', 'Y'), ('X', 'Y')],
            dim=dim, n_phases=16
        ).to(device)
        mlp_causal = MLPCausalModel(dim=dim).to(device)
        correlational = MLPBaseline(dim=dim).to(device)

        opt_phasor = torch.optim.Adam(phasor_model.parameters(), lr=1e-3)
        opt_mlp = torch.optim.Adam(mlp_causal.parameters(), lr=1e-3)
        opt_corr = torch.optim.Adam(correlational.parameters(), lr=1e-3)

        # Train
        for epoch in range(500):
            data_obs = generator.sample_observational(102)
            data_int = generator.sample_interventional(26)
            X = torch.cat([data_obs['X'], data_int['X']])
            Y = torch.cat([data_obs['Y'], data_int['Y']])
            Z = torch.cat([data_obs['Z'], data_int['Z']])

            values, _ = phasor_model.forward_causal({'Z': Z})
            loss_phasor = F.mse_loss(values['X'], X) + F.mse_loss(values['Y'], Y)
            opt_phasor.zero_grad(); loss_phasor.backward(); opt_phasor.step()

            loss_mlp = F.mse_loss(mlp_causal(X, Z), Y)
            opt_mlp.zero_grad(); loss_mlp.backward(); opt_mlp.step()

            loss_corr = F.mse_loss(correlational(X), Y)
            opt_corr.zero_grad(); loss_corr.backward(); opt_corr.step()

        # Evaluate
        with torch.no_grad():
            test_obs = generator.sample_observational(500)
            test_int = generator.sample_interventional(500)

            # ID
            values_obs, _ = phasor_model.forward_causal({'Z': test_obs['Z']})
            phasor_id = F.mse_loss(values_obs['Y'], test_obs['Y']).item()
            mlp_id = F.mse_loss(mlp_causal(test_obs['X'], test_obs['Z']), test_obs['Y']).item()
            corr_id = F.mse_loss(correlational(test_obs['X']), test_obs['Y']).item()

            # OOD
            y_phasor_int = phasor_model.get_mechanism_effect('X', 'Y', test_int['X']) + \
                           phasor_model.get_mechanism_effect('Z', 'Y', test_int['Z'])
            phasor_ood = F.mse_loss(y_phasor_int, test_int['Y']).item()
            mlp_ood = F.mse_loss(mlp_causal(test_int['X'], test_int['Z']), test_int['Y']).item()
            corr_ood = F.mse_loss(correlational(test_int['X']), test_int['Y']).item()

            results['phasor_deg'].append(phasor_ood / phasor_id)
            results['mlp_deg'].append(mlp_ood / mlp_id)
            results['corr_deg'].append(corr_ood / corr_id)

            # Causal effect
            x_high = torch.ones(100, dim, device=device) * 2
            x_low = torch.ones(100, dim, device=device) * -2
            true_effect = generator.true_causal_effect(x_high - x_low).mean().item()

            phasor_effect = phasor_model.get_mechanism_effect('X', 'Y', x_high).mean().item() - \
                           phasor_model.get_mechanism_effect('X', 'Y', x_low).mean().item()
            mlp_effect = mlp_causal.get_causal_effect(x_high).mean().item() - \
                        mlp_causal.get_causal_effect(x_low).mean().item()
            corr_effect = correlational(x_high).mean().item() - correlational(x_low).mean().item()

            results['phasor_effect_err'].append(abs(phasor_effect - true_effect))
            results['mlp_effect_err'].append(abs(mlp_effect - true_effect))
            results['corr_effect_err'].append(abs(corr_effect - true_effect))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean +/- std over {} trials)".format(n_trials))
    print("=" * 70)

    import numpy as np

    print("\nDegradation (OOD/ID ratio, lower = more robust):")
    print(f"  Phasor:       {np.mean(results['phasor_deg']):.2f} +/- {np.std(results['phasor_deg']):.2f}")
    print(f"  MLP Causal:   {np.mean(results['mlp_deg']):.2f} +/- {np.std(results['mlp_deg']):.2f}")
    print(f"  Correlational: {np.mean(results['corr_deg']):.2f} +/- {np.std(results['corr_deg']):.2f}")

    print("\nCausal Effect Error (lower = more accurate):")
    print(f"  Phasor:       {np.mean(results['phasor_effect_err']):.4f} +/- {np.std(results['phasor_effect_err']):.4f}")
    print(f"  MLP Causal:   {np.mean(results['mlp_effect_err']):.4f} +/- {np.std(results['mlp_effect_err']):.4f}")
    print(f"  Correlational: {np.mean(results['corr_effect_err']):.4f} +/- {np.std(results['corr_effect_err']):.4f}")


if __name__ == "__main__":
    run_benchmark()
    print("\n")
    run_multiple_trials(5)
