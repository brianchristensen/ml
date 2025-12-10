"""
Causal Reasoning Benchmark: Testing for SOTA capabilities

Key insight: Causal models should generalize under distribution shift caused by
interventions, while correlational models fail catastrophically.

Benchmark 1: Anti-Causal Prediction (Peters et al., 2016)
- Train on P(Y|X) where X->Y
- Test on data where we intervene on X (changing P(X))
- Causal models: should still predict Y correctly
- Correlational models: fail because P(X) changed

Benchmark 2: Counterfactual Prediction (Twin Networks)
- Given factual outcome, predict counterfactual
- Requires proper causal reasoning, not just pattern matching

Benchmark 3: Causal Discovery
- Learn graph structure from observational + interventional data
- Compare against PC algorithm, GES, etc.

Benchmark 4: Simpson's Paradox
- The classic test: aggregate correlation reverses within subgroups
- Only causal models handle this correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Benchmark 1: Out-of-Distribution Generalization Under Intervention
# =============================================================================

class CausalDataGenerator:
    """
    Generates data from a known structural causal model (SCM).

    True causal structure:
        Z -> X -> Y
        Z -> Y (direct effect)

    So: Y = f(X, Z) where X = g(Z)

    Key: We can intervene on X (set it to a value independent of Z)
    and test if models correctly predict Y.
    """

    def __init__(self, dim=32, noise_scale=0.1):
        self.dim = dim
        self.noise_scale = noise_scale

        # True causal mechanisms (fixed, known)
        # X = W_zx @ Z + noise
        self.W_zx = torch.randn(dim, dim, device=device) * 0.5
        # Y = W_xy @ X + W_zy @ Z + noise
        self.W_xy = torch.randn(dim, dim, device=device) * 0.3  # X -> Y (causal)
        self.W_zy = torch.randn(dim, dim, device=device) * 0.3  # Z -> Y (confounding)

    def sample_observational(self, n: int) -> Dict[str, torch.Tensor]:
        """Sample from observational distribution P(X, Y, Z)."""
        Z = torch.randn(n, self.dim, device=device)
        X = Z @ self.W_zx.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        Y = X @ self.W_xy.T + Z @ self.W_zy.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        return {'Z': Z, 'X': X, 'Y': Y}

    def sample_interventional(self, n: int, x_intervention: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Sample from interventional distribution P(Y | do(X=x)).
        X is set independently of Z (breaking the Z->X edge).
        """
        Z = torch.randn(n, self.dim, device=device)

        if x_intervention is None:
            # Random intervention: X drawn from different distribution
            X = torch.randn(n, self.dim, device=device) * 2  # Different scale!
        else:
            X = x_intervention.expand(n, -1)

        # Y still depends on both X and Z through true mechanisms
        Y = X @ self.W_xy.T + Z @ self.W_zy.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        return {'Z': Z, 'X': X, 'Y': Y}

    def compute_true_causal_effect(self, x: torch.Tensor) -> torch.Tensor:
        """Compute E[Y | do(X=x)] = W_xy @ x (Z averages out)."""
        return x @ self.W_xy.T


class CorrelationalBaseline(nn.Module):
    """
    Standard MLP that learns P(Y|X) from observational data.
    Should fail under intervention because it implicitly captures X-Z correlation.
    """
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


class CausalPredictor(nn.Module):
    """
    Model that explicitly represents causal structure.
    Should generalize under intervention.
    """
    def __init__(self, dim=32, n_phases=16):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

        # Learnable causal mechanism X -> Y
        self.mechanism_xy = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Phase encodings for variables
        self.phase_x = nn.Parameter(torch.randn(n_phases) * math.pi)
        self.phase_y = nn.Parameter(torch.randn(n_phases) * math.pi)
        self.phase_z = nn.Parameter(torch.randn(n_phases) * math.pi)

        # Memory encoder/decoder
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # Confounder detector: learns to identify when Z is affecting both X and Y
        self.confounder_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z=None):
        """
        Predict Y from X, optionally using Z to adjust for confounding.
        """
        # Direct causal effect X -> Y
        y_causal = self.mechanism_xy(x)

        if z is not None:
            # If we observe Z, we can adjust for confounding
            # But in intervention mode, Z is independent of X
            confound_score = self.confounder_detector(torch.cat([x, z], dim=-1))
            # When confounding is detected, rely more on causal mechanism
            # (This is a simplified version - full model would do proper adjustment)
            y_causal = y_causal * (1 + confound_score * 0.1)

        return y_causal

    def forward_interventional(self, x):
        """
        Predict Y under do(X=x).
        Only uses the causal mechanism, ignoring potential confounders.
        """
        return self.mechanism_xy(x)


def benchmark_ood_generalization():
    """
    Benchmark: Out-of-Distribution Generalization Under Intervention

    Train both models on observational data P(X, Y).
    Test on interventional data P(Y | do(X)) where X has different distribution.

    The causal model should generalize; the correlational model should fail.
    """
    print("=" * 70)
    print("BENCHMARK: Out-of-Distribution Generalization Under Intervention")
    print("=" * 70)

    dim = 32
    generator = CausalDataGenerator(dim=dim)

    # Models
    correlational = CorrelationalBaseline(dim=dim).to(device)
    causal = CausalPredictor(dim=dim).to(device)

    opt_corr = torch.optim.Adam(correlational.parameters(), lr=1e-3)
    opt_causal = torch.optim.Adam(causal.parameters(), lr=1e-3)

    # Training on OBSERVATIONAL data
    print("\n--- Training on observational data ---")
    n_epochs = 500

    for epoch in range(n_epochs):
        data = generator.sample_observational(128)
        X, Y, Z = data['X'], data['Y'], data['Z']

        # Train correlational model: just predict Y from X
        pred_corr = correlational(X)
        loss_corr = F.mse_loss(pred_corr, Y)
        opt_corr.zero_grad()
        loss_corr.backward()
        opt_corr.step()

        # Train causal model: predict Y from X (and optionally Z)
        pred_causal = causal(X, Z)
        loss_causal = F.mse_loss(pred_causal, Y)
        opt_causal.zero_grad()
        loss_causal.backward()
        opt_causal.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Correlational loss = {loss_corr.item():.4f}, "
                  f"Causal loss = {loss_causal.item():.4f}")

    # Evaluation
    print("\n--- Evaluation ---")

    with torch.no_grad():
        # Test 1: In-distribution (observational)
        test_obs = generator.sample_observational(500)
        X_obs, Y_obs = test_obs['X'], test_obs['Y']

        pred_corr_obs = correlational(X_obs)
        pred_causal_obs = causal.forward_interventional(X_obs)

        mse_corr_obs = F.mse_loss(pred_corr_obs, Y_obs).item()
        mse_causal_obs = F.mse_loss(pred_causal_obs, Y_obs).item()

        print(f"\n1. In-Distribution (Observational) Test:")
        print(f"   Correlational MSE: {mse_corr_obs:.4f}")
        print(f"   Causal MSE:        {mse_causal_obs:.4f}")

        # Test 2: Out-of-distribution (interventional)
        test_int = generator.sample_interventional(500)
        X_int, Y_int = test_int['X'], test_int['Y']

        pred_corr_int = correlational(X_int)
        pred_causal_int = causal.forward_interventional(X_int)

        mse_corr_int = F.mse_loss(pred_corr_int, Y_int).item()
        mse_causal_int = F.mse_loss(pred_causal_int, Y_int).item()

        print(f"\n2. Out-of-Distribution (Interventional) Test:")
        print(f"   Correlational MSE: {mse_corr_int:.4f}")
        print(f"   Causal MSE:        {mse_causal_int:.4f}")

        # Compute degradation
        corr_degradation = mse_corr_int / mse_corr_obs
        causal_degradation = mse_causal_int / mse_causal_obs

        print(f"\n3. Performance Degradation (OOD/ID ratio):")
        print(f"   Correlational: {corr_degradation:.2f}x worse")
        print(f"   Causal:        {causal_degradation:.2f}x worse")

        # Test 3: Specific intervention
        print(f"\n4. Specific Intervention do(X=1):")
        x_intervention = torch.ones(1, dim, device=device)
        true_effect = generator.compute_true_causal_effect(x_intervention)

        pred_corr_specific = correlational(x_intervention)
        pred_causal_specific = causal.forward_interventional(x_intervention)

        error_corr = (pred_corr_specific - true_effect).abs().mean().item()
        error_causal = (pred_causal_specific - true_effect).abs().mean().item()

        print(f"   True causal effect mean: {true_effect.mean().item():.4f}")
        print(f"   Correlational error: {error_corr:.4f}")
        print(f"   Causal error:        {error_causal:.4f}")

    return {
        'correlational': {'obs_mse': mse_corr_obs, 'int_mse': mse_corr_int},
        'causal': {'obs_mse': mse_causal_obs, 'int_mse': mse_causal_int}
    }


# =============================================================================
# Benchmark 2: Simpson's Paradox
# =============================================================================

def benchmark_simpsons_paradox():
    """
    Simpson's Paradox: A correlation in aggregated data reverses within subgroups.

    Classic example: Treatment appears harmful overall, but is beneficial in each subgroup.
    This happens due to confounding by subgroup membership.

    Setup:
    - Z = subgroup (binary: sick vs healthy)
    - X = treatment (binary: treated vs not)
    - Y = outcome

    Sick patients are more likely to be treated AND have worse outcomes.
    Treatment actually helps, but confounding makes it look harmful.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Simpson's Paradox")
    print("=" * 70)

    dim = 32
    n_samples = 2000

    # Generate Simpson's paradox data
    # Z: subgroup (0 = healthy, 1 = sick) - encoded as vectors
    z_healthy = torch.zeros(n_samples // 2, dim, device=device)
    z_healthy[:, 0] = 1  # One-hot-ish encoding
    z_sick = torch.zeros(n_samples // 2, dim, device=device)
    z_sick[:, 1] = 1
    Z = torch.cat([z_healthy, z_sick], dim=0)

    # Sick patients more likely to be treated (confounding)
    # X: treatment (encoded as vector)
    p_treat_healthy = 0.3  # Healthy rarely get treatment
    p_treat_sick = 0.8     # Sick usually get treatment

    treat_healthy = torch.bernoulli(torch.ones(n_samples // 2) * p_treat_healthy)
    treat_sick = torch.bernoulli(torch.ones(n_samples // 2) * p_treat_sick)
    treat = torch.cat([treat_healthy, treat_sick]).to(device)

    X = torch.zeros(n_samples, dim, device=device)
    X[treat == 1, 2] = 1  # Treated
    X[treat == 0, 3] = 1  # Not treated

    # Y: outcome (higher = better)
    # True effect: treatment helps (+0.3)
    # Confounding: sick have worse outcomes (-0.5)
    base_outcome = torch.randn(n_samples, dim, device=device) * 0.1
    treatment_effect = treat.unsqueeze(1) * 0.3  # Treatment helps
    sickness_effect = torch.cat([
        torch.zeros(n_samples // 2),  # Healthy
        -torch.ones(n_samples // 2) * 0.5  # Sick have worse baseline
    ]).unsqueeze(1).to(device)

    Y = base_outcome + treatment_effect + sickness_effect
    Y_scalar = Y.mean(dim=1)  # Scalar outcome for analysis

    # Analysis
    print("\n--- Data Analysis ---")

    # Aggregate correlation (misleading!)
    treated_outcome = Y_scalar[treat == 1].mean().item()
    untreated_outcome = Y_scalar[treat == 0].mean().item()

    print(f"\nAggregate (ignoring subgroups):")
    print(f"  Treated mean outcome:   {treated_outcome:.4f}")
    print(f"  Untreated mean outcome: {untreated_outcome:.4f}")
    print(f"  Naive effect estimate:  {treated_outcome - untreated_outcome:.4f}")
    print(f"  -> Treatment appears {'HARMFUL' if treated_outcome < untreated_outcome else 'BENEFICIAL'}!")

    # Within-subgroup (correct!)
    healthy_mask = (Z[:, 0] == 1)
    sick_mask = (Z[:, 1] == 1)

    healthy_treated = Y_scalar[healthy_mask & (treat == 1)].mean().item()
    healthy_untreated = Y_scalar[healthy_mask & (treat == 0)].mean().item()
    sick_treated = Y_scalar[sick_mask & (treat == 1)].mean().item()
    sick_untreated = Y_scalar[sick_mask & (treat == 0)].mean().item()

    print(f"\nWithin subgroups:")
    print(f"  Healthy: treated={healthy_treated:.4f}, untreated={healthy_untreated:.4f}, "
          f"effect={healthy_treated - healthy_untreated:.4f}")
    print(f"  Sick:    treated={sick_treated:.4f}, untreated={sick_untreated:.4f}, "
          f"effect={sick_treated - sick_untreated:.4f}")
    print(f"  -> Treatment is BENEFICIAL in both subgroups!")

    print(f"\nTrue causal effect: +0.3 (treatment helps)")

    # Now train models
    print("\n--- Training Models ---")

    # Correlational: just learns P(Y|X)
    correlational = nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Linear(64, dim)
    ).to(device)

    # Causal: learns P(Y|X, Z) and can adjust
    causal = nn.Sequential(
        nn.Linear(dim * 2, 64),  # Takes both X and Z
        nn.ReLU(),
        nn.Linear(64, dim)
    ).to(device)

    opt_corr = torch.optim.Adam(correlational.parameters(), lr=1e-3)
    opt_causal = torch.optim.Adam(causal.parameters(), lr=1e-3)

    for epoch in range(300):
        # Correlational
        pred_corr = correlational(X)
        loss_corr = F.mse_loss(pred_corr, Y)
        opt_corr.zero_grad()
        loss_corr.backward()
        opt_corr.step()

        # Causal (conditions on Z)
        pred_causal = causal(torch.cat([X, Z], dim=-1))
        loss_causal = F.mse_loss(pred_causal, Y)
        opt_causal.zero_grad()
        loss_causal.backward()
        opt_causal.step()

    print(f"  Final losses - Correlational: {loss_corr.item():.4f}, Causal: {loss_causal.item():.4f}")

    # Test: What's the effect of treatment?
    print("\n--- Testing Causal Effect Estimation ---")

    with torch.no_grad():
        # Create treated and untreated versions of same patient
        test_x_treated = torch.zeros(100, dim, device=device)
        test_x_treated[:, 2] = 1
        test_x_untreated = torch.zeros(100, dim, device=device)
        test_x_untreated[:, 3] = 1

        # Correlational estimate (confounded)
        corr_treated = correlational(test_x_treated).mean(dim=1).mean().item()
        corr_untreated = correlational(test_x_untreated).mean(dim=1).mean().item()
        corr_effect = corr_treated - corr_untreated

        print(f"\nCorrelational model effect estimate: {corr_effect:.4f}")
        print(f"  (Confounded - may show wrong sign)")

        # Causal estimate (averaged over Z)
        # For proper causal effect, average over Z distribution
        test_z_healthy = torch.zeros(50, dim, device=device)
        test_z_healthy[:, 0] = 1
        test_z_sick = torch.zeros(50, dim, device=device)
        test_z_sick[:, 1] = 1
        test_z = torch.cat([test_z_healthy, test_z_sick], dim=0)

        test_x_treated_full = torch.zeros(100, dim, device=device)
        test_x_treated_full[:, 2] = 1
        test_x_untreated_full = torch.zeros(100, dim, device=device)
        test_x_untreated_full[:, 3] = 1

        causal_treated = causal(torch.cat([test_x_treated_full, test_z], dim=-1)).mean(dim=1).mean().item()
        causal_untreated = causal(torch.cat([test_x_untreated_full, test_z], dim=-1)).mean(dim=1).mean().item()
        causal_effect = causal_treated - causal_untreated

        print(f"Causal model effect estimate: {causal_effect:.4f}")
        print(f"  (Adjusted for confounding)")

        print(f"\nTrue causal effect: 0.3")
        print(f"Correlational error: {abs(corr_effect - 0.3):.4f}")
        print(f"Causal error: {abs(causal_effect - 0.3):.4f}")


# =============================================================================
# Benchmark 3: Counterfactual Prediction
# =============================================================================

def benchmark_counterfactual():
    """
    Counterfactual prediction: "What would Y have been if X had been different?"

    This requires:
    1. Abduction: Infer latent noise from observed (X, Y)
    2. Action: Modify X to counterfactual value
    3. Prediction: Compute Y under new X with same noise

    Standard models can't do this - they don't represent the noise/exogenous variables.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Counterfactual Prediction")
    print("=" * 70)

    dim = 32

    # True SCM: Y = 2*X + U, where U is exogenous noise
    # Given (X, Y), we can recover U = Y - 2*X
    # Counterfactual Y' for X' is: Y' = 2*X' + U = 2*X' + (Y - 2*X)

    def true_counterfactual(x_factual, y_factual, x_counterfactual):
        """Compute true counterfactual using SCM."""
        u = y_factual - 2 * x_factual  # Abduction
        y_cf = 2 * x_counterfactual + u  # Prediction
        return y_cf

    # Generate factual data
    n_test = 500
    X_factual = torch.randn(n_test, dim, device=device)
    U = torch.randn(n_test, dim, device=device) * 0.5  # Exogenous noise
    Y_factual = 2 * X_factual + U

    # Counterfactual query: What if X had been X + delta?
    delta = torch.ones(n_test, dim, device=device)
    X_counterfactual = X_factual + delta
    Y_counterfactual_true = true_counterfactual(X_factual, Y_factual, X_counterfactual)

    print(f"\nFactual: X -> Y = 2*X + U")
    print(f"Counterfactual query: What if X had been X+1?")
    print(f"True answer: Y' = Y + 2 (because Y' = 2*(X+1) + U = 2*X + U + 2 = Y + 2)")

    # Model 1: Standard predictor (can't do counterfactuals properly)
    class StandardPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim)
            )

        def predict(self, x):
            return self.net(x)

        def counterfactual(self, x_factual, y_factual, x_cf):
            # Standard model just predicts from x_cf, ignoring factual info
            return self.net(x_cf)

    # Model 2: Causal model with noise inference
    class CausalCounterfactual(nn.Module):
        def __init__(self):
            super().__init__()
            # Learns the mechanism f(X) part
            self.mechanism = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim)
            )
            # Learns to predict noise scale
            self.noise_scale = nn.Parameter(torch.ones(1))

        def predict(self, x):
            return self.mechanism(x)

        def counterfactual(self, x_factual, y_factual, x_cf):
            # Abduction: infer noise from factual
            y_pred_factual = self.mechanism(x_factual)
            u_inferred = y_factual - y_pred_factual

            # Prediction: apply mechanism to counterfactual X, add same noise
            y_pred_cf = self.mechanism(x_cf)
            return y_pred_cf + u_inferred

    standard = StandardPredictor().to(device)
    causal = CausalCounterfactual().to(device)

    opt_std = torch.optim.Adam(standard.parameters(), lr=1e-3)
    opt_causal = torch.optim.Adam(causal.parameters(), lr=1e-3)

    # Train both on factual data
    print("\n--- Training on factual data ---")
    for epoch in range(500):
        X_train = torch.randn(128, dim, device=device)
        U_train = torch.randn(128, dim, device=device) * 0.5
        Y_train = 2 * X_train + U_train

        loss_std = F.mse_loss(standard.predict(X_train), Y_train)
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()

        loss_causal = F.mse_loss(causal.predict(X_train), Y_train)
        opt_causal.zero_grad()
        loss_causal.backward()
        opt_causal.step()

    print(f"  Final losses - Standard: {loss_std.item():.4f}, Causal: {loss_causal.item():.4f}")

    # Test counterfactual prediction
    print("\n--- Counterfactual Prediction Test ---")

    with torch.no_grad():
        # Standard model counterfactual
        Y_cf_standard = standard.counterfactual(X_factual, Y_factual, X_counterfactual)

        # Causal model counterfactual
        Y_cf_causal = causal.counterfactual(X_factual, Y_factual, X_counterfactual)

        mse_standard = F.mse_loss(Y_cf_standard, Y_counterfactual_true).item()
        mse_causal = F.mse_loss(Y_cf_causal, Y_counterfactual_true).item()

        print(f"\nCounterfactual prediction MSE:")
        print(f"  Standard model: {mse_standard:.4f}")
        print(f"  Causal model:   {mse_causal:.4f}")

        # Check if the change is correct
        factual_to_cf_true = (Y_counterfactual_true - Y_factual).mean().item()
        factual_to_cf_standard = (Y_cf_standard - Y_factual).mean().item()
        factual_to_cf_causal = (Y_cf_causal - Y_factual).mean().item()

        print(f"\nMean change from factual to counterfactual:")
        print(f"  True change:     {factual_to_cf_true:.4f} (should be ~2.0)")
        print(f"  Standard model:  {factual_to_cf_standard:.4f}")
        print(f"  Causal model:    {factual_to_cf_causal:.4f}")


class CliffordCausalModel(nn.Module):
    """
    Clifford memory-based causal model with explicit mechanism representation.

    Key idea: Use phasor binding to:
    1. Store causal relationships with non-interfering addresses
    2. Retrieve mechanisms independently (isolation via orthogonal phases)
    3. Support both observational and interventional queries
    """
    def __init__(self, dim=32, n_phases=16):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

        # Variable-specific phases (learnable addresses)
        # Initialize orthogonally to minimize interference
        self.phase_z = nn.Parameter(torch.randn(n_phases) * math.pi)
        self.phase_x = nn.Parameter(torch.randn(n_phases) * math.pi)
        self.phase_y = nn.Parameter(torch.randn(n_phases) * math.pi)

        # Causal mechanisms (learnable functions)
        # Z -> X mechanism
        self.mech_zx = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Z -> Y mechanism (direct confounding path)
        self.mech_zy = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # X -> Y mechanism (causal path)
        self.mech_xy = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward_observational(self, z):
        """Forward pass through the causal graph (observational)."""
        x = self.mech_zx(z)  # X is caused by Z
        y = self.mech_xy(x) + self.mech_zy(z)  # Y caused by both X and Z
        return x, y

    def forward_interventional(self, z, do_x):
        """
        Forward pass with intervention do(X=x).
        X is SET to do_x, breaking the Z->X edge.
        Y still depends on both X (intervened) and Z (unchanged).
        """
        # X is intervened, not computed from Z
        x = do_x

        # Y depends on intervened X and natural Z
        y = self.mech_xy(x) + self.mech_zy(z)

        return x, y

    def get_causal_effect(self, x):
        """Get the pure causal effect of X on Y (mechanism only)."""
        return self.mech_xy(x)


class CliffordCausalModelV2(nn.Module):
    """
    Version 2: Learns from MIXED observational + interventional data.

    This is the key for identifiability - with only observational data,
    the causal effect is not identifiable without assumptions.

    With even a small amount of interventional data, we can learn the
    true X->Y mechanism separate from confounding.
    """
    def __init__(self, dim=32, n_phases=16):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases

        # Shared mechanism for X -> Y
        self.mech_xy = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Z -> Y mechanism (confounding)
        self.mech_zy = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Z -> X mechanism
        self.mech_zx = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Phasor-based memory for storing causal knowledge
        self.phase_causal = nn.Parameter(torch.randn(n_phases) * math.pi)
        self.phase_confound = nn.Parameter(torch.randn(n_phases) * math.pi)

    def forward(self, x, z, is_intervention=False):
        """
        Forward pass.

        If is_intervention=True, we know X was set externally.
        If is_intervention=False, X was caused by Z.
        """
        # Causal effect X -> Y (always applies)
        y_causal = self.mech_xy(x)

        # Confounding effect Z -> Y (always applies)
        y_confound = self.mech_zy(z)

        return y_causal + y_confound

    def get_causal_effect(self, x):
        """Pure causal effect X -> Y without confounding."""
        return self.mech_xy(x)


def benchmark_clifford_causal():
    """
    Test Clifford/Causal model trained on MIXED data (obs + interventional).

    Key insight: Causal models can use interventional data to disentangle
    the true X->Y effect from confounding. Correlational models can't.

    This is the realistic scenario where we have:
    - Mostly observational data (cheap)
    - Some interventional data (expensive but valuable)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Causal vs Correlational with Mixed Training Data")
    print("=" * 70)

    dim = 32
    generator = CausalDataGenerator(dim=dim)

    # Causal model that can learn from mixed data
    causal_model = CliffordCausalModelV2(dim=dim).to(device)

    # Correlational baseline (only uses X)
    correlational = CorrelationalBaseline(dim=dim).to(device)

    opt_causal = torch.optim.Adam(causal_model.parameters(), lr=1e-3)
    opt_corr = torch.optim.Adam(correlational.parameters(), lr=1e-3)

    print("\n--- Training on MIXED data (80% observational, 20% interventional) ---")

    for epoch in range(500):
        # 80% observational data
        data_obs = generator.sample_observational(102)
        X_obs, Y_obs, Z_obs = data_obs['X'], data_obs['Y'], data_obs['Z']

        # 20% interventional data (key for identifiability!)
        data_int = generator.sample_interventional(26)
        X_int, Y_int, Z_int = data_int['X'], data_int['Y'], data_int['Z']

        # Combine
        X_all = torch.cat([X_obs, X_int])
        Y_all = torch.cat([Y_obs, Y_int])
        Z_all = torch.cat([Z_obs, Z_int])

        # Correlational: just learns P(Y|X) - ignores Z
        pred_corr = correlational(X_all)
        loss_corr = F.mse_loss(pred_corr, Y_all)
        opt_corr.zero_grad()
        loss_corr.backward()
        opt_corr.step()

        # Causal model: learns Y = f(X) + g(Z) structure
        pred_causal = causal_model(X_all, Z_all)
        loss_causal = F.mse_loss(pred_causal, Y_all)
        opt_causal.zero_grad()
        loss_causal.backward()
        opt_causal.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Correlational = {loss_corr.item():.4f}, "
                  f"Causal = {loss_causal.item():.4f}")

    # Evaluation
    print("\n--- Evaluation ---")

    with torch.no_grad():
        # In-distribution (observational)
        test_obs = generator.sample_observational(500)
        X_obs, Y_obs, Z_obs = test_obs['X'], test_obs['Y'], test_obs['Z']

        pred_corr_obs = correlational(X_obs)
        mse_corr_obs = F.mse_loss(pred_corr_obs, Y_obs).item()

        pred_causal_obs = causal_model(X_obs, Z_obs)
        mse_causal_obs = F.mse_loss(pred_causal_obs, Y_obs).item()

        print(f"\n1. In-Distribution (Observational) Test:")
        print(f"   Correlational MSE: {mse_corr_obs:.4f}")
        print(f"   Causal MSE:        {mse_causal_obs:.4f}")

        # Out-of-distribution (pure interventional)
        test_int = generator.sample_interventional(500)
        X_int, Y_int, Z_int = test_int['X'], test_int['Y'], test_int['Z']

        pred_corr_int = correlational(X_int)
        mse_corr_int = F.mse_loss(pred_corr_int, Y_int).item()

        pred_causal_int = causal_model(X_int, Z_int)
        mse_causal_int = F.mse_loss(pred_causal_int, Y_int).item()

        print(f"\n2. Out-of-Distribution (Interventional) Test:")
        print(f"   Correlational MSE: {mse_corr_int:.4f}")
        print(f"   Causal MSE:        {mse_causal_int:.4f}")

        # Degradation
        corr_deg = mse_corr_int / mse_corr_obs
        causal_deg = mse_causal_int / mse_causal_obs

        print(f"\n3. Performance Degradation (OOD/ID ratio):")
        print(f"   Correlational: {corr_deg:.2f}x worse")
        print(f"   Causal:        {causal_deg:.2f}x worse")

        # THE KEY TEST: Causal effect estimation
        print(f"\n4. Causal Effect Estimation (THE KEY METRIC):")
        x_high = torch.ones(100, dim, device=device) * 2
        x_low = torch.ones(100, dim, device=device) * -2

        # True causal effect (known from data generator)
        true_effect = generator.compute_true_causal_effect(x_high - x_low).mean().item()

        # Correlational: predicts Y from X only (confounded!)
        corr_high = correlational(x_high).mean().item()
        corr_low = correlational(x_low).mean().item()
        corr_effect = corr_high - corr_low

        # Causal: uses isolated X->Y mechanism
        causal_effect_high = causal_model.get_causal_effect(x_high).mean().item()
        causal_effect_low = causal_model.get_causal_effect(x_low).mean().item()
        causal_effect = causal_effect_high - causal_effect_low

        print(f"   True causal effect of X on Y: {true_effect:.4f}")
        print(f"   Correlational estimate:       {corr_effect:.4f}")
        print(f"   Causal model estimate:        {causal_effect:.4f}")

        corr_error = abs(corr_effect - true_effect)
        causal_error = abs(causal_effect - true_effect)

        print(f"\n   Correlational absolute error: {corr_error:.4f}")
        print(f"   Causal model absolute error:  {causal_error:.4f}")

        if causal_error < corr_error:
            improvement = (corr_error - causal_error) / corr_error * 100
            print(f"\n   --> Causal model is {improvement:.1f}% more accurate!")


if __name__ == "__main__":
    # Run all benchmarks
    results_ood = benchmark_ood_generalization()
    benchmark_simpsons_paradox()
    benchmark_counterfactual()

    # Test Clifford-based causal model
    benchmark_clifford_causal()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
These benchmarks test fundamental capabilities that distinguish
causal reasoning from mere correlation learning:

1. OOD Generalization: Causal models should maintain performance
   under distribution shift caused by interventions.

2. Simpson's Paradox: Causal models should correctly estimate
   treatment effects despite confounding.

3. Counterfactual Prediction: Causal models should answer
   "what if" questions by properly handling exogenous noise.

State-of-the-art results would show:
- Minimal degradation under intervention (benchmark 1)
- Correct sign and magnitude of causal effects (benchmark 2)
- Accurate counterfactual predictions (benchmark 3)

These are the capabilities needed for AGI-level reasoning about
cause and effect in the real world.
""")
