"""
Test: Does GELU in to_out help phasor on causal OOD task?

Finding from phasor_second_layer_analysis.py:
- 1L (GELU) beats 2 Layers on ASSOCIATIVE task (0.32 vs 0.48)
- But doesn't help on other tasks

Question: Does it help with OOD robustness?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PhasorLayerLinear(nn.Module):
    """Standard phasor layer with linear to_out."""
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


class PhasorLayerGELU(nn.Module):
    """Phasor layer with GELU in to_out (no expansion)."""
    def __init__(self, dim: int, n_planes: int = 16):
        super().__init__()
        self.dim = dim
        self.n_planes = n_planes
        self.key_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.key_encoder.weight)
        self.query_encoder = nn.Linear(dim, n_planes)
        nn.init.orthogonal_(self.query_encoder.weight)
        self.to_value = nn.Linear(dim, dim)
        # GELU between norm and linear
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class MultiPhasorMechanism(nn.Module):
    """Multi-layer phasor mechanism."""
    def __init__(self, dim: int, n_planes: int = 32, n_layers: int = 4, use_gelu: bool = False):
        super().__init__()
        layer_cls = PhasorLayerGELU if use_gelu else PhasorLayerLinear
        self.layers = nn.ModuleList([layer_cls(dim, n_planes) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class CausalModel(nn.Module):
    """Causal model with pluggable mechanism."""
    def __init__(self, dim=32, n_planes=32, n_layers=4, use_gelu=False):
        super().__init__()
        self.dim = dim
        self.mech_zx = MultiPhasorMechanism(dim, n_planes, n_layers, use_gelu)
        self.mech_zy = MultiPhasorMechanism(dim, n_planes, n_layers, use_gelu)
        self.mech_xy = MultiPhasorMechanism(dim, n_planes, n_layers, use_gelu)

    def forward(self, z: torch.Tensor):
        x = self.mech_zx(z)
        y = self.mech_xy(x) + self.mech_zy(z)
        return x, y

    def get_causal_effect(self, x: torch.Tensor) -> torch.Tensor:
        return self.mech_xy(x)


class CausalDataGenerator:
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
        X = torch.randn(n, self.dim, device=device) * 2
        Y = X @ self.W_xy.T + Z @ self.W_zy.T + torch.randn(n, self.dim, device=device) * self.noise_scale
        return {'Z': Z, 'X': X, 'Y': Y}

    def true_causal_effect(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_xy.T


def run_test():
    print("=" * 70)
    print("TEST: GELU in to_out for OOD Robustness")
    print("=" * 70)

    n_trials = 5
    dim = 32
    configs = [
        ('4L Linear', {'n_layers': 4, 'use_gelu': False}),
        ('4L GELU', {'n_layers': 4, 'use_gelu': True}),
        ('2L Linear', {'n_layers': 2, 'use_gelu': False}),
        ('2L GELU', {'n_layers': 2, 'use_gelu': True}),
        ('1L Linear', {'n_layers': 1, 'use_gelu': False}),
        ('1L GELU', {'n_layers': 1, 'use_gelu': True}),
    ]

    results = {name: {'deg': [], 'params': None} for name, _ in configs}

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        torch.manual_seed(trial * 42)
        generator = CausalDataGenerator(dim=dim)

        models = {}
        opts = {}
        for name, kwargs in configs:
            models[name] = CausalModel(dim=dim, n_planes=32, **kwargs).to(device)
            opts[name] = torch.optim.Adam(models[name].parameters(), lr=1e-3)
            if trial == 0:
                results[name]['params'] = sum(p.numel() for p in models[name].parameters())

        # Train
        for epoch in range(500):
            data_obs = generator.sample_observational(102)
            data_int = generator.sample_interventional(26)
            X = torch.cat([data_obs['X'], data_int['X']])
            Y = torch.cat([data_obs['Y'], data_int['Y']])
            Z = torch.cat([data_obs['Z'], data_int['Z']])

            for name in models:
                x_pred, y_pred = models[name](Z)
                loss = F.mse_loss(x_pred, X) + F.mse_loss(y_pred, Y)
                opts[name].zero_grad()
                loss.backward()
                opts[name].step()

        # Evaluate
        with torch.no_grad():
            test_obs = generator.sample_observational(500)
            test_int = generator.sample_interventional(500)

            for name in models:
                model = models[name]
                _, y_id = model(test_obs['Z'])
                id_mse = F.mse_loss(y_id, test_obs['Y']).item()

                y_ood = model.get_causal_effect(test_int['X']) + model.mech_zy(test_int['Z'])
                ood_mse = F.mse_loss(y_ood, test_int['Y']).item()

                results[name]['deg'].append(ood_mse / id_mse)

        # Print trial
        for name in models:
            deg = results[name]['deg'][-1]
            print(f"  {name:<12}: deg={deg:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<12} {'Params':>10} {'Deg (mean)':>12} {'Deg (std)':>10}")
    print("-" * 50)

    sorted_configs = sorted(configs, key=lambda c: np.mean(results[c[0]]['deg']))
    for name, _ in sorted_configs:
        r = results[name]
        print(f"{name:<12} {r['params']:>10,} {np.mean(r['deg']):>12.2f} {np.std(r['deg']):>10.2f}")

    # Compare GELU vs Linear at each depth
    print("\n" + "=" * 70)
    print("GELU vs LINEAR at each depth")
    print("=" * 70)
    for depth in [1, 2, 4]:
        linear_deg = np.mean(results[f'{depth}L Linear']['deg'])
        gelu_deg = np.mean(results[f'{depth}L GELU']['deg'])
        winner = "GELU" if gelu_deg < linear_deg else "Linear"
        diff = abs(gelu_deg - linear_deg) / linear_deg * 100
        print(f"  {depth}L: Linear={linear_deg:.2f}x, GELU={gelu_deg:.2f}x -> {winner} wins by {diff:.1f}%")


if __name__ == "__main__":
    run_test()
