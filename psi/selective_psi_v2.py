"""
Selective PSI v2 - More aggressive selectivity.

The previous version was too similar to vanilla PSI.
Let's make more dramatic changes based on what actually makes Mamba work:

1. Mamba's key insight: The B, C matrices (input/output projections) are input-dependent
2. The discretization (delta/dt) is input-dependent
3. This allows the model to selectively ignore or focus on different inputs

For PSI, the equivalent would be:
1. Make the phase encoding MORE input-dependent (not just additive omega)
2. Make the magnitude/write strength VERY selective
3. Add input-dependent output mixing (like Mamba's C matrix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectivePSIv2(nn.Module):
    """
    Selective PSI v2 - More aggressive selectivity.

    Key changes:
    1. Input-dependent phase basis (not just omega, but the whole encoding)
    2. Hard-ish selective gating with learned temperature
    3. Input-dependent output mixing
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # === INPUT-DEPENDENT PHASE BASIS ===
        # Instead of fixed omega + delta, make the entire phase transformation input-dependent
        self.to_phase_transform = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Integration scale (learned)
        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)

        # === SELECTIVE WRITE ===
        # Hard-ish gate with learned temperature
        self.to_write_gate = nn.Linear(dim, dim)
        self.gate_temperature = nn.Parameter(torch.ones(1) * 2.0)  # sharper sigmoid

        # Magnitude
        self.to_magnitude = nn.Linear(dim, dim)

        # === SELECTIVE READ (like Mamba's C matrix) ===
        # Input-dependent query transformation
        self.to_query = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Input-dependent output mixing
        self.to_output_mix = nn.Linear(dim, dim * 2)  # mix retrieved_real and retrieved_imag

        # Final output
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        batch, seq_len, dim = x.shape

        # === PHASE COMPUTATION ===
        # Fully input-dependent phase
        phase_delta = self.to_phase_transform(x)
        phase_delta_scaled = phase_delta * self.integration_scale.abs()
        phi = torch.cumsum(phase_delta_scaled, dim=1)

        # === SELECTIVE WRITE ===
        # Gate with temperature (sharper = more selective)
        write_logits = self.to_write_gate(x)
        write_gate = torch.sigmoid(write_logits * self.gate_temperature)

        # Magnitude
        magnitude = torch.sigmoid(self.to_magnitude(x)) * 5.0

        # Selective content
        content = write_gate * x * magnitude

        # Write to memory
        memory_real = torch.cumsum(content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(content * torch.sin(phi), dim=1)

        # Normalize
        accum_weight = torch.cumsum(write_gate * magnitude + 1e-8, dim=1)
        memory_real_norm = memory_real / torch.sqrt(accum_weight)
        memory_imag_norm = memory_imag / torch.sqrt(accum_weight)

        # === SELECTIVE READ ===
        # Input-dependent query
        query_phase = phi + self.to_query(x)

        cos_q = torch.cos(query_phase)
        sin_q = torch.sin(query_phase)

        retrieved_real = memory_real_norm * cos_q + memory_imag_norm * sin_q
        retrieved_imag = memory_imag_norm * cos_q - memory_real_norm * sin_q

        # === INPUT-DEPENDENT OUTPUT MIX ===
        # Like Mamba's C matrix - input determines how to combine the retrieved values
        mix_weights = self.to_output_mix(x)  # [B, L, dim*2]
        mix_real, mix_imag = mix_weights.chunk(2, dim=-1)
        mix_real = torch.sigmoid(mix_real)
        mix_imag = torch.sigmoid(mix_imag)

        # Weighted combination
        retrieved = mix_real * retrieved_real + mix_imag * retrieved_imag

        # Output
        out = self.to_out(retrieved)

        return x + out


class SelectivePSIv2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.psi = SelectivePSIv2(dim)

    def forward(self, x):
        return x + self.psi(self.norm(x))


class MambaLikePSI(nn.Module):
    """
    Even more Mamba-like: use actual SSM-style parameterization.

    Mamba's core: y = SSM(A, B, C, D)(x)
    where A is fixed (structured), B, C are input-dependent, D is skip connection

    PSI equivalent:
    - A: the phase evolution (cumsum structure)
    - B: how to write to memory (input-dependent)
    - C: how to read from memory (input-dependent)
    - D: skip connection (residual)
    """

    def __init__(self, dim, state_dim=None):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim or dim

        # B: input -> state (write)
        self.to_B = nn.Linear(dim, self.state_dim)

        # C: state -> output (read)
        self.to_C = nn.Linear(dim, self.state_dim)

        # Delta (dt): controls discretization
        self.to_dt = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Linear(dim // 4, self.state_dim)
        )
        # Initialize dt to be small
        nn.init.constant_(self.to_dt[-1].bias, -4.0)

        # Phase parameters (the "A" equivalent - but phase-based)
        self.omega_base = nn.Parameter(torch.randn(self.state_dim) * 0.1)

        # D: skip connection weight
        self.D = nn.Parameter(torch.ones(dim))

        # Output projection
        self.out_proj = nn.Linear(self.state_dim, dim)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        # Compute input-dependent B, C, dt
        B = self.to_B(x)  # [B, L, state_dim] - write strength
        C = self.to_C(x)  # [B, L, state_dim] - read strength
        dt = F.softplus(self.to_dt(x))  # [B, L, state_dim] - integration rate

        # Phase evolution (the SSM "A" matrix equivalent)
        omega = self.omega_base * dt
        phi = torch.cumsum(omega, dim=1)

        # Write to memory with input-dependent B
        content = B * x[..., :self.state_dim] if self.state_dim <= dim else B

        memory_real = torch.cumsum(content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(content * torch.sin(phi), dim=1)

        # Read with input-dependent C
        retrieved = C * (memory_real * torch.cos(phi) + memory_imag * torch.sin(phi))

        # Project and add skip
        y = self.out_proj(retrieved) + self.D * x

        return y


class MambaLikePSIBlock(nn.Module):
    def __init__(self, dim, state_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.psi = MambaLikePSI(dim, state_dim)

    def forward(self, x):
        return self.psi(self.norm(x))  # No extra residual, it's inside


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from scipy.integrate import odeint
    import time
    import sys
    sys.path.insert(0, '.')
    from psi import PSIBlock

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    def lorenz(state, t):
        x, y, z = state
        return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]

    def gen_data(n, steps, dt=0.01):
        trajs = []
        for _ in range(n):
            x0 = np.random.randn(3)*5 + [0,0,25]
            t = np.linspace(0, steps*dt, steps+1)
            trajs.append(odeint(lorenz, x0, t))
        return torch.tensor(np.array(trajs), dtype=torch.float32)

    class MultiStepPredictor(nn.Module):
        def __init__(self, state_dim, hidden_dim, K, block_cls, num_layers=2, **block_kwargs):
            super().__init__()
            self.K = K
            self.state_dim = state_dim
            self.input_proj = nn.Linear(state_dim, hidden_dim)
            self.blocks = nn.ModuleList([block_cls(hidden_dim, **block_kwargs) for _ in range(num_layers)])
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, K * state_dim)
            )
            self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.01)

        def forward(self, history):
            h = self.input_proj(history)
            for block in self.blocks:
                h = block(h)
            h_final = h[:, -1, :]
            dx_flat = self.output_proj(h_final)
            dx = dx_flat.view(-1, self.K, self.state_dim)
            x_last = history[:, -1, :]
            dx_scaled = dx * self.dt_scale
            future = x_last.unsqueeze(1) + torch.cumsum(dx_scaled, dim=1)
            return future

    # Generate data
    train_data = gen_data(500, 150).to(device)
    val_data = gen_data(100, 150).to(device)
    test_data = gen_data(100, 150).to(device)

    history_len = 50
    future_len = 100

    train_history = train_data[:, :history_len, :]
    train_future = train_data[:, history_len:history_len+future_len, :]
    val_history = val_data[:, :history_len, :]
    val_future = val_data[:, history_len:history_len+future_len, :]
    test_history = test_data[:, :history_len, :]
    test_future = test_data[:, history_len:history_len+future_len, :]

    print(f"History: {history_len}, Future: {future_len}\n")

    def train_and_eval(model, name):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        best_val = float('inf')
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            pred = model(train_history)
            loss = F.mse_loss(pred, train_future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(val_history)
                val_loss = F.mse_loss(pred_val, val_future)
                if val_loss < best_val:
                    best_val = val_loss

            if (epoch+1) % 25 == 0:
                print(f"  Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            pred_test = model(test_history)
            test_mse = F.mse_loss(pred_test, test_future).item()

        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model(test_history[:32])
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) / 50 * 1000

        return best_val.item(), test_mse, elapsed

    # Compare models
    models = {
        'Vanilla PSI': MultiStepPredictor(3, 64, future_len, PSIBlock, num_layers=2).to(device),
        'Selective v2': MultiStepPredictor(3, 64, future_len, SelectivePSIv2Block, num_layers=2).to(device),
        'Mamba-like': MultiStepPredictor(3, 64, future_len, MambaLikePSIBlock, num_layers=2).to(device),
    }

    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name} ({n_params:,} params)")
        print("-" * 40)
        val_mse, test_mse, time_ms = train_and_eval(model, name)
        results[name] = {'val': val_mse, 'test': test_mse, 'time': time_ms, 'params': n_params}
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Params':<12} {'Val MSE':<12} {'Test MSE':<12} {'Time(ms)':<10}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<20} {r['params']:<12,} {r['val']:<12.4f} {r['test']:<12.4f} {r['time']:<10.2f}")

    baseline = test_history[:, -1:, :].expand(-1, future_len, -1)
    baseline_mse = F.mse_loss(baseline, test_future).item()
    print(f"\nBaseline (repeat last): {baseline_mse:.4f}")

    best = min(results.items(), key=lambda x: x[1]['test'])
    print(f"Best model: {best[0]} ({best[1]['test']:.4f} MSE)")
