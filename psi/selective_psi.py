"""
Selective PSI - Making PSI more like Mamba with input-dependent selectivity.

Key insight from Mamba: The power of selective state spaces comes from
making the state transition HIGHLY input-dependent, not just learned.

In PSI terms:
- Current: omega/gate are functions of x, but the pattern is similar across sequences
- Selective: Make the phase dynamics more aggressively input-dependent,
  so different inputs write to completely different frequency slots

Changes:
1. Input-dependent delta (like Mamba's selective dt)
2. Input-dependent forget/retain mechanism
3. More expressive input-to-phase mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectivePSI(nn.Module):
    """
    Selective Phase-Space Integration.

    Key differences from vanilla PSI:
    1. Input-dependent delta scaling (like Mamba's selective dt)
    2. Selective write gate - content-aware filtering before memory
    3. Input-dependent phase spread - different inputs use different frequency bands
    """

    def __init__(self, dim, dt_rank=None, expand_factor=1):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank or max(16, dim // 8)
        self.inner_dim = dim * expand_factor

        # Project input to higher dim if expanding
        self.in_proj = nn.Linear(dim, self.inner_dim) if expand_factor > 1 else nn.Identity()

        # === SELECTIVE COMPONENTS (like Mamba) ===

        # Delta (dt) projection - controls integration rate per-input
        # Low rank for efficiency, then broadcast
        self.dt_proj_down = nn.Linear(self.inner_dim, self.dt_rank, bias=False)
        self.dt_proj_up = nn.Linear(self.dt_rank, self.inner_dim)

        # Selective gate - what to write to memory (input-dependent)
        self.selective_gate = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.Sigmoid()
        )

        # === PHASE COMPONENTS ===

        # Phase init - more expressive, input-dependent
        self.to_phase_init = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

        # Omega (phase velocity) - base + input-dependent modulation
        self.omega_base = nn.Parameter(torch.randn(self.inner_dim) * 0.1)
        self.to_omega_delta = nn.Linear(self.inner_dim, self.inner_dim)

        # Magnitude for memory write
        self.to_magnitude = nn.Linear(self.inner_dim, self.inner_dim)

        # Query offset for retrieval
        self.to_query_offset = nn.Linear(self.inner_dim, self.inner_dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(self.inner_dim * 2),
            nn.Linear(self.inner_dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

        # Initialize dt bias to be small positive (like Mamba)
        nn.init.constant_(self.dt_proj_up.bias, -4.0)  # sigmoid(-4) ≈ 0.018

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Project to inner dim
        h = self.in_proj(x)  # [B, L, inner_dim]

        # === SELECTIVE DT (like Mamba) ===
        # Input-dependent integration rate
        dt_low = self.dt_proj_down(h)  # [B, L, dt_rank]
        dt = F.softplus(self.dt_proj_up(dt_low))  # [B, L, inner_dim], positive

        # === SELECTIVE GATE ===
        # What portion of input to write to memory
        select_gate = self.selective_gate(h)  # [B, L, inner_dim]

        # === PHASE DYNAMICS ===
        # Base omega + input-dependent delta
        omega = self.omega_base + self.to_omega_delta(h)

        # Scale omega by selective dt
        omega_scaled = omega * dt * 0.01  # small scale for stability

        # Integrate phase
        phi_init = self.to_phase_init(h)
        phi = phi_init + torch.cumsum(omega_scaled, dim=1)

        # === MEMORY WRITE (selective) ===
        magnitude = torch.sigmoid(self.to_magnitude(h)) * 5.0

        # Selective content - gate controls what gets written
        content = select_gate * h * magnitude

        # Write to phase-modulated memory
        memory_real = torch.cumsum(content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(content * torch.sin(phi), dim=1)

        # Normalize
        accumulated_magnitude = torch.cumsum(magnitude * select_gate, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_norm = memory_real / sqrt_magnitude
        memory_imag_norm = memory_imag / sqrt_magnitude

        # === MEMORY READ ===
        query_offset = self.to_query_offset(h)
        phi_query = phi + query_offset

        cos_q = torch.cos(phi_query)
        sin_q = torch.sin(phi_query)

        retrieved_real = memory_real_norm * cos_q + memory_imag_norm * sin_q
        retrieved_imag = memory_imag_norm * cos_q - memory_real_norm * sin_q

        # === OUTPUT ===
        context = torch.cat([retrieved_real, retrieved_imag], dim=-1)
        out = self.to_out(context)

        return x + out


class SelectivePSIBlock(nn.Module):
    """Selective PSI block with pre-norm."""

    def __init__(self, dim, dt_rank=None, expand_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.selective_psi = SelectivePSI(dim, dt_rank, expand_factor)

    def forward(self, x):
        return x + self.selective_psi(self.norm(x))


# =============================================================================
# Benchmark against vanilla PSI
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

    # Lorenz data
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
        """Multi-step predictor using any block type."""
        def __init__(self, state_dim, hidden_dim, K, block_cls, num_layers=2):
            super().__init__()
            self.K = K
            self.state_dim = state_dim
            self.input_proj = nn.Linear(state_dim, hidden_dim)
            self.blocks = nn.ModuleList([block_cls(hidden_dim) for _ in range(num_layers)])
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

    print(f"History: {history_len}, Future: {future_len}")
    print()

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

        # Test
        model.eval()
        with torch.no_grad():
            pred_test = model(test_history)
            test_mse = F.mse_loss(pred_test, test_future).item()

        # Speed
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
        'Selective PSI': MultiStepPredictor(3, 64, future_len, SelectivePSIBlock, num_layers=2).to(device),
    }

    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name} ({n_params:,} params)")
        print("-" * 40)
        val_mse, test_mse, time_ms = train_and_eval(model, name)
        results[name] = {'val': val_mse, 'test': test_mse, 'time': time_ms, 'params': n_params}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Params':<12} {'Val MSE':<12} {'Test MSE':<12} {'Time(ms)':<10}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<20} {r['params']:<12,} {r['val']:<12.4f} {r['test']:<12.4f} {r['time']:<10.2f}")

    # Baseline
    baseline = test_history[:, -1:, :].expand(-1, future_len, -1)
    baseline_mse = F.mse_loss(baseline, test_future).item()
    print(f"\nBaseline (repeat last): {baseline_mse:.4f}")

    best = min(results.items(), key=lambda x: x[1]['test'])
    print(f"Best model: {best[0]} ({best[1]['test']:.4f} MSE)")
