"""
PSI vs Mamba - Dynamics Prediction Benchmark

Tests on Lorenz attractor:
- Multi-step rollout prediction (where integration-like computation matters)
- Long context utilization
- Error accumulation over time
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
import gc

from mambapy.mamba import Mamba, MambaConfig

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================================
# PSI Model (Unified Position/Content)
# ============================================================================

class PositionOnlyPhasorBlock(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        base_phases = torch.randn(max_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)
        value = self.to_value(x)
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)
        return x + self.to_out(retrieved)


class ContentOnlyPhasorBlock(nn.Module):
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators)
        )
        self.to_amp = nn.Sequential(nn.Linear(dim, n_oscillators), nn.Softplus())
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        amp = self.to_amp(x) + 0.1
        key_phasor = amp * torch.exp(1j * key_phase)
        query_phasor = amp * torch.exp(1j * query_phase)
        V = self.to_value(x).to(torch.complex64)
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved / norm
        return x + self.to_out(retrieved)


class PSIDynamicsModel(nn.Module):
    def __init__(self, state_dim=3, dim=128, n_layers=4, n_oscillators=64, max_len=256):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.embed = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(PositionOnlyPhasorBlock(dim, max_len))
            else:
                self.blocks.append(ContentOnlyPhasorBlock(dim, n_oscillators))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, state_dim)

    def forward(self, x):
        """x: [B, L, state_dim] -> [B, L, state_dim]"""
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))

    def predict_trajectory(self, context, num_steps):
        """Autoregressive rollout from context."""
        self.eval()
        predictions = []
        current = context.clone()

        with torch.no_grad():
            for _ in range(num_steps):
                out = self.forward(current)
                next_state = out[:, -1, :]  # Predict from last position
                predictions.append(next_state)
                # Shift context and add prediction
                current = torch.cat([current[:, 1:, :], next_state.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)  # [B, num_steps, state_dim]


# ============================================================================
# Mamba Dynamics Model
# ============================================================================

class MambaDynamicsModel(nn.Module):
    def __init__(self, state_dim=3, dim=128, n_layers=4, d_state=16):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.embed = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        config = MambaConfig(
            d_model=dim,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=2,
            d_conv=4,
            pscan=True,
            use_cuda=False
        )
        self.mamba = Mamba(config)
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, state_dim)

    def forward(self, x):
        h = self.embed(x)
        h = self.mamba(h)
        return self.head(self.norm_out(h))

    def predict_trajectory(self, context, num_steps):
        self.eval()
        predictions = []
        current = context.clone()

        with torch.no_grad():
            for _ in range(num_steps):
                out = self.forward(current)
                next_state = out[:, -1, :]
                predictions.append(next_state)
                current = torch.cat([current[:, 1:, :], next_state.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)


# ============================================================================
# Lorenz System
# ============================================================================

def lorenz_step(state, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([x + dx*dt, y + dy*dt, z + dz*dt])


def generate_lorenz_trajectory(length, dt=0.01):
    state = np.array([
        np.random.uniform(-15, 15),
        np.random.uniform(-20, 20),
        np.random.uniform(5, 40)
    ])
    trajectory = np.zeros((length, 3), dtype=np.float32)
    for i in range(length):
        trajectory[i] = state
        state = lorenz_step(state, dt=dt)
    return trajectory


def generate_dataset(n_trajectories, length, dt=0.01):
    data = np.zeros((n_trajectories, length, 3), dtype=np.float32)
    for i in range(n_trajectories):
        data[i] = generate_lorenz_trajectory(length, dt)
    # Normalize
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True) + 1e-8
    return (data - mean) / std, mean, std


# ============================================================================
# Training and Evaluation
# ============================================================================

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def train_model(model, train_data, epochs, lr, context_len=20, batch_size=64):
    """Train on next-state prediction."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    n_traj, traj_len, state_dim = train_data.shape

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Sample random windows
        for _ in range(100):  # 100 batches per epoch
            # Random trajectory and position
            traj_idx = np.random.randint(0, n_traj, batch_size)
            start_idx = np.random.randint(0, traj_len - context_len - 1, batch_size)

            # Build batch
            contexts = []
            targets = []
            for i in range(batch_size):
                ctx = train_data[traj_idx[i], start_idx[i]:start_idx[i]+context_len]
                tgt = train_data[traj_idx[i], start_idx[i]+1:start_idx[i]+context_len+1]
                contexts.append(ctx)
                targets.append(tgt)

            context = torch.tensor(np.array(contexts), device=device)
            target = torch.tensor(np.array(targets), device=device)

            pred = model(context)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.6f}")

    return total_loss / n_batches


def evaluate_rollout(model, val_data, context_len=20, rollout_steps=50, n_samples=100):
    """Evaluate multi-step rollout error."""
    model.eval()
    n_traj, traj_len, state_dim = val_data.shape

    step_errors = np.zeros(rollout_steps)

    with torch.no_grad():
        for _ in range(n_samples):
            traj_idx = np.random.randint(0, n_traj)
            start_idx = np.random.randint(0, traj_len - context_len - rollout_steps)

            context = val_data[traj_idx, start_idx:start_idx+context_len]
            gt_future = val_data[traj_idx, start_idx+context_len:start_idx+context_len+rollout_steps]

            context_tensor = torch.tensor(context, device=device).unsqueeze(0)
            predictions = model.predict_trajectory(context_tensor, rollout_steps)
            predictions = predictions[0].cpu().numpy()

            for step in range(rollout_steps):
                step_errors[step] += np.mean((predictions[step] - gt_future[step])**2)

    step_errors /= n_samples
    return step_errors


def main():
    print("=" * 80)
    print("PSI vs MAMBA - DYNAMICS PREDICTION BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Generate data
    print("Generating Lorenz trajectories...")
    train_data, mean, std = generate_dataset(200, 200)  # Smaller dataset
    val_data = (generate_dataset(50, 200)[0] - mean) / std  # Use same normalization
    print(f"Train: {train_data.shape}, Val: {val_data.shape}")
    print()

    # Hyperparameters
    state_dim = 3
    dim = 128
    n_layers = 4
    context_len = 20
    rollout_steps = 50
    epochs = 10  # Reduced for faster testing

    results = {}

    # ========== PSI ==========
    print("-" * 80)
    print("PSI (Unified Phasor)")
    print("-" * 80)

    best_psi = None
    best_psi_lr = None

    for lr in [1e-3, 5e-4]:  # Reduced configs
        print(f"\n  Testing lr={lr}...")
        reset_memory()

        model = PSIDynamicsModel(state_dim, dim, n_layers).to(device)
        params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        train_model(model, train_data, epochs, lr, context_len)
        train_time = time.time() - start_time
        train_mem = get_gpu_memory_mb()

        step_errors = evaluate_rollout(model, val_data, context_len, rollout_steps)

        # Summary metrics
        avg_error = step_errors.mean()
        final_error = step_errors[-1]
        error_growth = step_errors[-1] / (step_errors[0] + 1e-8)

        print(f"    Avg MSE: {avg_error:.6f}, Final MSE: {final_error:.6f}, Growth: {error_growth:.1f}x")

        if best_psi is None or avg_error < best_psi['avg_error']:
            best_psi = {
                'params': params,
                'train_time': train_time,
                'train_mem': train_mem,
                'step_errors': step_errors,
                'avg_error': avg_error,
                'final_error': final_error,
                'error_growth': error_growth
            }
            best_psi_lr = lr

        del model
        reset_memory()

    results['PSI'] = best_psi
    print(f"\n  Best PSI: lr={best_psi_lr}, avg_error={best_psi['avg_error']:.6f}")

    # ========== Mamba ==========
    print()
    print("-" * 80)
    print("Mamba")
    print("-" * 80)

    mamba_configs = [
        {'lr': 1e-3, 'd_state': 16},
        {'lr': 1e-3, 'd_state': 64},
    ]

    best_mamba = None
    best_mamba_cfg = None

    for cfg in mamba_configs:
        print(f"\n  Testing d_state={cfg['d_state']}, lr={cfg['lr']}...")
        reset_memory()

        model = MambaDynamicsModel(state_dim, dim, n_layers, d_state=cfg['d_state']).to(device)
        params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        train_model(model, train_data, epochs, cfg['lr'], context_len)
        train_time = time.time() - start_time
        train_mem = get_gpu_memory_mb()

        step_errors = evaluate_rollout(model, val_data, context_len, rollout_steps)

        avg_error = step_errors.mean()
        final_error = step_errors[-1]
        error_growth = step_errors[-1] / (step_errors[0] + 1e-8)

        print(f"    Avg MSE: {avg_error:.6f}, Final MSE: {final_error:.6f}, Growth: {error_growth:.1f}x")

        if best_mamba is None or avg_error < best_mamba['avg_error']:
            best_mamba = {
                'params': params,
                'train_time': train_time,
                'train_mem': train_mem,
                'step_errors': step_errors,
                'avg_error': avg_error,
                'final_error': final_error,
                'error_growth': error_growth
            }
            best_mamba_cfg = cfg

        del model
        reset_memory()

    results['Mamba'] = best_mamba
    print(f"\n  Best Mamba: d_state={best_mamba_cfg['d_state']}, lr={best_mamba_cfg['lr']}, avg_error={best_mamba['avg_error']:.6f}")

    # ========== Summary ==========
    print()
    print("=" * 80)
    print("SUMMARY: Lorenz Multi-Step Prediction (50-step rollout)")
    print("=" * 80)
    print()

    print(f"{'Metric':<25} {'PSI':>15} {'Mamba':>15} {'Winner':>10}")
    print("-" * 65)

    # Average error
    psi_avg = results['PSI']['avg_error']
    mamba_avg = results['Mamba']['avg_error']
    winner = 'PSI' if psi_avg < mamba_avg * 0.95 else ('Mamba' if mamba_avg < psi_avg * 0.95 else 'Tie')
    print(f"{'Avg MSE':<25} {psi_avg:>15.6f} {mamba_avg:>15.6f} {winner:>10}")

    # Final step error
    psi_final = results['PSI']['final_error']
    mamba_final = results['Mamba']['final_error']
    winner = 'PSI' if psi_final < mamba_final * 0.95 else ('Mamba' if mamba_final < psi_final * 0.95 else 'Tie')
    print(f"{'Final Step MSE':<25} {psi_final:>15.6f} {mamba_final:>15.6f} {winner:>10}")

    # Error growth
    psi_growth = results['PSI']['error_growth']
    mamba_growth = results['Mamba']['error_growth']
    winner = 'PSI' if psi_growth < mamba_growth * 0.9 else ('Mamba' if mamba_growth < psi_growth * 0.9 else 'Tie')
    print(f"{'Error Growth (50 steps)':<25} {psi_growth:>14.1f}x {mamba_growth:>14.1f}x {winner:>10}")

    # Parameters
    psi_params = results['PSI']['params']
    mamba_params = results['Mamba']['params']
    winner = 'PSI' if psi_params < mamba_params else 'Mamba'
    print(f"{'Parameters':<25} {psi_params:>15,} {mamba_params:>15,} {winner:>10}")

    # Training time
    psi_time = results['PSI']['train_time']
    mamba_time = results['Mamba']['train_time']
    winner = 'PSI' if psi_time < mamba_time else 'Mamba'
    print(f"{'Train Time (s)':<25} {psi_time:>15.1f} {mamba_time:>15.1f} {winner:>10}")

    # Memory
    psi_mem = results['PSI']['train_mem']
    mamba_mem = results['Mamba']['train_mem']
    winner = 'PSI' if psi_mem < mamba_mem else 'Mamba'
    print(f"{'Train Memory (MB)':<25} {psi_mem:>15.1f} {mamba_mem:>15.1f} {winner:>10}")

    print("-" * 65)
    print()

    # Step-by-step comparison
    print("Error at specific rollout steps:")
    print(f"{'Step':<10} {'PSI MSE':>15} {'Mamba MSE':>15} {'Winner':>10}")
    print("-" * 50)
    for step in [1, 5, 10, 20, 30, 50]:
        psi_err = results['PSI']['step_errors'][step-1]
        mamba_err = results['Mamba']['step_errors'][step-1]
        winner = 'PSI' if psi_err < mamba_err * 0.95 else ('Mamba' if mamba_err < psi_err * 0.95 else 'Tie')
        print(f"{step:<10} {psi_err:>15.6f} {mamba_err:>15.6f} {winner:>10}")

    print()

    # Verdict
    psi_wins = sum([
        psi_avg < mamba_avg * 0.95,
        psi_final < mamba_final * 0.95,
        psi_growth < mamba_growth * 0.9,
        psi_params < mamba_params,
        psi_time < mamba_time,
        psi_mem < mamba_mem
    ])
    mamba_wins = sum([
        mamba_avg < psi_avg * 0.95,
        mamba_final < psi_final * 0.95,
        mamba_growth < psi_growth * 0.9,
        mamba_params < psi_params,
        mamba_time < psi_time,
        mamba_mem < psi_mem
    ])

    print(f"PSI wins: {psi_wins}, Mamba wins: {mamba_wins}")

    if psi_wins > mamba_wins:
        print("\nVERDICT: PSI outperforms Mamba on dynamics prediction!")
    elif mamba_wins > psi_wins:
        print("\nVERDICT: Mamba outperforms PSI on dynamics prediction")
    else:
        print("\nVERDICT: PSI and Mamba are comparable on dynamics prediction")


if __name__ == "__main__":
    main()
