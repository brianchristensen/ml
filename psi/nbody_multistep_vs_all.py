"""
N-body Multi-step PSI vs LSTM vs Transformer on Longer Sequences

Tests PSI Multi-Step (single forward pass) against baselines
on longer prediction horizons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import math

from psi import PSIBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Data Generation
# =============================================================================

def nbody_dynamics(t, state, n_bodies):
    """Compute derivatives for N-body gravitational system."""
    state = state.reshape(n_bodies, 5)
    deriv = np.zeros_like(state)

    for i in range(n_bodies):
        m_i, x_i, y_i, vx_i, vy_i = state[i]
        deriv[i, 0] = 0
        deriv[i, 1] = vx_i
        deriv[i, 2] = vy_i

        ax, ay = 0.0, 0.0
        for j in range(n_bodies):
            if i != j:
                m_j, x_j, y_j, vx_j, vy_j = state[j]
                dx = x_j - x_i
                dy = y_j - y_i
                r = np.sqrt(dx**2 + dy**2)
                ax += m_j * dx / (r**3 + 1e-8)
                ay += m_j * dy / (r**3 + 1e-8)

        deriv[i, 3] = ax
        deriv[i, 4] = ay

    return deriv.flatten()


def random_nbody_config(n_bodies, orbit_noise=5e-2):
    """Generate random initial configuration for N-body system."""
    state = np.zeros((n_bodies, 5))
    state[:, 0] = 1.0

    for i in range(n_bodies):
        r = np.random.rand() * 0.5 + 0.75
        theta = 2 * np.pi * i / n_bodies + np.random.randn() * 0.3
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        state[i, 1:3] = [x, y]

        vx = -y / (r**1.5) * (1 + orbit_noise * np.random.randn())
        vy = x / (r**1.5) * (1 + orbit_noise * np.random.randn())
        state[i, 3:5] = [vx, vy]

    return state


def generate_nbody_data(n_bodies=3, trials=500, timesteps=150, t_span=[0, 30]):
    """Generate N-body gravitational dynamics dataset."""
    print(f"Generating {trials} {n_bodies}-body trajectories ({timesteps} steps)...")

    x = np.zeros((trials, timesteps, n_bodies, 5))

    for trial in range(trials):
        if (trial + 1) % 200 == 0:
            print(f"  {trial + 1}/{trials}")

        state0 = random_nbody_config(n_bodies)
        t_eval = np.linspace(t_span[0], t_span[1], timesteps)

        sol = solve_ivp(
            lambda t, y: nbody_dynamics(t, y, n_bodies),
            t_span,
            state0.flatten(),
            t_eval=t_eval,
            rtol=1e-9
        )

        for i in range(timesteps):
            x[trial, i] = sol.y[:, i].reshape(n_bodies, 5)

    return x


# =============================================================================
# Dataset
# =============================================================================

class NBodyMultiStepDataset(Dataset):
    """Dataset that returns context and K future states."""

    def __init__(self, trajectories, context_len=20, K=50, split='train', train_frac=0.8):
        n_bodies = trajectories.shape[2]
        state_features = []
        for i in range(n_bodies):
            state_features.append(trajectories[:, :, i, 1:])

        self.trajectories = np.concatenate(state_features, axis=-1).astype(np.float32)
        self.n_bodies = n_bodies
        self.K = K

        self.mean = self.trajectories.mean(axis=(0, 1))
        self.std = self.trajectories.std(axis=(0, 1))
        self.trajectories = (self.trajectories - self.mean) / (self.std + 1e-8)

        num_train = int(len(self.trajectories) * train_frac)
        if split == 'train':
            self.trajectories = self.trajectories[:num_train]
        else:
            self.trajectories = self.trajectories[num_train:]

        self.context_len = context_len
        self.seqs_per_traj = max(1, self.trajectories.shape[1] - context_len - K)

        print(f"{split.upper()}: {len(self.trajectories)} trajectories, {len(self)} sequences (K={K})")

    def __len__(self):
        return len(self.trajectories) * self.seqs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.seqs_per_traj
        pos = idx % self.seqs_per_traj

        trajectory = self.trajectories[traj_idx]
        context = trajectory[pos:pos + self.context_len]
        future = trajectory[pos + self.context_len:pos + self.context_len + self.K]

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'future': torch.tensor(future, dtype=torch.float32)
        }


# =============================================================================
# Models
# =============================================================================

class PSIMultiStep(nn.Module):
    """PSI model that predicts K future states at once using cumsum integration."""

    def __init__(self, state_dim, dim=128, num_layers=6, K=50, max_len=50):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.K = K

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))
        self.time_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.blocks = nn.ModuleList([PSIBlock(dim=dim) for _ in range(num_layers)])

        self.derivative_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

        self.dt_scale = nn.Parameter(torch.ones(state_dim) * 0.1)

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, states):
        batch_size, seq_len, _ = states.shape

        x = self.state_embedding(states)
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        for block in self.blocks:
            x = block(x)

        final_hidden = x[:, -1:, :]
        h = final_hidden.expand(-1, self.K, -1) + self.time_embed.expand(batch_size, -1, -1)

        for block in self.blocks:
            h = block(h)

        dx = self.derivative_head(h)
        last_state = states[:, -1, :]
        future = last_state.unsqueeze(1) + torch.cumsum(dx * self.dt_scale, dim=1)

        return future

    def rollout(self, context, num_steps):
        """Single forward pass for entire rollout when K >= num_steps."""
        if num_steps <= self.K:
            future = self.forward(context)
            return future[:, :num_steps, :]
        else:
            # Chain predictions if needed
            predictions = []
            current_context = context.clone()
            steps_remaining = num_steps

            while steps_remaining > 0:
                future = self.forward(current_context)
                take = min(self.K, steps_remaining)
                predictions.append(future[:, :take, :])

                current_context = torch.cat([
                    current_context[:, take:, :],
                    future[:, :take, :]
                ], dim=1)
                steps_remaining -= take

            return torch.cat(predictions, dim=1)


class ManualLSTMCell(nn.Module):
    """Pure PyTorch LSTM cell - no cuDNN optimization."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=-1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMSingleStep(nn.Module):
    """Manual LSTM for single-step prediction."""

    def __init__(self, state_dim, dim=128, num_layers=6):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim
        self.num_layers = num_layers

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.cells = nn.ModuleList([
            ManualLSTMCell(dim, dim) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

    def forward(self, states):
        batch, seq_len, _ = states.shape
        x = self.state_embedding(states)

        h = [torch.zeros(batch, self.dim, device=states.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.dim, device=states.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]

        return self.output_head(h[-1])

    def rollout(self, context, num_steps):
        predictions = []
        current_context = context.clone()

        for _ in range(num_steps):
            next_state = self.forward(current_context)
            predictions.append(next_state)
            current_context = torch.cat([
                current_context[:, 1:, :],
                next_state.unsqueeze(1)
            ], dim=1)

        return torch.stack(predictions, dim=1)


class TransformerPredictor(nn.Module):
    """Transformer for single-step prediction."""

    def __init__(self, state_dim, dim=128, num_layers=6, num_heads=4, max_len=100):
        super().__init__()
        self.state_dim = state_dim
        self.dim = dim

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, state_dim)
        )

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, states):
        batch_size, seq_len, _ = states.shape

        x = self.state_embedding(states)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=states.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        return self.output_head(x[:, -1, :])

    def rollout(self, context, num_steps):
        predictions = []
        current_context = context.clone()

        for _ in range(num_steps):
            next_state = self.forward(current_context)
            predictions.append(next_state)
            current_context = torch.cat([
                current_context[:, 1:, :],
                next_state.unsqueeze(1)
            ], dim=1)

        return torch.stack(predictions, dim=1)


# =============================================================================
# Training
# =============================================================================

def train_single_step(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """Train single-step model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            context = batch['context'].to(device)
            future = batch['future'].to(device)
            target = future[:, 0, :]

            optimizer.zero_grad()
            pred = model(context)
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(device)
                future = batch['future'].to(device)
                target = future[:, 0, :]
                pred = model(context)
                val_loss += F.mse_loss(pred, target).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

    return best_val, history


def train_multi_step(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """Train multi-step model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            context = batch['context'].to(device)
            future = batch['future'].to(device)

            optimizer.zero_grad()
            pred = model(context)
            loss = F.mse_loss(pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(device)
                future = batch['future'].to(device)
                pred = model(context)
                val_loss += F.mse_loss(pred, future).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

    return best_val, history


def evaluate_rollout(model, test_trajectories, context_len, mean, std, n_bodies, rollout_steps):
    """Evaluate rollout accuracy and speed."""
    model.eval()

    errors = []
    times = []

    for i in range(min(50, len(test_trajectories))):
        traj = test_trajectories[i]
        context = np.concatenate([traj[:context_len, j, 1:] for j in range(n_bodies)], axis=-1)
        context_norm = (context - mean) / (std + 1e-8)
        context_tensor = torch.tensor(context_norm, dtype=torch.float32).unsqueeze(0).to(device)

        gt = np.concatenate([traj[context_len:context_len+rollout_steps, j, 1:] for j in range(n_bodies)], axis=-1)
        gt_norm = (gt - mean) / (std + 1e-8)

        with torch.no_grad():
            start = time.time()
            pred = model.rollout(context_tensor, rollout_steps)
            torch.cuda.synchronize() if device == 'cuda' else None
            elapsed = time.time() - start

        pred_np = pred[0].cpu().numpy()
        mse = np.mean((pred_np - gt_norm)**2)
        errors.append(mse)
        times.append(elapsed)

    return np.mean(errors), np.std(errors), np.mean(times), np.std(times)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("N-BODY: PSI MULTI-STEP vs LSTM vs TRANSFORMER (LONG SEQUENCES)")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Parameters
    n_bodies = 3
    trials = 500
    timesteps = 150  # Longer trajectories
    context_len = 20
    rollout_steps_list = [30, 50, 80]  # Test multiple horizons
    K = max(rollout_steps_list)  # PSI predicts all at once
    dim = 128
    num_layers = 6
    epochs = 50
    batch_size = 128
    lr = 1e-4

    # Generate data
    print("1. GENERATING DATA")
    print("-" * 40)
    x = generate_nbody_data(n_bodies, trials, timesteps)
    print()

    state_dim = n_bodies * 4

    # Create datasets
    train_dataset = NBodyMultiStepDataset(x, context_len, K=K, split='train')
    val_dataset = NBodyMultiStepDataset(x, context_len, K=K, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Models
    models = {
        f'PSI Multi-Step (K={K})': (
            PSIMultiStep(state_dim, dim, num_layers, K, context_len).to(device),
            'multi'
        ),
        'LSTM Single-Step': (
            LSTMSingleStep(state_dim, dim, num_layers).to(device),
            'single'
        ),
        'Transformer Single-Step': (
            TransformerPredictor(state_dim, dim, num_layers, num_heads=4, max_len=context_len+10).to(device),
            'single'
        ),
    }

    results = {}
    test_trajectories = x[int(len(x) * 0.8):]

    for name, (model, train_type) in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print('='*60)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        start_time = time.time()

        if train_type == 'multi':
            best_val, history = train_multi_step(model, train_loader, val_loader, epochs, lr)
        else:
            best_val, history = train_single_step(model, train_loader, val_loader, epochs, lr)

        train_time = time.time() - start_time

        # Evaluate at multiple rollout horizons
        rollout_results = {}
        for rollout_steps in rollout_steps_list:
            mse_mean, mse_std, time_mean, time_std = evaluate_rollout(
                model, test_trajectories, context_len,
                train_dataset.mean, train_dataset.std, n_bodies, rollout_steps
            )
            rollout_results[rollout_steps] = {
                'mse': mse_mean,
                'mse_std': mse_std,
                'time': time_mean,
                'time_std': time_std
            }
            print(f"  {rollout_steps}-step: MSE={mse_mean:.6f}, Time={time_mean*1000:.2f}ms")

        results[name] = {
            'params': n_params,
            'train_time': train_time,
            'best_val': best_val,
            'rollouts': rollout_results,
            'history': history
        }

        print(f"\n  Training time: {train_time:.1f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY ROLLOUT HORIZON")
    print("=" * 80)

    for rollout_steps in rollout_steps_list:
        print(f"\n{rollout_steps}-STEP ROLLOUT:")
        print(f"{'Model':<30} {'MSE':>12} {'Time (ms)':>12} {'Speedup':>10}")
        print("-" * 70)

        # Find baseline (LSTM) time for speedup calc
        lstm_time = results['LSTM Single-Step']['rollouts'][rollout_steps]['time']

        for name, r in results.items():
            rr = r['rollouts'][rollout_steps]
            speedup = lstm_time / rr['time']
            print(f"{name:<30} {rr['mse']:>12.6f} {rr['time']*1000:>12.2f} {speedup:>9.1f}x")

    # Overall comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<30} {'Params':>12} {'Train Time':>12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30} {r['params']:>12,} {r['train_time']:>11.1f}s")

    # PSI vs others summary
    psi_name = f'PSI Multi-Step (K={K})'
    print("\n" + "=" * 80)
    print("PSI MULTI-STEP ADVANTAGES")
    print("=" * 80)

    for rollout_steps in rollout_steps_list:
        print(f"\n{rollout_steps}-step horizon:")
        psi_mse = results[psi_name]['rollouts'][rollout_steps]['mse']
        psi_time = results[psi_name]['rollouts'][rollout_steps]['time']

        for other in ['LSTM Single-Step', 'Transformer Single-Step']:
            other_mse = results[other]['rollouts'][rollout_steps]['mse']
            other_time = results[other]['rollouts'][rollout_steps]['time']

            acc_improvement = (other_mse - psi_mse) / other_mse * 100
            speedup = other_time / psi_time

            print(f"  vs {other}: {acc_improvement:+.1f}% accuracy, {speedup:.1f}x faster")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        f'PSI Multi-Step (K={K})': 'coral',
        'LSTM Single-Step': 'steelblue',
        'Transformer Single-Step': 'seagreen'
    }

    # Learning curves
    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(r['history']['val'], label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # MSE by rollout horizon
    ax = axes[0, 1]
    x_pos = np.arange(len(rollout_steps_list))
    width = 0.25
    for i, (name, r) in enumerate(results.items()):
        mses = [r['rollouts'][rs]['mse'] for rs in rollout_steps_list]
        ax.bar(x_pos + i*width - width, mses, width, label=name, color=colors[name])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{rs}-step' for rs in rollout_steps_list])
    ax.set_ylabel('Rollout MSE')
    ax.set_title('Accuracy by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Time by rollout horizon
    ax = axes[1, 0]
    for i, (name, r) in enumerate(results.items()):
        times = [r['rollouts'][rs]['time']*1000 for rs in rollout_steps_list]
        ax.bar(x_pos + i*width - width, times, width, label=name, color=colors[name])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{rs}-step' for rs in rollout_steps_list])
    ax.set_ylabel('Rollout Time (ms)')
    ax.set_title('Speed by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Speedup vs horizon
    ax = axes[1, 1]
    lstm_times = [results['LSTM Single-Step']['rollouts'][rs]['time'] for rs in rollout_steps_list]
    trans_times = [results['Transformer Single-Step']['rollouts'][rs]['time'] for rs in rollout_steps_list]
    psi_times = [results[psi_name]['rollouts'][rs]['time'] for rs in rollout_steps_list]

    ax.plot(rollout_steps_list, [lt/pt for lt, pt in zip(lstm_times, psi_times)],
            'o-', label='vs LSTM', color='steelblue', linewidth=2, markersize=8)
    ax.plot(rollout_steps_list, [tt/pt for tt, pt in zip(trans_times, psi_times)],
            's-', label='vs Transformer', color='seagreen', linewidth=2, markersize=8)
    ax.set_xlabel('Rollout Steps')
    ax.set_ylabel('PSI Speedup Factor')
    ax.set_title('PSI Multi-Step Speedup vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nbody_multistep_vs_all.png', dpi=150)
    plt.close()
    print("\nSaved nbody_multistep_vs_all.png")


if __name__ == "__main__":
    main()
