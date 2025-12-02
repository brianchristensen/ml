"""
Control Systems Benchmark: Where PSI's Integration Should Shine

Steve Brunton-inspired tasks testing learned dynamics and control.

KEY HYPOTHESIS:
PSI's cumsum operation is a learned integrator - this should excel at:
1. System identification (learning ODEs from data)
2. State estimation (learned Kalman filtering)
3. Trajectory prediction (extrapolating dynamics)

These tasks DON'T require associative recall - just integration!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIBlock(nn.Module):
    """PSI block with chunked cumsum for extrapolation-safe operation."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        gv = g * v

        # Pad to multiple of chunk_size for efficient parallel processing
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            gv_padded = F.pad(gv, (0, 0, 0, pad_len))
            g_padded = F.pad(g, (0, 0, 0, pad_len))
        else:
            gv_padded = gv
            g_padded = g

        padded_len = gv_padded.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape for parallel chunk processing
        gv_chunked = gv_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        g_chunked = g_padded.view(batch_size, num_chunks, self.chunk_size, dim)

        # Parallel cumsum across all chunks
        cumsum_v = torch.cumsum(gv_chunked, dim=2)
        cumsum_g = torch.cumsum(g_chunked, dim=2) + 1e-6

        # Running average within each chunk
        mem = cumsum_v / cumsum_g

        # Reshape back and remove padding
        mem = mem.view(batch_size, padded_len, dim)
        if pad_len > 0:
            mem = mem[:, :seq_len, :]

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=128, num_layers=4, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=128, num_layers=4, num_heads=4, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=128, num_layers=4, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.head(self.norm(h))


# =============================================================================
# Dynamical Systems Datasets
# =============================================================================

class SpringMassDataset(Dataset):
    """
    Simple harmonic oscillator: m*x'' + k*x = 0
    State: [x, v] where v = x'
    Dynamics: x' = v, v' = -k/m * x

    Task: Given noisy observations of x, predict x and v.
    This requires INTEGRATING velocity to get position.
    """
    def __init__(self, n_examples=2000, seq_len=50, dt=0.1, noise=0.1):
        self.examples = []

        for _ in range(n_examples):
            # Random initial conditions
            x0 = np.random.uniform(-2, 2)
            v0 = np.random.uniform(-2, 2)

            # Random spring constant
            k_over_m = np.random.uniform(0.5, 2.0)
            omega = np.sqrt(k_over_m)

            # Analytical solution
            t = np.arange(seq_len) * dt
            A = np.sqrt(x0**2 + (v0/omega)**2)
            phi = np.arctan2(-v0/omega, x0)

            x = A * np.cos(omega * t + phi)
            v = -A * omega * np.sin(omega * t + phi)

            # Add noise to observations
            x_noisy = x + np.random.randn(seq_len) * noise

            # Input: noisy position observations
            # Target: true [x, v] state
            self.examples.append({
                'input': torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(-1),
                'target': torch.tensor(np.stack([x, v], axis=-1), dtype=torch.float32)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DampedOscillatorDataset(Dataset):
    """
    Damped harmonic oscillator: x'' + 2*zeta*omega*x' + omega^2*x = 0

    Task: Predict future trajectory given initial observations.
    """
    def __init__(self, n_examples=2000, seq_len=50, dt=0.1, noise=0.05):
        self.examples = []

        for _ in range(n_examples):
            x0 = np.random.uniform(-2, 2)
            v0 = np.random.uniform(-2, 2)

            omega = np.random.uniform(1.0, 3.0)
            zeta = np.random.uniform(0.1, 0.5)  # Underdamped

            omega_d = omega * np.sqrt(1 - zeta**2)

            t = np.arange(seq_len) * dt

            # Underdamped solution
            A = np.sqrt(x0**2 + ((v0 + zeta*omega*x0)/omega_d)**2)
            phi = np.arctan2(x0*omega_d, v0 + zeta*omega*x0)

            x = A * np.exp(-zeta*omega*t) * np.sin(omega_d*t + phi)
            v = np.gradient(x, dt)

            x_noisy = x + np.random.randn(seq_len) * noise

            self.examples.append({
                'input': torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(-1),
                'target': torch.tensor(np.stack([x, v], axis=-1), dtype=torch.float32)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class LorenzDataset(Dataset):
    """
    Lorenz system - chaotic dynamics!
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = x*y - beta*z

    Task: Predict next state given current state sequence.
    """
    def __init__(self, n_examples=2000, seq_len=50, dt=0.01, noise=0.1):
        self.examples = []

        sigma = 10.0
        rho = 28.0
        beta = 8/3

        for _ in range(n_examples):
            # Random initial conditions near attractor
            state = np.array([
                np.random.uniform(-20, 20),
                np.random.uniform(-30, 30),
                np.random.uniform(5, 45)
            ])

            trajectory = [state.copy()]

            # Integrate using RK4
            for _ in range(seq_len - 1):
                def lorenz(s):
                    return np.array([
                        sigma * (s[1] - s[0]),
                        s[0] * (rho - s[2]) - s[1],
                        s[0] * s[1] - beta * s[2]
                    ])

                k1 = lorenz(state)
                k2 = lorenz(state + dt/2 * k1)
                k3 = lorenz(state + dt/2 * k2)
                k4 = lorenz(state + dt * k3)

                state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(state.copy())

            trajectory = np.array(trajectory)

            # Normalize
            trajectory = (trajectory - trajectory.mean(axis=0)) / (trajectory.std(axis=0) + 1e-6)

            # Add noise to input
            noisy_input = trajectory + np.random.randn(*trajectory.shape) * noise

            self.examples.append({
                'input': torch.tensor(noisy_input, dtype=torch.float32),
                'target': torch.tensor(trajectory, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PendulumDataset(Dataset):
    """
    Nonlinear pendulum: theta'' + (g/L)*sin(theta) = 0

    Task: Predict trajectory given initial observations.
    """
    def __init__(self, n_examples=2000, seq_len=50, dt=0.05, noise=0.05):
        self.examples = []

        g_over_L = 9.81  # Normalized

        for _ in range(n_examples):
            theta0 = np.random.uniform(-np.pi/2, np.pi/2)
            omega0 = np.random.uniform(-2, 2)

            state = np.array([theta0, omega0])
            trajectory = [state.copy()]

            for _ in range(seq_len - 1):
                # RK4 integration
                def pendulum(s):
                    return np.array([s[1], -g_over_L * np.sin(s[0])])

                k1 = pendulum(state)
                k2 = pendulum(state + dt/2 * k1)
                k3 = pendulum(state + dt/2 * k2)
                k4 = pendulum(state + dt * k3)

                state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(state.copy())

            trajectory = np.array(trajectory)

            # Add noise
            noisy_input = trajectory + np.random.randn(*trajectory.shape) * noise

            self.examples.append({
                'input': torch.tensor(noisy_input, dtype=torch.float32),
                'target': torch.tensor(trajectory, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class VanDerPolDataset(Dataset):
    """
    Van der Pol oscillator - nonlinear with limit cycle
    x'' - mu*(1 - x^2)*x' + x = 0

    Classic example in nonlinear dynamics!
    """
    def __init__(self, n_examples=2000, seq_len=100, dt=0.05, noise=0.05):
        self.examples = []

        for _ in range(n_examples):
            mu = np.random.uniform(0.5, 2.0)
            x0 = np.random.uniform(-3, 3)
            v0 = np.random.uniform(-3, 3)

            state = np.array([x0, v0])
            trajectory = [state.copy()]

            for _ in range(seq_len - 1):
                def vdp(s):
                    return np.array([s[1], mu*(1 - s[0]**2)*s[1] - s[0]])

                k1 = vdp(state)
                k2 = vdp(state + dt/2 * k1)
                k3 = vdp(state + dt/2 * k2)
                k4 = vdp(state + dt * k3)

                state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(state.copy())

            trajectory = np.array(trajectory)

            # Normalize
            trajectory = (trajectory - trajectory.mean(axis=0)) / (trajectory.std(axis=0) + 1e-6)

            noisy_input = trajectory + np.random.randn(*trajectory.shape) * noise

            self.examples.append({
                'input': torch.tensor(noisy_input, dtype=torch.float32),
                'target': torch.tensor(trajectory, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DoubleIntegratorControlDataset(Dataset):
    """
    Double integrator: x'' = u (control input)
    State: [x, v], Control: u

    Task: Given desired trajectory, output control signal.
    This is INVERSE dynamics - requires differentiation!

    But we'll frame it as: given state trajectory, predict what control was applied.
    """
    def __init__(self, n_examples=2000, seq_len=50, dt=0.1, noise=0.05):
        self.examples = []

        for _ in range(n_examples):
            # Generate random smooth control signal
            # Use sum of sinusoids for smooth control
            n_freqs = np.random.randint(2, 5)
            freqs = np.random.uniform(0.1, 1.0, n_freqs)
            amps = np.random.uniform(-1, 1, n_freqs)
            phases = np.random.uniform(0, 2*np.pi, n_freqs)

            t = np.arange(seq_len) * dt
            u = np.sum([a * np.sin(f * t + p) for a, f, p in zip(amps, freqs, phases)], axis=0)

            # Integrate to get velocity and position
            v = np.cumsum(u) * dt
            x = np.cumsum(v) * dt

            # Stack state
            state = np.stack([x, v], axis=-1)

            # Add noise
            noisy_state = state + np.random.randn(*state.shape) * noise

            self.examples.append({
                'input': torch.tensor(noisy_state, dtype=torch.float32),
                'target': torch.tensor(u, dtype=torch.float32).unsqueeze(-1)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Training and Evaluation
# =============================================================================

def collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {'input': inputs, 'target': targets}


def train_model(model, train_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = F.mse_loss(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader):
    model.eval()
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            mse = F.mse_loss(outputs, targets, reduction='sum')

            total_mse += mse.item()
            total_samples += targets.numel()

    return total_mse / total_samples


def evaluate_multistep(model, test_loader, n_context=10, n_predict=20):
    """
    Evaluate multi-step prediction:
    Given first n_context steps, autoregressively predict n_predict steps.

    Note: For tasks where input_dim != output_dim, we use teacher forcing
    since autoregressive doesn't make sense (e.g., state estimation tasks).
    """
    model.eval()
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            batch_size, seq_len, input_dim = inputs.shape
            output_dim = targets.shape[-1]

            if seq_len < n_context + n_predict:
                continue

            # If input and output dims match, we can do autoregressive
            if input_dim == output_dim:
                # Start with context
                current = inputs[:, :n_context, :]
                predictions = []

                # Autoregressive prediction
                for i in range(n_predict):
                    output = model(current)
                    next_pred = output[:, -1:, :]  # Take last prediction
                    predictions.append(next_pred)
                    current = torch.cat([current, next_pred], dim=1)

                predictions = torch.cat(predictions, dim=1)
                ground_truth = targets[:, n_context:n_context+n_predict, :]
            else:
                # Teacher forcing - just evaluate on later timesteps
                output = model(inputs)
                predictions = output[:, n_context:n_context+n_predict, :]
                ground_truth = targets[:, n_context:n_context+n_predict, :]

            mse = F.mse_loss(predictions, ground_truth, reduction='sum')
            total_mse += mse.item()
            total_samples += ground_truth.numel()

    return total_mse / total_samples if total_samples > 0 else float('inf')


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("CONTROL SYSTEMS BENCHMARK: Testing PSI on Dynamics")
    print("=" * 80)
    print(f"Device: {device}")
    print()
    print("Hypothesis: PSI's cumsum = learned integrator -> should excel at dynamics")
    print()

    dim = 128
    num_layers = 4

    # Define tasks
    tasks = [
        ('Spring-Mass (state est.)', SpringMassDataset, {'seq_len': 50, 'noise': 0.1}, 1, 2, 100),
        ('Damped Oscillator', DampedOscillatorDataset, {'seq_len': 50, 'noise': 0.05}, 1, 2, 100),
        ('Pendulum (nonlinear)', PendulumDataset, {'seq_len': 50, 'noise': 0.05}, 2, 2, 100),
        ('Van der Pol', VanDerPolDataset, {'seq_len': 100, 'noise': 0.05}, 2, 2, 100),
        ('Lorenz (chaotic)', LorenzDataset, {'seq_len': 50, 'noise': 0.1}, 3, 3, 100),
        ('Double Integrator Ctrl', DoubleIntegratorControlDataset, {'seq_len': 50, 'noise': 0.05}, 2, 1, 100),
    ]

    results = {}

    for task_name, dataset_cls, dataset_kwargs, input_dim, output_dim, epochs in tasks:
        print("=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        train_data = dataset_cls(n_examples=3000, **dataset_kwargs)
        test_data = dataset_cls(n_examples=500, **dataset_kwargs)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

        task_results = {}

        for name, model_fn in [
            ('PSI', lambda: PSIModel(input_dim, output_dim, dim, num_layers)),
            ('Transformer', lambda: TransformerModel(input_dim, output_dim, dim, num_layers)),
            ('LSTM', lambda: LSTMModel(input_dim, output_dim, dim, num_layers)),
        ]:
            model = model_fn().to(device)

            start_time = time.time()
            train_model(model, train_loader, epochs=epochs)
            train_time = time.time() - start_time

            mse = evaluate_model(model, test_loader)

            # Also evaluate multi-step prediction
            multistep_mse = evaluate_multistep(model, test_loader, n_context=10, n_predict=20)

            task_results[name] = {
                'mse': mse,
                'multistep_mse': multistep_mse,
                'time': train_time
            }
            print(f"  {name}: MSE={mse:.6f}, Multistep={multistep_mse:.6f} ({train_time:.1f}s)")

        results[task_name] = task_results
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Control Systems Benchmark")
    print("=" * 80)
    print()

    print("SINGLE-STEP PREDICTION (MSE, lower is better):")
    print(f"{'Task':<28} {'PSI':>12} {'Transformer':>12} {'LSTM':>12} {'Winner':>10}")
    print("-" * 80)

    for task_name, task_results in results.items():
        psi = task_results['PSI']['mse']
        trans = task_results['Transformer']['mse']
        lstm = task_results['LSTM']['mse']

        best = min(psi, trans, lstm)
        margin = best * 0.05  # 5% margin

        if psi <= best + margin and trans > best + margin and lstm > best + margin:
            winner = 'PSI'
        elif trans <= best + margin and psi > best + margin and lstm > best + margin:
            winner = 'Trans'
        elif lstm <= best + margin and psi > best + margin and trans > best + margin:
            winner = 'LSTM'
        else:
            winner = 'Tie'

        print(f"{task_name:<28} {psi:>12.6f} {trans:>12.6f} {lstm:>12.6f} {winner:>10}")

    print()
    print("MULTI-STEP PREDICTION (MSE over 20 steps, lower is better):")
    print(f"{'Task':<28} {'PSI':>12} {'Transformer':>12} {'LSTM':>12} {'Winner':>10}")
    print("-" * 80)

    psi_wins = 0
    trans_wins = 0
    lstm_wins = 0

    for task_name, task_results in results.items():
        psi = task_results['PSI']['multistep_mse']
        trans = task_results['Transformer']['multistep_mse']
        lstm = task_results['LSTM']['multistep_mse']

        best = min(psi, trans, lstm)
        margin = best * 0.1  # 10% margin for multi-step

        if psi <= best + margin and trans > best + margin and lstm > best + margin:
            winner = 'PSI'
            psi_wins += 1
        elif trans <= best + margin and psi > best + margin and lstm > best + margin:
            winner = 'Trans'
            trans_wins += 1
        elif lstm <= best + margin and psi > best + margin and trans > best + margin:
            winner = 'LSTM'
            lstm_wins += 1
        else:
            winner = 'Tie'

        print(f"{task_name:<28} {psi:>12.6f} {trans:>12.6f} {lstm:>12.6f} {winner:>10}")

    print("-" * 80)
    print(f"MULTI-STEP WINS: PSI={psi_wins}, Transformer={trans_wins}, LSTM={lstm_wins}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (task_name, task_results) in enumerate(results.items()):
        ax = axes[idx]

        models = ['PSI', 'Transformer', 'LSTM']
        single_step = [task_results[m]['mse'] for m in models]
        multi_step = [task_results[m]['multistep_mse'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, single_step, width, label='Single-step', color='steelblue')
        ax.bar(x + width/2, multi_step, width, label='Multi-step (20)', color='coral')

        ax.set_ylabel('MSE')
        ax.set_title(task_name)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('control_systems_benchmark.png', dpi=150)
    plt.close()
    print("\nSaved control_systems_benchmark.png")

    # Key insight
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
    These tasks test learned dynamics - NO associative recall needed!

    If PSI excels here, it suggests the architecture is well-suited for:
    - Robotics (state estimation, trajectory prediction)
    - Physics simulation (learned integrators)
    - Control systems (model predictive control)
    - Signal processing (filtering, denoising)

    Key question: Does PSI's cumsum give it an edge in MULTI-STEP prediction?
    Multi-step prediction requires stable integration over time - exactly what
    PSI's architecture should be good at.
    """)


if __name__ == "__main__":
    main()
