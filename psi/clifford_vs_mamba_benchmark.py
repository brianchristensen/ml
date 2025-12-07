"""
Comprehensive benchmark: Orthogonal Clifford PSI vs Mamba

Tests on:
1. Multi-query associative recall (where Clifford showed 28.1%)
2. Lorenz dynamics prediction (PSI's strength domain)
3. Scaling with number of pairs/sequence length

Mamba implementation verified against:
- Original paper: https://arxiv.org/abs/2312.00752
- Visual guide: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ============================================================================
# Import Orthogonal Clifford PSI from canonical source
# ============================================================================

from clifford_memory_v2 import OrthogonalModel as OrthogonalCliffordModel
from clifford_memory_v2 import OrthogonalBivectorBlock


class OrthogonalCliffordDynamics(nn.Module):
    """Dynamics predictor with Orthogonal Clifford blocks."""
    def __init__(self, state_dim, dim=128, n_layers=4, n_orthogonal_sets=4, planes_per_set=16,
                 use_positional_plane=False, max_len=512, pos_planes=16):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, dim)
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_orthogonal_sets, planes_per_set,
                                   use_positional_plane, max_len, pos_planes)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, state_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.output_proj(h[:, -1, :])

    def predict_trajectory(self, context, num_steps):
        predictions = []
        current = context.clone()
        with torch.no_grad():
            for _ in range(num_steps):
                pred = self.forward(current)
                predictions.append(pred)
                current = torch.cat([current[:, 1:, :], pred.unsqueeze(1)], dim=1)
        return torch.stack(predictions, dim=1)


# ============================================================================
# Mamba (S6 SSM)
# ============================================================================

class MambaBlock(nn.Module):
    """Simplified Mamba block with selective state space."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = dim * expand

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )

        # dt, B, C projections from x_inner
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # A is diagonal, initialized to be negative (for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # [B, L, d_inner] each

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_inner = F.silu(x_conv)

        # Project to get dt, B, C
        delta = F.softplus(self.dt_proj(x_inner))  # [B, L, d_inner]
        B_t = self.B_proj(x_inner)  # [B, L, d_state]
        C_t = self.C_proj(x_inner)  # [B, L, d_state]

        A = -torch.exp(self.A_log)  # [d_state]

        # SSM recurrence
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            # Discretize A and B
            dt = delta[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            dA = torch.exp(dt * A)  # [B, d_inner, d_state]
            dB = dt * B_t[:, t, :].unsqueeze(1)  # [B, 1, d_state] -> broadcast

            h = dA * h + dB * x_inner[:, t, :].unsqueeze(-1)  # [B, d_inner, d_state]
            y = (h * C_t[:, t, :].unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]

        y = y + x_inner * self.D
        y = y * F.silu(z)
        return x + self.out_proj(y)


class MambaModel(nn.Module):
    """Language model with Mamba blocks."""
    def __init__(self, vocab_size, dim=64, n_layers=2, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


class MambaDynamics(nn.Module):
    """Dynamics predictor with Mamba blocks."""
    def __init__(self, state_dim, dim=128, n_layers=4, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, state_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.output_proj(h[:, -1, :])

    def predict_trajectory(self, context, num_steps):
        predictions = []
        current = context.clone()
        with torch.no_grad():
            for _ in range(num_steps):
                pred = self.forward(current)
                predictions.append(pred)
                current = torch.cat([current[:, 1:, :], pred.unsqueeze(1)], dim=1)
        return torch.stack(predictions, dim=1)


# ============================================================================
# Benchmark Tasks
# ============================================================================

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    """Generate multi-query associative recall task."""
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + n_queries * 2
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    query_positions = []

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]

        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2

        query_indices = np.random.choice(n_pairs, n_queries, replace=False)
        for qi, query_idx in enumerate(query_indices):
            data[b, pos] = QUERY_TOKEN
            pos += 1
            query_k, query_v = pairs[query_idx]
            data[b, pos] = query_k
            targets[b, pos] = query_v
            if b == 0:
                query_positions.append(pos)
            pos += 1

    return data, targets, query_positions


def lorenz_step(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx * dt, y + dy * dt, z + dz * dt


def generate_lorenz_trajectory(length=200, dt=0.01):
    x = np.random.uniform(-15, 15)
    y = np.random.uniform(-20, 20)
    z = np.random.uniform(5, 40)
    trajectory = np.zeros((length, 3), dtype=np.float32)
    for i in range(length):
        trajectory[i] = [x, y, z]
        x, y, z = lorenz_step(x, y, z, dt=dt)
    return trajectory


def generate_lorenz_dataset(num_trajectories=1000, trajectory_length=200, dt=0.01):
    trajectories = np.zeros((num_trajectories, trajectory_length, 3), dtype=np.float32)
    for i in range(num_trajectories):
        trajectories[i] = generate_lorenz_trajectory(trajectory_length, dt=dt)
    mean = trajectories.mean(axis=(0, 1))
    std = trajectories.std(axis=(0, 1))
    trajectories = (trajectories - mean) / (std + 1e-8)
    return trajectories


# ============================================================================
# Training Functions
# ============================================================================

def train_recall(model, vocab_size, n_pairs, n_queries, epochs=300, lr=1e-3):
    """Train on associative recall and return best accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        data, targets, positions = generate_multi_query_recall(64, n_pairs, n_queries, vocab_size, device)
        logits = model(data)
        loss = sum(criterion(logits[:, pos, :vocab_size], targets[:, pos]) for pos in positions) / len(positions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item()
                             for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc

    return best_acc


def train_dynamics(model, train_data, val_data, context_len=20, epochs=20, lr=1e-3):
    """Train on Lorenz dynamics and return validation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        # Sample training batch
        indices = np.random.choice(len(train_data), 256, replace=False)
        total_loss = 0

        for idx in indices:
            traj = train_data[idx]
            pos = np.random.randint(0, len(traj) - context_len - 1)
            context = torch.tensor(traj[pos:pos+context_len], dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor(traj[pos+context_len], dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(context)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx in range(min(100, len(val_data))):
                traj = val_data[idx]
                pos = np.random.randint(0, len(traj) - context_len - 1)
                context = torch.tensor(traj[pos:pos+context_len], dtype=torch.float32).unsqueeze(0).to(device)
                target = torch.tensor(traj[pos+context_len], dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(context)
                val_loss += criterion(pred, target).item()

        val_loss /= 100
        if val_loss < best_val:
            best_val = val_loss

        if (epoch + 1) % 5 == 0:
            print(f'    Epoch {epoch+1}: train={total_loss/256:.6f}, val={val_loss:.6f}')

    return best_val


def eval_multistep(model, val_data, context_len=20, num_steps=50, num_samples=100):
    """Evaluate multi-step rollout error."""
    model.eval()
    step_errors = np.zeros(num_steps)

    with torch.no_grad():
        for _ in range(num_samples):
            idx = np.random.randint(0, len(val_data))
            traj = val_data[idx]
            pos = np.random.randint(0, len(traj) - context_len - num_steps)

            context = torch.tensor(traj[pos:pos+context_len], dtype=torch.float32).unsqueeze(0).to(device)
            gt = traj[pos+context_len:pos+context_len+num_steps]

            preds = model.predict_trajectory(context, num_steps)
            preds = preds[0].cpu().numpy()

            for step in range(num_steps):
                step_errors[step] += np.mean((preds[step] - gt[step]) ** 2)

    return step_errors / num_samples


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print('=' * 70)
    print('BENCHMARK: Orthogonal Clifford PSI vs Mamba')
    print('=' * 70)
    print()

    # =========================================
    # Task 1: Multi-Query Associative Recall
    # =========================================
    print('=' * 70)
    print('TASK 1: Multi-Query Associative Recall')
    print('=' * 70)
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2
    epochs = 400

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    # Orthogonal Clifford (baseline - no positional plane)
    print('-' * 70)
    print('Orthogonal Clifford PSI (4 sets x 16 planes)')
    print('-' * 70)
    clifford = OrthogonalCliffordModel(vocab_size + 1, dim, n_layers,
                                       n_orthogonal_sets=4, planes_per_set=16).to(device)
    cliff_params = sum(p.numel() for p in clifford.parameters())
    print(f'Parameters: {cliff_params:,}')

    start = time.time()
    clifford_recall = train_recall(clifford, vocab_size, n_pairs, n_queries, epochs)
    cliff_time = time.time() - start
    print(f'Best accuracy: {clifford_recall:.1f}% (trained in {cliff_time:.1f}s)')
    print()

    # Orthogonal Clifford + Positional Plane (our best variant)
    print('-' * 70)
    print('Clifford PSI + Positional Plane (4 sets x 16 planes + 16 pos planes)')
    print('-' * 70)
    clifford_pos = OrthogonalCliffordModel(vocab_size + 1, dim, n_layers,
                                           n_orthogonal_sets=4, planes_per_set=16,
                                           use_positional_plane=True, pos_planes=16).to(device)
    cliff_pos_params = sum(p.numel() for p in clifford_pos.parameters())
    print(f'Parameters: {cliff_pos_params:,}')

    start = time.time()
    clifford_pos_recall = train_recall(clifford_pos, vocab_size, n_pairs, n_queries, epochs)
    cliff_pos_time = time.time() - start
    print(f'Best accuracy: {clifford_pos_recall:.1f}% (trained in {cliff_pos_time:.1f}s)')
    print()

    # Mamba
    print('-' * 70)
    print('Mamba (d_state=16)')
    print('-' * 70)
    mamba = MambaModel(vocab_size + 1, dim, n_layers, d_state=16).to(device)
    mamba_params = sum(p.numel() for p in mamba.parameters())
    print(f'Parameters: {mamba_params:,}')

    start = time.time()
    mamba_recall = train_recall(mamba, vocab_size, n_pairs, n_queries, epochs)
    mamba_time = time.time() - start
    print(f'Best accuracy: {mamba_recall:.1f}% (trained in {mamba_time:.1f}s)')
    print()

    # Scaling test with more pairs
    print('-' * 70)
    print('Scaling: More pairs (harder recall)')
    print('-' * 70)

    scaling_results = []
    for test_pairs in [4, 8, 12, 16]:
        test_queries = min(4, test_pairs)

        # Clifford + PosPlane (best variant)
        cliff = OrthogonalCliffordModel(vocab_size + 1, dim, n_layers,
                                        n_orthogonal_sets=4, planes_per_set=16,
                                        use_positional_plane=True, pos_planes=16).to(device)
        cliff_acc = train_recall(cliff, vocab_size, test_pairs, test_queries, 300)

        # Mamba
        mam = MambaModel(vocab_size + 1, dim, n_layers, d_state=16).to(device)
        mam_acc = train_recall(mam, vocab_size, test_pairs, test_queries, 300)

        scaling_results.append((test_pairs, cliff_acc, mam_acc))
        print(f'  {test_pairs} pairs: Clifford+Pos={cliff_acc:.1f}%, Mamba={mam_acc:.1f}%')

    print()

    # =========================================
    # Task 2: Lorenz Dynamics
    # =========================================
    print('=' * 70)
    print('TASK 2: Lorenz Dynamics Prediction')
    print('=' * 70)
    print()

    print('Generating Lorenz dataset...')
    train_data = generate_lorenz_dataset(2000, 200)
    val_data = generate_lorenz_dataset(200, 200)

    context_len = 20
    dyn_dim = 128
    dyn_layers = 4
    dyn_epochs = 15

    # Orthogonal Clifford Dynamics + Positional Plane
    print('-' * 70)
    print('Orthogonal Clifford Dynamics + Positional Plane')
    print('-' * 70)
    cliff_dyn = OrthogonalCliffordDynamics(3, dyn_dim, dyn_layers,
                                           n_orthogonal_sets=4, planes_per_set=16,
                                           use_positional_plane=True, pos_planes=16).to(device)
    cliff_dyn_params = sum(p.numel() for p in cliff_dyn.parameters())
    print(f'Parameters: {cliff_dyn_params:,}')

    cliff_val = train_dynamics(cliff_dyn, train_data, val_data, context_len, dyn_epochs)
    cliff_multistep = eval_multistep(cliff_dyn, val_data, context_len, 50, 100)
    print(f'Final val loss: {cliff_val:.6f}')
    print(f'Multi-step MSE: step1={cliff_multistep[0]:.4f}, step10={cliff_multistep[9]:.4f}, step50={cliff_multistep[49]:.4f}')
    print()

    # Mamba Dynamics
    print('-' * 70)
    print('Mamba Dynamics')
    print('-' * 70)
    mamba_dyn = MambaDynamics(3, dyn_dim, dyn_layers, d_state=16).to(device)
    mamba_dyn_params = sum(p.numel() for p in mamba_dyn.parameters())
    print(f'Parameters: {mamba_dyn_params:,}')

    mamba_val = train_dynamics(mamba_dyn, train_data, val_data, context_len, dyn_epochs)
    mamba_multistep = eval_multistep(mamba_dyn, val_data, context_len, 50, 100)
    print(f'Final val loss: {mamba_val:.6f}')
    print(f'Multi-step MSE: step1={mamba_multistep[0]:.4f}, step10={mamba_multistep[9]:.4f}, step50={mamba_multistep[49]:.4f}')
    print()

    # =========================================
    # Summary
    # =========================================
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()

    print('ASSOCIATIVE RECALL (higher is better):')
    print(f'  Clifford (baseline):   {clifford_recall:.1f}%')
    print(f'  Clifford + PosPlane:   {clifford_pos_recall:.1f}%')
    print(f'  Mamba:                 {mamba_recall:.1f}%')
    diff = clifford_pos_recall - mamba_recall
    if diff > 0:
        print(f'  -> Clifford+Pos wins by {diff:.1f}%')
    else:
        print(f'  -> Mamba wins by {-diff:.1f}%')
    pos_improvement = clifford_pos_recall - clifford_recall
    print(f'  -> Positional plane improvement: +{pos_improvement:.1f}%')
    print()

    print('LORENZ DYNAMICS (lower MSE is better):')
    print(f'  Single-step val loss:')
    print(f'    Clifford: {cliff_val:.6f}')
    print(f'    Mamba:    {mamba_val:.6f}')
    print()
    print(f'  Multi-step rollout (step 50):')
    print(f'    Clifford: {cliff_multistep[49]:.4f}')
    print(f'    Mamba:    {mamba_multistep[49]:.4f}')

    if cliff_multistep[49] < mamba_multistep[49]:
        print(f'  -> Clifford wins on long-horizon dynamics')
    else:
        print(f'  -> Mamba wins on long-horizon dynamics')

    print()
    print('SCALING (pairs -> accuracy):')
    print(f'{"Pairs":>6} {"Clifford":>10} {"Mamba":>10} {"Winner":>10}')
    print('-' * 40)
    for pairs, c_acc, m_acc in scaling_results:
        winner = "Clifford" if c_acc > m_acc else "Mamba"
        print(f'{pairs:>6} {c_acc:>9.1f}% {m_acc:>9.1f}% {winner:>10}')

    print()
    print('PARAMETER EFFICIENCY:')
    print(f'  Recall - Clifford: {cliff_params:,} params')
    print(f'  Recall - Mamba:    {mamba_params:,} params')
    print(f'  Dynamics - Clifford: {cliff_dyn_params:,} params')
    print(f'  Dynamics - Mamba:    {mamba_dyn_params:,} params')


if __name__ == "__main__":
    main()
