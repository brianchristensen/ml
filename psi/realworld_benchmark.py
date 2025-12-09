"""
Real-World Application Benchmark for Clifford PSI vs Mamba (V2)

Redesigned with INTERLEAVED multi-source time-series approach.
Key insight: All signals from all sources get sorted by time and
interleaved into sequences - this works well for phasor models.

Tests scenarios where Clifford's O(n) + associative recall shine:
1. Multi-sensor IoT with interleaved readings + query
2. Multi-agent navigation with interleaved observations
3. Long-horizon control with context recall

All tests emphasize: O(n) scaling + associative recall + dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# Import models
# ============================================================================

from clifford_memory import OrthogonalModel as CliffordModel
from clifford_memory import OrthogonalBivectorBlock

# Mamba block
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = dim * expand

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_inner = F.silu(x_conv)

        delta = F.softplus(self.dt_proj(x_inner))  # [B, L, d_inner]
        B_t = self.B_proj(x_inner)  # [B, L, d_state]
        C_t = self.C_proj(x_inner)  # [B, L, d_state]
        A = -torch.exp(self.A_log)  # [d_state]

        # Vectorized SSM scan using parallel associative scan approximation
        # For simplicity, use cumsum-based approximation (faster than loop)
        dt_expanded = delta.unsqueeze(-1)  # [B, L, d_inner, 1]
        dA = torch.exp(dt_expanded * A)  # [B, L, d_inner, d_state]
        dB = dt_expanded * B_t.unsqueeze(2)  # [B, L, d_inner, d_state]

        # Input contribution
        input_contrib = dB * x_inner.unsqueeze(-1)  # [B, L, d_inner, d_state]

        # Approximate scan with cumsum (works well for small dt)
        # This is faster than sequential loop
        log_dA = dt_expanded * A  # [B, L, d_inner, d_state]
        cumsum_log_dA = torch.cumsum(log_dA, dim=1)
        decay_factors = torch.exp(cumsum_log_dA)

        # Scale inputs by inverse decay, cumsum, then rescale
        scaled_input = input_contrib * torch.exp(-cumsum_log_dA)
        h_cumsum = torch.cumsum(scaled_input, dim=1)
        h = h_cumsum * decay_factors  # [B, L, d_inner, d_state]

        # Output
        y = (h * C_t.unsqueeze(2)).sum(-1)  # [B, L, d_inner]
        y = y + x_inner * self.D
        y = y * F.silu(z)
        return x + self.out_proj(y)


class MambaModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, d_state=16):
        super().__init__()
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


# ============================================================================
# Task 1: Multi-Sensor IoT with Interleaved Readings
# ============================================================================
# Key insight: All sensor readings are interleaved by timestamp
# Each reading is: (sensor_id, value, optional_label)
# Query: Given sensor_id + value pattern, recall the label

def generate_interleaved_iot(batch_size, n_sensors, n_timesteps, vocab_size, device):
    """
    Generate interleaved IoT sensor data - FIXED VERSION.

    Structure: Each reading is a TRIPLET [sensor_id, value, label]
    - Label is an anomaly type (0-7) or 8 for "normal/no anomaly"
    - Query: [QUERY] [sensor_id] [value] -> predict label

    This is a proper associative recall task where the binding IS in the sequence.
    """
    QUERY_TOKEN = vocab_size
    n_labels = 8
    NO_LABEL = n_labels  # "normal" reading

    # Each timestep is 3 tokens: sensor, value, label
    seq_len = n_timesteps * 3 + 3  # +3 for query
    value_vocab = vocab_size - n_sensors - n_labels - 1  # -1 for NO_LABEL token

    # Generate readings
    sensors = torch.randint(0, n_sensors, (batch_size, n_timesteps), device=device)
    values = torch.randint(0, value_vocab, (batch_size, n_timesteps), device=device)

    # 20% of readings have anomaly labels, 80% are "normal"
    has_anomaly = torch.rand(batch_size, n_timesteps, device=device) < 0.2
    anomaly_types = torch.randint(0, n_labels, (batch_size, n_timesteps), device=device)
    labels = torch.where(has_anomaly, anomaly_types, torch.full_like(anomaly_types, NO_LABEL))

    # Build data tensor with triplets
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for t in range(n_timesteps):
        data[:, t * 3] = sensors[:, t]
        data[:, t * 3 + 1] = n_sensors + values[:, t]
        data[:, t * 3 + 2] = n_sensors + value_vocab + labels[:, t]

    # Query: pick a random ANOMALY timestep (not normal)
    query_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    valid_target = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for b in range(batch_size):
        anomaly_positions = has_anomaly[b].nonzero(as_tuple=True)[0]
        if len(anomaly_positions) > 0:
            query_idx[b] = anomaly_positions[torch.randint(len(anomaly_positions), (1,))]
            valid_target[b] = True
        else:
            # No anomalies - pick any position, target will be masked
            query_idx[b] = 0

    query_sensors = sensors.gather(1, query_idx.unsqueeze(1)).squeeze(1)
    query_values = values.gather(1, query_idx.unsqueeze(1)).squeeze(1)
    query_labels = labels.gather(1, query_idx.unsqueeze(1)).squeeze(1)

    # Set query tokens
    query_pos = n_timesteps * 3
    data[:, query_pos] = QUERY_TOKEN
    data[:, query_pos + 1] = query_sensors
    data[:, query_pos + 2] = n_sensors + query_values

    # Target is the anomaly type (0-7), masked to 0 if no valid anomaly
    targets = query_labels * valid_target.long()

    return data, targets, n_labels


def test_interleaved_iot():
    print('=' * 70)
    print('TASK 1: Interleaved Multi-Sensor IoT')
    print('=' * 70)
    print('Sensors report in interleaved time order.')
    print('Must recall event labels from earlier readings.')
    print()

    vocab_size = 64
    n_sensors = 8
    dim = 96
    n_layers = 2
    epochs = 400

    results = {}

    for n_timesteps in [20, 50, 100]:  # Reduced for faster testing with 3x tokens
        print(f'\n--- {n_timesteps} timesteps (seq_len={n_timesteps*3+3}) ---')

        _, _, n_labels = generate_interleaved_iot(1, n_sensors, n_timesteps, vocab_size, device)

        # Clifford
        cliff = CliffordModel(vocab_size + 1, dim, n_layers,
                             n_orthogonal_sets=4, planes_per_set=16,
                             use_positional_plane=True, pos_planes=16).to(device)

        optimizer = torch.optim.AdamW(cliff.parameters(), lr=1e-3, weight_decay=0.01)
        best_cliff = 0

        for epoch in range(epochs):
            cliff.train()
            data, targets, _ = generate_interleaved_iot(32, n_sensors, n_timesteps, vocab_size, device)
            logits = cliff(data)
            loss = F.cross_entropy(logits[:, -1, :n_labels], targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cliff.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                cliff.eval()
                with torch.no_grad():
                    data, targets, _ = generate_interleaved_iot(200, n_sensors, n_timesteps, vocab_size, device)
                    logits = cliff(data)
                    acc = (logits[:, -1, :n_labels].argmax(-1) == targets).float().mean().item() * 100
                    best_cliff = max(best_cliff, acc)

        # Mamba
        mamba = MambaModel(vocab_size + 1, dim, n_layers, d_state=16).to(device)
        optimizer = torch.optim.AdamW(mamba.parameters(), lr=1e-3, weight_decay=0.01)
        best_mamba = 0

        for epoch in range(epochs):
            mamba.train()
            data, targets, _ = generate_interleaved_iot(32, n_sensors, n_timesteps, vocab_size, device)
            logits = mamba(data)
            loss = F.cross_entropy(logits[:, -1, :n_labels], targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                mamba.eval()
                with torch.no_grad():
                    data, targets, _ = generate_interleaved_iot(200, n_sensors, n_timesteps, vocab_size, device)
                    logits = mamba(data)
                    acc = (logits[:, -1, :n_labels].argmax(-1) == targets).float().mean().item() * 100
                    best_mamba = max(best_mamba, acc)

        random_baseline = 100.0 / n_labels
        print(f'  Random: {random_baseline:.1f}%')
        print(f'  Clifford: {best_cliff:.1f}%')
        print(f'  Mamba: {best_mamba:.1f}%')
        results[n_timesteps] = (best_cliff, best_mamba)

    return results


# ============================================================================
# Task 2: Multi-Agent Navigation with Interleaved Observations
# ============================================================================
# Multiple agents exploring, observations interleaved by time
# Query: What did agent X see at location Y?

def generate_interleaved_navigation(batch_size, n_agents, n_steps_per_agent, grid_size, n_objects, device):
    """
    Multiple agents navigate and observe objects (vectorized).
    Observations interleaved by timestamp across all agents.
    """
    vocab_size = n_agents + grid_size * grid_size + n_objects + 1
    QUERY_TOKEN = vocab_size - 1

    total_steps = n_agents * n_steps_per_agent
    seq_len = total_steps * 3 + 3

    # Object positions (batch_size x n_objects)
    obj_positions = torch.randint(0, grid_size * grid_size, (batch_size, n_objects), device=device)

    # Agent initial positions
    agent_pos = torch.randint(0, grid_size * grid_size, (batch_size, n_agents), device=device)

    # Pre-generate all moves
    moves_x = torch.randint(-1, 2, (batch_size, n_steps_per_agent, n_agents), device=device)
    moves_y = torch.randint(-1, 2, (batch_size, n_steps_per_agent, n_agents), device=device)

    # Build observations
    all_agents = []
    all_positions = []
    all_objects = []
    observations_with_objects = []  # (batch_idx, step, agent, pos, obj)

    for step in range(n_steps_per_agent):
        for agent in range(n_agents):
            all_agents.append(torch.full((batch_size,), agent, dtype=torch.long, device=device))
            all_positions.append(agent_pos[:, agent].clone())

            # Check if at object position
            pos = agent_pos[:, agent]
            obj_seen = torch.full((batch_size,), n_objects, dtype=torch.long, device=device)
            for obj_idx in range(n_objects):
                match = (pos == obj_positions[:, obj_idx])
                obj_seen = torch.where(match, torch.full_like(obj_seen, obj_idx), obj_seen)

            all_objects.append(obj_seen)

            # Track observations with objects for query selection
            has_obj = obj_seen < n_objects

            # Move agents
            x = agent_pos[:, agent] % grid_size
            y = agent_pos[:, agent] // grid_size
            x = torch.clamp(x + moves_x[:, step, agent], 0, grid_size - 1)
            y = torch.clamp(y + moves_y[:, step, agent], 0, grid_size - 1)
            agent_pos[:, agent] = y * grid_size + x

    # Stack observations
    all_agents = torch.stack(all_agents, dim=1)  # [B, total_steps]
    all_positions = torch.stack(all_positions, dim=1)  # [B, total_steps]
    all_objects = torch.stack(all_objects, dim=1)  # [B, total_steps]

    # Build data tensor
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    data[:, 0::3][:, :total_steps] = all_agents
    data[:, 1::3][:, :total_steps] = n_agents + all_positions
    data[:, 2::3][:, :total_steps] = n_agents + grid_size * grid_size + all_objects

    # Query: pick a random step where agent saw an object
    saw_object = all_objects < n_objects  # [B, total_steps]
    has_any = saw_object.any(dim=1)

    # For each batch, pick a random valid observation
    query_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    for b in range(batch_size):
        valid = saw_object[b].nonzero(as_tuple=True)[0]
        if len(valid) > 0:
            query_idx[b] = valid[torch.randint(len(valid), (1,))]

    query_agents = all_agents.gather(1, query_idx.unsqueeze(1)).squeeze(1)
    query_positions = all_positions.gather(1, query_idx.unsqueeze(1)).squeeze(1)
    query_objects = all_objects.gather(1, query_idx.unsqueeze(1)).squeeze(1)

    # Set query
    query_pos = total_steps * 3
    data[:, query_pos] = QUERY_TOKEN
    data[:, query_pos + 1] = query_agents
    data[:, query_pos + 2] = n_agents + query_positions

    targets = query_objects * has_any.long()

    return data, targets, vocab_size, n_objects


def test_interleaved_navigation():
    print('=' * 70)
    print('TASK 2: Interleaved Multi-Agent Navigation')
    print('=' * 70)
    print('Multiple agents explore, observations interleaved by time.')
    print('Query: What did agent X see at position Y?')
    print()

    n_agents = 4
    grid_size = 6
    n_objects = 8
    dim = 96
    n_layers = 2
    epochs = 500

    results = {}

    for n_steps in [10, 30, 50]:
        print(f'\n--- {n_steps} steps per agent ({n_agents * n_steps * 3} obs tokens) ---')

        _, _, vocab_size, _ = generate_interleaved_navigation(
            1, n_agents, n_steps, grid_size, n_objects, device)

        # Clifford
        cliff = CliffordModel(vocab_size, dim, n_layers,
                             n_orthogonal_sets=4, planes_per_set=16,
                             use_positional_plane=True, pos_planes=16).to(device)

        optimizer = torch.optim.AdamW(cliff.parameters(), lr=1e-3, weight_decay=0.01)
        best_cliff = 0

        for epoch in range(epochs):
            cliff.train()
            data, targets, _, _ = generate_interleaved_navigation(
                32, n_agents, n_steps, grid_size, n_objects, device)
            logits = cliff(data)
            loss = F.cross_entropy(logits[:, -1, :n_objects], targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cliff.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                cliff.eval()
                with torch.no_grad():
                    data, targets, _, _ = generate_interleaved_navigation(
                        200, n_agents, n_steps, grid_size, n_objects, device)
                    logits = cliff(data)
                    acc = (logits[:, -1, :n_objects].argmax(-1) == targets).float().mean().item() * 100
                    best_cliff = max(best_cliff, acc)

        # Mamba
        mamba = MambaModel(vocab_size, dim, n_layers, d_state=16).to(device)
        optimizer = torch.optim.AdamW(mamba.parameters(), lr=1e-3, weight_decay=0.01)
        best_mamba = 0

        for epoch in range(epochs):
            mamba.train()
            data, targets, _, _ = generate_interleaved_navigation(
                32, n_agents, n_steps, grid_size, n_objects, device)
            logits = mamba(data)
            loss = F.cross_entropy(logits[:, -1, :n_objects], targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                mamba.eval()
                with torch.no_grad():
                    data, targets, _, _ = generate_interleaved_navigation(
                        200, n_agents, n_steps, grid_size, n_objects, device)
                    logits = mamba(data)
                    acc = (logits[:, -1, :n_objects].argmax(-1) == targets).float().mean().item() * 100
                    best_mamba = max(best_mamba, acc)

        random_baseline = 100.0 / n_objects
        print(f'  Random: {random_baseline:.1f}%')
        print(f'  Clifford: {best_cliff:.1f}%')
        print(f'  Mamba: {best_mamba:.1f}%')
        results[n_steps] = (best_cliff, best_mamba)

    return results


# ============================================================================
# Task 3: Continuous Dynamics + Discrete Event Recall (Hybrid)
# ============================================================================
# Continuous time series with interleaved discrete events
# Must predict next state AND recall past event labels

class CliffordContinuousHybrid(nn.Module):
    """Clifford model for continuous + discrete hybrid input."""
    def __init__(self, input_dim, dim, n_layers, n_event_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.event_embed = nn.Embedding(n_event_classes + 1, dim)  # +1 for "no event"
        self.blocks = nn.ModuleList([
            OrthogonalBivectorBlock(dim, n_orthogonal_sets=4, planes_per_set=16,
                                   use_positional_plane=True, pos_planes=16)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.dynamics_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, input_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_event_classes))

    def forward(self, continuous, events):
        # continuous: [B, L, input_dim]
        # events: [B, L] - event class at each timestep (or n_classes for "no event")
        h = self.input_proj(continuous) + self.event_embed(events)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        dynamics_pred = self.dynamics_head(h)
        event_pred = self.event_head(h)
        return dynamics_pred, event_pred


class MambaContinuousHybrid(nn.Module):
    """Mamba model for continuous + discrete hybrid input."""
    def __init__(self, input_dim, dim, n_layers, n_event_classes, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.event_embed = nn.Embedding(n_event_classes + 1, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.dynamics_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, input_dim))
        self.event_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_event_classes))

    def forward(self, continuous, events):
        h = self.input_proj(continuous) + self.event_embed(events)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        dynamics_pred = self.dynamics_head(h)
        event_pred = self.event_head(h)
        return dynamics_pred, event_pred


def generate_hybrid_dynamics(batch_size, seq_len, n_event_classes, device):
    """
    Generate hybrid continuous dynamics + discrete events (vectorized).

    Continuous: Damped oscillator with perturbations when events occur
    Events: Discrete labels that affect the dynamics
    Task: Predict next state AND recall which event type caused current perturbation
    """
    state_dim = 4  # position, velocity for 2D oscillator
    omega = 0.5
    damping = 0.1
    dt = 0.1

    # Pre-generate all random values
    initial_states = torch.randn(batch_size, state_dim, device=device) * 0.5
    has_event = torch.rand(batch_size, seq_len, device=device) < 0.1
    event_classes = torch.randint(0, n_event_classes, (batch_size, seq_len), device=device)
    perturbations = torch.randn(batch_size, seq_len, state_dim, device=device)

    # Initialize tensors
    continuous = torch.zeros(batch_size, seq_len, state_dim, device=device)
    events = torch.full((batch_size, seq_len), n_event_classes, dtype=torch.long, device=device)
    dynamics_targets = torch.zeros(batch_size, seq_len, state_dim, device=device)
    event_targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Set events where they occur
    events = torch.where(has_event, event_classes, events)

    # Simulate dynamics (need sequential for state propagation, but vectorized across batch)
    state = initial_states.clone()
    last_event = torch.full((batch_size,), -100, dtype=torch.long, device=device)
    last_event_class = torch.zeros(batch_size, dtype=torch.long, device=device)

    for t in range(seq_len):
        continuous[:, t] = state

        # Apply perturbations where events occur
        event_mask = has_event[:, t].unsqueeze(1)  # [B, 1]
        scale = (event_classes[:, t].float() + 1) * 0.2  # [B]
        perturbation = perturbations[:, t] * scale.unsqueeze(1)  # [B, state_dim]
        state = state + perturbation * event_mask

        # Track last event for recall
        event_occurred = has_event[:, t]
        last_event = torch.where(event_occurred, torch.full_like(last_event, t), last_event)
        last_event_class = torch.where(event_occurred, event_classes[:, t], last_event_class)

        # Damped oscillator dynamics
        x = state[:, 0]
        vx = state[:, 1]
        y = state[:, 2]
        vy = state[:, 3]

        ax = -omega**2 * x - damping * vx
        ay = -omega**2 * y - damping * vy

        new_state = torch.stack([
            x + vx * dt,
            vx + ax * dt,
            y + vy * dt,
            vy + ay * dt
        ], dim=1)

        dynamics_targets[:, t] = new_state
        state = new_state

        # Event recall: if last event was within 10 steps
        recent = (t - last_event) < 10
        event_targets[:, t] = torch.where(recent, last_event_class, torch.zeros_like(last_event_class))

    return continuous, events, dynamics_targets, event_targets


def test_hybrid_dynamics():
    print('=' * 70)
    print('TASK 3: Hybrid Dynamics + Event Recall')
    print('=' * 70)
    print('Continuous dynamics with interleaved discrete events.')
    print('Must predict next state AND recall past event types.')
    print()

    state_dim = 4
    n_event_classes = 6
    dim = 96
    n_layers = 2
    epochs = 400

    results = {}

    for seq_len in [50, 100, 200]:
        print(f'\n--- Sequence length: {seq_len} ---')

        # Clifford
        cliff = CliffordContinuousHybrid(state_dim, dim, n_layers, n_event_classes).to(device)
        optimizer = torch.optim.AdamW(cliff.parameters(), lr=1e-3, weight_decay=0.01)
        best_cliff_dyn = float('inf')
        best_cliff_event = 0

        for epoch in range(epochs):
            cliff.train()
            continuous, events, dyn_targets, event_targets = generate_hybrid_dynamics(
                32, seq_len, n_event_classes, device)
            dyn_pred, event_pred = cliff(continuous, events)

            # Combined loss
            dyn_loss = F.mse_loss(dyn_pred[:, :-1], dyn_targets[:, :-1])
            event_loss = F.cross_entropy(
                event_pred[:, :-1].reshape(-1, n_event_classes),
                event_targets[:, :-1].reshape(-1)
            )
            loss = dyn_loss + 0.5 * event_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cliff.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                cliff.eval()
                with torch.no_grad():
                    continuous, events, dyn_targets, event_targets = generate_hybrid_dynamics(
                        200, seq_len, n_event_classes, device)
                    dyn_pred, event_pred = cliff(continuous, events)

                    dyn_mse = F.mse_loss(dyn_pred[:, :-1], dyn_targets[:, :-1]).item()
                    event_acc = (event_pred[:, :-1].argmax(-1) == event_targets[:, :-1]).float().mean().item() * 100

                    best_cliff_dyn = min(best_cliff_dyn, dyn_mse)
                    best_cliff_event = max(best_cliff_event, event_acc)

        # Mamba
        mamba = MambaContinuousHybrid(state_dim, dim, n_layers, n_event_classes).to(device)
        optimizer = torch.optim.AdamW(mamba.parameters(), lr=1e-3, weight_decay=0.01)
        best_mamba_dyn = float('inf')
        best_mamba_event = 0

        for epoch in range(epochs):
            mamba.train()
            continuous, events, dyn_targets, event_targets = generate_hybrid_dynamics(
                32, seq_len, n_event_classes, device)
            dyn_pred, event_pred = mamba(continuous, events)

            dyn_loss = F.mse_loss(dyn_pred[:, :-1], dyn_targets[:, :-1])
            event_loss = F.cross_entropy(
                event_pred[:, :-1].reshape(-1, n_event_classes),
                event_targets[:, :-1].reshape(-1)
            )
            loss = dyn_loss + 0.5 * event_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                mamba.eval()
                with torch.no_grad():
                    continuous, events, dyn_targets, event_targets = generate_hybrid_dynamics(
                        200, seq_len, n_event_classes, device)
                    dyn_pred, event_pred = mamba(continuous, events)

                    dyn_mse = F.mse_loss(dyn_pred[:, :-1], dyn_targets[:, :-1]).item()
                    event_acc = (event_pred[:, :-1].argmax(-1) == event_targets[:, :-1]).float().mean().item() * 100

                    best_mamba_dyn = min(best_mamba_dyn, dyn_mse)
                    best_mamba_event = max(best_mamba_event, event_acc)

        print(f'  Clifford: Dynamics MSE={best_cliff_dyn:.4f}, Event Recall={best_cliff_event:.1f}%')
        print(f'  Mamba:    Dynamics MSE={best_mamba_dyn:.4f}, Event Recall={best_mamba_event:.1f}%')
        results[seq_len] = ((best_cliff_dyn, best_cliff_event), (best_mamba_dyn, best_mamba_event))

    return results


# ============================================================================
# Task 4: Inference Speed Benchmark
# ============================================================================

def benchmark_inference_speed():
    print('=' * 70)
    print('TASK 4: Inference Speed vs Sequence Length')
    print('=' * 70)
    print('Tests O(n) scaling advantage')
    print()

    vocab_size = 64
    dim = 128
    n_layers = 4
    batch_size = 1

    cliff = CliffordModel(vocab_size, dim, n_layers,
                         n_orthogonal_sets=4, planes_per_set=16,
                         use_positional_plane=True, pos_planes=16).to(device)
    mamba = MambaModel(vocab_size, dim, n_layers, d_state=16).to(device)

    cliff.eval()
    mamba.eval()

    print(f'{"Seq Len":>10} {"Clifford (ms)":>15} {"Mamba (ms)":>15} {"Ratio":>10}')
    print('-' * 55)

    for seq_len in [100, 500, 1000, 2000, 4000]:
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            _ = cliff(data)
            _ = mamba(data)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Clifford timing
        start = time.time()
        n_runs = 10
        with torch.no_grad():
            for _ in range(n_runs):
                _ = cliff(data)
        if device == 'cuda':
            torch.cuda.synchronize()
        cliff_time = (time.time() - start) / n_runs * 1000

        # Mamba timing
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = mamba(data)
        if device == 'cuda':
            torch.cuda.synchronize()
        mamba_time = (time.time() - start) / n_runs * 1000

        ratio = mamba_time / cliff_time if cliff_time > 0 else 0
        print(f'{seq_len:>10} {cliff_time:>15.2f} {mamba_time:>15.2f} {ratio:>10.2f}x')


# ============================================================================
# Main
# ============================================================================

def main():
    print('=' * 70)
    print('REAL-WORLD APPLICATION BENCHMARK V2')
    print('Interleaved Multi-Source Time Series Approach')
    print('Clifford PSI vs Mamba on Practical Tasks')
    print('=' * 70)
    print()

    iot_results = test_interleaved_iot()
    print()

    nav_results = test_interleaved_navigation()
    print()

    hybrid_results = test_hybrid_dynamics()
    print()

    benchmark_inference_speed()
    print()

    # Summary
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()

    print('INTERLEAVED IoT SENSOR RECALL:')
    for n_steps, (cliff, mamba) in iot_results.items():
        winner = "Clifford" if cliff > mamba else "Mamba" if mamba > cliff else "Tie"
        print(f'  {n_steps} timesteps: Clifford={cliff:.1f}% Mamba={mamba:.1f}% -> {winner}')

    print()
    print('INTERLEAVED MULTI-AGENT NAVIGATION:')
    for steps, (cliff, mamba) in nav_results.items():
        winner = "Clifford" if cliff > mamba else "Mamba" if mamba > cliff else "Tie"
        print(f'  {steps} steps/agent: Clifford={cliff:.1f}% Mamba={mamba:.1f}% -> {winner}')

    print()
    print('HYBRID DYNAMICS + EVENT RECALL:')
    for seq_len, ((c_dyn, c_event), (m_dyn, m_event)) in hybrid_results.items():
        dyn_winner = "Clifford" if c_dyn < m_dyn else "Mamba"
        event_winner = "Clifford" if c_event > m_event else "Mamba"
        print(f'  len={seq_len}: Dyn MSE (C={c_dyn:.4f} M={m_dyn:.4f}) -> {dyn_winner}')
        print(f'           Event (C={c_event:.1f}% M={m_event:.1f}%) -> {event_winner}')

    # Count wins
    cliff_wins = 0
    total = 0

    for _, (cliff, mamba) in iot_results.items():
        total += 1
        if cliff > mamba:
            cliff_wins += 1

    for _, (cliff, mamba) in nav_results.items():
        total += 1
        if cliff > mamba:
            cliff_wins += 1

    for _, ((c_dyn, c_event), (m_dyn, m_event)) in hybrid_results.items():
        total += 2  # Two metrics
        if c_dyn < m_dyn:
            cliff_wins += 1
        if c_event > m_event:
            cliff_wins += 1

    print()
    print(f'Clifford wins {cliff_wins}/{total} comparisons')

    if cliff_wins > total // 2:
        print()
        print('VERDICT: Clifford PSI shows advantage on interleaved')
        print('         multi-source time series with associative recall')


if __name__ == "__main__":
    main()
