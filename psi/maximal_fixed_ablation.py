"""
Ablation: Maximally Fixed Model

NO trainable parameters except the phase memory accumulation itself.
All projections are fixed random matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class MaximallyFixedBlock(nn.Module):
    """
    Phase memory block with ZERO trainable parameters.
    All projections are fixed random matrices.
    """

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.pos_planes = pos_planes
        self.ltm_planes = pos_planes

        # ALL FIXED projections
        self.register_buffer('key_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('query_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('value_proj', torch.randn(dim, dim) / math.sqrt(dim))
        self.register_buffer('out_proj', torch.randn(dim, dim) / math.sqrt(dim))

        # LTM projections (fixed)
        self.register_buffer('ltm_key_proj', torch.randn(dim * 3, self.ltm_planes) / math.sqrt(dim * 3))
        self.register_buffer('ltm_value_proj', torch.randn(dim, dim) / math.sqrt(dim))

        # Positional frequencies (fixed)
        freqs = torch.exp(torch.arange(0, pos_planes).float() * (-math.log(10000.0) / pos_planes))
        self.register_buffer('pos_freqs', freqs)

        # Fixed scalar weights
        self.register_buffer('pos_weight', torch.tensor(0.5))
        self.register_buffer('content_weight', torch.tensor(0.5))
        self.register_buffer('ltm_weight', torch.tensor(0.5))

        # Fixed surprise parameters
        self.register_buffer('surprise_scale', torch.tensor(5.0))
        self.register_buffer('surprise_bias', torch.tensor(0.0))
        self.register_buffer('resonance_scale', torch.tensor(5.0))
        self.register_buffer('resonance_threshold', torch.tensor(0.3))
        self.register_buffer('ltm_resonance_scale', torch.tensor(5.0))
        self.register_buffer('ltm_resonance_threshold', torch.tensor(0.3))
        self.register_buffer('ltm_surprise_scale', torch.tensor(5.0))
        self.register_buffer('ltm_surprise_bias', torch.tensor(0.0))
        self.register_buffer('ltm_decay', torch.tensor(0.999))

        # Persistent LTM state
        self.register_buffer('ltm_key_memory', torch.zeros(self.ltm_planes, dtype=torch.complex64))
        self.register_buffer('ltm_binding_memory', torch.zeros(self.ltm_planes, dim, dtype=torch.complex64))
        self.register_buffer('ltm_count', torch.tensor(0.0))

        self._pending_ltm_update = None

    def forward(self, x):
        B, L, D = x.shape

        # Apply pending LTM updates
        if self._pending_ltm_update is not None:
            update = self._pending_ltm_update
            self.ltm_key_memory.mul_(update['decay'])
            self.ltm_binding_memory.mul_(update['decay'])
            self.ltm_key_memory.add_(update['new_keys'])
            self.ltm_binding_memory.add_(update['new_bindings'])
            self.ltm_count.add_(update['count_delta'])
            self._pending_ltm_update = None

        # Fixed projections for phases
        key_phase = torch.tanh(x @ self.key_proj) * math.pi
        query_phase = torch.tanh(x @ self.query_proj) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        # Fixed value projection
        V = (x @ self.value_proj).to(torch.complex64)

        # Process content sets
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        # Equal weights for all sets
        weights = torch.ones(self.n_sets, device=x.device) / self.n_sets

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        if self.n_sets > 1:
            joint_key = torch.prod(key_phasor, dim=2)
            joint_query = torch.prod(query_phasor, dim=2)

            V_real = x @ self.value_proj

            # Surprise-gated writing
            key_memory = torch.zeros_like(joint_key)
            key_memory[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)

            resonance = key_memory * joint_query.conj()
            resonance_magnitude = resonance.abs().mean(dim=-1, keepdim=True)

            positions = torch.arange(L, device=x.device, dtype=torch.float32).clamp(min=1.0)
            normalized_resonance = resonance_magnitude / positions.view(1, -1, 1).sqrt()

            surprise = 0.5 * (1.0 - torch.tanh(self.resonance_scale * (normalized_resonance - self.resonance_threshold)))
            write_gate = torch.sigmoid(self.surprise_scale * (surprise - 0.5) + self.surprise_bias)

            V_real_gated = V_real * write_gate
            V_gated = V_real_gated.to(torch.complex64)

            bound = joint_key.unsqueeze(-1) * V_gated.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * joint_query.conj().unsqueeze(-1)
            cross_bank_retrieved = retrieved.sum(dim=2).real

            bound_all = key_phasor.unsqueeze(-1) * V_gated.unsqueeze(2).unsqueeze(3)
            B_, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B_, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B_, L_seq, n_s, pp, D_dim)

            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real

            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

            cross_weight = 1.0 / (self.n_sets + 1)
            total_retrieved = cross_weight * (total_retrieved + cross_bank_retrieved)
        else:
            bound_all = key_phasor.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)
            B_, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B_, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B_, L_seq, n_s, pp, D_dim)

            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real

            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

        # Positional planes
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
        pos_phase = pos * self.pos_freqs * 2 * math.pi
        pos_phasor = torch.exp(1j * pos_phase)

        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)

        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real

        # Fixed blending of content and position
        total_retrieved = self.content_weight * total_retrieved + self.pos_weight * retrieved_pos

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)

        # LTM with fixed projection
        cumsum = torch.cumsum(x, dim=1)
        positions_for_mean = torch.arange(1, L + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
        running_mean = cumsum / positions_for_mean

        cumsum_sq = torch.cumsum(x ** 2, dim=1)
        running_var = (cumsum_sq / positions_for_mean) - (running_mean ** 2)
        running_std = (running_var.clamp(min=1e-8)).sqrt()

        ltm_key_input = torch.cat([x, running_mean, running_std], dim=-1)

        ltm_key_phase = torch.tanh(ltm_key_input @ self.ltm_key_proj) * math.pi
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = x @ self.ltm_value_proj

        # LTM surprise
        key_resonance = self.ltm_key_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()
        total_resonance = key_resonance.sum(dim=-1)
        resonance_magnitude = total_resonance.abs()

        count_norm = (self.ltm_count.detach().clamp(min=1.0) * math.sqrt(self.ltm_planes))
        normalized_resonance = resonance_magnitude / count_norm

        ltm_surprise = 0.5 * (1.0 - torch.tanh(self.ltm_resonance_scale * (normalized_resonance - self.ltm_resonance_threshold)))
        ltm_surprise = ltm_surprise.unsqueeze(-1)

        ltm_write_gate = torch.sigmoid(self.ltm_surprise_scale * (ltm_surprise - 0.5) + self.ltm_surprise_bias)

        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)

        # LTM retrieval
        persistent_retrieved = self.ltm_binding_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj().unsqueeze(-1)
        persistent_retrieved = persistent_retrieved.sum(dim=2).real
        persistent_norm = (self.ltm_count.detach().clamp(min=1.0) * self.ltm_planes).sqrt()
        ltm_retrieved = persistent_retrieved / persistent_norm

        # Pending LTM update
        if self.training:
            with torch.no_grad():
                decay = self.ltm_decay.clamp(min=0.9, max=0.9999)
                new_keys = (ltm_write_gate.detach() * ltm_key_phasor.detach()).mean(dim=(0, 1))
                new_bindings = gated_binding.detach().mean(dim=(0, 1))
                avg_gate = ltm_write_gate.detach().mean().item()

                self._pending_ltm_update = {
                    'decay': decay,
                    'new_keys': new_keys,
                    'new_bindings': new_bindings,
                    'count_delta': avg_gate * B * L * (1 - decay)
                }

        total_retrieved = total_retrieved + self.ltm_weight * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        # Fixed output projection
        out = total_retrieved / norm
        out = out @ self.out_proj

        return x + out

    def reset_ltm(self):
        self.ltm_key_memory.zero_()
        self.ltm_binding_memory.zero_()
        self.ltm_count.zero_()


class MaximallyFixedModel(nn.Module):
    """
    Model with NO trainable parameters in projections.
    Only the phase memory accumulation adapts.
    """

    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # FIXED input/output projections
        self.register_buffer('input_proj', torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim))
        self.register_buffer('output_proj', torch.randn(hidden_dim, input_dim) / math.sqrt(hidden_dim))

        self.blocks = nn.ModuleList([
            MaximallyFixedBlock(hidden_dim, n_orthogonal_sets, planes_per_set,
                               pos_planes=pos_planes)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # Fixed input projection
        h = x @ self.input_proj

        for block in self.blocks:
            h = block(h)

        # Fixed output projection
        return h @ self.output_proj

    def reset_ltm(self):
        for block in self.blocks:
            block.reset_ltm()


# =============================================================================
# Data Generation
# =============================================================================

def generate_lorenz(batch_size, seq_len, dt=0.01):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x = np.random.uniform(-15, 15, batch_size)
    y = np.random.uniform(-20, 20, batch_size)
    z = np.random.uniform(10, 40, batch_size)
    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        trajectories[:, t, 0] = x
        trajectories[:, t, 1] = y
        trajectories[:, t, 2] = z
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
    return trajectories


def generate_chen(batch_size, seq_len, dt=0.002):
    a, b, c = 35.0, 3.0, 28.0
    x = np.random.uniform(-10, 10, batch_size)
    y = np.random.uniform(-10, 10, batch_size)
    z = np.random.uniform(10, 30, batch_size)
    trajectories = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    for t in range(seq_len):
        trajectories[:, t, 0] = x
        trajectories[:, t, 1] = y
        trajectories[:, t, 2] = z
        dx = a * (y - x)
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
    return trajectories


def light_normalize(trajs, scale=100.0):
    return trajs / scale


def create_sequences(trajectories, context_len=20):
    n_traj, traj_len, dim = trajectories.shape
    n_seqs = traj_len - context_len - 1
    contexts = []
    targets = []
    for i in range(n_traj):
        for t in range(n_seqs):
            contexts.append(trajectories[i, t:t+context_len])
            targets.append(trajectories[i, t+context_len])
    return np.array(contexts), np.array(targets)


def evaluate(model, contexts, targets, batch_size=256):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx = torch.tensor(contexts[i:i+batch_size], device=device)
            tgt = torch.tensor(targets[i:i+batch_size], device=device)
            pred = model(ctx)[:, -1, :]
            loss = F.mse_loss(pred, tgt)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return total, trainable, buffers


def run_experiment():
    print("=" * 70)
    print("ABLATION: Maximally Fixed Model (No Trainable Parameters)")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    context_len = 20

    # Generate data
    print("\nGenerating datasets...")
    lorenz_train = light_normalize(generate_lorenz(100, 100))
    lorenz_test = light_normalize(generate_lorenz(30, 100))
    chen_train = light_normalize(generate_chen(100, 100))
    chen_test = light_normalize(generate_chen(30, 100))

    lorenz_ctx_train, lorenz_tgt_train = create_sequences(lorenz_train, context_len)
    lorenz_ctx_test, lorenz_tgt_test = create_sequences(lorenz_test, context_len)
    chen_ctx_train, chen_tgt_train = create_sequences(chen_train, context_len)
    chen_ctx_test, chen_tgt_test = create_sequences(chen_test, context_len)

    print(f"  Lorenz: {len(lorenz_ctx_train)} train, {len(lorenz_ctx_test)} test")
    print(f"  Chen: {len(chen_ctx_train)} train, {len(chen_ctx_test)} test")

    # Create maximally fixed model
    model = MaximallyFixedModel(
        input_dim=3,
        hidden_dim=128,
        n_layers=4,
        n_orthogonal_sets=4,
        planes_per_set=16
    ).to(device)
    model.reset_ltm()

    total_params, trainable_params, buffer_size = count_params(model)
    print(f"\n  Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print(f"  Buffers (fixed): {buffer_size:,}")
    print(f"  ALL projections are FIXED - only LTM accumulation adapts")

    # Phase 1: "Train" on Lorenz (just accumulate in LTM)
    print("\n" + "=" * 60)
    print("Phase 1: Accumulating Lorenz in LTM...")
    print("=" * 60)

    # No optimizer needed - just forward passes to accumulate LTM
    model.train()  # Enable LTM updates
    n_accumulation_passes = 5

    for pass_idx in range(n_accumulation_passes):
        # Forward pass through training data to accumulate LTM
        for i in range(0, len(lorenz_ctx_train), 128):
            ctx = torch.tensor(lorenz_ctx_train[i:i+128], device=device)
            _ = model(ctx)  # Just forward, no backward

        if (pass_idx + 1) % 2 == 0:
            val_loss = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Pass {pass_idx+1}: val={val_loss:.6f}")

    lorenz_after_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    print(f"\n  Lorenz after Phase A: MSE={lorenz_after_A:.6f}")

    # Phase 2: "Train" on Chen
    print("\n" + "=" * 60)
    print("Phase 2: Accumulating Chen in LTM...")
    print("=" * 60)

    for pass_idx in range(n_accumulation_passes):
        for i in range(0, len(chen_ctx_train), 128):
            ctx = torch.tensor(chen_ctx_train[i:i+128], device=device)
            _ = model(ctx)

        if (pass_idx + 1) % 2 == 0:
            chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
            lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
            print(f"  Pass {pass_idx+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

    lorenz_after_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
    chen_final = evaluate(model, chen_ctx_test, chen_tgt_test)

    forgetting = (lorenz_after_B - lorenz_after_A) / lorenz_after_A * 100

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Maximally Fixed Model")
    print("=" * 70)
    print(f"  Lorenz after A: {lorenz_after_A:.6f}")
    print(f"  Lorenz after B: {lorenz_after_B:.6f}")
    print(f"  Forgetting: {forgetting:+.1f}%")
    print(f"  Chen final: {chen_final:.6f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("  Config                       Forgetting    Can Learn?")
    print("  " + "-" * 55)
    print(f"  Normalized + NN encoders:     +10781%        Yes")
    print(f"  No-norm + NN encoders:        +1850%         Yes")
    print(f"  No-norm + Fixed encoders:     +927%          Yes")
    print(f"  Maximally fixed:              {forgetting:+.0f}%          {'Yes' if chen_final < 0.01 else 'Limited'}")

    if forgetting < 100:
        print("\n  SUCCESS: Near-zero forgetting with pure memory accumulation!")
    elif forgetting < 500:
        print("\n  GOOD: Low forgetting, memory mechanism working.")
    else:
        print("\n  PARTIAL: Some forgetting remains in LTM accumulation itself.")

    if lorenz_after_A > 0.01:
        print("\n  WARNING: Model struggles to learn - random projections may be too lossy.")
        print("  May need more planes or better projection initialization.")

    return {
        'lorenz_A': lorenz_after_A,
        'lorenz_B': lorenz_after_B,
        'chen': chen_final,
        'forgetting': forgetting
    }


if __name__ == "__main__":
    results = run_experiment()
