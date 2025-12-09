"""
Systematic Ablation: Fix components one by one

Starting from "fixed encoders" (927% forgetting), progressively fix more
components to find the minimal trainable set needed for learning.

Trainable components in "fixed encoders" version:
1. input_proj (NN: 3 -> 128 -> 128)
2. output_proj (NN: 128 -> 128 -> 3)
3. to_value (Linear: 128 -> 128)
4. to_out (LayerNorm + Linear: 128 -> 128)
5. content_pos_gate (NN: 128 -> 64 -> 1)
6. norms (LayerNorm per layer)
7. ltm_value_proj (Linear: 128 -> 128)
8. Various scalar parameters (set_weights, pos_weight, surprise params, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class AblationBlock(nn.Module):
    """Block with configurable fixed/trainable components"""

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16,
                 fix_value=False, fix_out=False, fix_content_pos_gate=False,
                 fix_ltm_value=False, fix_scalars=False):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.pos_planes = pos_planes
        self.ltm_planes = pos_planes

        # Key/Query encoders are ALWAYS fixed (from previous ablation)
        self.register_buffer('key_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('query_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('ltm_key_proj', torch.randn(dim * 3, self.ltm_planes) / math.sqrt(dim * 3))

        # to_value: fixed or trainable
        if fix_value:
            self.register_buffer('value_proj', torch.randn(dim, dim) / math.sqrt(dim))
            self.to_value = lambda x: x @ self.value_proj
        else:
            self.to_value = nn.Linear(dim, dim)

        # to_out: fixed or trainable
        if fix_out:
            self.register_buffer('out_proj', torch.randn(dim, dim) / math.sqrt(dim))
            self.to_out = lambda x: x @ self.out_proj
        else:
            self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # content_pos_gate: fixed or trainable
        if fix_content_pos_gate:
            self.content_pos_gate = lambda x: torch.full((x.shape[0], x.shape[1], 1), 0.5, device=x.device)
        else:
            self.content_pos_gate = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            )

        # ltm_value_proj: fixed or trainable
        if fix_ltm_value:
            self.register_buffer('ltm_value_proj_mat', torch.randn(dim, dim) / math.sqrt(dim))
            self.ltm_value_proj = lambda x: x @ self.ltm_value_proj_mat
        else:
            self.ltm_value_proj = nn.Linear(dim, dim)

        # Scalar parameters: fixed or trainable
        if fix_scalars:
            self.register_buffer('set_weights', torch.ones(n_orthogonal_sets) / n_orthogonal_sets)
            self.register_buffer('pos_weight', torch.tensor(0.5))
            self.register_buffer('surprise_scale', torch.tensor(5.0))
            self.register_buffer('surprise_bias', torch.tensor(0.0))
            self.register_buffer('resonance_scale', torch.tensor(5.0))
            self.register_buffer('resonance_threshold', torch.tensor(0.3))
            self.register_buffer('ltm_resonance_scale', torch.tensor(5.0))
            self.register_buffer('ltm_resonance_threshold', torch.tensor(0.3))
            self.register_buffer('ltm_surprise_scale', torch.tensor(5.0))
            self.register_buffer('ltm_surprise_bias', torch.tensor(0.0))
            self.register_buffer('ltm_weight', torch.tensor(0.5))
            self.register_buffer('ltm_decay', torch.tensor(0.999))
            self._fixed_scalars = True
        else:
            self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))
            self.pos_weight = nn.Parameter(torch.ones(1))
            self.surprise_scale = nn.Parameter(torch.tensor(5.0))
            self.surprise_bias = nn.Parameter(torch.tensor(0.0))
            self.resonance_scale = nn.Parameter(torch.tensor(5.0))
            self.resonance_threshold = nn.Parameter(torch.tensor(0.3))
            self.ltm_resonance_scale = nn.Parameter(torch.tensor(5.0))
            self.ltm_resonance_threshold = nn.Parameter(torch.tensor(0.3))
            self.ltm_surprise_scale = nn.Parameter(torch.tensor(5.0))
            self.ltm_surprise_bias = nn.Parameter(torch.tensor(0.0))
            self.ltm_weight = nn.Parameter(torch.tensor(0.5))
            self.ltm_decay = nn.Parameter(torch.tensor(0.999))
            self._fixed_scalars = False

        # Positional frequencies (always fixed)
        freqs = torch.exp(torch.arange(0, pos_planes).float() * (-math.log(10000.0) / pos_planes))
        self.register_buffer('pos_freqs', freqs)

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

        V = self.to_value(x).to(torch.complex64)

        # Process content sets
        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        if self._fixed_scalars:
            weights = self.set_weights
        else:
            weights = F.softmax(self.set_weights, dim=0)

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        if self.n_sets > 1:
            joint_key = torch.prod(key_phasor, dim=2)
            joint_query = torch.prod(query_phasor, dim=2)

            V_real = self.to_value(x)

            # Surprise-gated writing
            key_memory = torch.zeros_like(joint_key)
            key_memory[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)

            resonance = key_memory * joint_query.conj()
            resonance_magnitude = resonance.abs().mean(dim=-1, keepdim=True)

            positions = torch.arange(L, device=x.device, dtype=torch.float32).clamp(min=1.0)
            normalized_resonance = resonance_magnitude / positions.view(1, -1, 1).sqrt()

            if self._fixed_scalars:
                scale = self.resonance_scale
                threshold = self.resonance_threshold
            else:
                scale = self.resonance_scale.clamp(min=1.0, max=20.0)
                threshold = self.resonance_threshold.clamp(min=0.1, max=0.9)

            surprise = 0.5 * (1.0 - torch.tanh(scale * (normalized_resonance - threshold)))
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

        gate = self.content_pos_gate(x)
        if self._fixed_scalars:
            pos_contribution = self.pos_weight * retrieved_pos
        else:
            pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos

        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)

        # LTM
        cumsum = torch.cumsum(x, dim=1)
        positions_for_mean = torch.arange(1, L + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
        running_mean = cumsum / positions_for_mean

        cumsum_sq = torch.cumsum(x ** 2, dim=1)
        running_var = (cumsum_sq / positions_for_mean) - (running_mean ** 2)
        running_std = (running_var.clamp(min=1e-8)).sqrt()

        ltm_key_input = torch.cat([x, running_mean, running_std], dim=-1)

        ltm_key_phase = torch.tanh(ltm_key_input @ self.ltm_key_proj) * math.pi
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = self.ltm_value_proj(x)

        # LTM surprise
        key_resonance = self.ltm_key_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()
        total_resonance = key_resonance.sum(dim=-1)
        resonance_magnitude = total_resonance.abs()

        count_norm = (self.ltm_count.detach().clamp(min=1.0) * math.sqrt(self.ltm_planes))
        normalized_resonance = resonance_magnitude / count_norm

        if self._fixed_scalars:
            ltm_scale = self.ltm_resonance_scale
            ltm_threshold = self.ltm_resonance_threshold
        else:
            ltm_scale = self.ltm_resonance_scale.clamp(min=1.0, max=20.0)
            ltm_threshold = self.ltm_resonance_threshold.clamp(min=0.1, max=0.9)

        ltm_surprise = 0.5 * (1.0 - torch.tanh(ltm_scale * (normalized_resonance - ltm_threshold)))
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
                if self._fixed_scalars:
                    decay = self.ltm_decay.clamp(min=0.9, max=0.9999)
                else:
                    decay = self.ltm_decay.detach().clamp(min=0.9, max=0.9999)
                new_keys = (ltm_write_gate.detach() * ltm_key_phasor.detach()).mean(dim=(0, 1))
                new_bindings = gated_binding.detach().mean(dim=(0, 1))
                avg_gate = ltm_write_gate.detach().mean().item()

                self._pending_ltm_update = {
                    'decay': decay,
                    'new_keys': new_keys,
                    'new_bindings': new_bindings,
                    'count_delta': avg_gate * B * L * (1 - decay)
                }

        if self._fixed_scalars:
            total_retrieved = total_retrieved + self.ltm_weight * ltm_retrieved
        else:
            total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

    def reset_ltm(self):
        self.ltm_key_memory.zero_()
        self.ltm_binding_memory.zero_()
        self.ltm_count.zero_()


class AblationModel(nn.Module):
    """Model with configurable fixed/trainable components"""

    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16,
                 fix_input_proj=False, fix_output_proj=False, fix_norms=False,
                 fix_value=False, fix_out=False, fix_content_pos_gate=False,
                 fix_ltm_value=False, fix_scalars=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        if fix_input_proj:
            self.register_buffer('input_proj_mat', torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim))
            self.input_proj = lambda x: x @ self.input_proj_mat
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Output projection
        if fix_output_proj:
            self.register_buffer('output_proj_mat', torch.randn(hidden_dim, input_dim) / math.sqrt(hidden_dim))
            self.output_proj = lambda x: x @ self.output_proj_mat
        else:
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim)
            )

        # Blocks
        self.blocks = nn.ModuleList([
            AblationBlock(hidden_dim, n_orthogonal_sets, planes_per_set,
                         pos_planes=pos_planes,
                         fix_value=fix_value, fix_out=fix_out,
                         fix_content_pos_gate=fix_content_pos_gate,
                         fix_ltm_value=fix_ltm_value, fix_scalars=fix_scalars)
            for _ in range(n_layers)
        ])

        # Norms
        if fix_norms:
            # No normalization
            self.norms = [lambda x: x for _ in range(n_layers)]
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.output_proj(h)

    def reset_ltm(self):
        for block in self.blocks:
            block.reset_ltm()


# =============================================================================
# Data Generation (same as before)
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


def train_epoch(model, contexts, targets, optimizer, batch_size=64):
    model.train()
    n_samples = len(contexts)
    indices = np.random.permutation(n_samples)
    total_loss = 0
    n_batches = 0
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        ctx = torch.tensor(contexts[batch_idx], device=device)
        tgt = torch.tensor(targets[batch_idx], device=device)
        optimizer.zero_grad()
        pred = model(ctx)[:, -1, :]
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


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


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_ablation(name, lorenz_ctx_train, lorenz_tgt_train, lorenz_ctx_test, lorenz_tgt_test,
                        chen_ctx_train, chen_tgt_train, chen_ctx_test, chen_tgt_test, **kwargs):
    """Run a single ablation configuration"""
    model = AblationModel(
        input_dim=3, hidden_dim=128, n_layers=4,
        n_orthogonal_sets=4, planes_per_set=16,
        **kwargs
    ).to(device)
    model.reset_ltm()

    trainable = count_trainable(model)

    # Only optimize if there are trainable params
    if trainable > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Phase 1: Lorenz
        for epoch in range(5):
            train_epoch(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)

        lorenz_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)

        # Phase 2: Chen
        for epoch in range(5):
            train_epoch(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)

        lorenz_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        chen_F = evaluate(model, chen_ctx_test, chen_tgt_test)
    else:
        # No training possible
        lorenz_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        lorenz_B = lorenz_A
        chen_F = evaluate(model, chen_ctx_test, chen_tgt_test)

    forgetting = (lorenz_B - lorenz_A) / max(lorenz_A, 1e-8) * 100

    return {
        'name': name,
        'trainable': trainable,
        'lorenz_A': lorenz_A,
        'lorenz_B': lorenz_B,
        'chen': chen_F,
        'forgetting': forgetting,
        'can_learn': lorenz_A < 0.01
    }


def run_experiment():
    print("=" * 70)
    print("SYSTEMATIC ABLATION: Fix Components One by One")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Generate data
    print("\nGenerating datasets...")
    lorenz_train = light_normalize(generate_lorenz(100, 100))
    lorenz_test = light_normalize(generate_lorenz(30, 100))
    chen_train = light_normalize(generate_chen(100, 100))
    chen_test = light_normalize(generate_chen(30, 100))

    lorenz_ctx_train, lorenz_tgt_train = create_sequences(lorenz_train, 20)
    lorenz_ctx_test, lorenz_tgt_test = create_sequences(lorenz_test, 20)
    chen_ctx_train, chen_tgt_train = create_sequences(chen_train, 20)
    chen_ctx_test, chen_tgt_test = create_sequences(chen_test, 20)

    data = (lorenz_ctx_train, lorenz_tgt_train, lorenz_ctx_test, lorenz_tgt_test,
            chen_ctx_train, chen_tgt_train, chen_ctx_test, chen_tgt_test)

    # Ablation configurations
    configs = [
        ("Baseline (fixed key/query only)", {}),
        ("+ fix scalars", {"fix_scalars": True}),
        ("+ fix content_pos_gate", {"fix_content_pos_gate": True}),
        ("+ fix ltm_value", {"fix_ltm_value": True}),
        ("+ fix to_value", {"fix_value": True}),
        ("+ fix to_out", {"fix_out": True}),
        ("+ fix norms", {"fix_norms": True}),
        ("+ fix input_proj", {"fix_input_proj": True}),
        ("+ fix output_proj", {"fix_output_proj": True}),
        # Combinations
        ("fix value+out", {"fix_value": True, "fix_out": True}),
        ("fix input+output proj", {"fix_input_proj": True, "fix_output_proj": True}),
        ("fix all block internals", {"fix_value": True, "fix_out": True, "fix_content_pos_gate": True,
                                     "fix_ltm_value": True, "fix_scalars": True}),
        ("fix all except input/output proj", {"fix_value": True, "fix_out": True, "fix_content_pos_gate": True,
                                               "fix_ltm_value": True, "fix_scalars": True, "fix_norms": True}),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\nTesting: {name}...")
        result = run_single_ablation(name, *data, **kwargs)
        results.append(result)
        print(f"  Trainable: {result['trainable']:,}, Forgetting: {result['forgetting']:+.0f}%, "
              f"Lorenz: {result['lorenz_A']:.6f} -> {result['lorenz_B']:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<40} {'Params':>10} {'Forget':>10} {'Learn?':>8}")
    print("-" * 70)

    for r in results:
        learn = "Yes" if r['can_learn'] else "No"
        print(f"{r['name']:<40} {r['trainable']:>10,} {r['forgetting']:>+9.0f}% {learn:>8}")

    # Find best config that can still learn
    learnable = [r for r in results if r['can_learn']]
    if learnable:
        best = min(learnable, key=lambda x: x['forgetting'])
        print(f"\nBest learnable config: {best['name']}")
        print(f"  Forgetting: {best['forgetting']:+.0f}%")
        print(f"  Trainable params: {best['trainable']:,}")

    return results


if __name__ == "__main__":
    results = run_experiment()
