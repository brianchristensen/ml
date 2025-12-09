"""
Geometry-Preserving Output Projections Ablation

Hypothesis: Orthogonal init helped because it preserves geometry (distances/angles).
Testing various geometry-preserving projection methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# =============================================================================
# Geometry-Preserving Output Projection Modules
# =============================================================================

class NormalizedLinear(nn.Module):
    """Linear layer with rows constrained to unit norm during training"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Initialize with orthogonal
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # Normalize weight rows to unit norm (preserves angles)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, weight_normalized, self.bias)


class HouseholderLinear(nn.Module):
    """
    Linear projection using Householder reflections.
    Product of k Householder reflections is exactly orthogonal.
    Parameterized by k vectors instead of full matrix.
    """
    def __init__(self, in_features, out_features, n_reflections=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Use min(in, out) reflections by default
        self.n_reflections = n_reflections or min(in_features, out_features)

        # Each reflection is defined by a unit vector v: H = I - 2*v*v^T
        # We learn the vectors (will be normalized in forward)
        self.reflection_vectors = nn.Parameter(
            torch.randn(self.n_reflections, in_features)
        )
        # Final projection to output dim (if different from input)
        if in_features != out_features:
            self.final_proj = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        else:
            self.final_proj = None

    def forward(self, x):
        # Apply Householder reflections sequentially
        # H_k @ ... @ H_1 @ x
        result = x
        for i in range(self.n_reflections):
            v = F.normalize(self.reflection_vectors[i], dim=0)  # Unit vector
            # Householder reflection: x - 2 * (x @ v) * v
            proj = (result @ v).unsqueeze(-1) * v  # [B, L, D]
            result = result - 2 * proj

        # Project to output dimension
        if self.final_proj is not None:
            result = result @ self.final_proj.T
        else:
            result = result[..., :self.out_features]

        return result


class StereographicProjection(nn.Module):
    """
    Stereographic projection from high-dim sphere to low-dim space.
    Conformal (preserves local angles).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable projection center and scale
        self.center = nn.Parameter(torch.zeros(in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        # Fixed random projection matrix for dimension reduction
        proj_mat = torch.randn(in_features, out_features) / math.sqrt(in_features)
        self.register_buffer('proj_mat', proj_mat)

    def forward(self, x):
        # Normalize input to sphere
        x_centered = x - self.center
        x_norm = F.normalize(x_centered, p=2, dim=-1)

        # Stereographic projection: project from north pole
        # For point p on sphere, stereographic projection is p[:-1] / (1 - p[-1])
        # We use a soft version to avoid division issues
        denom = (1 - x_norm[..., -1:]).clamp(min=0.1)
        stereo = x_norm[..., :-1] / denom

        # Project to output dim
        out = stereo @ self.proj_mat[:-1, :]
        return out * self.scale


class AveragingProjection(nn.Module):
    """
    Simple averaging: group hidden dims and average each group.
    Zero learnable params in the projection itself.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = in_features // out_features
        # Optional learnable scale/bias
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        B, L, D = x.shape
        # Truncate to fit evenly into groups
        D_used = self.group_size * self.out_features
        x_truncated = x[..., :D_used]
        # Reshape and average
        x_grouped = x_truncated.view(B, L, self.out_features, self.group_size)
        out = x_grouped.mean(dim=-1)
        return out * self.scale + self.bias


class CosineReadout(nn.Module):
    """
    Output is cosine similarity to learned prototype vectors.
    Purely angle-based, no magnitude information.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Learnable prototype vectors (one per output dim)
        self.prototypes = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1) * 10.0)  # Learnable temperature
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Initialize prototypes orthogonally
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x):
        # Normalize both input and prototypes
        x_norm = F.normalize(x, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        # Cosine similarity
        cos_sim = x_norm @ proto_norm.T  # [B, L, out_features]
        return cos_sim * self.scale + self.bias


class SVDProjection(nn.Module):
    """
    SVD-based projection: initialize with principal directions,
    then optionally allow fine-tuning while staying near orthogonal.
    """
    def __init__(self, in_features, out_features, trainable=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize with random orthogonal matrix (will be set from data if available)
        weight = torch.randn(out_features, in_features)
        nn.init.orthogonal_(weight)

        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer('weight', weight)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.trainable = trainable

    def forward(self, x):
        if self.trainable:
            # Apply soft orthogonality: project weight to nearest orthogonal matrix
            # Using polar decomposition approximation
            weight = self.weight
            # One step of Newton iteration toward orthogonal: W = 1.5*W - 0.5*W@W.T@W
            wwt = weight @ weight.T
            weight_ortho = 1.5 * weight - 0.5 * (wwt @ weight)
            return F.linear(x, weight_ortho, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


# =============================================================================
# Block and Model (same as best_combo_ablation but with geometry projections)
# =============================================================================

class GeometryBlock(nn.Module):
    """Block with fixed key/query encoders and fixed content_pos_gate"""

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.pos_planes = pos_planes
        self.ltm_planes = pos_planes

        # Fixed key/query projections
        self.register_buffer('key_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('query_proj', torch.randn(dim, self.total_planes) / math.sqrt(dim))
        self.register_buffer('ltm_key_proj', torch.randn(dim * 3, self.ltm_planes) / math.sqrt(dim * 3))

        # Trainable value/output
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # Fixed content_pos_gate
        self.register_buffer('fixed_gate', torch.tensor(0.5))

        # Trainable LTM value proj
        self.ltm_value_proj = nn.Linear(dim, dim)

        # Trainable scalar parameters
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

        # Positional frequencies
        freqs = torch.exp(torch.arange(0, pos_planes).float() * (-math.log(10000.0) / pos_planes))
        self.register_buffer('pos_freqs', freqs)

        # Persistent LTM state
        self.register_buffer('ltm_key_memory', torch.zeros(self.ltm_planes, dtype=torch.complex64))
        self.register_buffer('ltm_binding_memory', torch.zeros(self.ltm_planes, dim, dtype=torch.complex64))
        self.register_buffer('ltm_count', torch.tensor(0.0))

        self._pending_ltm_update = None

    def forward(self, x):
        B, L, D = x.shape

        if self._pending_ltm_update is not None:
            update = self._pending_ltm_update
            self.ltm_key_memory.mul_(update['decay'])
            self.ltm_binding_memory.mul_(update['decay'])
            self.ltm_key_memory.add_(update['new_keys'])
            self.ltm_binding_memory.add_(update['new_bindings'])
            self.ltm_count.add_(update['count_delta'])
            self._pending_ltm_update = None

        key_phase = torch.tanh(x @ self.key_proj) * math.pi
        query_phase = torch.tanh(x @ self.query_proj) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        weights = F.softmax(self.set_weights, dim=0)

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        if self.n_sets > 1:
            joint_key = torch.prod(key_phasor, dim=2)
            joint_query = torch.prod(query_phasor, dim=2)

            V_real = self.to_value(x)

            key_memory = torch.zeros_like(joint_key)
            key_memory[:, 1:, :] = torch.cumsum(joint_key[:, :-1, :], dim=1)

            resonance = key_memory * joint_query.conj()
            resonance_magnitude = resonance.abs().mean(dim=-1, keepdim=True)

            positions = torch.arange(L, device=x.device, dtype=torch.float32).clamp(min=1.0)
            normalized_resonance = resonance_magnitude / positions.view(1, -1, 1).sqrt()

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

        pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos
        total_retrieved = self.fixed_gate * total_retrieved + (1 - self.fixed_gate) * pos_contribution

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

        key_resonance = self.ltm_key_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj()
        total_resonance = key_resonance.sum(dim=-1)
        resonance_magnitude = total_resonance.abs()

        count_norm = (self.ltm_count.detach().clamp(min=1.0) * math.sqrt(self.ltm_planes))
        normalized_resonance = resonance_magnitude / count_norm

        ltm_scale = self.ltm_resonance_scale.clamp(min=1.0, max=20.0)
        ltm_threshold = self.ltm_resonance_threshold.clamp(min=0.1, max=0.9)

        ltm_surprise = 0.5 * (1.0 - torch.tanh(ltm_scale * (normalized_resonance - ltm_threshold)))
        ltm_surprise = ltm_surprise.unsqueeze(-1)

        ltm_write_gate = torch.sigmoid(self.ltm_surprise_scale * (ltm_surprise - 0.5) + self.ltm_surprise_bias)

        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)

        persistent_retrieved = self.ltm_binding_memory.detach().unsqueeze(0).unsqueeze(0) * ltm_key_phasor.conj().unsqueeze(-1)
        persistent_retrieved = persistent_retrieved.sum(dim=2).real
        persistent_norm = (self.ltm_count.detach().clamp(min=1.0) * self.ltm_planes).sqrt()
        ltm_retrieved = persistent_retrieved / persistent_norm

        if self.training:
            with torch.no_grad():
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

        total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        return x + self.to_out(total_retrieved / norm)

    def reset_ltm(self):
        self.ltm_key_memory.zero_()
        self.ltm_binding_memory.zero_()
        self.ltm_count.zero_()


class GeometryModel(nn.Module):
    """Model with configurable geometry-preserving output projection"""

    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16,
                 output_mode='linear_ortho'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Trainable input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection - geometry preserving options
        if output_mode == 'linear_ortho':
            # Baseline: linear with orthogonal init
            self.output_proj = nn.Linear(hidden_dim, input_dim)
            nn.init.orthogonal_(self.output_proj.weight)

        elif output_mode == 'normalized':
            # Normalized linear: rows constrained to unit norm
            self.output_proj = NormalizedLinear(hidden_dim, input_dim)

        elif output_mode == 'householder':
            # Householder reflections (exactly orthogonal)
            self.output_proj = HouseholderLinear(hidden_dim, input_dim, n_reflections=8)

        elif output_mode == 'stereographic':
            # Stereographic projection (conformal)
            self.output_proj = StereographicProjection(hidden_dim, input_dim)

        elif output_mode == 'averaging':
            # Simple averaging of hidden dim groups
            self.output_proj = AveragingProjection(hidden_dim, input_dim)

        elif output_mode == 'cosine':
            # Cosine similarity to prototypes
            self.output_proj = CosineReadout(hidden_dim, input_dim)

        elif output_mode == 'svd_trainable':
            # SVD-style with soft orthogonality constraint during training
            self.output_proj = SVDProjection(hidden_dim, input_dim, trainable=True)

        elif output_mode == 'svd_fixed':
            # SVD-style with fixed orthogonal matrix
            self.output_proj = SVDProjection(hidden_dim, input_dim, trainable=False)

        elif output_mode == 'full':
            # Full NN (baseline for comparison)
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim)
            )

        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")

        self.blocks = nn.ModuleList([
            GeometryBlock(hidden_dim, n_orthogonal_sets, planes_per_set, pos_planes=pos_planes)
            for _ in range(n_layers)
        ])
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


def run_experiment():
    print("=" * 70)
    print("GEOMETRY-PRESERVING OUTPUT PROJECTIONS")
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

    print(f"  Lorenz: {len(lorenz_ctx_train)} train, {len(lorenz_ctx_test)} test")
    print(f"  Chen: {len(chen_ctx_train)} train, {len(chen_ctx_test)} test")

    results = {}

    configs = [
        ("Full NN (baseline)", "full"),
        ("Linear + ortho init", "linear_ortho"),
        ("Normalized Linear", "normalized"),
        ("Householder (exact ortho)", "householder"),
        ("Stereographic", "stereographic"),
        ("Averaging", "averaging"),
        ("Cosine Readout", "cosine"),
        ("SVD trainable", "svd_trainable"),
        ("SVD fixed", "svd_fixed"),
    ]

    for name, output_mode in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        model = GeometryModel(
            input_dim=3, hidden_dim=128, n_layers=4,
            n_orthogonal_sets=4, planes_per_set=16,
            output_mode=output_mode
        ).to(device)
        model.reset_ltm()

        trainable = count_trainable(model)
        print(f"  Trainable params: {trainable:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Phase 1: Lorenz
        print("\n  Phase 1: Learning Lorenz...")
        for epoch in range(5):
            loss = train_epoch(model, lorenz_ctx_train, lorenz_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 2 == 0:
                val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"    Epoch {epoch+1}: train={loss:.6f}, val={val:.6f}")

        lorenz_A = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        print(f"\n  Lorenz after A: {lorenz_A:.6f}")

        # Phase 2: Chen
        print("\n  Phase 2: Learning Chen...")
        for epoch in range(5):
            loss = train_epoch(model, chen_ctx_train, chen_tgt_train, optimizer, batch_size=128)
            if (epoch + 1) % 2 == 0:
                chen_val = evaluate(model, chen_ctx_test, chen_tgt_test)
                lorenz_val = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
                print(f"    Epoch {epoch+1}: Chen={chen_val:.6f}, Lorenz={lorenz_val:.6f}")

        lorenz_B = evaluate(model, lorenz_ctx_test, lorenz_tgt_test)
        chen_F = evaluate(model, chen_ctx_test, chen_tgt_test)

        forgetting = (lorenz_B - lorenz_A) / lorenz_A * 100

        results[name] = {
            'trainable': trainable,
            'lorenz_A': lorenz_A,
            'lorenz_B': lorenz_B,
            'chen': chen_F,
            'forgetting': forgetting
        }

        print(f"\n  Results:")
        print(f"    Lorenz A: {lorenz_A:.6f}")
        print(f"    Lorenz B: {lorenz_B:.6f}")
        print(f"    Forgetting: {forgetting:+.1f}%")
        print(f"    Chen final: {chen_F:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GEOMETRY-PRESERVING PROJECTIONS")
    print("=" * 70)
    print(f"{'Config':<30} {'Params':>10} {'Lorenz A':>12} {'Lorenz B':>12} {'Forget':>10}")
    print("-" * 76)

    # Sort by forgetting
    for name, r in sorted(results.items(), key=lambda x: x[1]['forgetting']):
        print(f"{name:<30} {r['trainable']:>10,} {r['lorenz_A']:>12.6f} {r['lorenz_B']:>12.6f} {r['forgetting']:>+9.0f}%")

    # Find best
    best = min(results.items(), key=lambda x: x[1]['forgetting'])
    best_learning = min(results.items(), key=lambda x: x[1]['lorenz_A'])

    print(f"\nBest forgetting: {best[0]} ({best[1]['forgetting']:+.0f}%)")
    print(f"Best learning: {best_learning[0]} (Lorenz A: {best_learning[1]['lorenz_A']:.6f})")

    return results


if __name__ == "__main__":
    results = run_experiment()
