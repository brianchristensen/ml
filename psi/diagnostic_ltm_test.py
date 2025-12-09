"""
Diagnostic LTM Test - Inspect Surprise and Cross-Bank Binding

This test hooks into the model internals to output actual values from:
1. Surprise gating (Titans-style prediction error)
2. Cross-bank binding (joint key product across banks)

Uses REDUNDANT data patterns (repeated subsequences) so we can verify
that surprise gating actually becomes selective (low gate for seen patterns,
high gate for novel patterns).

Goal: Verify these features are actually contributing to the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")


# ============================================================================
# Instrumented Block - captures diagnostic values
# ============================================================================

class InstrumentedBivectorBlock(nn.Module):
    """
    OrthogonalBivectorBlock with diagnostic outputs.
    Captures surprise values and cross-bank binding statistics.
    """

    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=4, pos_planes=16):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set
        self.pos_planes = pos_planes

        # Content-based phase encoders
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_planes)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.total_planes)
        )

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))
        self.retention_logits = nn.Parameter(torch.ones(n_orthogonal_sets) * 9.2)
        self.pos_retention_logit = nn.Parameter(torch.tensor(9.2))

        freqs = torch.exp(torch.arange(0, pos_planes).float() *
                        (-math.log(10000.0) / pos_planes))
        self.register_buffer('pos_freqs', freqs)

        self.pos_weight = nn.Parameter(torch.ones(1))
        self.content_pos_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # LTM parameters
        self.ltm_planes = pos_planes
        self.ltm_key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.ltm_planes)
        )
        self.ltm_value_proj = nn.Linear(dim, dim)
        self.surprise_scale = nn.Parameter(torch.tensor(1.0))
        self.surprise_bias = nn.Parameter(torch.tensor(0.0))
        self.ltm_weight = nn.Parameter(torch.tensor(0.5))
        self.surprise_momentum = nn.Parameter(torch.tensor(0.9))

        # Diagnostic storage
        self.last_diagnostics = {}

    def forward(self, x, capture_diagnostics=False):
        B, L, D = x.shape

        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi

        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)

        V = self.to_value(x).to(torch.complex64)

        key_phasor = key_phasor.view(B, L, self.n_sets, self.planes_per_set)
        query_phasor = query_phasor.view(B, L, self.n_sets, self.planes_per_set)

        weights = F.softmax(self.set_weights, dim=0)

        total_retrieved = torch.zeros(B, L, D, device=x.device)

        # Initialize diagnostics
        cross_bank_surprise = None
        cross_bank_write_gate = None
        joint_key_magnitude = None
        joint_key_phase_std = None

        if self.n_sets > 1:
            # Cross-bank binding
            joint_key = torch.prod(key_phasor, dim=2)  # [B, L, planes_per_set]
            joint_query = torch.prod(query_phasor, dim=2)

            if capture_diagnostics:
                # Capture joint key statistics
                joint_key_magnitude = joint_key.abs().mean().item()
                joint_key_phase = joint_key.angle()
                joint_key_phase_std = joint_key_phase.std().item()

            V_real = self.to_value(x)

            # Surprise computation using COSINE SIMILARITY
            binding_for_pred = joint_key.unsqueeze(-1) * V_real.unsqueeze(-2).to(joint_key.dtype)
            memory_shifted = torch.zeros_like(binding_for_pred)
            memory_shifted[:, 1:, :, :] = torch.cumsum(binding_for_pred[:, :-1, :, :], dim=1)

            prediction = memory_shifted * joint_key.conj().unsqueeze(-1)
            prediction = prediction.sum(dim=2).real

            # CONTENT-BASED SURPRISE via Cosine Similarity
            pred_norm_cb = F.normalize(prediction, dim=-1, eps=1e-8)
            value_norm_cb = F.normalize(V_real, dim=-1, eps=1e-8)
            familiarity_cb = (pred_norm_cb * value_norm_cb).sum(dim=-1, keepdim=True)

            # Handle empty memory (position 0)
            pred_mag_cb = prediction.norm(dim=-1, keepdim=True)
            has_mem_cb = (pred_mag_cb > 1e-6).float()
            familiarity_cb = familiarity_cb * has_mem_cb

            # Convert familiarity to surprise: [-1, 1] -> [0, 1]
            surprise = (1.0 - familiarity_cb) / 2.0
            write_gate = torch.sigmoid(self.surprise_scale * (surprise - 0.5) + self.surprise_bias)

            if capture_diagnostics:
                cross_bank_surprise = surprise.mean().item()
                cross_bank_write_gate = write_gate.mean().item()

            V_real_gated = V_real * write_gate
            V_gated = V_real_gated.to(torch.complex64)

            bound = joint_key.unsqueeze(-1) * V_gated.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * joint_query.conj().unsqueeze(-1)
            cross_bank_retrieved = retrieved.sum(dim=2).real

            # Individual bank retrievals
            bound_all = key_phasor.unsqueeze(-1) * V_gated.unsqueeze(2).unsqueeze(3)
            B, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B, L_seq, n_s, pp, D_dim)
            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real
            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

            cross_weight = 1.0 / (self.n_sets + 1)
            total_retrieved = cross_weight * (total_retrieved + cross_bank_retrieved)
        else:
            bound_all = key_phasor.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)
            B, L_seq, n_s, pp, D_dim = bound_all.shape
            bound_flat = bound_all.view(B, L_seq, n_s * pp, D_dim)
            memory_flat = torch.cumsum(bound_flat, dim=1)
            memory_all = memory_flat.view(B, L_seq, n_s, pp, D_dim)
            retrieved_all = memory_all * query_phasor.conj().unsqueeze(-1)
            retrieved_per_bank = retrieved_all.sum(dim=3).real
            weighted_retrieved = retrieved_per_bank * weights.view(1, 1, -1, 1)
            total_retrieved = weighted_retrieved.sum(dim=2)

        # Positional memory
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
        pos_phase = pos * self.pos_freqs * 2 * math.pi
        pos_phasor = torch.exp(1j * pos_phase)

        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)

        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)
        pos_retention = torch.sigmoid(self.pos_retention_logit)
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real

        gate = self.content_pos_gate(x)
        pos_contribution = torch.sigmoid(self.pos_weight) * retrieved_pos
        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)

        # LTM with surprise gating
        ltm_key_phase = torch.tanh(self.ltm_key_encoder(x)) * math.pi
        ltm_key_phasor = torch.exp(1j * ltm_key_phase)
        ltm_value = self.ltm_value_proj(x)

        ltm_value_complex = ltm_value.to(torch.complex64)
        binding = ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)

        ltm_memory_shifted = torch.zeros_like(binding)
        ltm_memory_shifted[:, 1:, :, :] = torch.cumsum(binding[:, :-1, :, :], dim=1)

        ltm_prediction = ltm_memory_shifted * ltm_key_phasor.conj().unsqueeze(-1)
        ltm_prediction = ltm_prediction.sum(dim=2).real

        # CONTENT-BASED SURPRISE via Cosine Similarity
        pred_norm_vec = F.normalize(ltm_prediction, dim=-1, eps=1e-8)
        value_norm_vec = F.normalize(ltm_value, dim=-1, eps=1e-8)

        # Cosine similarity for familiarity
        familiarity = (pred_norm_vec * value_norm_vec).sum(dim=-1, keepdim=True)

        # Handle position 0 where memory is empty
        pred_magnitude = ltm_prediction.norm(dim=-1, keepdim=True)
        has_memory = (pred_magnitude > 1e-6).float()
        familiarity = familiarity * has_memory

        # Convert familiarity to surprise: [-1, 1] -> [0, 1]
        ltm_surprise = (1.0 - familiarity) / 2.0
        ltm_write_gate = torch.sigmoid(self.surprise_scale * (ltm_surprise - 0.5) + self.surprise_bias)

        # Capture LTM diagnostics
        ltm_surprise_val = ltm_surprise.mean().item() if capture_diagnostics else None
        ltm_write_gate_val = ltm_write_gate.mean().item() if capture_diagnostics else None

        ltm_value_complex = ltm_value.to(torch.complex64)
        gated_binding = ltm_write_gate.unsqueeze(-2) * ltm_key_phasor.unsqueeze(-1) * ltm_value_complex.unsqueeze(-2)

        ltm_memory = torch.cumsum(gated_binding, dim=1)
        ltm_retrieved = ltm_memory * ltm_key_phasor.conj().unsqueeze(-1)
        ltm_retrieved = ltm_retrieved.sum(dim=2).real

        ltm_norm = torch.sqrt(positions * self.ltm_planes).view(1, L, 1)
        ltm_retrieved = ltm_retrieved / ltm_norm

        total_retrieved = total_retrieved + torch.sigmoid(self.ltm_weight) * ltm_retrieved

        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)

        if capture_diagnostics:
            self.last_diagnostics = {
                'cross_bank_surprise': cross_bank_surprise,
                'cross_bank_write_gate': cross_bank_write_gate,
                'joint_key_magnitude': joint_key_magnitude,
                'joint_key_phase_std': joint_key_phase_std,
                'ltm_surprise': ltm_surprise_val,
                'ltm_write_gate': ltm_write_gate_val,
                'content_pos_gate': gate.mean().item(),
                'ltm_weight': torch.sigmoid(self.ltm_weight).item(),
                # Per-position surprise for detailed analysis
                'ltm_surprise_per_pos': ltm_surprise.mean(dim=0).squeeze().detach().cpu().numpy(),
                'ltm_gate_per_pos': ltm_write_gate.mean(dim=0).squeeze().detach().cpu().numpy(),
            }

        return x + self.to_out(total_retrieved / norm)


class InstrumentedModel(nn.Module):
    """Model with instrumented blocks for diagnostics"""
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=4,
                 n_orthogonal_sets=4, planes_per_set=16, pos_planes=16):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            InstrumentedBivectorBlock(hidden_dim, n_orthogonal_sets, planes_per_set, pos_planes)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, capture_diagnostics=False):
        h = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h), capture_diagnostics=capture_diagnostics)
        return self.output_proj(h)

    def get_all_diagnostics(self):
        """Aggregate diagnostics from all blocks"""
        all_diag = {}
        for i, block in enumerate(self.blocks):
            for key, val in block.last_diagnostics.items():
                all_diag[f'block{i}_{key}'] = val
        return all_diag


# ============================================================================
# Data generation - REDUNDANT PATTERNS for surprise testing
# ============================================================================

def generate_redundant_sequences(batch_size, seq_len, dim=3, n_patterns=4, noise=0.01):
    """
    Generate sequences with REPEATED patterns.

    Structure: [pattern_A, pattern_B, pattern_A, pattern_C, pattern_A, ...]

    Pattern A repeats multiple times - surprise should be LOW for repeats.
    Novel patterns (B, C, etc.) should have HIGH surprise.

    This lets us verify that surprise gating is actually selective.
    """
    # Create base patterns (fixed random vectors)
    np.random.seed(42)  # Reproducible patterns
    base_patterns = np.random.randn(n_patterns, dim).astype(np.float32)
    base_patterns = base_patterns / (np.linalg.norm(base_patterns, axis=1, keepdims=True) + 1e-8)

    seqs = np.zeros((batch_size, seq_len, dim), dtype=np.float32)
    pattern_ids = np.zeros((batch_size, seq_len), dtype=np.int32)  # Track which pattern

    for b in range(batch_size):
        # Pattern 0 is the "repeated" pattern, others are "novel"
        # Structure: repeat pattern 0 frequently, intersperse others
        for t in range(seq_len):
            if t % 3 == 0:
                # Repeated pattern (should have low surprise after first occurrence)
                p_id = 0
            else:
                # Novel pattern (should have high surprise)
                p_id = np.random.randint(1, n_patterns)

            pattern_ids[b, t] = p_id
            seqs[b, t] = base_patterns[p_id] + np.random.randn(dim).astype(np.float32) * noise

    return seqs, pattern_ids, base_patterns


def generate_copy_with_redundancy(batch_size, seq_len, dim=3):
    """
    Generate sequences where the task is to predict the next value,
    but with explicit redundancy (same values repeat).

    Structure: [v1, v1, v1, v2, v2, v1, v1, v3, ...]

    When v1 appears again, surprise should be LOW.
    When new value appears, surprise should be HIGH.
    """
    seqs = np.zeros((batch_size, seq_len, dim), dtype=np.float32)

    # Create a small vocabulary of values
    np.random.seed(42)
    vocab_size = 8
    vocab = np.random.randn(vocab_size, dim).astype(np.float32)
    vocab = vocab / (np.linalg.norm(vocab, axis=1, keepdims=True) + 1e-8)

    for b in range(batch_size):
        # Each sequence has a "dominant" value that repeats often
        dominant = np.random.randint(0, vocab_size)

        for t in range(seq_len):
            if np.random.rand() < 0.6:  # 60% repeat dominant
                v_id = dominant
            else:  # 40% random
                v_id = np.random.randint(0, vocab_size)

            seqs[b, t] = vocab[v_id] + np.random.randn(dim).astype(np.float32) * 0.01

    return seqs


def make_batches(trajs, context_len=10, batch_size=64):
    n_traj, traj_len, dim = trajs.shape
    contexts, targets = [], []

    for i in range(n_traj):
        for t in range(traj_len - context_len - 1):
            contexts.append(trajs[i, t:t+context_len])
            targets.append(trajs[i, t+context_len])

    contexts = np.array(contexts)
    targets = np.array(targets)

    idx = np.random.permutation(len(contexts))
    contexts, targets = contexts[idx], targets[idx]

    batches = []
    for i in range(0, len(contexts), batch_size):
        ctx = torch.tensor(contexts[i:i+batch_size], device=device)
        tgt = torch.tensor(targets[i:i+batch_size], device=device)
        batches.append((ctx, tgt))
    return batches


# ============================================================================
# Diagnostic test
# ============================================================================

def run_diagnostic_test():
    print("=" * 70)
    print("DIAGNOSTIC LTM TEST - REDUNDANT PATTERN DATA")
    print("=" * 70)
    print()
    print("This test captures and displays internal values from:")
    print("  1. Surprise gating (Titans-style prediction error)")
    print("  2. Cross-bank binding (joint key product across banks)")
    print()
    print("Data structure: [P0, P1, P0, P2, P0, P3, P0, ...]")
    print("  - P0 repeats at positions 0, 3, 6, 9, ... (should have LOW surprise)")
    print("  - P1, P2, P3 are novel (should have HIGH surprise)")
    print()

    # Generate redundant pattern data
    print("Generating redundant pattern data...")
    train_seqs, train_pattern_ids, base_patterns = generate_redundant_sequences(100, 30, dim=3)
    val_seqs, val_pattern_ids, _ = generate_redundant_sequences(30, 30, dim=3)

    context_len = 10
    train_batches = make_batches(train_seqs, context_len, batch_size=64)
    val_batches = make_batches(val_seqs, context_len, batch_size=64)

    print(f"  Train: {len(train_batches)} batches")
    print(f"  Val: {len(val_batches)} batches")
    print(f"  Base patterns: {len(base_patterns)}")
    print()

    # Create model
    model = InstrumentedModel(
        input_dim=3,
        hidden_dim=64,
        n_layers=2,
        n_orthogonal_sets=4,
        planes_per_set=8
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Collect diagnostics before training
    print("=" * 70)
    print("BEFORE TRAINING (random weights)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        ctx, tgt = val_batches[0]
        _ = model(ctx, capture_diagnostics=True)
        diag = model.get_all_diagnostics()

        print()
        print("Block 0 Diagnostics:")
        print(f"  Cross-bank surprise:     {diag['block0_cross_bank_surprise']:.6f}")
        print(f"  Cross-bank write gate:   {diag['block0_cross_bank_write_gate']:.4f}")
        print(f"  Joint key magnitude:     {diag['block0_joint_key_magnitude']:.4f}")
        print(f"  Joint key phase std:     {diag['block0_joint_key_phase_std']:.4f}")
        print(f"  LTM surprise:            {diag['block0_ltm_surprise']:.6f}")
        print(f"  LTM write gate:          {diag['block0_ltm_write_gate']:.4f}")
        print(f"  Content-pos gate:        {diag['block0_content_pos_gate']:.4f}")
        print(f"  LTM weight (sigmoid):    {diag['block0_ltm_weight']:.4f}")
        print()
        print("Block 1 Diagnostics:")
        print(f"  Cross-bank surprise:     {diag['block1_cross_bank_surprise']:.6f}")
        print(f"  Cross-bank write gate:   {diag['block1_cross_bank_write_gate']:.4f}")
        print(f"  LTM surprise:            {diag['block1_ltm_surprise']:.6f}")
        print(f"  LTM write gate:          {diag['block1_ltm_write_gate']:.4f}")

    # Train
    print()
    print("=" * 70)
    print("TRAINING (10 epochs)")
    print("=" * 70)
    print()

    diag_history = []

    for epoch in range(10):
        model.train()
        train_loss = 0
        for ctx, tgt in train_batches:
            optimizer.zero_grad()
            pred = model(ctx)[:, -1, :]
            loss = F.mse_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation with diagnostics
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ctx, tgt in val_batches:
                pred = model(ctx, capture_diagnostics=True)[:, -1, :]
                val_loss += F.mse_loss(pred, tgt).item()

        train_loss /= len(train_batches)
        val_loss /= len(val_batches)

        # Capture diagnostics at this epoch
        diag = model.get_all_diagnostics()
        diag_history.append(diag)

        print(f"Epoch {epoch+1:2d}: train={train_loss:.6f}, val={val_loss:.6f}")
        print(f"         CB_surprise={diag['block0_cross_bank_surprise']:.4f}, "
              f"CB_gate={diag['block0_cross_bank_write_gate']:.3f}, "
              f"LTM_surprise={diag['block0_ltm_surprise']:.4f}, "
              f"LTM_gate={diag['block0_ltm_write_gate']:.3f}")

    # Final diagnostics
    print()
    print("=" * 70)
    print("AFTER TRAINING")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        ctx, tgt = val_batches[0]
        _ = model(ctx, capture_diagnostics=True)
        diag = model.get_all_diagnostics()

        print()
        print("Block 0 Diagnostics:")
        print(f"  Cross-bank surprise:     {diag['block0_cross_bank_surprise']:.6f}")
        print(f"  Cross-bank write gate:   {diag['block0_cross_bank_write_gate']:.4f}")
        print(f"  Joint key magnitude:     {diag['block0_joint_key_magnitude']:.4f}")
        print(f"  Joint key phase std:     {diag['block0_joint_key_phase_std']:.4f}")
        print(f"  LTM surprise:            {diag['block0_ltm_surprise']:.6f}")
        print(f"  LTM write gate:          {diag['block0_ltm_write_gate']:.4f}")
        print(f"  Content-pos gate:        {diag['block0_content_pos_gate']:.4f}")
        print(f"  LTM weight (sigmoid):    {diag['block0_ltm_weight']:.4f}")
        print()
        print("Block 1 Diagnostics:")
        print(f"  Cross-bank surprise:     {diag['block1_cross_bank_surprise']:.6f}")
        print(f"  Cross-bank write gate:   {diag['block1_cross_bank_write_gate']:.4f}")
        print(f"  LTM surprise:            {diag['block1_ltm_surprise']:.6f}")
        print(f"  LTM write gate:          {diag['block1_ltm_write_gate']:.4f}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Check if surprise is varying meaningfully
    init_diag = diag_history[0]
    final_diag = diag_history[-1]

    # Surprise analysis
    cb_surprise_change = final_diag['block0_cross_bank_surprise'] - init_diag['block0_cross_bank_surprise']
    ltm_surprise_change = final_diag['block0_ltm_surprise'] - init_diag['block0_ltm_surprise']

    print("Surprise Evolution (Block 0):")
    print(f"  Cross-bank surprise: {init_diag['block0_cross_bank_surprise']:.4f} -> {final_diag['block0_cross_bank_surprise']:.4f} ({cb_surprise_change:+.4f})")
    print(f"  LTM surprise:        {init_diag['block0_ltm_surprise']:.4f} -> {final_diag['block0_ltm_surprise']:.4f} ({ltm_surprise_change:+.4f})")
    print()

    # Write gate analysis
    cb_gate_change = final_diag['block0_cross_bank_write_gate'] - init_diag['block0_cross_bank_write_gate']
    ltm_gate_change = final_diag['block0_ltm_write_gate'] - init_diag['block0_ltm_write_gate']

    print("Write Gate Evolution (Block 0):")
    print(f"  Cross-bank gate: {init_diag['block0_cross_bank_write_gate']:.4f} -> {final_diag['block0_cross_bank_write_gate']:.4f} ({cb_gate_change:+.4f})")
    print(f"  LTM gate:        {init_diag['block0_ltm_write_gate']:.4f} -> {final_diag['block0_ltm_write_gate']:.4f} ({ltm_gate_change:+.4f})")
    print()

    # Interpretation
    print("Interpretation:")

    if abs(cb_surprise_change) < 0.01 and final_diag['block0_cross_bank_write_gate'] > 0.45 and final_diag['block0_cross_bank_write_gate'] < 0.55:
        print("  [~] Cross-bank write gate is near 0.5 - surprise gating may not be selective")
    elif final_diag['block0_cross_bank_write_gate'] > 0.7:
        print("  [+] Cross-bank write gate > 0.7 - model is writing most values (high novelty)")
    elif final_diag['block0_cross_bank_write_gate'] < 0.3:
        print("  [+] Cross-bank write gate < 0.3 - model is selective (low novelty)")
    else:
        print("  [+] Cross-bank write gate is moderate - surprise gating is active")

    if final_diag['block0_joint_key_magnitude'] < 0.1:
        print("  [!] Joint key magnitude is very low - cross-bank binding may be vanishing")
    elif final_diag['block0_joint_key_magnitude'] > 0.5:
        print("  [+] Joint key magnitude is healthy - cross-bank binding is active")
    else:
        print("  [~] Joint key magnitude is moderate - cross-bank binding partially active")

    if final_diag['block0_ltm_weight'] < 0.3:
        print("  [~] LTM weight < 0.3 - LTM contribution is low")
    elif final_diag['block0_ltm_weight'] > 0.6:
        print("  [+] LTM weight > 0.6 - LTM is contributing significantly")
    else:
        print("  [+] LTM weight is moderate - LTM is contributing")

    if final_diag['block0_content_pos_gate'] > 0.7:
        print("  [+] Content-pos gate > 0.7 - prioritizing content over position")
    elif final_diag['block0_content_pos_gate'] < 0.3:
        print("  [~] Content-pos gate < 0.3 - prioritizing position over content")
    else:
        print("  [+] Content-pos gate is balanced - mixing content and position")

    # Per-position surprise analysis
    print()
    print("=" * 70)
    print("PER-POSITION SURPRISE ANALYSIS")
    print("=" * 70)
    print()
    print("Data pattern: P0 at positions 0,3,6,9 (repeated), others are novel")
    print()

    # Get per-position data from final diagnostics
    if 'block0_ltm_surprise_per_pos' in diag:
        surprise_per_pos = diag['block0_ltm_surprise_per_pos']
        gate_per_pos = diag['block0_ltm_gate_per_pos']

        # Positions where P0 (repeated pattern) appears: 0, 3, 6, 9
        repeated_positions = [i for i in range(len(surprise_per_pos)) if i % 3 == 0]
        novel_positions = [i for i in range(len(surprise_per_pos)) if i % 3 != 0]

        # Calculate mean surprise at repeated vs novel positions
        if len(repeated_positions) > 1:  # Skip position 0 (no history)
            repeated_surprise = np.mean([surprise_per_pos[i] for i in repeated_positions[1:]])
            repeated_gate = np.mean([gate_per_pos[i] for i in repeated_positions[1:]])
        else:
            repeated_surprise = 0
            repeated_gate = 0

        novel_surprise = np.mean([surprise_per_pos[i] for i in novel_positions])
        novel_gate = np.mean([gate_per_pos[i] for i in novel_positions])

        print(f"Surprise at REPEATED positions (P0):  {repeated_surprise:.4f}")
        print(f"Surprise at NOVEL positions (P1-P3):  {novel_surprise:.4f}")
        print(f"Ratio (novel/repeated):               {novel_surprise/max(repeated_surprise, 0.0001):.2f}x")
        print()
        print(f"Write gate at REPEATED positions:     {repeated_gate:.4f}")
        print(f"Write gate at NOVEL positions:        {novel_gate:.4f}")
        print()

        # Show per-position breakdown
        print("Per-position breakdown (first 10):")
        print("  Pos  Pattern  Surprise  Gate")
        print("  " + "-" * 35)
        for i in range(min(10, len(surprise_per_pos))):
            pattern = "P0 (rep)" if i % 3 == 0 else f"P{1 + (i % 3)} (nov)"
            print(f"  {i:3d}  {pattern:8s}  {surprise_per_pos[i]:8.4f}  {gate_per_pos[i]:.4f}")

        print()

        # Interpretation
        if novel_surprise > repeated_surprise * 1.2:
            print("[+] GOOD: Novel patterns have higher surprise than repeated!")
            print("    Surprise gating IS distinguishing between seen/unseen patterns.")
        elif abs(novel_surprise - repeated_surprise) < 0.1 * max(novel_surprise, repeated_surprise):
            print("[!] PROBLEM: Surprise is similar for repeated and novel patterns.")
            print("    Surprise gating is NOT distinguishing between seen/unseen.")
        else:
            print("[~] MIXED: Some difference in surprise, but not strong.")

    else:
        print("  [!] Per-position diagnostics not available")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return diag_history


if __name__ == "__main__":
    run_diagnostic_test()
