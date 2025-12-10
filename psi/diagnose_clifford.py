"""
Diagnose Clifford model:
1. Why is positional addressing not working for copy task?
2. Where is the time being spent?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from clifford_memory import OrthogonalModel, OrthogonalBivectorBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================================
# Timing Diagnostics
# ============================================================================

class TimedOrthogonalBivectorBlock(nn.Module):
    """Instrumented version of OrthogonalBivectorBlock for timing"""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.timings = {}

    def forward(self, x):
        B, L, D = x.shape
        block = self.block

        if device == 'cuda':
            torch.cuda.synchronize()

        # Time key/query encoding
        t0 = time.time()
        key_phase = torch.tanh(block.key_encoder(x)) * math.pi
        query_phase = torch.tanh(block.query_encoder(x)) * math.pi
        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['key_query_encode'] = time.time() - t0

        # Time value projection
        t0 = time.time()
        V = block.to_value(x).to(torch.complex64)
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['value_proj'] = time.time() - t0

        # Time content memory (the main cumsum loop)
        t0 = time.time()
        key_phasor = key_phasor.view(B, L, block.n_sets, block.planes_per_set)
        query_phasor = query_phasor.view(B, L, block.n_sets, block.planes_per_set)
        weights = F.softmax(block.set_weights, dim=0)
        total_retrieved = torch.zeros(B, L, D, device=x.device)

        for s in range(block.n_sets):
            key_s = key_phasor[:, :, s, :]
            query_s = query_phasor[:, :, s, :]
            bound = key_s.unsqueeze(-1) * V.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * query_s.conj().unsqueeze(-1)
            retrieved = retrieved.sum(dim=2).real
            total_retrieved = total_retrieved + weights[s] * retrieved
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['content_memory'] = time.time() - t0

        # Time positional memory
        t0 = time.time()
        pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
        pos_phase = pos * block.pos_freqs * 2 * math.pi
        pos_phasor = torch.exp(1j * pos_phase)
        pos_key = pos_phasor.unsqueeze(0).expand(B, -1, -1)
        pos_query = pos_phasor.unsqueeze(0).expand(B, -1, -1)
        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['positional_memory'] = time.time() - t0

        # Time gating
        t0 = time.time()
        gate = block.content_pos_gate(x)
        pos_contribution = torch.sigmoid(block.pos_weight) * retrieved_pos
        total_retrieved = gate * total_retrieved + (1 - gate) * pos_contribution
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['gating'] = time.time() - t0

        # Time output
        t0 = time.time()
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * block.planes_per_set).view(1, L, 1)
        result = x + block.to_out(total_retrieved / norm)
        if device == 'cuda':
            torch.cuda.synchronize()
        self.timings['output'] = time.time() - t0

        return result


def time_breakdown(seq_len=128, batch_size=8, n_runs=5):
    """Profile where time is spent in Clifford model"""
    print(f"\n{'='*60}")
    print(f"TIMING BREAKDOWN (seq_len={seq_len}, batch={batch_size})")
    print('='*60)

    model = OrthogonalModel(
        vocab_size=64,
        dim=64,
        n_layers=4,
        n_orthogonal_sets=4,
        planes_per_set=8
    ).to(device)

    # Wrap blocks with timing
    timed_blocks = [TimedOrthogonalBivectorBlock(block) for block in model.blocks]

    x = torch.randint(0, 64, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        h = model.embed(x)
        for tb, norm in zip(timed_blocks, model.norms):
            h = tb(norm(h))
        _ = model.head(model.norm_out(h))

    # Collect timings
    all_timings = {k: [] for k in ['key_query_encode', 'value_proj', 'content_memory',
                                    'positional_memory', 'gating', 'output']}

    for _ in range(n_runs):
        with torch.no_grad():
            h = model.embed(x)
            for tb, norm in zip(timed_blocks, model.norms):
                h = tb(norm(h))
            _ = model.head(model.norm_out(h))

        # Aggregate across layers
        for key in all_timings:
            total = sum(tb.timings[key] for tb in timed_blocks)
            all_timings[key].append(total)

    # Print results
    print(f"\n{'Component':<25} {'Time (ms)':>12} {'%':>8}")
    print('-' * 45)

    total_time = 0
    for key in all_timings:
        avg_time = np.mean(all_timings[key]) * 1000
        total_time += avg_time

    for key in all_timings:
        avg_time = np.mean(all_timings[key]) * 1000
        pct = avg_time / total_time * 100
        print(f"{key:<25} {avg_time:>10.2f}ms {pct:>7.1f}%")

    print('-' * 45)
    print(f"{'TOTAL (blocks only)':<25} {total_time:>10.2f}ms")


# ============================================================================
# Positional Addressing Diagnostics
# ============================================================================

def diagnose_positional_retrieval():
    """Test if positional addressing actually works"""
    print(f"\n{'='*60}")
    print("POSITIONAL ADDRESSING DIAGNOSIS")
    print('='*60)

    dim = 64
    block = OrthogonalBivectorBlock(dim, n_orthogonal_sets=4, planes_per_set=8, pos_planes=16).to(device)
    block.eval()

    # Create simple test: different values at different positions
    seq_len = 32
    batch_size = 1

    # Input: each position has a unique vector
    x = torch.randn(batch_size, seq_len, dim, device=device)
    # Make positions very distinct
    for i in range(seq_len):
        x[0, i, :] = 0
        x[0, i, i % dim] = 1.0  # One-hot-ish

    with torch.no_grad():
        # Get the gate values - how much is position vs content?
        gate = block.content_pos_gate(x)
        print(f"\nContent-Position Gate values:")
        print(f"  Mean: {gate.mean().item():.3f}")
        print(f"  Std:  {gate.std().item():.3f}")
        print(f"  Min:  {gate.min().item():.3f}")
        print(f"  Max:  {gate.max().item():.3f}")
        print(f"  (gate=1 means 100% content, gate=0 means 100% position)")

        # Check pos_weight
        pos_weight = torch.sigmoid(block.pos_weight).item()
        print(f"\nPositional weight (sigmoid): {pos_weight:.3f}")

        # Run full forward to see retrieval
        V = block.to_value(x).to(torch.complex64)

        # Positional retrieval only
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        pos_phase = pos * block.pos_freqs * 2 * math.pi
        pos_phasor = torch.exp(1j * pos_phase)
        pos_key = pos_phasor.unsqueeze(0).expand(batch_size, -1, -1)
        pos_query = pos_phasor.unsqueeze(0).expand(batch_size, -1, -1)

        bound_pos = pos_key.unsqueeze(-1) * V.unsqueeze(-2)
        memory_pos = torch.cumsum(bound_pos, dim=1)
        retrieved_pos = memory_pos * pos_query.conj().unsqueeze(-1)
        retrieved_pos = retrieved_pos.sum(dim=2).real

        # Check if positional retrieval at position i retrieves value from position i
        print(f"\nPositional retrieval self-similarity:")
        print(f"  (Does querying position i retrieve value from position i?)")

        # For each position, what's the similarity to each stored value?
        similarities = []
        for query_pos in [0, 5, 10, 15, 20, 25, 30]:
            query_vec = retrieved_pos[0, query_pos]  # What we retrieved

            # Compare to all stored values
            sims = []
            for stored_pos in range(seq_len):
                stored_vec = V[0, stored_pos].real
                sim = F.cosine_similarity(query_vec.unsqueeze(0), stored_vec.unsqueeze(0)).item()
                sims.append(sim)

            best_match = np.argmax(sims)
            similarities.append((query_pos, best_match, sims[query_pos], max(sims)))

        print(f"\n  {'Query Pos':<12} {'Best Match':<12} {'Self Sim':<12} {'Best Sim':<12}")
        print(f"  {'-'*48}")
        for qp, bm, ss, bs in similarities:
            match_str = "YES" if qp == bm else f"NO (got {bm})"
            print(f"  {qp:<12} {match_str:<12} {ss:<12.3f} {bs:<12.3f}")


def diagnose_copy_task():
    """Test model on simplified copy task"""
    print(f"\n{'='*60}")
    print("COPY TASK DIAGNOSIS")
    print('='*60)

    model = OrthogonalModel(
        vocab_size=64,
        dim=64,
        n_layers=4,
        n_orthogonal_sets=4,
        planes_per_set=8
    ).to(device)
    model.eval()

    # Simple copy: [1, 2, 3, 0, 0, 0, 0, 0] -> predict [1, 2, 3] at positions 5, 6, 7
    seq = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]], device=device)

    with torch.no_grad():
        logits = model(seq)
        preds = logits.argmax(dim=-1)

        print(f"\nInput:      {seq[0].tolist()}")
        print(f"Predictions: {preds[0].tolist()}")

        # What are the top predictions at positions 5, 6, 7?
        print(f"\nTop-5 predictions at recall positions:")
        for pos in [5, 6, 7]:
            probs = F.softmax(logits[0, pos], dim=-1)
            top5 = torch.topk(probs, 5)
            print(f"  Position {pos}: {list(zip(top5.indices.tolist(), [f'{p:.2f}' for p in top5.values.tolist()]))}")
            expected = seq[0, pos - 5].item() if pos - 5 < 3 else 0
            print(f"    Expected: {expected}, Got prob: {probs[expected].item():.3f}")


if __name__ == "__main__":
    # Timing breakdown
    for seq_len in [128, 512, 2048]:
        time_breakdown(seq_len=seq_len)

    # Positional diagnostics
    diagnose_positional_retrieval()

    # Copy task diagnostics
    diagnose_copy_task()
