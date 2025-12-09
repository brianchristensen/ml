"""
Unified Dual-Addressing Phasor Model V2

Key insight from failed V1: Adding pos+content phases dilutes both signals.

New approach: PARALLEL memory streams, each specialized:
1. Positional stream: Fixed random phases (like phasor_optimal.py)
2. Content stream: Learned key-query phases (like phase_binding_memory.py)

Both streams operate in O(n) and their outputs are combined via learned gating.

This is more like having two specialists that can be consulted independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class PositionalPhasorStream(nn.Module):
    """
    Position-based retrieval via fixed random phases.
    Specialized for: copy task, position-to-position retrieval.

    From phasor_optimal.py - random phases are ~orthogonal in high dim.
    """
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        self.dim = dim

        # Fixed random phases per dimension (NOT learned)
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_value = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)  # [1, L, D]

        value = self.to_value(x)  # [B, L, D]

        # Bind: multiply value by exp(i*phase)
        cos_p = torch.cos(phases)
        sin_p = torch.sin(phases)

        bound_real = value * cos_p
        bound_imag = value * sin_p

        # Cumsum for causal memory
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind with same phase (self-retrieval)
        retrieved = mem_real * cos_p + mem_imag * sin_p

        # Normalize by sqrt(position)
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions).view(1, L, 1)

        return retrieved / norm


class ContentPhasorStream(nn.Module):
    """
    Content-based retrieval via learned key-query phases.
    Specialized for: associative recall, key-value binding.

    From phase_binding_memory.py - phases derived from content.
    """
    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Separate key and query encoders (crucial for content addressing!)
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, n_oscillators)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, n_oscillators)
        )

        self.to_value = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators

        # Encode keys and queries to phases
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, K]
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi  # [B, L, K]

        value = self.to_value(x)  # [B, L, D]

        # Bind: value * exp(i * key_phase)
        # [B, L, K, 1] * [B, L, 1, D] -> [B, L, K, D]
        cos_k = torch.cos(key_phase).unsqueeze(-1)
        sin_k = torch.sin(key_phase).unsqueeze(-1)
        value_expanded = value.unsqueeze(-2)

        bound_real = cos_k * value_expanded
        bound_imag = sin_k * value_expanded

        # Cumsum for causal memory [B, L, K, D]
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind with query phase (conjugate = negate the phase)
        cos_q = torch.cos(query_phase).unsqueeze(-1)
        sin_q = torch.sin(query_phase).unsqueeze(-1)

        # Conjugate multiplication: (a+bi)(c-di) = ac+bd + i(bc-ad)
        # We want real part: mem_real * cos_q + mem_imag * sin_q
        retrieved_per_osc = mem_real * cos_q + mem_imag * sin_q  # [B, L, K, D]

        # Sum over oscillators
        retrieved = retrieved_per_osc.sum(dim=2)  # [B, L, D]

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)

        return retrieved / norm


class DualStreamPhasorBlock(nn.Module):
    """
    Combines positional and content streams with learned gating.

    The model can learn to rely on:
    - Positional stream for position-based tasks (copy)
    - Content stream for content-based tasks (associative recall)
    - Both for tasks that need both
    """
    def __init__(self, dim, n_oscillators=32, max_seq_len=8192):
        super().__init__()
        self.dim = dim

        # Two parallel streams
        self.pos_stream = PositionalPhasorStream(dim, max_seq_len)
        self.content_stream = ContentPhasorStream(dim, n_oscillators)

        # Learned gating: decides how much to use each stream
        # Input-dependent gating allows task-specific routing
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # Get retrievals from both streams
        pos_retrieved = self.pos_stream(x)  # [B, L, D]
        content_retrieved = self.content_stream(x)  # [B, L, D]

        # Compute gates per position (can vary throughout sequence)
        gates = self.gate(x)  # [B, L, 2]
        pos_gate = gates[:, :, 0:1]  # [B, L, 1]
        content_gate = gates[:, :, 1:2]  # [B, L, 1]

        # Combine streams
        combined = pos_gate * pos_retrieved + content_gate * content_retrieved

        return x + self.to_out(combined)


class DualStreamPhasorLM(nn.Module):
    """
    Language model using dual-stream phasor blocks.
    """
    def __init__(self, vocab_size, dim=256, num_layers=4, n_oscillators=32,
                 max_seq_len=8192, ffn_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            DualStreamPhasorBlock(dim, n_oscillators, max_seq_len)
            for _ in range(num_layers)
        ])

        # FFN after each block
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Linear(dim * ffn_mult, dim)
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, x, target_indices=None):
        h = self.embed(x)

        for layer, ffn in zip(self.layers, self.ffns):
            h = layer(h)
            h = h + ffn(h)

        return self.head(self.norm_out(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device


class DualStreamPhasorRecall(nn.Module):
    """
    Associative recall model using dual-stream phasors.
    """
    def __init__(self, vocab_size, dim=128, n_oscillators=64, num_layers=4,
                 query_token=None, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.query_token = query_token if query_token is not None else vocab_size

        self.embed = nn.Embedding(vocab_size + 1, dim)

        self.layers = nn.ModuleList([
            DualStreamPhasorBlock(dim, n_oscillators, max_seq_len)
            for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x, target_indices=None):
        if x.dim() == 3:
            indices = x.argmax(dim=-1)
        else:
            indices = x

        h = self.embed(indices)

        for layer, ffn in zip(self.layers, self.ffns):
            h = layer(h)
            h = h + ffn(h)

        return self.output(self.norm_out(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Benchmarks
# ============================================================================

def test_copy_task():
    """Test on copy task (position-based retrieval)."""
    print("=" * 70)
    print("COPY TASK BENCHMARK")
    print("=" * 70)
    print()

    vocab_size = 20
    dim = 128
    num_layers = 2  # Minimal like phasor_optimal
    n_oscillators = 32

    model = DualStreamPhasorLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        n_oscillators=n_oscillators,
        max_seq_len=8192,
        ffn_mult=2
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train on short sequences
    print("Training on sequences 100-500...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        n_batches = 50

        for _ in range(n_batches):
            batch_size = 16
            seq_len = np.random.randint(100, 501)

            x = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(
                logits[:, :-1].reshape(-1, vocab_size),
                x[:, 1:].reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    # Evaluate
    print()
    print("Testing generalization to longer sequences:")
    print(f"{'Length':<10} {'Accuracy':>10}")
    print("-" * 25)

    model.eval()
    for test_len in [500, 1000, 2000, 4000]:
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(10):
                x = torch.randint(3, vocab_size, (8, test_len), device=device)
                logits = model(x)
                preds = logits[:, :-1].argmax(dim=-1)
                targets = x[:, 1:]

                correct += (preds == targets).sum().item()
                total += targets.numel()

        acc = correct / total * 100
        print(f"{test_len:<10} {acc:>9.1f}%")

    return model


def test_associative_recall():
    """Test on associative recall (content-based retrieval)."""
    print()
    print("=" * 70)
    print("ASSOCIATIVE RECALL BENCHMARK")
    print("=" * 70)
    print()

    vocab_size = 50
    n_pairs = 20
    n_queries = 5
    QUERY = 3

    model = DualStreamPhasorRecall(
        vocab_size=vocab_size,
        dim=128,
        n_oscillators=64,
        num_layers=4,
        query_token=QUERY
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Task: {n_pairs} key-value pairs, {n_queries} queries")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def generate_batch(batch_size):
        available_tokens = list(range(4, vocab_size))

        inputs = []
        targets = []

        for _ in range(batch_size):
            np.random.shuffle(available_tokens)
            keys = available_tokens[:n_pairs]
            values = [available_tokens[np.random.randint(n_pairs, len(available_tokens))]
                     for _ in range(n_pairs)]

            kv_dict = dict(zip(keys, values))

            seq = []
            tgt = []
            for k, v in zip(keys, values):
                seq.extend([k, v])
                tgt.extend([0, 0])

            for _ in range(n_queries):
                query_key = keys[np.random.randint(n_pairs)]
                seq.extend([QUERY, query_key])
                tgt.extend([0, kv_dict[query_key]])

            inputs.append(seq)
            targets.append(tgt)

        return (
            torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device)
        )

    print("Training...")
    best_acc = 0.0
    for epoch in range(80):
        model.train()
        total_loss = 0
        n_batches = 30

        for _ in range(n_batches):
            inputs, targets = generate_batch(32)

            optimizer.zero_grad()
            logits = model(inputs)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            inputs, targets = generate_batch(200)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)

            mask = targets != 0
            if mask.sum() > 0:
                acc = (preds[mask] == targets[mask]).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, acc={acc:.1f}% (best: {best_acc:.1f}%)")

    print()
    print(f"Final accuracy: {best_acc:.1f}%")

    # Test generalization
    print()
    print("Generalization to more key-value pairs:")
    print(f"{'Pairs':<10} {'Accuracy':>10}")
    print("-" * 25)

    for test_pairs in [30, 40]:
        if test_pairs > (vocab_size - 4) // 2:
            continue

        def generate_test_batch(batch_size, np_test):
            available_tokens = list(range(4, vocab_size))
            inputs = []
            targets = []

            for _ in range(batch_size):
                np.random.shuffle(available_tokens)
                keys = available_tokens[:np_test]
                values = [available_tokens[np.random.randint(np_test, len(available_tokens))]
                         for _ in range(np_test)]

                kv_dict = dict(zip(keys, values))

                seq = []
                tgt = []
                for k, v in zip(keys, values):
                    seq.extend([k, v])
                    tgt.extend([0, 0])

                nq = max(5, np_test // 4)
                for _ in range(nq):
                    query_key = keys[np.random.randint(np_test)]
                    seq.extend([QUERY, query_key])
                    tgt.extend([0, kv_dict[query_key]])

                inputs.append(seq)
                targets.append(tgt)

            max_len = max(len(s) for s in inputs)
            inputs = [s + [0] * (max_len - len(s)) for s in inputs]
            targets = [t + [0] * (max_len - len(t)) for t in targets]

            return (
                torch.tensor(inputs, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device)
            )

        model.eval()
        with torch.no_grad():
            inputs, targets = generate_test_batch(100, test_pairs)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)

            mask = targets != 0
            acc = (preds[mask] == targets[mask]).float().mean().item() * 100

        print(f"{test_pairs:<10} {acc:>9.1f}%")

    return model


def analyze_gate_usage(model, task='copy'):
    """Analyze how gates are used for different tasks."""
    print()
    print("=" * 70)
    print(f"GATE ANALYSIS: {task.upper()} TASK")
    print("=" * 70)

    model.eval()

    if task == 'copy':
        # Generate copy task data
        x = torch.randint(3, 20, (1, 100), device=device)
        h = model.embed(x)

        print("Gate values (positional vs content) through layers:")
        for i, (layer, ffn) in enumerate(zip(model.layers, model.ffns)):
            gates = layer.gate(h)
            pos_gate = gates[:, :, 0].mean().item()
            content_gate = gates[:, :, 1].mean().item()
            print(f"  Layer {i+1}: pos={pos_gate:.3f}, content={content_gate:.3f}")
            h = layer(h)
            h = h + ffn(h)


if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED DUAL-STREAM PHASOR MODEL V2")
    print("Parallel positional + content streams with learned gating")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Test both tasks
    copy_model = test_copy_task()

    # Analyze gate usage
    analyze_gate_usage(copy_model, 'copy')

    recall_model = test_associative_recall()

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
