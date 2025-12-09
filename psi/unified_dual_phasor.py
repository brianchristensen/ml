"""
Unified Dual-Addressing Phasor Model

Combines two addressing modes in O(n):
1. POSITIONAL: Fixed random phases for position-based retrieval (copy task)
2. CONTENT: Learned phases from input for content-based retrieval (associative recall)

The key insight: phase = pos_phase + content_phase
- pos_phase: Fixed random phases (like phasor_optimal.py)
- content_phase: Learned projection of input (like phase_binding_memory.py)

Both modes use O(n) cumsum for causal accumulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class DualAddressingPhasor(nn.Module):
    """
    Dual-mode phasor with both positional and content addressing.

    Mode 1 (Positional): Fixed random phases enable position-to-position retrieval
    Mode 2 (Content): Learned phases enable content-based key-value binding

    Total phase = alpha * pos_phase + beta * content_phase
    Where alpha/beta are learnable per-layer scalars.
    """

    def __init__(self, dim, n_oscillators=32, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Positional phases: FIXED, not learned (enables length generalization)
        # Use structured frequencies like RoPE for better extrapolation
        freqs = 1.0 / (10000.0 ** (torch.arange(0, n_oscillators, dtype=torch.float32) / n_oscillators))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        pos_phases = torch.outer(positions, freqs)  # [max_len, K]
        self.register_buffer('pos_phases', pos_phases)

        # Content phases: learned projection
        self.to_content_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, n_oscillators)
        )

        # Learnable mixing weights for the two modes
        self.pos_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.content_scale = nn.Parameter(torch.ones(1) * 0.5)

        # Value projection
        self.to_value = nn.Linear(dim, dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        """
        x: [B, L, D] -> [B, L, D]

        O(n) complexity via cumsum.
        """
        B, L, D = x.shape
        K = self.n_oscillators

        # Get positional phases [L, K]
        pos_phase = self.pos_phases[:L]

        # Get content phases [B, L, K]
        content_phase = torch.tanh(self.to_content_phase(x)) * math.pi

        # Combine: total_phase = pos_scale * pos + content_scale * content
        # Broadcasting: [L, K] + [B, L, K] -> [B, L, K]
        total_phase = self.pos_scale * pos_phase.unsqueeze(0) + self.content_scale * content_phase

        # Value projection [B, L, D]
        value = self.to_value(x)

        # Complex binding: value * exp(i * phase)
        # For efficiency, work with real/imag separately
        cos_phase = torch.cos(total_phase)  # [B, L, K]
        sin_phase = torch.sin(total_phase)  # [B, L, K]

        # Bind: [B, L, K, 1] * [B, L, 1, D] -> [B, L, K, D]
        bound_real = cos_phase.unsqueeze(-1) * value.unsqueeze(-2)
        bound_imag = sin_phase.unsqueeze(-1) * value.unsqueeze(-2)

        # Causal accumulation via cumsum [B, L, K, D]
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind with same phase (query = key for self-attention-like behavior)
        # [B, L, K, D] * [B, L, K, 1] -> sum over K -> [B, L, D]
        retrieved_real = (mem_real * cos_phase.unsqueeze(-1) + mem_imag * sin_phase.unsqueeze(-1)).sum(dim=2)

        # Normalize
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        retrieved = retrieved_real / norm

        return x + self.to_out(retrieved)


class DualPhasorBlock(nn.Module):
    """
    Full block with dual-addressing phasor + FFN.
    """

    def __init__(self, dim, n_oscillators=32, max_seq_len=8192, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.phasor = DualAddressingPhasor(dim, n_oscillators, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, x):
        x = x + self.phasor(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class UnifiedDualPhasorLM(nn.Module):
    """
    Language model using dual-addressing phasor blocks.

    Combines strengths of:
    - phasor_optimal.py (positional retrieval for copy)
    - phase_binding_memory.py (content retrieval for associative recall)

    Maintains O(n) complexity.
    """

    def __init__(self, vocab_size, dim=256, num_layers=4, n_oscillators=32,
                 max_seq_len=8192, ffn_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            DualPhasorBlock(dim, n_oscillators, max_seq_len, ffn_mult)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, x, target_indices=None):
        """x: [B, L] token indices -> [B, L, vocab_size] logits"""
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h)

        return self.head(self.norm_out(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids, max_length, temperature=1.0, top_k=50):
        """Autoregressive generation."""
        self.eval()
        generated = input_ids.clone()

        while generated.shape[1] < max_length:
            context = generated[:, -self.max_seq_len:]
            logits = self(context)[:, -1, :] / temperature

            if top_k > 0:
                top_logits, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = F.softmax(top_logits, dim=-1)
                next_token = top_idx.gather(-1, torch.multinomial(probs, 1))
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


class UnifiedDualPhasorRecall(nn.Module):
    """
    Associative recall model using dual-addressing phasors.

    Input format: [k1, v1, k2, v2, ..., kN, vN, QUERY, q1, QUERY, q2, ...]
    """

    def __init__(self, vocab_size, dim=128, n_oscillators=64, num_layers=4,
                 query_token=None, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.query_token = query_token if query_token is not None else vocab_size

        self.embed = nn.Embedding(vocab_size + 1, dim)

        self.layers = nn.ModuleList([
            DualPhasorBlock(dim, n_oscillators, max_seq_len, ffn_mult=2)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x, target_indices=None):
        """
        x: [B, L] token indices or [B, L, V] one-hot
        Returns: logits [B, L, vocab_size]
        """
        if x.dim() == 3:
            indices = x.argmax(dim=-1)
        else:
            indices = x

        h = self.embed(indices)

        for layer in self.layers:
            h = layer(h)

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
    num_layers = 4
    n_oscillators = 32

    model = UnifiedDualPhasorLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        n_oscillators=n_oscillators,
        max_seq_len=8192
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train on short sequences (100-500)
    print("Training on sequences 100-500...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        n_batches = 50

        for _ in range(n_batches):
            batch_size = 16
            seq_len = np.random.randint(100, 501)

            # Generate random sequences
            x = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)

            optimizer.zero_grad()
            logits = model(x)

            # Predict next token (shifted)
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

    # Evaluate on various lengths
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

    model = UnifiedDualPhasorRecall(
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
        """Generate associative recall batch."""
        available_tokens = list(range(4, vocab_size))  # 0-3 reserved

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
    for epoch in range(60):
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

    # Test generalization to more pairs
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


if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED DUAL-ADDRESSING PHASOR MODEL")
    print("Combining positional + content addressing in O(n)")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Test both tasks
    copy_model = test_copy_task()
    recall_model = test_associative_recall()

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
