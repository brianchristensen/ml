"""
Test using full dim as phases (like phasor_optimal.py) instead of n_phases.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


class FullDimHybridBlock(nn.Module):
    """
    Uses full dim as phases (like phasor_optimal.py) instead of n_phases.
    This removes the n_phases hyperparameter entirely.
    """
    def __init__(self, dim, max_seq_len=16384):
        super().__init__()
        self.dim = dim

        # ===== Positional Memory =====
        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)
        self.pos_value = nn.Linear(dim, dim)

        # ===== Content-Based KV Memory =====
        self.key_encoder = nn.Linear(dim, dim)
        self.kv_value = nn.Linear(dim, dim)

        # Learned gate: is this a value position?
        self.value_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # ===== Blending =====
        self.blend_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape

        # ========== Positional Memory (full dim phases) ==========
        pos_phases = self.pos_phases[:L]  # [L, D]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        pos_values = self.pos_value(x)  # [B, L, D]

        # Bind with position using real arithmetic
        pos_bound_cos = pos_cos * pos_values  # [B, L, D]
        pos_bound_sin = pos_sin * pos_values

        pos_mem_cos = torch.cumsum(pos_bound_cos, dim=1)
        pos_mem_sin = torch.cumsum(pos_bound_sin, dim=1)

        # Retrieve by position (conjugate = negate sin)
        pos_retrieved = pos_cos * pos_mem_cos + pos_sin * pos_mem_sin
        pos_retrieved = pos_retrieved / math.sqrt(D)

        # ========== Content-Based KV Memory (full dim phases) ==========
        key_phases = torch.tanh(self.key_encoder(x)) * math.pi  # [B, L, D]
        key_cos = torch.cos(key_phases)
        key_sin = torch.sin(key_phases)

        kv_values = self.kv_value(x)  # [B, L, D]

        # Learn which positions are values
        x_shifted = torch.roll(x, shifts=1, dims=1)
        x_shifted[:, 0] = 0
        gate_input = torch.cat([x, x_shifted], dim=-1)
        value_gates = self.value_gate(gate_input)  # [B, L, 1]

        # Shift key phasors to align with values
        key_cos_shifted = torch.roll(key_cos, shifts=1, dims=1)
        key_sin_shifted = torch.roll(key_sin, shifts=1, dims=1)
        key_cos_shifted[:, 0] = 0
        key_sin_shifted[:, 0] = 0

        # Bind key phase with value, gated
        kv_bound_cos = key_cos_shifted * kv_values * value_gates
        kv_bound_sin = key_sin_shifted * kv_values * value_gates

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(value_gates, dim=1).clamp(min=1)

        # Retrieve by content (conjugate query)
        kv_retrieved = key_cos * kv_mem_cos + key_sin * kv_mem_sin
        kv_retrieved = kv_retrieved / (torch.sqrt(gate_cumsum) * math.sqrt(D))

        # ========== Blend ==========
        blend_weights = F.softmax(self.blend_gate(x), dim=-1)

        blended = (
            blend_weights[..., 0:1] * pos_retrieved +
            blend_weights[..., 1:2] * kv_retrieved
        )

        return x + self.to_out(blended)


class FullDimHybridModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, max_seq_len=16384):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            FullDimHybridBlock(dim, max_seq_len) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


# Tasks
def generate_copy_task(batch_size, seq_len, vocab_size, n_to_copy=8):
    half = seq_len // 2
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    for b in range(batch_size):
        tokens = torch.randint(1, min(vocab_size, 32), (n_to_copy,))
        sequences[b, :n_to_copy] = tokens
        for i in range(n_to_copy):
            targets[b, half + i] = tokens[i]
    return sequences, targets


def generate_induction_task(batch_size, seq_len, vocab_size, n_pairs=8):
    key_vocab = vocab_size // 2
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randperm(key_vocab)[:n_pairs]
        vals = torch.randint(key_vocab, key_vocab + key_vocab, (n_pairs,))
        for i in range(n_pairs):
            sequences[b, i * 2] = keys[i]
            sequences[b, i * 2 + 1] = vals[i]
        pair_end = n_pairs * 2
        delim_end = pair_end + 10
        query_order = torch.randperm(n_pairs)
        for i, idx in enumerate(query_order):
            pos = delim_end + i * 2
            if pos + 1 < seq_len:
                sequences[b, pos] = keys[idx]
                sequences[b, pos + 1] = vals[idx]
                targets[b, pos] = vals[idx]
    return sequences, targets


def train_and_eval(model, task_fn, n_epochs=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    best_val = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for _ in range(50):
            seq, tgt = task_fn(64, 128, 64)
            seq, tgt = seq.to(device), tgt.to(device)
            logits = model(seq)
            loss = F.cross_entropy(logits.view(-1, 64), tgt.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(20):
                seq, tgt = task_fn(64, 128, 64)
                seq, tgt = seq.to(device), tgt.to(device)
                logits = model(seq)
                mask = tgt != -100
                if mask.sum() > 0:
                    preds = logits.argmax(dim=-1)
                    correct += ((preds == tgt) & mask).sum().item()
                    total += mask.sum().item()
        val_acc = correct / total if total > 0 else 0
        best_val = max(best_val, val_acc)

        if epoch % 3 == 0 or epoch == 1:
            print(f'    Epoch {epoch:2d}: loss={total_loss/50:.3f}, val={val_acc:.1%}')
        if val_acc >= 0.99:
            print(f'    Converged at epoch {epoch}!')
            break
    return best_val


if __name__ == "__main__":
    from kv_memory import FastHybridMemoryModel

    print('=' * 70)
    print('FULL DIM PHASES vs n_phases=32')
    print('=' * 70)

    for n_layers in [2, 3, 4]:
        print(f'\n=== {n_layers} LAYERS ===')

        # Full dim model
        model_full = FullDimHybridModel(vocab_size=64, dim=64, n_layers=n_layers).to(device)
        n_params_full = sum(p.numel() for p in model_full.parameters())
        print(f'  FullDimHybrid: {n_params_full:,} params')

        # n_phases=32 model (FastHybrid, no refiner)
        model_32 = FastHybridMemoryModel(vocab_size=64, dim=64, n_layers=n_layers, n_phases=32).to(device)
        n_params_32 = sum(p.numel() for p in model_32.parameters())
        print(f'  FastHybrid (n_phases=32): {n_params_32:,} params')

        print(f'\n  Associative Recall:')
        print(f'    FullDim:')
        assoc_full = train_and_eval(FullDimHybridModel(vocab_size=64, dim=64, n_layers=n_layers).to(device), generate_induction_task)
        print(f'    n_phases=32:')
        assoc_32 = train_and_eval(FastHybridMemoryModel(vocab_size=64, dim=64, n_layers=n_layers, n_phases=32).to(device), generate_induction_task)

        print(f'\n  Copy:')
        print(f'    FullDim:')
        copy_full = train_and_eval(FullDimHybridModel(vocab_size=64, dim=64, n_layers=n_layers).to(device), generate_copy_task)
        print(f'    n_phases=32:')
        copy_32 = train_and_eval(FastHybridMemoryModel(vocab_size=64, dim=64, n_layers=n_layers, n_phases=32).to(device), generate_copy_task)

        print(f'\n  Summary ({n_layers} layers):')
        print(f'    FullDim ({n_params_full:,} params): Assoc={assoc_full:.1%}, Copy={copy_full:.1%}')
        print(f'    n_phases=32 ({n_params_32:,} params): Assoc={assoc_32:.1%}, Copy={copy_32:.1%}')
