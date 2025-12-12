"""
Slim Phasor value_dim sweep - find optimal tradeoff between speed and accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class SlimPhasorBlock(nn.Module):
    def __init__(self, dim, n_phases=128, value_dim=8, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_phases = n_phases
        self.value_dim = value_dim

        pos_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('pos_phases', pos_phases)

        self.mem1_value = nn.Linear(dim, dim)
        self.mem1_out = nn.Linear(dim, dim)
        self.offset_predictor = nn.Linear(dim * 2, dim)
        self.to_magnitude = nn.Linear(dim, dim)
        self.magnitude_scale = nn.Parameter(torch.tensor(5.0))

        self.key_encoder = nn.Linear(dim, n_phases)
        self.value_encoder = nn.Linear(dim, value_dim)
        self.storage_key = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_phases)
        )
        self.store_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.kv_out = nn.Linear(value_dim, dim)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1
        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * weighted_v1, dim=1)
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * weighted_v1, dim=1)
        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * math.pi
        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin
        pos_ret = (mem1_cos * query_cos + mem1_sin * query_sin) / math.sqrt(D)
        pos_out = self.mem1_out(pos_ret)

        query_phase = torch.tanh(self.key_encoder(x)) * math.pi
        values = self.value_encoder(x)
        store_gate = self.store_gate(x)

        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions
        storage_phase = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * math.pi

        store_cos = torch.cos(storage_phase)
        store_sin = torch.sin(storage_phase)
        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)
        kv_mem_cos = kv_mem_cos / torch.sqrt(gate_cumsum).unsqueeze(-1)
        kv_mem_sin = kv_mem_sin / torch.sqrt(gate_cumsum).unsqueeze(-1)

        q_cos = torch.cos(query_phase)
        q_sin = torch.sin(query_phase)
        kv_retrieved = (
            q_cos.unsqueeze(-1) * kv_mem_cos +
            q_sin.unsqueeze(-1) * kv_mem_sin
        ).sum(dim=2) / math.sqrt(self.n_phases)

        kv_out = self.kv_out(kv_retrieved)
        trajectory = x * torch.cos(query_phase).mean(dim=-1, keepdim=True)
        combined = torch.cat([pos_out, kv_out, trajectory], dim=-1)
        return x + self.to_out(combined)


class SlimPhasorModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=4, n_phases=128, value_dim=8, max_seq_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SlimPhasorBlock(dim, n_phases, value_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))


def generate_assoc_task(batch_size, seq_len, vocab_size, n_pairs=8):
    """Associative recall task."""
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


def train_and_eval(value_dim, n_epochs=20):
    """Train and evaluate a model with given value_dim."""
    model = SlimPhasorModel(
        vocab_size=64, dim=64, n_layers=4, n_phases=128,
        value_dim=value_dim, max_seq_len=256
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    for epoch in range(n_epochs):
        model.train()
        for _ in range(50):
            seq, tgt = generate_assoc_task(64, 128, 64)
            seq, tgt = seq.to(device), tgt.to(device)
            logits = model(seq)
            loss = F.cross_entropy(logits.view(-1, 64), tgt.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = generate_assoc_task(64, 128, 64)
            seq, tgt = seq.to(device), tgt.to(device)
            logits = model(seq)
            mask = tgt != -100
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1)
                correct += ((preds == tgt) & mask).sum().item()
                total += mask.sum().item()

    acc = correct / total if total > 0 else 0

    # Speed
    x = torch.randint(0, 64, (32, 256), device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    tps = (32 * 256 * 10) / (time.time() - start)

    return acc, tps


if __name__ == "__main__":
    print("=" * 70)
    print("SLIM PHASOR VALUE_DIM SWEEP")
    print("=" * 70)
    print()
    print("value_dim | Mem Factor | Assoc Acc  | Speed (tok/s)")
    print("-" * 55)

    D = 64  # full dimension for comparison
    for value_dim in [8, 16, 32, 48, 64]:
        mem_factor = value_dim / D
        acc, tps = train_and_eval(value_dim)
        print(f"{value_dim:>9} | {mem_factor:>9.0%} | {acc:>9.1%} | {tps:>12.0f}")

    print()
    print("Done!")
