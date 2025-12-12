"""
Long Context Test: Can Phasor's O(n) complexity help it learn from longer sequences?

Hypothesis: Phasor with longer context windows might close the gap with Transformer,
since it can process more context without quadratic cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from phasor import PhasorModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class TransformerLM(nn.Module):
    """Standard transformer for language modeling."""
    def __init__(self, vocab_size=256, dim=128, n_layers=4, n_heads=4, max_seq_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


# Load data
print("Loading enwik8...")
with open('data/enwik8', 'rb') as f:
    data = list(f.read(2000000))

split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1]) for i in ix])
    return x.to(device), y.to(device)


def train_epoch(model, optimizer, train_data, batch_size, seq_len, n_batches=200):
    model.train()
    total_loss = 0
    for _ in range(n_batches):
        x, y = get_batch(train_data, batch_size, seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batches


def eval_bpc(model, val_data, batch_size, seq_len, n_batches=50):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size, seq_len)
            loss = F.cross_entropy(model(x).view(-1, 256), y.view(-1))
            total_loss += loss.item()
    return (total_loss / n_batches) / np.log(2)


def measure_throughput(model, seq_len, batch_size=32, n_runs=10):
    """Measure tokens/second."""
    model.eval()
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * n_runs) / elapsed
    return tokens_per_sec


# Configuration
TF_SEQ_LEN = 128  # Transformer stays at 128 (quadratic cost)
PHASOR_SEQ_LENS = [128, 256, 512, 1024]  # Phasor can go longer (linear cost)
N_EPOCHS = 20
DIM = 128
N_LAYERS = 4

print("\n" + "=" * 70)
print("LONG CONTEXT EXPERIMENT")
print("=" * 70)
print(f"Transformer: fixed at seq_len={TF_SEQ_LEN}")
print(f"Phasor: testing seq_lens={PHASOR_SEQ_LENS}")
print()

# Train Transformer baseline
print("Training Transformer (seq_len=128)...")
transformer = TransformerLM(vocab_size=256, dim=DIM, n_layers=N_LAYERS, max_seq_len=TF_SEQ_LEN).to(device)
tf_opt = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(1, N_EPOCHS + 1):
    train_epoch(transformer, tf_opt, train_data, batch_size=32, seq_len=TF_SEQ_LEN)
    if epoch % 5 == 0:
        bpc = eval_bpc(transformer, val_data, batch_size=32, seq_len=TF_SEQ_LEN)
        print(f"  Epoch {epoch}: {bpc:.3f} BPC")

tf_bpc = eval_bpc(transformer, val_data, batch_size=32, seq_len=TF_SEQ_LEN)
tf_throughput = measure_throughput(transformer, TF_SEQ_LEN)
print(f"Transformer final: {tf_bpc:.3f} BPC, {tf_throughput/1000:.1f}K tok/sec")

# Test Phasor at different sequence lengths
print("\n" + "-" * 70)
results = []

for phasor_seq_len in PHASOR_SEQ_LENS:
    print(f"\nTraining Phasor (seq_len={phasor_seq_len})...")

    # Adjust batch size for memory
    if phasor_seq_len <= 256:
        batch_size = 32
    elif phasor_seq_len <= 512:
        batch_size = 16
    else:
        batch_size = 8

    phasor = PhasorModel(vocab_size=256, dim=DIM, n_layers=N_LAYERS, max_seq_len=phasor_seq_len + 128).to(device)
    phasor_opt = torch.optim.AdamW(phasor.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(1, N_EPOCHS + 1):
        train_epoch(phasor, phasor_opt, train_data, batch_size=batch_size, seq_len=phasor_seq_len)
        if epoch % 5 == 0:
            bpc = eval_bpc(phasor, val_data, batch_size=batch_size, seq_len=phasor_seq_len)
            print(f"  Epoch {epoch}: {bpc:.3f} BPC")

    phasor_bpc = eval_bpc(phasor, val_data, batch_size=batch_size, seq_len=phasor_seq_len)
    phasor_throughput = measure_throughput(phasor, phasor_seq_len, batch_size=batch_size)

    gap = phasor_bpc - tf_bpc
    results.append({
        'seq_len': phasor_seq_len,
        'bpc': phasor_bpc,
        'gap': gap,
        'throughput': phasor_throughput
    })

    print(f"Phasor (seq={phasor_seq_len}): {phasor_bpc:.3f} BPC, gap={gap:+.3f}, {phasor_throughput/1000:.1f}K tok/sec")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Does longer context help Phasor?")
print("=" * 70)
print(f"\nTransformer baseline: {tf_bpc:.3f} BPC (seq_len=128)")
print()
print(f"{'Phasor seq_len':>15} | {'BPC':>8} | {'Gap vs TF':>10} | {'Throughput':>12}")
print("-" * 55)
for r in results:
    print(f"{r['seq_len']:>15} | {r['bpc']:>8.3f} | {r['gap']:>+10.3f} | {r['throughput']/1000:>10.1f}K/s")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

best_result = min(results, key=lambda x: x['bpc'])
baseline_result = next(r for r in results if r['seq_len'] == 128)

if best_result['bpc'] < baseline_result['bpc'] - 0.05:
    improvement = baseline_result['bpc'] - best_result['bpc']
    print(f"SUCCESS: Longer context helps! Best at seq_len={best_result['seq_len']}")
    print(f"  Improvement: {improvement:.3f} BPC over Phasor@128")
    if best_result['gap'] < baseline_result['gap']:
        print(f"  Gap vs Transformer closed from {baseline_result['gap']:+.3f} to {best_result['gap']:+.3f}")
else:
    print("Longer context did NOT significantly help Phasor.")
    print("The issue may be in the architecture, not just context length.")
