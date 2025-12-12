"""
Scaling Study: Track BPC gap between Phasor and Transformer over extended training.

Key question: Does the gap stay constant, shrink, or grow as we train longer?
This tells us if Phasor's inductive bias is fundamentally limiting or just slower to learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    data = list(f.read(2000000))  # 2MB for longer test

split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1]) for i in ix])
    return x.to(device), y.to(device)


def train_epoch(model, optimizer, n_batches=200):
    model.train()
    total_loss = 0
    for _ in range(n_batches):
        x, y = get_batch(train_data, 32, 128)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batches


def eval_bpc(model, n_batches=100):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, 32, 128)
            loss = F.cross_entropy(model(x).view(-1, 256), y.view(-1))
            total_loss += loss.item()
    return (total_loss / n_batches) / np.log(2)


# Create models
print("\nCreating models...")
phasor = PhasorModel(vocab_size=256, dim=128, n_layers=2, max_seq_len=256).to(device)
transformer = TransformerLM(vocab_size=256, dim=128, n_layers=4).to(device)

phasor_opt = torch.optim.AdamW(phasor.parameters(), lr=1e-3, weight_decay=0.01)
tf_opt = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=0.01)

phasor_params = sum(p.numel() for p in phasor.parameters())
tf_params = sum(p.numel() for p in transformer.parameters())
print(f"Phasor params: {phasor_params:,}")
print(f"Transformer params: {tf_params:,}")

# Training
print("\n" + "=" * 60)
print("EXTENDED TRAINING - TRACKING GAP BEHAVIOR")
print("=" * 60)
print(f"{'Epoch':>6} | {'TF BPC':>8} | {'Phasor BPC':>10} | {'Gap':>8}")
print("-" * 45)

gaps = []
for epoch in range(1, 31):
    train_epoch(transformer, tf_opt)
    train_epoch(phasor, phasor_opt)

    if epoch % 5 == 0 or epoch == 1:
        tf_bpc = eval_bpc(transformer)
        phasor_bpc = eval_bpc(phasor)
        gap = phasor_bpc - tf_bpc
        gaps.append((epoch, gap))
        print(f"{epoch:>6} | {tf_bpc:>8.3f} | {phasor_bpc:>10.3f} | {gap:>+8.3f}")

# Analysis
print("\n" + "=" * 60)
print("GAP ANALYSIS")
print("=" * 60)

first_gap = gaps[0][1]
last_gap = gaps[-1][1]
gap_change = last_gap - first_gap

print(f"Initial gap (epoch 1):  {first_gap:+.3f} BPC")
print(f"Final gap (epoch 30):   {last_gap:+.3f} BPC")
print(f"Gap change:             {gap_change:+.3f} BPC")

if abs(gap_change) < 0.05:
    print("\n-> Gap is STABLE: Phasor has a fixed overhead but learns at same rate")
elif gap_change < -0.05:
    print("\n-> Gap is SHRINKING: Phasor catches up with more training!")
else:
    print("\n-> Gap is GROWING: Phasor falls further behind with scale")

# Final results
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
tf_final = eval_bpc(transformer)
phasor_final = eval_bpc(phasor)
print(f"Transformer: {tf_final:.3f} BPC")
print(f"Phasor:      {phasor_final:.3f} BPC")
print(f"Gap:         {phasor_final - tf_final:+.3f} BPC")

# What's needed for better BPC
print("\n" + "=" * 60)
print("ROADMAP TO LOWER BPC")
print("=" * 60)
print("""
Current:  ~3.0 BPC with 1M params on 2MB data

To reach ~2.0 BPC:
  - Scale to 10-50M parameters
  - Train on full enwik8 (100MB)
  - Use seq_len 512-1024
  - Train for 50-100 epochs

To reach ~1.0 BPC (SOTA):
  - 100M+ parameters
  - Rotary embeddings, SwiGLU, RMSNorm
  - Careful LR scheduling with warmup
  - Billions of tokens
""")
