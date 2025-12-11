"""
enwik8 Character-Level Language Modeling Benchmark
Compares Phasor O(n) vs Transformer O(n²)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import urllib.request
import zipfile

from phasor import PhasorModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Download enwik8 if not present
DATA_DIR = 'data'
ENWIK8_PATH = os.path.join(DATA_DIR, 'enwik8')

if not os.path.exists(ENWIK8_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)
    print('Downloading enwik8...')
    url = 'http://mattmahoney.net/dc/enwik8.zip'
    zip_path = os.path.join(DATA_DIR, 'enwik8.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print('Downloaded.')

# Load data
print("Loading enwik8...")
with open(ENWIK8_PATH, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Character-level encoding
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

# Encode
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Split: 90M train, 5M val, 5M test
train_data = data[:90_000_000]
val_data = data[90_000_000:95_000_000]
test_data = data[95_000_000:]

print(f'Vocab size: {vocab_size}')
print(f'Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}')


# Transformer baseline
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=8, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)
        return self.output(h)


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


def train_epoch(model, optimizer, data, batch_size, seq_len, steps):
    model.train()
    total_loss = 0
    for step in range(steps):
        x, y = get_batch(data, batch_size, seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{steps}, loss: {total_loss/(step+1):.4f}")

    return total_loss / steps


def evaluate(model, data, batch_size, seq_len, steps=50):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(data, batch_size, seq_len)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / steps


if __name__ == "__main__":
    print('\n' + '='*70)
    print('ENWIK8 CHARACTER-LEVEL LANGUAGE MODELING')
    print('='*70)

    # Config
    batch_size = 32
    seq_len = 256
    n_steps = 500
    n_epochs = 4

    results = {}

    # Test Phasor
    print('\n--- Phasor Model ---')
    model_phasor = PhasorModel(
        vocab_size=vocab_size, dim=256, n_layers=6, n_phases=64, max_seq_len=seq_len
    ).to(device)
    n_params_phasor = sum(p.numel() for p in model_phasor.parameters())
    print(f'Parameters: {n_params_phasor:,}')

    optimizer_phasor = torch.optim.AdamW(model_phasor.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(1, n_epochs + 1):
        print(f'  Epoch {epoch}:')
        train_loss = train_epoch(model_phasor, optimizer_phasor, train_data, batch_size, seq_len, n_steps)
        val_loss = evaluate(model_phasor, val_data, batch_size, seq_len)
        bpc = val_loss / np.log(2)
        print(f'  -> train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, val_bpc={bpc:.3f}')

    results['Phasor'] = {'params': n_params_phasor, 'val_loss': val_loss, 'bpc': bpc}

    # Test Transformer
    print('\n--- Transformer Model ---')
    model_tf = TransformerLM(
        vocab_size=vocab_size, dim=256, n_layers=6, n_heads=8, max_seq_len=seq_len
    ).to(device)
    n_params_tf = sum(p.numel() for p in model_tf.parameters())
    print(f'Parameters: {n_params_tf:,}')

    optimizer_tf = torch.optim.AdamW(model_tf.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(1, n_epochs + 1):
        print(f'  Epoch {epoch}:')
        train_loss = train_epoch(model_tf, optimizer_tf, train_data, batch_size, seq_len, n_steps)
        val_loss = evaluate(model_tf, val_data, batch_size, seq_len)
        bpc = val_loss / np.log(2)
        print(f'  -> train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, val_bpc={bpc:.3f}')

    results['Transformer'] = {'params': n_params_tf, 'val_loss': val_loss, 'bpc': bpc}

    # Timing comparison
    print('\n--- Timing at seq_len=256, batch_size=32 ---')
    x_test = torch.randint(0, vocab_size, (32, 256), device=device)

    for name, model in [('Phasor', model_phasor), ('Transformer', model_tf)]:
        model.eval()
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x_test)
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = model(x_test)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / 20
        print(f'{name}: {elapsed*1000:.2f}ms per batch')

    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'{"Model":<15} {"Parameters":>15} {"Val Loss":>12} {"Val BPC":>10}')
    print('-'*55)
    for name, r in results.items():
        print(f'{name:<15} {r["params"]:>15,} {r["val_loss"]:>12.3f} {r["bpc"]:>10.3f}')
