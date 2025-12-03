"""
Test if batch size improvements scale to longer sequences.

Key finding from previous test:
- batch=128, LR=3e-3 converges in ~400 epochs for seq_len=128

Now test:
1. Does this scale to seq_len=256, 512?
2. Can we push further with batch=256?
3. Optimal LR with large batch?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PureOrthoPhasor(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        self.key_mod = nn.Linear(dim, dim)
        self.query_mod = nn.Linear(dim, dim)
        self.mod_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_mod = self.key_mod(key) * self.mod_scale
        query_mod = self.query_mod(query) * self.mod_scale

        key_phase = phases + key_mod
        query_phase = phases + query_mod

        br = value * torch.cos(key_phase)
        bi = value * torch.sin(key_phase)

        mr = torch.cumsum(br, dim=1)
        mi = torch.cumsum(bi, dim=1)

        ret = (mr * torch.cos(query_phase) + mi * torch.sin(query_phase)) / math.sqrt(D)
        return x + self.to_out(ret)


class MinimalOrthoModel(nn.Module):
    def __init__(self, vocab_size, dim=256, num_layers=2, max_seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                PureOrthoPhasor(dim, max_seq_len)
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def test_scaling(seq_len, batch_size, lr, max_epochs=800):
    """Test training convergence with given hyperparameters."""
    vocab_size = 10
    dim = 256

    def generate_batch(bs):
        first_tokens = torch.randint(0, vocab_size, (bs,))
        middle = torch.randint(0, vocab_size, (bs, seq_len - 2))
        x = torch.cat([
            first_tokens.unsqueeze(1),
            middle,
            torch.zeros(bs, 1, dtype=torch.long)
        ], dim=1)
        return x, first_tokens

    model = MinimalOrthoModel(vocab_size, dim=dim, num_layers=2, max_seq_len=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start = time.time()
    converged_epoch = None

    model.train()
    for epoch in range(max_epochs):
        x, first_tokens = generate_batch(batch_size)
        x = x.to(device)
        first_tokens = first_tokens.to(device)

        # Create targets
        y = torch.cat([
            x[:, 1:-1],
            torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            first_tokens.unsqueeze(1)
        ], dim=1)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(5):
                    tx, tf = generate_batch(batch_size)
                    tx, tf = tx.to(device), tf.to(device)
                    logits = model(tx)
                    pred = logits[:, -1].argmax(dim=-1)
                    correct += (pred == tf).sum().item()
                    total += batch_size

            acc = correct / total * 100
            elapsed = time.time() - start
            print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%, time={elapsed:.1f}s")
            model.train()

            if acc >= 95 and converged_epoch is None:
                converged_epoch = epoch + 1

    return converged_epoch, time.time() - start


def main():
    print("=" * 70)
    print("SCALING TEST: Large Batch Training")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Test 1: seq_len=256 with our best settings
    print("=" * 70)
    print("TEST 1: seq_len=256 with batch=128, LR=3e-3")
    print("=" * 70)
    epoch, t = test_scaling(256, 128, 3e-3, max_epochs=1000)
    print(f"  --> Converged at epoch {epoch}" if epoch else "  --> Did not converge")
    print()

    # Test 2: Try even larger batch
    print("=" * 70)
    print("TEST 2: seq_len=256 with batch=256, LR=5e-3")
    print("=" * 70)
    epoch, t = test_scaling(256, 256, 5e-3, max_epochs=1000)
    print(f"  --> Converged at epoch {epoch}" if epoch else "  --> Did not converge")
    print()

    # Test 3: seq_len=512
    print("=" * 70)
    print("TEST 3: seq_len=512 with batch=128, LR=3e-3")
    print("=" * 70)
    epoch, t = test_scaling(512, 128, 3e-3, max_epochs=1500)
    print(f"  --> Converged at epoch {epoch}" if epoch else "  --> Did not converge")
    print()

    # Test 4: Linear LR scaling rule (batch 2x -> LR sqrt(2)x)
    print("=" * 70)
    print("TEST 4: seq_len=256 with batch=256, LR=4.2e-3 (sqrt scaling)")
    print("=" * 70)
    epoch, t = test_scaling(256, 256, 4.2e-3, max_epochs=1000)
    print(f"  --> Converged at epoch {epoch}" if epoch else "  --> Did not converge")


if __name__ == "__main__":
    main()
