"""
Optimal Orthogonal Phasor Configuration

Key findings from ablation study:
- 2 phasor layers (no FFN) is minimal architecture that works
- Random fixed phases + learned modulation = breakthrough
- batch=256, lr=5e-3 is optimal training config
- Converges: 128->400, 256->500, 512->700 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PureOrthoPhasor(nn.Module):
    """
    Orthogonal phasor with random fixed phases.

    Key insight: Random phases are approximately orthogonal in high dimensions.
    This enables O(n) associative retrieval via interference.
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        # Random phases - FIXED, not learned
        base_phases = torch.randn(max_seq_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)

        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)

        value = self.to_value(x)

        # Bind: multiply value by complex exponential
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)

        # Cumsum = O(n) memory accumulation
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)

        # Unbind: same phase retrieves original value
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        retrieved = retrieved / math.sqrt(D)

        return x + self.to_out(retrieved)


class OptimalPhasorModel(nn.Module):
    """
    Minimal architecture: 2 phasor layers, no FFN.

    Why 2 layers: "Layer 1 retrieves, Layer 2 decodes"
    The two retrieval operations seem essential.
    """
    def __init__(self, vocab_size, dim=256, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.phasor1 = PureOrthoPhasor(dim, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.phasor2 = PureOrthoPhasor(dim, max_seq_len)
        self.norm3 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = h + self.phasor1(self.norm1(h))
        h = h + self.phasor2(self.norm2(h))
        return self.head(self.norm3(h))


def train_copy_task(seq_len, max_epochs=1500):
    """Train on long-range copy task."""
    vocab_size = 10
    dim = 256
    batch_size = 256
    lr = 5e-3

    def generate_batch(bs, sl):
        first_tokens = torch.randint(0, vocab_size, (bs,))
        middle = torch.randint(0, vocab_size, (bs, sl - 2))
        x = torch.cat([
            first_tokens.unsqueeze(1),
            middle,
            torch.zeros(bs, 1, dtype=torch.long)
        ], dim=1)
        return x, first_tokens

    print(f"Training seq_len={seq_len}")
    model = OptimalPhasorModel(vocab_size, dim=dim, max_seq_len=seq_len+10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    model.train()
    for epoch in range(max_epochs):
        x, first_tokens = generate_batch(batch_size, seq_len)
        x = x.to(device)
        y = torch.cat([
            torch.zeros(batch_size, seq_len - 1, dtype=torch.long),
            first_tokens.unsqueeze(1)
        ], dim=1).to(device)

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
                    tx, tf = generate_batch(128, seq_len)
                    tx = tx.to(device)
                    pred = model(tx)[:, -1].argmax(dim=-1).cpu()
                    correct += (pred == tf).sum().item()
                    total += 128

            acc = correct / total * 100
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1f}%")
            model.train()

            if acc >= 95:
                print(f"  -> CONVERGED at epoch {epoch+1}")
                return epoch + 1

    print("  -> Did not converge")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMAL PHASOR MODEL")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    for seq_len in [128, 256, 512, 1024]:
        train_copy_task(seq_len)
        print()
