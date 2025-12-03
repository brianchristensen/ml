"""
Gradient Flow Diagnosis for HRR Phasor

Why doesn't the loss move from 2.30?
Let's trace gradients through the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DiagnosticHRRPhasor(nn.Module):
    """HRR Phasor with gradient hooks for diagnosis."""
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_query = nn.Linear(dim, dim)

        self.to_key_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.to_query_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        self.phase_scale = nn.Parameter(torch.ones(dim) * math.pi)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        # Storage for gradient magnitudes
        self.grad_magnitudes = {}

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        key_phase = self.to_key_phase(key) * self.phase_scale
        query_phase = self.to_query_phase(query) * self.phase_scale

        # Bind
        bound_real = value * torch.cos(key_phase)
        bound_imag = value * torch.sin(key_phase)

        # Register hooks to capture gradients
        if bound_real.requires_grad:
            bound_real.register_hook(lambda g: self._save_grad('bound_real', g))
            bound_imag.register_hook(lambda g: self._save_grad('bound_imag', g))

        # Chunked cumsum
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            bound_real = F.pad(bound_real, (0, 0, 0, pad_len))
            bound_imag = F.pad(bound_imag, (0, 0, 0, pad_len))
            query_phase = F.pad(query_phase, (0, 0, 0, pad_len))

        padded_len = bound_real.shape[1]
        num_chunks = padded_len // self.chunk_size

        br = bound_real.view(batch_size, num_chunks, self.chunk_size, dim)
        bi = bound_imag.view(batch_size, num_chunks, self.chunk_size, dim)
        qp = query_phase.view(batch_size, num_chunks, self.chunk_size, dim)

        # Cumsum
        mem_real = torch.cumsum(br, dim=2)
        mem_imag = torch.cumsum(bi, dim=2)

        if mem_real.requires_grad:
            mem_real.register_hook(lambda g: self._save_grad('mem_real', g))
            mem_imag.register_hook(lambda g: self._save_grad('mem_imag', g))

        # Unbind
        cos_q = torch.cos(qp)
        sin_q = torch.sin(qp)
        retrieved_real = mem_real * cos_q + mem_imag * sin_q

        if retrieved_real.requires_grad:
            retrieved_real.register_hook(lambda g: self._save_grad('retrieved', g))

        retrieved = retrieved_real.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        retrieved = retrieved / math.sqrt(dim)

        out = self.to_out(retrieved)
        return x + out

    def _save_grad(self, name, grad):
        self.grad_magnitudes[name] = grad.abs().mean().item()


class DiagnosticModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DiagnosticBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    def get_grad_magnitudes(self):
        mags = {}
        for i, block in enumerate(self.blocks):
            for name, val in block.hrr.grad_magnitudes.items():
                mags[f'block{i}_{name}'] = val
        return mags


class DiagnosticBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.hrr = DiagnosticHRRPhasor(dim, chunk_size)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.hrr(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def diagnose():
    print("=" * 60)
    print("GRADIENT FLOW DIAGNOSIS")
    print("=" * 60)

    vocab_size = 10
    seq_len = 64
    dim = 64

    def generate_batch(batch_size=32):
        first_tokens = torch.randint(0, vocab_size, (batch_size,))
        middle = torch.randint(0, vocab_size, (batch_size, seq_len - 2))
        x = torch.cat([
            first_tokens.unsqueeze(1),
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long)
        ], dim=1)
        y = torch.cat([
            middle,
            torch.zeros(batch_size, 1, dtype=torch.long),
            first_tokens.unsqueeze(1)
        ], dim=1)
        return x, y, first_tokens

    model = DiagnosticModel(vocab_size, dim=dim, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nTraining and monitoring gradients...")
    print()

    for epoch in range(20):
        x, y, _ = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Get gradient magnitudes
        grad_mags = model.get_grad_magnitudes()

        # Also check parameter gradients
        param_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grads[name] = param.grad.abs().mean().item()

        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
            print("  Activation gradients:")
            for name, val in sorted(grad_mags.items()):
                print(f"    {name}: {val:.6f}")
            print("  Key parameter gradients:")
            for name in ['embed.weight', 'head.weight', 'blocks.0.hrr.to_key.weight',
                        'blocks.0.hrr.to_query.weight', 'blocks.0.hrr.phase_scale']:
                if name in param_grads:
                    print(f"    {name}: {param_grads[name]:.6f}")
            print()

    # Check gradient at specific positions
    print("=" * 60)
    print("POSITION-SPECIFIC GRADIENT ANALYSIS")
    print("=" * 60)
    print("Checking if gradient reaches position 0 from loss at position 63")
    print()

    # Simple test: can gradient flow from last position to first?
    model.eval()
    x, y, first_tokens = generate_batch(1)
    x, y = x.to(device), y.to(device)

    # Make embedding require grad for diagnosis
    embed_out = model.embed(x)
    embed_out.retain_grad()

    h = embed_out
    for block in model.blocks:
        h = block(h)
    logits = model.head(model.norm(h))

    # Loss only at last position
    loss_last = F.cross_entropy(logits[0, -1:], y[0, -1:])
    loss_last.backward()

    grad_at_pos0 = embed_out.grad[0, 0].abs().mean().item()
    grad_at_pos63 = embed_out.grad[0, -1].abs().mean().item()

    print(f"Gradient magnitude at position 0:  {grad_at_pos0:.8f}")
    print(f"Gradient magnitude at position 63: {grad_at_pos63:.8f}")
    print(f"Ratio (pos63/pos0): {grad_at_pos63 / (grad_at_pos0 + 1e-10):.1f}x")

    if grad_at_pos0 < 1e-6:
        print("\n!! Gradient at position 0 is essentially ZERO !!")
        print("The model cannot learn to use information from position 0")
    elif grad_at_pos0 < grad_at_pos63 / 100:
        print("\n!! Gradient at position 0 is 100x smaller than position 63 !!")
        print("Severe gradient decay - hard to learn long-range dependencies")
    else:
        print("\nGradient flow looks reasonable")

    # Check gradient per position
    print("\nGradient magnitude by position:")
    grads_by_pos = embed_out.grad[0].abs().mean(dim=-1).cpu().numpy()
    for i in [0, 1, 2, 10, 30, 60, 61, 62, 63]:
        print(f"  pos {i:2d}: {grads_by_pos[i]:.8f}")


if __name__ == "__main__":
    diagnose()
