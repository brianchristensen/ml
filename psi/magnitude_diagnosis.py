"""
Magnitude Diagnosis: Is the HRR signal too weak?

Check the relative magnitudes of:
- Input x
- HRR output (before residual)
- Retrieved signal
- Memory contents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class MagnitudeDiagnosticHRR(nn.Module):
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

        self.magnitudes = {}

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        self.magnitudes['input_x'] = x.abs().mean().item()

        key = self.to_key(x)
        value = self.to_value(x)
        query = self.to_query(x)

        self.magnitudes['key'] = key.abs().mean().item()
        self.magnitudes['value'] = value.abs().mean().item()
        self.magnitudes['query'] = query.abs().mean().item()

        key_phase = self.to_key_phase(key) * self.phase_scale
        query_phase = self.to_query_phase(query) * self.phase_scale

        self.magnitudes['key_phase'] = key_phase.abs().mean().item()
        self.magnitudes['query_phase'] = query_phase.abs().mean().item()

        # Bind
        cos_k = torch.cos(key_phase)
        sin_k = torch.sin(key_phase)
        bound_real = value * cos_k
        bound_imag = value * sin_k

        self.magnitudes['bound_real'] = bound_real.abs().mean().item()
        self.magnitudes['bound_imag'] = bound_imag.abs().mean().item()

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

        # Cumsum (no normalization)
        mem_real = torch.cumsum(br, dim=2)
        mem_imag = torch.cumsum(bi, dim=2)

        self.magnitudes['mem_real'] = mem_real.abs().mean().item()
        self.magnitudes['mem_imag'] = mem_imag.abs().mean().item()

        # Check memory at last position specifically (where retrieval happens)
        self.magnitudes['mem_real_last'] = mem_real[:, :, -1, :].abs().mean().item()
        self.magnitudes['mem_imag_last'] = mem_imag[:, :, -1, :].abs().mean().item()

        # Unbind
        cos_q = torch.cos(qp)
        sin_q = torch.sin(qp)
        retrieved_real = mem_real * cos_q + mem_imag * sin_q

        self.magnitudes['retrieved_raw'] = retrieved_real.abs().mean().item()

        retrieved = retrieved_real.view(batch_size, padded_len, dim)
        if pad_len > 0:
            retrieved = retrieved[:, :seq_len]

        # Scale by sqrt(dim)
        retrieved_scaled = retrieved / math.sqrt(dim)
        self.magnitudes['retrieved_scaled'] = retrieved_scaled.abs().mean().item()

        out = self.to_out(retrieved_scaled)
        self.magnitudes['hrr_output'] = out.abs().mean().item()

        final = x + out
        self.magnitudes['final_output'] = final.abs().mean().item()

        return final


class DiagnosticModel(nn.Module):
    def __init__(self, vocab_size, dim=64, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.hrr = MagnitudeDiagnosticHRR(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.hrr(h)
        return self.head(self.norm(h))


def diagnose_magnitudes():
    print("=" * 60)
    print("MAGNITUDE DIAGNOSIS")
    print("=" * 60)
    print("Is the HRR signal too weak compared to the residual?")
    print()

    vocab_size = 10
    seq_len = 64
    dim = 64

    model = DiagnosticModel(vocab_size, dim=dim, num_layers=1).to(device)

    # Before training
    x = torch.randint(0, vocab_size, (1, seq_len)).to(device)

    print("BEFORE TRAINING:")
    with torch.no_grad():
        _ = model(x)

    for name, val in model.hrr.magnitudes.items():
        print(f"  {name:20s}: {val:.6f}")

    # Train a bit
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
        return x, y

    print("\nTraining...")
    for epoch in range(50):
        x, y = generate_batch(32)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\nAFTER 50 EPOCHS:")
    with torch.no_grad():
        x, _ = generate_batch(1)
        x = x.to(device)
        _ = model(x)

    for name, val in model.hrr.magnitudes.items():
        print(f"  {name:20s}: {val:.6f}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    mags = model.hrr.magnitudes
    hrr_vs_input = mags['hrr_output'] / (mags['input_x'] + 1e-10)

    print(f"\nHRR output / Input ratio: {hrr_vs_input:.4f}")

    if hrr_vs_input < 0.1:
        print("!! HRR contribution is <10% of input - effectively ignored !!")
    elif hrr_vs_input < 0.5:
        print("HRR contribution is weak but present")
    else:
        print("HRR contribution is substantial")

    # Check if memory grows too large (instability)
    print(f"\nMemory magnitude at last position: {mags['mem_real_last']:.4f}")
    print(f"This is {mags['mem_real_last']/mags['bound_real']:.1f}x the per-token bound signal")

    if mags['mem_real_last'] > 100:
        print("!! Memory exploding - need normalization !!")
    elif mags['mem_real_last'] < 1:
        print("Memory magnitude seems reasonable")


if __name__ == "__main__":
    diagnose_magnitudes()
