"""
Matched state capacity test: PSI vs Mamba
PSI has ~8,320 floats state, so we test Mamba with d_state=32 (~8,192 floats)
"""
import torch
import torch.nn as nn
import numpy as np
import math
import gc

from mambapy.mamba import Mamba, MambaConfig

device = 'cuda'

# PSI Model
class PositionOnlyPhasorBlock(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        base_phases = torch.randn(max_len, dim) * math.pi
        self.register_buffer('base_phases', base_phases)
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        phases = self.base_phases[:L].unsqueeze(0)
        value = self.to_value(x)
        bound_real = value * torch.cos(phases)
        bound_imag = value * torch.sin(phases)
        mem_real = torch.cumsum(bound_real, dim=1)
        mem_imag = torch.cumsum(bound_imag, dim=1)
        retrieved = mem_real * torch.cos(phases) + mem_imag * torch.sin(phases)
        return x + self.to_out(retrieved / math.sqrt(D))

class ContentOnlyPhasorBlock(nn.Module):
    def __init__(self, dim, n_oscillators=64):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators
        self.key_encoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators))
        self.query_encoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_oscillators))
        self.to_amp = nn.Sequential(nn.Linear(dim, n_oscillators), nn.Softplus())
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        K = self.n_oscillators
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        amp = self.to_amp(x) + 0.1
        key_phasor = amp * torch.exp(1j * key_phase)
        query_phasor = amp * torch.exp(1j * query_phase)
        V = self.to_value(x).to(torch.complex64)
        bound = key_phasor.unsqueeze(-1) * V.unsqueeze(-2)
        memory = torch.cumsum(bound, dim=1)
        retrieved = memory * query_phasor.conj().unsqueeze(-1)
        retrieved = retrieved.sum(dim=2).real
        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * K).view(1, L, 1)
        return x + self.to_out(retrieved / norm)

class PSIModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, n_oscillators=64, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(PositionOnlyPhasorBlock(dim, max_len))
            else:
                self.blocks.append(ContentOnlyPhasorBlock(dim, n_oscillators))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))

class MambaModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        config = MambaConfig(d_model=dim, n_layers=n_layers, d_state=d_state,
                            expand_factor=2, d_conv=4, pscan=True, use_cuda=False)
        self.mamba = Mamba(config)
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.mamba(h)
        return self.head(self.norm_out(h))

def generate_multi_query_recall(batch_size, n_pairs, n_queries, vocab_size, device):
    QUERY_TOKEN = vocab_size
    seq_len = n_pairs * 2 + n_queries * 2
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    query_positions = []
    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i + 1]) for i in range(n_pairs)]
        pos = 0
        for k, v in pairs:
            data[b, pos] = k
            data[b, pos + 1] = v
            pos += 2
        query_indices = np.random.choice(n_pairs, n_queries, replace=False)
        for qi, query_idx in enumerate(query_indices):
            data[b, pos] = QUERY_TOKEN
            pos += 1
            query_k, query_v = pairs[query_idx]
            data[b, pos] = query_k
            targets[b, pos] = query_v
            if b == 0:
                query_positions.append(pos)
            pos += 1
    return data, targets, query_positions

def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        data, targets, positions = generate_multi_query_recall(64, n_pairs, n_queries, vocab_size, device)
        logits = model(data)
        loss = sum(criterion(logits[:, pos, :vocab_size], targets[:, pos]) for pos in positions) / len(positions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item() for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc
    return best_acc

def main():
    print('=' * 60)
    print('MATCHED STATE SIZE COMPARISON: Multi-Query Recall')
    print('=' * 60)
    print()

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    dim = 64
    n_layers = 2

    # PSI state: 8,320 floats
    # Mamba d_state=32: 32 * 128 * 2 = 8,192 floats (close match!)

    print('PSI (state: ~8,320 floats)')
    psi = PSIModel(vocab_size + 1, dim, n_layers).to(device)
    psi_params = sum(p.numel() for p in psi.parameters())
    print(f'  Parameters: {psi_params:,}')
    psi_acc = train_and_eval(psi, vocab_size, n_pairs, n_queries)
    print(f'  Accuracy: {psi_acc:.1f}%')
    del psi
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print('Mamba d_state=16 (state: ~4,096 floats)')
    mamba16 = MambaModel(vocab_size + 1, dim, n_layers, d_state=16).to(device)
    m16_params = sum(p.numel() for p in mamba16.parameters())
    print(f'  Parameters: {m16_params:,}')
    m16_acc = train_and_eval(mamba16, vocab_size, n_pairs, n_queries)
    print(f'  Accuracy: {m16_acc:.1f}%')
    del mamba16
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print('Mamba d_state=32 (state: ~8,192 floats - MATCHED)')
    mamba32 = MambaModel(vocab_size + 1, dim, n_layers, d_state=32).to(device)
    m32_params = sum(p.numel() for p in mamba32.parameters())
    print(f'  Parameters: {m32_params:,}')
    m32_acc = train_and_eval(mamba32, vocab_size, n_pairs, n_queries)
    print(f'  Accuracy: {m32_acc:.1f}%')

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()
    print(f'PSI              (state: 8,320):  {psi_acc:.1f}%')
    print(f'Mamba d_state=16 (state: 4,096):  {m16_acc:.1f}%')
    print(f'Mamba d_state=32 (state: 8,192):  {m32_acc:.1f}%')
    print()
    if psi_acc > m32_acc + 2:
        print('VERDICT: PSI wins even with matched state size!')
        print('This suggests the retrieval mechanism matters, not just capacity.')
    elif m32_acc > psi_acc + 2:
        print('VERDICT: With matched state, Mamba catches up.')
        print('PSI advantage was mainly from larger state.')
    else:
        print('VERDICT: Comparable with matched state size.')

if __name__ == "__main__":
    main()
