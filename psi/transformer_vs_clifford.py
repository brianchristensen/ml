"""
Compare Transformer vs Clifford PSI on Associative Recall.
Also test: position-encoded dynamics with dedicated phase planes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Transformer baseline
# ============================================================================

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(64, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, n_heads, dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos(pos)
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)

# ============================================================================
# Clifford PSI (from benchmark)
# ============================================================================

class OrthogonalBivectorBlock(nn.Module):
    def __init__(self, dim, n_orthogonal_sets=4, planes_per_set=16):
        super().__init__()
        self.dim = dim
        self.n_sets = n_orthogonal_sets
        self.planes_per_set = planes_per_set
        self.total_planes = n_orthogonal_sets * planes_per_set

        self.set_weights = nn.Parameter(torch.ones(n_orthogonal_sets))
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.total_planes)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.total_planes)
        )
        self.to_value = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, L, D = x.shape
        key_phase = torch.tanh(self.key_encoder(x)) * math.pi
        query_phase = torch.tanh(self.query_encoder(x)) * math.pi
        key_phase = key_phase.view(B, L, self.n_sets, self.planes_per_set)
        query_phase = query_phase.view(B, L, self.n_sets, self.planes_per_set)
        key_phasor = torch.exp(1j * key_phase)
        query_phasor = torch.exp(1j * query_phase)
        V = self.to_value(x).to(torch.complex64)
        weights = F.softmax(self.set_weights, dim=0)
        total_retrieved = torch.zeros(B, L, D, device=x.device, dtype=torch.float32)

        for s in range(self.n_sets):
            key_s = key_phasor[:, :, s, :]
            query_s = query_phasor[:, :, s, :]
            bound = key_s.unsqueeze(-1) * V.unsqueeze(-2)
            memory = torch.cumsum(bound, dim=1)
            retrieved = memory * query_s.conj().unsqueeze(-1)
            retrieved = retrieved.sum(dim=2).real
            total_retrieved = total_retrieved + weights[s] * retrieved

        positions = torch.arange(1, L + 1, device=x.device, dtype=torch.float32)
        norm = torch.sqrt(positions * self.planes_per_set).view(1, L, 1)
        return x + self.to_out(total_retrieved / norm)

class OrthogonalCliffordModel(nn.Module):
    def __init__(self, vocab_size, dim=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([OrthogonalBivectorBlock(dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = block(norm(h))
        return self.head(self.norm_out(h))

# ============================================================================
# Task generation
# ============================================================================

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

def train_and_eval(model, vocab_size, n_pairs, n_queries, epochs=500, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                data, targets, positions = generate_multi_query_recall(500, n_pairs, n_queries, vocab_size, device)
                logits = model(data)
                correct = sum((logits[:, pos, :vocab_size].argmax(dim=-1) == targets[:, pos]).sum().item()
                             for pos in positions)
                acc = correct / (500 * len(positions)) * 100
                if acc > best_acc:
                    best_acc = acc
                print(f'    Epoch {epoch+1}: acc={acc:.1f}%')
    return best_acc

# ============================================================================
# Main comparison
# ============================================================================

def main():
    print('='*70)
    print('COMPARISON: Transformer vs Clifford PSI on Associative Recall')
    print('='*70)

    vocab_size = 32
    n_pairs = 8
    n_queries = 4
    epochs = 500

    print(f'Task: Store {n_pairs} pairs, query {n_queries}')
    print(f'Random baseline: {100/vocab_size:.1f}%')
    print()

    # Transformer
    print('Training Transformer (2 layers, 4 heads)...')
    transformer = TransformerModel(vocab_size + 1, dim=64, n_heads=4, n_layers=2).to(device)
    t_params = sum(p.numel() for p in transformer.parameters())
    print(f'  Parameters: {t_params:,}')
    t_acc = train_and_eval(transformer, vocab_size, n_pairs, n_queries, epochs)
    print(f'  Best: {t_acc:.1f}%')
    print()

    # Clifford PSI with more epochs
    print('Training Clifford PSI (4 sets x 16 planes, 500 epochs)...')
    clifford = OrthogonalCliffordModel(vocab_size + 1, dim=128, n_layers=2).to(device)
    c_params = sum(p.numel() for p in clifford.parameters())
    print(f'  Parameters: {c_params:,}')
    c_acc = train_and_eval(clifford, vocab_size, n_pairs, n_queries, epochs)
    print(f'  Best: {c_acc:.1f}%')
    print()

    print('='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Random:      {100/vocab_size:.1f}%')
    print(f'Transformer: {t_acc:.1f}%  ({t_params:,} params)')
    print(f'Clifford:    {c_acc:.1f}%  ({c_params:,} params)')
    print()
    print(f'Gap to transformer: {t_acc - c_acc:.1f}%')

if __name__ == "__main__":
    main()
