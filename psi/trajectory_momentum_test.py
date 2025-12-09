"""
Trajectory Momentum Test: Isolating PSI's Unique Property

HYPOTHESIS: PSI encodes trajectory "momentum" in its hidden state, allowing it
to extrapolate direction even when explicit evidence is removed.

KEY INSIGHT from previous test:
- With only 2 tokens of context, PSI got 88% vs Transformer's 59%
- This suggests PSI retains trajectory information in its cumsum state

NEW TESTS designed to isolate this:

1. MOMENTUM AFTER ERASURE:
   - Train on full trajectories
   - Test: Show trajectory, then add "erasure" tokens, then predict
   - If PSI has momentum, it should predict correctly despite erasure
   - Transformer's attention to erasure tokens should disrupt its memory

2. VELOCITY INFERENCE:
   - Show trajectory with VARYING step sizes (+1, +2, +3, etc.)
   - Test: Can model infer the "velocity" (step size) and extrapolate?
   - PSI's continuous state should naturally encode velocity
   - Transformer must explicitly compute differences

3. TRAJECTORY REVERSAL DETECTION:
   - Train on trajectories that may reverse direction mid-sequence
   - Test: Detect reversal point with minimal lookahead
   - PSI should feel the "deceleration" before reversal
   - Transformer needs explicit token comparison

4. HIDDEN STATE PROBING:
   - Train linear probes on hidden states to predict direction
   - Compare how "linearly readable" the trajectory info is
   - PSI's cumsum should make direction explicitly encoded
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.value = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)

        # Cumsum memory - this is where momentum lives
        cumsum_v = torch.cumsum(g * v, dim=1)
        cumsum_g = torch.cumsum(g, dim=1) + 1e-6
        mem = cumsum_v / cumsum_g

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, max_len=256):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]

        hiddens = [h]
        for block in self.blocks:
            h = block(h)
            hiddens.append(h)

        h = self.norm(h)

        if return_hidden:
            return self.head(h), hiddens
        return self.head(h)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, num_heads=4, max_len=256):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]

        if return_hidden:
            # Can't easily get intermediate states from nn.Transformer
            # Just return input and output embeddings
            hiddens = [h]

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)

        if return_hidden:
            hiddens.append(h)

        h = self.norm(h)

        if return_hidden:
            return self.head(h), hiddens
        return self.head(h)


# =============================================================================
# TEST 1: Momentum After Erasure
# =============================================================================

class ErasureDataset(Dataset):
    """
    Train: Normal trajectories (ascending/descending)
    Test: Trajectory + ERASURE tokens + predict

    The erasure tokens are meant to "distract" attention-based models
    while PSI's cumsum state should maintain trajectory momentum.

    Format:
    - Training: [0, 1, 2, 3, 4, 5] -> predict 6
    - Test: [0, 1, 2, 3, 4, 5, E, E, E] -> predict 6 (or 9 if model can extrapolate through erasure)
    """

    def __init__(self, n_examples=2000, traj_len=6, vocab_size=30,
                 mode='train', n_erasure=0, erasure_token=None):
        self.examples = []
        self.vocab_size = vocab_size
        self.ERASURE = erasure_token if erasure_token else vocab_size  # Special token
        self.total_vocab = vocab_size + 1

        for _ in range(n_examples):
            direction = np.random.choice(['asc', 'desc'])

            if direction == 'asc':
                start = np.random.randint(0, vocab_size - traj_len - 1)
                traj = list(range(start, start + traj_len))
                target = start + traj_len  # Next in sequence
            else:
                start = np.random.randint(traj_len, vocab_size - 1)
                traj = list(range(start, start - traj_len, -1))
                target = start - traj_len

            # Ensure target is valid
            if target < 0 or target >= vocab_size:
                continue

            # Add erasure tokens for test mode
            if n_erasure > 0:
                input_seq = traj + [self.ERASURE] * n_erasure
            else:
                input_seq = traj

            self.examples.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'direction': direction,
                'clean_traj': torch.tensor(traj, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 2: Velocity Inference
# =============================================================================

class VelocityDataset(Dataset):
    """
    Trajectories with varying step sizes (velocities).

    Examples:
    - Velocity +1: [0, 1, 2, 3, 4] -> 5
    - Velocity +2: [0, 2, 4, 6, 8] -> 10
    - Velocity -1: [10, 9, 8, 7, 6] -> 5
    - Velocity +3: [0, 3, 6, 9, 12] -> 15

    Model must infer the velocity and extrapolate.
    """

    def __init__(self, n_examples=2000, traj_len=5, vocab_size=50,
                 velocities=[1, 2, 3, -1, -2, -3]):
        self.examples = []
        self.vocab_size = vocab_size
        self.velocities = velocities

        for _ in range(n_examples):
            vel = np.random.choice(velocities)

            # Find valid starting point
            if vel > 0:
                max_start = vocab_size - vel * traj_len - 1
                if max_start <= 0:
                    continue
                start = np.random.randint(0, max_start)
            else:
                min_start = abs(vel) * traj_len
                if min_start >= vocab_size:
                    continue
                start = np.random.randint(min_start, vocab_size)

            traj = [start + vel * i for i in range(traj_len)]
            target = start + vel * traj_len

            # Validate
            if any(t < 0 or t >= vocab_size for t in traj):
                continue
            if target < 0 or target >= vocab_size:
                continue

            self.examples.append({
                'input': torch.tensor(traj, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'velocity': vel
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 3: Reversal Detection
# =============================================================================

class ReversalDataset(Dataset):
    """
    Trajectories that may reverse direction. Predict next token.

    Examples:
    - No reversal: [0, 1, 2, 3, 4, 5] -> 6
    - Reversal at 3: [0, 1, 2, 3, 2, 1] -> 0
    - Reversal at 4: [5, 4, 3, 2, 3, 4] -> 5

    Key test: Can model detect the reversal and predict correctly?
    Early detection (before reversal completes) tests momentum.
    """

    def __init__(self, n_examples=2000, traj_len=8, vocab_size=30,
                 reversal_prob=0.5):
        self.examples = []
        self.vocab_size = vocab_size

        for _ in range(n_examples):
            has_reversal = np.random.random() < reversal_prob

            if has_reversal:
                # Start ascending or descending, then reverse
                direction = np.random.choice(['asc', 'desc'])
                reversal_point = np.random.randint(3, traj_len - 2)

                if direction == 'asc':
                    start = np.random.randint(0, vocab_size // 2)
                    # Go up, then down
                    traj = list(range(start, start + reversal_point))
                    peak = traj[-1]
                    traj.extend(list(range(peak - 1, peak - (traj_len - reversal_point) - 1, -1)))
                    # Next would continue descending
                    target = traj[-1] - 1
                else:
                    start = np.random.randint(vocab_size // 2, vocab_size)
                    # Go down, then up
                    traj = list(range(start, start - reversal_point, -1))
                    valley = traj[-1]
                    traj.extend(list(range(valley + 1, valley + (traj_len - reversal_point) + 1)))
                    # Next would continue ascending
                    target = traj[-1] + 1
            else:
                # Simple trajectory, no reversal
                direction = np.random.choice(['asc', 'desc'])
                if direction == 'asc':
                    start = np.random.randint(0, vocab_size - traj_len - 1)
                    traj = list(range(start, start + traj_len))
                    target = traj[-1] + 1
                else:
                    start = np.random.randint(traj_len, vocab_size - 1)
                    traj = list(range(start, start - traj_len, -1))
                    target = traj[-1] - 1

            # Validate
            if any(t < 0 or t >= vocab_size for t in traj):
                continue
            if target < 0 or target >= vocab_size:
                continue
            if len(traj) != traj_len:
                continue

            self.examples.append({
                'input': torch.tensor(traj, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'has_reversal': has_reversal
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 4: Hidden State Probing
# =============================================================================

class DirectionProbe(nn.Module):
    """Linear probe to predict direction from hidden state."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 2)  # 2 classes: asc/desc

    def forward(self, h):
        return self.linear(h)


def train_probe(model, dataset, probe_position='last'):
    """
    Train a linear probe to predict direction from hidden states.

    Args:
        model: Trained PSI or Transformer model
        dataset: Dataset with 'direction' labels
        probe_position: Which position to probe ('last', 'middle', 'first')

    Returns:
        Probe accuracy (how linearly readable is direction?)
    """
    model.eval()

    # Collect hidden states and labels
    hiddens = []
    labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        collate_fn=lambda b: {
                            'input': torch.stack([x['input'] for x in b]),
                            'direction': [x['direction'] for x in b]
                        })

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            _, hidden_list = model(inputs, return_hidden=True)

            # Get hidden state from last layer
            h = hidden_list[-1]  # [batch, seq_len, dim]

            # Select position
            if probe_position == 'last':
                h_probe = h[:, -1, :]
            elif probe_position == 'middle':
                h_probe = h[:, h.shape[1] // 2, :]
            else:  # first
                h_probe = h[:, 0, :]

            hiddens.append(h_probe.cpu())
            labels.extend([1 if d == 'asc' else 0 for d in batch['direction']])

    hiddens = torch.cat(hiddens, dim=0)
    labels = torch.tensor(labels)

    # Train linear probe
    probe = DirectionProbe(hiddens.shape[-1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)

    # Simple training
    for _ in range(100):
        optimizer.zero_grad()
        logits = probe(hiddens)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = probe(hiddens).argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

    return acc


# =============================================================================
# Training
# =============================================================================

def collate_fn(batch):
    max_len = max(item['input'].shape[0] for item in batch)
    inputs = []
    targets = []

    for item in batch:
        inp = item['input']
        if inp.shape[0] < max_len:
            inp = F.pad(inp, (0, max_len - inp.shape[0]))
        inputs.append(inp)
        targets.append(item['target'])

    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets)
    }


def train_model(model, train_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits[:, -1], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            logits = model(inputs)
            preds = logits[:, -1].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

    return correct / total


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("TRAJECTORY MOMENTUM TEST: Isolating PSI's Unique Property")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    dim = 128
    num_layers = 4
    results = {}

    # =========================================================================
    # TEST 1: Momentum After Erasure
    # =========================================================================
    print("=" * 70)
    print("TEST 1: MOMENTUM AFTER ERASURE")
    print("Train on clean trajectories, test with erasure tokens inserted")
    print("=" * 70)
    print()

    vocab_size = 32
    train_data = ErasureDataset(n_examples=3000, traj_len=6, vocab_size=vocab_size,
                                 mode='train', n_erasure=0)
    test_clean = ErasureDataset(n_examples=500, traj_len=6, vocab_size=vocab_size,
                                 mode='test', n_erasure=0)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader_clean = DataLoader(test_clean, batch_size=64, shuffle=False, collate_fn=collate_fn)

    erasure_results = {}

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size + 1, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size + 1, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=40)

        # Test on clean
        acc_clean = evaluate_model(model, test_loader_clean)
        print(f"    Clean accuracy: {acc_clean:.1%}")

        # Test with increasing erasure
        erasure_counts = [1, 2, 3, 4, 5]
        erasure_accs = [acc_clean]

        for n_erase in erasure_counts:
            test_erased = ErasureDataset(n_examples=500, traj_len=6, vocab_size=vocab_size,
                                          mode='test', n_erasure=n_erase,
                                          erasure_token=vocab_size)
            test_loader_erased = DataLoader(test_erased, batch_size=64, shuffle=False,
                                            collate_fn=collate_fn)
            acc = evaluate_model(model, test_loader_erased)
            erasure_accs.append(acc)
            print(f"    With {n_erase} erasure tokens: {acc:.1%}")

        erasure_results[name] = {
            'clean': acc_clean,
            'erased': dict(zip(erasure_counts, erasure_accs[1:]))
        }

        results[f'erasure_{name}'] = erasure_results[name]

    print()

    # =========================================================================
    # TEST 2: Velocity Inference
    # =========================================================================
    print("=" * 70)
    print("TEST 2: VELOCITY INFERENCE")
    print("Can model infer step size and extrapolate?")
    print("=" * 70)
    print()

    vocab_size = 60
    train_data = VelocityDataset(n_examples=4000, traj_len=5, vocab_size=vocab_size,
                                  velocities=[1, 2, -1, -2])
    test_seen = VelocityDataset(n_examples=500, traj_len=5, vocab_size=vocab_size,
                                 velocities=[1, 2, -1, -2])
    test_unseen = VelocityDataset(n_examples=500, traj_len=5, vocab_size=vocab_size,
                                   velocities=[3, -3])  # Unseen velocities!

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader_seen = DataLoader(test_seen, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader_unseen = DataLoader(test_unseen, batch_size=64, shuffle=False, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=50)

        acc_seen = evaluate_model(model, test_loader_seen)
        acc_unseen = evaluate_model(model, test_loader_unseen)

        print(f"    Seen velocities (1,2,-1,-2): {acc_seen:.1%}")
        print(f"    Unseen velocities (3,-3): {acc_unseen:.1%}")
        print(f"    Generalization gap: {(acc_seen - acc_unseen) / acc_seen * 100:.1f}%")

        results[f'velocity_{name}_seen'] = acc_seen
        results[f'velocity_{name}_unseen'] = acc_unseen

    print()

    # =========================================================================
    # TEST 3: Reversal Detection
    # =========================================================================
    print("=" * 70)
    print("TEST 3: REVERSAL DETECTION")
    print("Can model detect and adapt to trajectory reversals?")
    print("=" * 70)
    print()

    vocab_size = 40
    train_data = ReversalDataset(n_examples=4000, traj_len=8, vocab_size=vocab_size,
                                  reversal_prob=0.5)
    test_data = ReversalDataset(n_examples=500, traj_len=8, vocab_size=vocab_size,
                                 reversal_prob=0.5)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=50)

        # Overall accuracy
        acc = evaluate_model(model, test_loader)
        print(f"    Overall accuracy: {acc:.1%}")

        # Breakdown by reversal/no-reversal
        model.eval()
        correct_rev = 0
        total_rev = 0
        correct_no_rev = 0
        total_no_rev = 0

        with torch.no_grad():
            for item in test_data.examples:
                inp = item['input'].unsqueeze(0).to(device)
                target = item['target'].item()
                has_rev = item['has_reversal']

                logits = model(inp)
                pred = logits[0, -1].argmax().item()

                if has_rev:
                    if pred == target:
                        correct_rev += 1
                    total_rev += 1
                else:
                    if pred == target:
                        correct_no_rev += 1
                    total_no_rev += 1

        acc_rev = correct_rev / total_rev if total_rev > 0 else 0
        acc_no_rev = correct_no_rev / total_no_rev if total_no_rev > 0 else 0

        print(f"    No reversal: {acc_no_rev:.1%}")
        print(f"    With reversal: {acc_rev:.1%}")

        results[f'reversal_{name}_overall'] = acc
        results[f'reversal_{name}_no_rev'] = acc_no_rev
        results[f'reversal_{name}_with_rev'] = acc_rev

    print()

    # =========================================================================
    # TEST 4: Hidden State Probing
    # =========================================================================
    print("=" * 70)
    print("TEST 4: HIDDEN STATE PROBING")
    print("How linearly readable is direction from hidden states?")
    print("=" * 70)
    print()

    vocab_size = 30
    train_data = ErasureDataset(n_examples=2000, traj_len=6, vocab_size=vocab_size,
                                 mode='train', n_erasure=0)
    probe_data = ErasureDataset(n_examples=500, traj_len=6, vocab_size=vocab_size,
                                 mode='test', n_erasure=0)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size + 1, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size + 1, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=30)

        print(f"  Probing {name} hidden states...")

        for pos in ['first', 'middle', 'last']:
            acc = train_probe(model, probe_data, probe_position=pos)
            print(f"    Position '{pos}': {acc:.1%}")
            results[f'probe_{name}_{pos}'] = acc

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Trajectory Momentum Evidence")
    print("=" * 80)
    print()

    print("1. ERASURE TEST (momentum retention):")
    print(f"   PSI with 3 erasure tokens: {results.get('erasure_PSI', {}).get('erased', {}).get(3, 0):.1%}")
    print(f"   Trans with 3 erasure tokens: {results.get('erasure_Transformer', {}).get('erased', {}).get(3, 0):.1%}")
    print()

    print("2. VELOCITY TEST (continuous dynamics):")
    print(f"   PSI unseen velocity generalization: {results.get('velocity_PSI_unseen', 0):.1%}")
    print(f"   Trans unseen velocity generalization: {results.get('velocity_Transformer_unseen', 0):.1%}")
    print()

    print("3. REVERSAL TEST (trajectory state tracking):")
    print(f"   PSI with reversal: {results.get('reversal_PSI_with_rev', 0):.1%}")
    print(f"   Trans with reversal: {results.get('reversal_Transformer_with_rev', 0):.1%}")
    print()

    print("4. PROBING TEST (linear readability of direction):")
    print(f"   PSI at first position: {results.get('probe_PSI_first', 0):.1%}")
    print(f"   Trans at first position: {results.get('probe_Transformer_first', 0):.1%}")
    print(f"   PSI at last position: {results.get('probe_PSI_last', 0):.1%}")
    print(f"   Trans at last position: {results.get('probe_Transformer_last', 0):.1%}")
    print()

    # Plot erasure degradation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Erasure test
    ax = axes[0, 0]
    erasure_counts = [0, 1, 2, 3, 4, 5]
    psi_erasure = [results['erasure_PSI']['clean']]
    psi_erasure.extend([results['erasure_PSI']['erased'].get(i, 0) for i in erasure_counts[1:]])
    trans_erasure = [results['erasure_Transformer']['clean']]
    trans_erasure.extend([results['erasure_Transformer']['erased'].get(i, 0) for i in erasure_counts[1:]])

    ax.plot(erasure_counts, psi_erasure, 'o-', label='PSI', color='steelblue', linewidth=2)
    ax.plot(erasure_counts, trans_erasure, 's-', label='Transformer', color='coral', linewidth=2)
    ax.set_xlabel('Number of Erasure Tokens')
    ax.set_ylabel('Accuracy')
    ax.set_title('Momentum After Erasure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Velocity test
    ax = axes[0, 1]
    names = ['PSI', 'Transformer']
    seen = [results['velocity_PSI_seen'], results['velocity_Transformer_seen']]
    unseen = [results['velocity_PSI_unseen'], results['velocity_Transformer_unseen']]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, seen, width, label='Seen velocities', color='steelblue')
    ax.bar(x + width/2, unseen, width, label='Unseen velocities', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Velocity Generalization')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # Reversal test
    ax = axes[1, 0]
    no_rev = [results['reversal_PSI_no_rev'], results['reversal_Transformer_no_rev']]
    with_rev = [results['reversal_PSI_with_rev'], results['reversal_Transformer_with_rev']]

    ax.bar(x - width/2, no_rev, width, label='No reversal', color='steelblue')
    ax.bar(x + width/2, with_rev, width, label='With reversal', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Reversal Detection')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # Probing test
    ax = axes[1, 1]
    positions = ['first', 'middle', 'last']
    psi_probe = [results.get(f'probe_PSI_{p}', 0) for p in positions]
    trans_probe = [results.get(f'probe_Transformer_{p}', 0) for p in positions]

    x = np.arange(len(positions))
    ax.bar(x - width/2, psi_probe, width, label='PSI', color='steelblue')
    ax.bar(x + width/2, trans_probe, width, label='Transformer', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_xlabel('Probe Position')
    ax.set_ylabel('Direction Probe Accuracy')
    ax.set_title('Linear Readability of Direction')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('trajectory_momentum_test.png', dpi=150)
    plt.close()
    print("Saved trajectory_momentum_test.png")

    # Conclusions
    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
    If PSI has trajectory momentum:
    1. Erasure test: PSI should degrade MORE SLOWLY with erasure tokens
       (momentum carries through distraction)

    2. Velocity test: PSI should generalize BETTER to unseen velocities
       (continuous state encodes velocity naturally)

    3. Reversal test: Similar or PSI advantage on reversals
       (cumsum state tracks trajectory changes)

    4. Probing test: Direction should be MORE linearly readable from PSI,
       especially at EARLY positions (momentum encoded from start)
    """)


if __name__ == "__main__":
    main()
