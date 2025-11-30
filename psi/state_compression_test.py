"""
State Compression Test: Does PSI compress trajectory history into persistent state?

KEY INSIGHT from ambiguous trajectory test:
- PSI got 88% with only 2 tokens of context (vs Transformer's 59%)
- The model was TRAINED on full trajectories but TESTED on truncated ones
- PSI retained trajectory information even when early tokens were removed

HYPOTHESIS: PSI's cumsum operation compresses trajectory history into a
continuous state that persists, while Transformer needs explicit tokens
to reconstruct via attention.

TEST DESIGN:

1. TRUNCATION TEST (Direct replication of the finding):
   - Train on full sequences: [0,1,2,3,4,5,6,7] -> 8 or [15,14,13,12,11,10,9,8] -> 7
   - Test: progressively truncate from the START
   - Measure: How quickly does accuracy degrade?

2. HISTORY ENCODING TEST:
   - Train on sequences where early tokens determine a "mode"
   - Later tokens are identical across modes
   - Test: Can model predict correctly with only late tokens visible?

3. CUMULATIVE INFORMATION TEST:
   - Sequence where information accumulates over time
   - Final prediction requires integrating ALL previous tokens
   - Test: Can model do this with partial context?

4. STATE TRANSFER TEST:
   - Train model, then extract hidden state at position N
   - Initialize new sequence with that state
   - Test: Does the state carry the trajectory information?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
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
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if return_hidden:
            return self.head(h), h
        return self.head(h)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, num_heads=4, max_len=256):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
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
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.norm(h)
        if return_hidden:
            return self.head(h), h
        return self.head(h)


# =============================================================================
# TEST 1: Truncation Test (replicating ambiguous finding)
# =============================================================================

class TruncationDataset(Dataset):
    """
    Full trajectories for training.
    Ascending: [0,1,2,3,4,5,6,7] -> 8
    Descending: [15,14,13,12,11,10,9,8] -> 7

    For testing, we truncate from the start and see if model can still predict.
    """

    def __init__(self, n_examples=2000, traj_len=8, vocab_size=20):
        self.examples = []
        self.vocab_size = vocab_size

        for _ in range(n_examples):
            direction = np.random.choice(['asc', 'desc'])

            if direction == 'asc':
                start = np.random.randint(0, vocab_size - traj_len - 1)
                traj = list(range(start, start + traj_len))
                target = start + traj_len
            else:
                start = np.random.randint(traj_len, vocab_size - 1)
                traj = list(range(start, start - traj_len, -1))
                target = start - traj_len

            if target < 0 or target >= vocab_size:
                continue

            self.examples.append({
                'input': torch.tensor(traj, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'direction': direction
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 2: History Encoding (mode determined by early tokens)
# =============================================================================

class HistoryEncodingDataset(Dataset):
    """
    Early tokens set a "mode", later tokens are identical across modes.

    Mode A (starts with 0): [0, 5, 6, 7, 8] -> 10
    Mode B (starts with 1): [1, 5, 6, 7, 8] -> 20

    The only difference is the FIRST token, but it determines the answer.
    Later tokens [5,6,7,8] are identical.

    Test: Can model predict correctly when only seeing [5,6,7,8]?
    (It cannot without state compression of the mode)
    """

    def __init__(self, n_examples=2000, n_modes=4, shared_len=5, vocab_size=50):
        self.examples = []
        self.vocab_size = vocab_size
        self.n_modes = n_modes

        # Define modes: each mode has a unique start token and target
        self.mode_tokens = list(range(n_modes))  # [0, 1, 2, 3]
        self.mode_targets = [10 + i * 5 for i in range(n_modes)]  # [10, 15, 20, 25]

        # Shared sequence (same for all modes)
        self.shared_seq = list(range(n_modes, n_modes + shared_len))  # [4, 5, 6, 7, 8]

        for _ in range(n_examples):
            mode = np.random.randint(0, n_modes)
            mode_token = self.mode_tokens[mode]
            target = self.mode_targets[mode]

            full_seq = [mode_token] + self.shared_seq

            self.examples.append({
                'input': torch.tensor(full_seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'mode': mode,
                'mode_token': mode_token
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 3: Cumulative XOR (information must be integrated)
# =============================================================================

class CumulativeXORDataset(Dataset):
    """
    Sequence of bits. Target is XOR of ALL bits.

    [0, 1, 1, 0, 1] -> XOR = 1
    [1, 1, 0, 0, 1] -> XOR = 1
    [0, 0, 1, 1, 0] -> XOR = 0

    Every bit matters - can't skip any.
    Test: With truncated input, model must have compressed the XOR state.
    """

    def __init__(self, n_examples=2000, seq_len=8):
        self.examples = []
        self.seq_len = seq_len

        for _ in range(n_examples):
            bits = np.random.randint(0, 2, size=seq_len).tolist()
            target = sum(bits) % 2  # XOR is sum mod 2

            self.examples.append({
                'input': torch.tensor(bits, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'bits': bits
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# TEST 4: Parity with Varying Prefix (tests state compression directly)
# =============================================================================

class ParityPrefixDataset(Dataset):
    """
    More controlled version: prefix determines parity offset.

    Prefix "00": answer is parity of suffix
    Prefix "01": answer is parity of suffix + 1
    Prefix "10": answer is parity of suffix + 0
    Prefix "11": answer is parity of suffix + 1

    So the prefix encodes an offset (0 or 1) that must be remembered.
    The suffix is random bits.

    Training: Full sequences with prefix + suffix
    Testing: Only suffix, but model should have learned to use prefix state
    """

    def __init__(self, n_examples=2000, suffix_len=6, vocab_size=4):
        self.examples = []
        self.suffix_len = suffix_len
        # vocab: 0, 1, 2, 3 where 2=prefix_end marker, 3=query marker

        for _ in range(n_examples):
            # Random prefix (2 bits)
            prefix = np.random.randint(0, 2, size=2).tolist()
            prefix_offset = prefix[0] ^ prefix[1]  # XOR of prefix bits

            # Random suffix (suffix_len bits)
            suffix = np.random.randint(0, 2, size=suffix_len).tolist()
            suffix_parity = sum(suffix) % 2

            # Target: suffix parity XOR prefix offset
            target = (suffix_parity + prefix_offset) % 2

            # Full sequence: prefix + [2] + suffix + [3]
            full_seq = prefix + [2] + suffix + [3]

            self.examples.append({
                'input': torch.tensor(full_seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'prefix': prefix,
                'suffix': suffix,
                'prefix_offset': prefix_offset
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Training and Evaluation
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
    return {'input': torch.stack(inputs), 'target': torch.stack(targets)}


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


def evaluate_truncated(model, dataset, truncate_from_start):
    """Evaluate with first N tokens removed."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in dataset.examples:
            full_input = item['input']
            target = item['target'].item()

            if truncate_from_start >= len(full_input):
                continue

            # Remove first N tokens
            truncated = full_input[truncate_from_start:]
            inp = truncated.unsqueeze(0).to(device)

            logits = model(inp)
            pred = logits[0, -1].argmax().item()

            if pred == target:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


def evaluate_mode_accuracy(model, dataset, truncate_prefix=False):
    """Evaluate history encoding task, optionally removing mode token."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in dataset.examples:
            full_input = item['input']
            target = item['target'].item()

            if truncate_prefix:
                # Remove the mode token (first token)
                inp = full_input[1:].unsqueeze(0).to(device)
            else:
                inp = full_input.unsqueeze(0).to(device)

            logits = model(inp)
            pred = logits[0, -1].argmax().item()

            if pred == target:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("STATE COMPRESSION TEST: Does PSI compress trajectory into state?")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    dim = 128
    num_layers = 4
    results = {}

    # =========================================================================
    # TEST 1: Truncation (Replicating Ambiguous Finding)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: TRUNCATION (State Compression Under Removal)")
    print("Train on full trajectories, test with start tokens removed")
    print("=" * 70)
    print()

    vocab_size = 25
    traj_len = 8

    train_data = TruncationDataset(n_examples=3000, traj_len=traj_len, vocab_size=vocab_size)
    test_data = TruncationDataset(n_examples=500, traj_len=traj_len, vocab_size=vocab_size)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    truncation_results = {}

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=50)

        # Test with varying truncation
        print(f"  Testing {name} with truncation:")
        accs = []
        for truncate in range(traj_len):
            acc = evaluate_truncated(model, test_data, truncate)
            accs.append(acc)
            remaining = traj_len - truncate
            print(f"    {remaining} tokens remaining: {acc:.1%}")

        truncation_results[name] = accs
        results[f'truncation_{name}'] = accs

    print()

    # =========================================================================
    # TEST 2: History Encoding (Mode Determined by First Token)
    # =========================================================================
    print("=" * 70)
    print("TEST 2: HISTORY ENCODING (Early Token Sets Mode)")
    print("First token determines answer, rest is shared across modes")
    print("=" * 70)
    print()

    vocab_size = 50
    n_modes = 4

    train_data = HistoryEncodingDataset(n_examples=4000, n_modes=n_modes,
                                         shared_len=5, vocab_size=vocab_size)
    test_data = HistoryEncodingDataset(n_examples=500, n_modes=n_modes,
                                        shared_len=5, vocab_size=vocab_size)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=50)

        # Test with full input
        acc_full = evaluate_mode_accuracy(model, test_data, truncate_prefix=False)
        # Test without mode token
        acc_no_mode = evaluate_mode_accuracy(model, test_data, truncate_prefix=True)

        print(f"    With mode token: {acc_full:.1%}")
        print(f"    Without mode token: {acc_no_mode:.1%}")
        print(f"    Random baseline (1/{n_modes}): {1/n_modes:.1%}")

        results[f'history_{name}_full'] = acc_full
        results[f'history_{name}_no_mode'] = acc_no_mode

    print()

    # =========================================================================
    # TEST 3: Cumulative XOR
    # =========================================================================
    print("=" * 70)
    print("TEST 3: CUMULATIVE XOR (All Bits Matter)")
    print("Predict XOR of all bits - requires integrating entire sequence")
    print("=" * 70)
    print()

    seq_len = 8

    train_data = CumulativeXORDataset(n_examples=4000, seq_len=seq_len)
    test_data = CumulativeXORDataset(n_examples=500, seq_len=seq_len)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    xor_results = {}

    for name, model_fn in [
        ('PSI', lambda: PSIModel(2, dim, num_layers)),  # vocab=2 (bits)
        ('Transformer', lambda: TransformerModel(2, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=60)

        # Test with varying truncation
        print(f"  Testing {name} with truncation:")
        accs = []
        for truncate in range(seq_len):
            acc = evaluate_truncated(model, test_data, truncate)
            accs.append(acc)
            remaining = seq_len - truncate
            print(f"    {remaining} bits remaining: {acc:.1%}")

        xor_results[name] = accs
        results[f'xor_{name}'] = accs

    print()

    # =========================================================================
    # TEST 4: Parity with Prefix
    # =========================================================================
    print("=" * 70)
    print("TEST 4: PARITY WITH PREFIX (Mode Offset)")
    print("Prefix determines an offset, suffix provides bits")
    print("=" * 70)
    print()

    suffix_len = 6

    train_data = ParityPrefixDataset(n_examples=4000, suffix_len=suffix_len)
    test_data = ParityPrefixDataset(n_examples=500, suffix_len=suffix_len)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(4, dim, num_layers)),  # vocab=4 (0,1,2,3)
        ('Transformer', lambda: TransformerModel(4, dim, num_layers))
    ]:
        print(f"  Training {name}...")
        model = model_fn().to(device)
        train_model(model, train_loader, epochs=60)

        # Test with full sequence
        model.eval()
        correct_full = 0
        correct_no_prefix = 0
        total = 0

        with torch.no_grad():
            for item in test_data.examples:
                full_input = item['input']
                target = item['target'].item()

                # Full input
                inp_full = full_input.unsqueeze(0).to(device)
                pred_full = model(inp_full)[0, -1].argmax().item()

                # Without prefix (start after the marker at position 2)
                # Full: [p0, p1, 2, s0, s1, ..., 3]
                # Without prefix: [s0, s1, ..., 3]
                inp_no_prefix = full_input[3:].unsqueeze(0).to(device)
                pred_no_prefix = model(inp_no_prefix)[0, -1].argmax().item()

                if pred_full == target:
                    correct_full += 1
                if pred_no_prefix == target:
                    correct_no_prefix += 1
                total += 1

        acc_full = correct_full / total
        acc_no_prefix = correct_no_prefix / total

        print(f"    With prefix: {acc_full:.1%}")
        print(f"    Without prefix: {acc_no_prefix:.1%}")
        print(f"    Random baseline: 50.0%")

        results[f'parity_{name}_full'] = acc_full
        results[f'parity_{name}_no_prefix'] = acc_no_prefix

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: State Compression Evidence")
    print("=" * 80)
    print()

    # Truncation comparison
    print("1. TRUNCATION TEST (trajectory state retention):")
    print("   Tokens remaining | PSI      | Transformer | PSI Advantage")
    print("   " + "-" * 55)
    for i, (psi_acc, trans_acc) in enumerate(zip(truncation_results.get('PSI', []),
                                                   truncation_results.get('Transformer', []))):
        remaining = traj_len - i
        adv = psi_acc - trans_acc
        adv_str = f"+{adv*100:.1f}%" if adv > 0 else f"{adv*100:.1f}%"
        print(f"   {remaining:^15} | {psi_acc:^8.1%} | {trans_acc:^11.1%} | {adv_str:^13}")

    print()
    print("2. HISTORY ENCODING TEST (mode token state):")
    print(f"   PSI with mode token: {results.get('history_PSI_full', 0):.1%}")
    print(f"   PSI without mode token: {results.get('history_PSI_no_mode', 0):.1%}")
    print(f"   Transformer with mode token: {results.get('history_Transformer_full', 0):.1%}")
    print(f"   Transformer without mode token: {results.get('history_Transformer_no_mode', 0):.1%}")

    print()
    print("3. XOR TEST (cumulative state):")
    print("   Bits remaining | PSI      | Transformer")
    print("   " + "-" * 40)
    for i, (psi_acc, trans_acc) in enumerate(zip(xor_results.get('PSI', []),
                                                   xor_results.get('Transformer', []))):
        remaining = seq_len - i
        print(f"   {remaining:^13} | {psi_acc:^8.1%} | {trans_acc:^11.1%}")

    print()
    print("4. PARITY PREFIX TEST (prefix state):")
    print(f"   PSI with prefix: {results.get('parity_PSI_full', 0):.1%}")
    print(f"   PSI without prefix: {results.get('parity_PSI_no_prefix', 0):.1%}")
    print(f"   Transformer with prefix: {results.get('parity_Transformer_full', 0):.1%}")
    print(f"   Transformer without prefix: {results.get('parity_Transformer_no_prefix', 0):.1%}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Truncation plot
    ax = axes[0, 0]
    x = list(range(traj_len, 0, -1))  # Remaining tokens
    ax.plot(x, truncation_results.get('PSI', []), 'o-', label='PSI',
            color='steelblue', linewidth=2, markersize=8)
    ax.plot(x, truncation_results.get('Transformer', []), 's-', label='Transformer',
            color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Tokens Remaining (start removed)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Truncation Test: State Retention')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.invert_xaxis()

    # History encoding
    ax = axes[0, 1]
    names = ['PSI', 'Transformer']
    full = [results.get('history_PSI_full', 0), results.get('history_Transformer_full', 0)]
    no_mode = [results.get('history_PSI_no_mode', 0), results.get('history_Transformer_no_mode', 0)]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, full, width, label='With mode token', color='steelblue')
    ax.bar(x + width/2, no_mode, width, label='Without mode token', color='coral')
    ax.axhline(y=0.25, color='gray', linestyle='--', label='Random (1/4)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    ax.set_title('History Encoding: Mode State')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # XOR plot
    ax = axes[1, 0]
    x = list(range(seq_len, 0, -1))
    ax.plot(x, xor_results.get('PSI', []), 'o-', label='PSI',
            color='steelblue', linewidth=2, markersize=8)
    ax.plot(x, xor_results.get('Transformer', []), 's-', label='Transformer',
            color='coral', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax.set_xlabel('Bits Remaining')
    ax.set_ylabel('Accuracy')
    ax.set_title('XOR Test: Cumulative State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.invert_xaxis()

    # Parity prefix
    ax = axes[1, 1]
    full = [results.get('parity_PSI_full', 0), results.get('parity_Transformer_full', 0)]
    no_prefix = [results.get('parity_PSI_no_prefix', 0), results.get('parity_Transformer_no_prefix', 0)]
    ax.bar(x - width/2, full, width, label='With prefix', color='steelblue')
    ax.bar(x + width/2, no_prefix, width, label='Without prefix', color='coral')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Parity Prefix: State Transfer')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('state_compression_test.png', dpi=150)
    plt.close()
    print("\nSaved state_compression_test.png")

    # Key finding
    print()
    print("=" * 80)
    print("KEY FINDING")
    print("=" * 80)

    # Calculate average advantage in truncation test
    psi_accs = truncation_results.get('PSI', [])
    trans_accs = truncation_results.get('Transformer', [])
    if psi_accs and trans_accs:
        advantages = [p - t for p, t in zip(psi_accs, trans_accs)]
        avg_adv = np.mean(advantages)
        max_adv = max(advantages)
        max_adv_idx = advantages.index(max_adv)
        remaining_at_max = traj_len - max_adv_idx

        print(f"""
State Compression Evidence:

TRUNCATION TEST:
- Average PSI advantage: {avg_adv*100:+.1f}%
- Maximum PSI advantage: {max_adv*100:+.1f}% (at {remaining_at_max} tokens remaining)

If PSI > Transformer with few tokens remaining, PSI has compressed
trajectory state that persists even when explicit tokens are removed.

This matches the original finding: PSI 88% vs Transformer 59% with 2-token context.
""")


if __name__ == "__main__":
    main()
