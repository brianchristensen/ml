"""
HARD Compositional Tasks: Stress-Testing PSI's Integration Capability

These tasks are designed to be HARDER than the previous ones:
1. Longer sequences (up to 50+ tokens)
2. Chained operations (multiple integration steps)
3. Recursive structures (deep nesting)
4. Algorithmic tasks (sorting, graph reachability)

KEY QUESTION: Where does PSI's parallelizable cumsum beat both:
- Transformer (fails on integration)
- LSTM (sequential, can't parallelize)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

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
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, num_heads=4, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.embedding(x) + self.pos_embed[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, dim=128, num_layers=4, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        h, _ = self.lstm(h)
        return self.head(self.norm(h))


# =============================================================================
# HARD Task Datasets
# =============================================================================

class LongXORDataset(Dataset):
    """XOR over LONG sequences (32-64 bits) - stresses state maintenance."""
    def __init__(self, n_examples=2000, min_len=32, max_len=64):
        self.examples = []
        for _ in range(n_examples):
            seq_len = np.random.randint(min_len, max_len + 1)
            bits = np.random.randint(0, 2, size=seq_len).tolist()
            target = sum(bits) % 2
            self.examples.append({
                'input': torch.tensor(bits, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'seq_len': seq_len
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DeepBracketDataset(Dataset):
    """
    Bracket matching with DEEP nesting (up to depth 10+).
    This requires tracking a counter that can go quite high.
    """
    def __init__(self, n_examples=2000, max_depth=15, seq_len=30):
        self.examples = []
        for _ in range(n_examples):
            # Generate balanced sequence with controlled depth
            seq, max_d = self._generate_balanced(seq_len // 2)

            # Sometimes corrupt it
            corrupt = np.random.random() < 0.5
            if corrupt and len(seq) > 2:
                # Swap two random positions
                i, j = np.random.choice(len(seq), 2, replace=False)
                seq[i], seq[j] = seq[j], seq[i]

            # Check validity
            valid = self._is_balanced(seq)
            target = 0 if valid else 1  # 0=balanced, 1=unbalanced

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            })

    def _generate_balanced(self, n_pairs):
        """Generate balanced sequence with n_pairs of brackets."""
        seq = []
        depth = 0
        max_depth = 0
        opens_remaining = n_pairs
        closes_remaining = n_pairs

        while opens_remaining > 0 or closes_remaining > 0:
            can_open = opens_remaining > 0
            can_close = closes_remaining > 0 and depth > 0

            if can_open and can_close:
                if np.random.random() < 0.5:
                    seq.append(0)  # open
                    opens_remaining -= 1
                    depth += 1
                    max_depth = max(max_depth, depth)
                else:
                    seq.append(1)  # close
                    closes_remaining -= 1
                    depth -= 1
            elif can_open:
                seq.append(0)
                opens_remaining -= 1
                depth += 1
                max_depth = max(max_depth, depth)
            elif can_close:
                seq.append(1)
                closes_remaining -= 1
                depth -= 1

        return seq, max_depth

    def _is_balanced(self, seq):
        depth = 0
        for b in seq:
            depth += 1 if b == 0 else -1
            if depth < 0:
                return False
        return depth == 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ChainedXORDataset(Dataset):
    """
    Chained XOR: Compute XOR of first half, XOR of second half, then XOR those together.

    This requires TWO integration steps, testing compositional computation.
    """
    def __init__(self, n_examples=2000, half_len=8):
        self.examples = []
        for _ in range(n_examples):
            first_half = np.random.randint(0, 2, size=half_len).tolist()
            second_half = np.random.randint(0, 2, size=half_len).tolist()

            xor1 = sum(first_half) % 2
            xor2 = sum(second_half) % 2
            final_xor = xor1 ^ xor2

            # Use token 2 as separator
            seq = first_half + [2] + second_half

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(final_xor, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class NestedModularSumDataset(Dataset):
    """
    Nested modular sum:
    Given groups separated by delimiters, compute sum of each group mod M,
    then sum those results mod M.

    Example: [1,2,3 | 4,5 | 6] -> (6 mod 5) + (9 mod 5) + (6 mod 5) = 1 + 4 + 1 = 6 mod 5 = 1
    """
    def __init__(self, n_examples=2000, n_groups=3, group_len=4, max_val=5, mod=7):
        self.examples = []
        self.mod = mod
        self.delimiter = max_val  # Use max_val as delimiter

        for _ in range(n_examples):
            groups = []
            seq = []
            for g in range(n_groups):
                group = np.random.randint(0, max_val, size=group_len).tolist()
                groups.append(group)
                seq.extend(group)
                if g < n_groups - 1:
                    seq.append(self.delimiter)

            # Compute nested modular sum
            group_sums = [sum(g) % mod for g in groups]
            final = sum(group_sums) % mod

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(final, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SortingVerificationDataset(Dataset):
    """
    Is the sequence sorted (ascending)?

    Requires tracking running max and comparing each new element.
    """
    def __init__(self, n_examples=2000, seq_len=12, max_val=20):
        self.examples = []
        for _ in range(n_examples):
            # Half sorted, half random
            if np.random.random() < 0.5:
                seq = sorted(np.random.randint(0, max_val, size=seq_len).tolist())
            else:
                seq = np.random.randint(0, max_val, size=seq_len).tolist()

            # Check if sorted
            is_sorted = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(int(is_sorted), dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RunningMaxDiffDataset(Dataset):
    """
    Output the running difference between current value and running max.

    Requires: tracking max AND computing difference at each step.
    """
    def __init__(self, n_examples=2000, seq_len=10, max_val=10):
        self.examples = []
        self.max_val = max_val

        for _ in range(n_examples):
            seq = np.random.randint(0, max_val, size=seq_len).tolist()

            # Compute running max diff
            running_max = 0
            diffs = []
            for v in seq:
                running_max = max(running_max, v)
                diff = running_max - v  # Always non-negative
                diffs.append(diff)

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(diffs, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class CumulativeXORDataset(Dataset):
    """
    Output running XOR at each position.

    This is the SEQUENCE version of XOR - should be PSI's sweet spot.
    """
    def __init__(self, n_examples=2000, seq_len=16):
        self.examples = []
        for _ in range(n_examples):
            bits = np.random.randint(0, 2, size=seq_len).tolist()

            # Compute cumulative XOR
            cumxor = []
            running = 0
            for b in bits:
                running ^= b
                cumxor.append(running)

            self.examples.append({
                'input': torch.tensor(bits, dtype=torch.long),
                'target': torch.tensor(cumxor, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MultipleCountersDataset(Dataset):
    """
    Count occurrences of MULTIPLE tokens (0, 1, 2) and output the maximum count.

    Requires tracking 3 separate counters and comparing them.
    """
    def __init__(self, n_examples=2000, seq_len=15, n_tokens=3):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, n_tokens, size=seq_len).tolist()

            # Count each token
            counts = [seq.count(i) for i in range(n_tokens)]
            max_count = max(counts)

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(max_count, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class AlternatingPatternDataset(Dataset):
    """
    Does the sequence follow alternating pattern (0,1,0,1,... or 1,0,1,0,...)?

    Requires tracking expected next value and comparing.
    """
    def __init__(self, n_examples=2000, seq_len=16):
        self.examples = []
        for _ in range(n_examples):
            # 50% alternating, 50% random
            if np.random.random() < 0.5:
                start = np.random.randint(0, 2)
                seq = [(start + i) % 2 for i in range(seq_len)]
            else:
                seq = np.random.randint(0, 2, size=seq_len).tolist()

            # Check if alternating
            is_alternating = True
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    is_alternating = False
                    break

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(int(is_alternating), dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class StackSimulationDataset(Dataset):
    """
    Simulate a stack: 0=push, 1=pop, 2=query (output current stack size mod 8).

    Final output is the stack size after all operations (clamped to 0).
    """
    def __init__(self, n_examples=2000, seq_len=20):
        self.examples = []
        for _ in range(n_examples):
            # Generate random operations
            ops = np.random.randint(0, 2, size=seq_len).tolist()  # 0=push, 1=pop

            # Simulate stack
            stack_size = 0
            for op in ops:
                if op == 0:
                    stack_size += 1
                else:
                    stack_size = max(0, stack_size - 1)

            self.examples.append({
                'input': torch.tensor(ops, dtype=torch.long),
                'target': torch.tensor(min(stack_size, 15), dtype=torch.long)  # Cap at 15
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class LongRangeXORDataset(Dataset):
    """
    XOR of first token and last token only.

    This tests LONG-RANGE dependency without full integration.
    Hypothesis: Transformer might do better here (can attend directly).
    """
    def __init__(self, n_examples=2000, seq_len=32):
        self.examples = []
        for _ in range(n_examples):
            # Middle tokens are random noise
            seq = np.random.randint(0, 2, size=seq_len).tolist()

            # XOR of first and last only
            target = seq[0] ^ seq[-1]

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
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

    # Handle both scalar and sequence targets
    if targets[0].dim() == 0:
        targets = torch.stack(targets)
    else:
        max_target_len = max(t.shape[0] for t in targets)
        padded_targets = []
        for t in targets:
            if t.shape[0] < max_target_len:
                t = F.pad(t, (0, max_target_len - t.shape[0]))
            padded_targets.append(t)
        targets = torch.stack(padded_targets)

    return {'input': torch.stack(inputs), 'target': targets}


def train_model(model, train_loader, epochs=50, lr=1e-3, task_type='final'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            logits = model(inputs)

            if task_type == 'final':
                loss = F.cross_entropy(logits[:, -1], targets)
            else:  # sequence
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()


def evaluate_model(model, test_loader, task_type='final'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            logits = model(inputs)

            if task_type == 'final':
                preds = logits[:, -1].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)
            else:  # sequence
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

    return correct / total


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("HARD COMPOSITIONAL TASKS: Stress-Testing PSI")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    dim = 128
    num_layers = 4

    # Define hard tasks
    # Format: (name, dataset_cls, kwargs, vocab_size, task_type, epochs)
    tasks = [
        # Integration stress tests
        ('Long XOR (32-64 bits)', LongXORDataset, {'min_len': 32, 'max_len': 64}, 2, 'final', 100),
        ('Cumulative XOR (seq)', CumulativeXORDataset, {'seq_len': 16}, 2, 'sequence', 80),

        # Chained operations
        ('Chained XOR', ChainedXORDataset, {'half_len': 8}, 3, 'final', 80),
        ('Nested Modular Sum', NestedModularSumDataset, {'n_groups': 3, 'group_len': 4, 'mod': 7}, 7, 'final', 80),

        # Stack/counter operations
        ('Deep Brackets (depth 15)', DeepBracketDataset, {'max_depth': 15, 'seq_len': 30}, 2, 'final', 100),
        ('Stack Simulation', StackSimulationDataset, {'seq_len': 20}, 16, 'final', 80),

        # Comparison tasks
        ('Sorting Verification', SortingVerificationDataset, {'seq_len': 12, 'max_val': 20}, 20, 'final', 80),
        ('Running Max Diff', RunningMaxDiffDataset, {'seq_len': 10, 'max_val': 10}, 10, 'sequence', 80),

        # Multi-counter (max_count up to seq_len, so 16 possible classes)
        ('Multiple Counters (3)', MultipleCountersDataset, {'seq_len': 15, 'n_tokens': 3}, 16, 'final', 80),
        ('Alternating Pattern', AlternatingPatternDataset, {'seq_len': 16}, 2, 'final', 60),

        # Long-range (Transformer should win)
        ('Long-Range XOR (first^last)', LongRangeXORDataset, {'seq_len': 32}, 2, 'final', 60),
    ]

    results = {}

    for task_name, dataset_cls, dataset_kwargs, vocab_size, task_type, epochs in tasks:
        print("=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        train_data = dataset_cls(n_examples=5000, **dataset_kwargs)
        test_data = dataset_cls(n_examples=500, **dataset_kwargs)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

        task_results = {}

        for name, model_fn in [
            ('PSI', lambda vs=vocab_size: PSIModel(vs, dim, num_layers, max_len=100)),
            ('Transformer', lambda vs=vocab_size: TransformerModel(vs, dim, num_layers, max_len=100)),
            ('LSTM', lambda vs=vocab_size: LSTMModel(vs, dim, num_layers)),
        ]:
            model = model_fn().to(device)

            start_time = time.time()
            train_model(model, train_loader, epochs=epochs, task_type=task_type)
            train_time = time.time() - start_time

            acc = evaluate_model(model, test_loader, task_type=task_type)

            task_results[name] = {'acc': acc, 'time': train_time}
            print(f"  {name}: {acc:.1%} (trained in {train_time:.1f}s)")

        results[task_name] = task_results
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Hard Compositional Task Performance")
    print("=" * 80)
    print()

    print(f"{'Task':<32} {'PSI':>8} {'Trans':>8} {'LSTM':>8} {'Winner':>12}")
    print("-" * 75)

    psi_wins = 0
    trans_wins = 0
    lstm_wins = 0
    ties = 0

    for task_name, task_results in results.items():
        psi = task_results['PSI']['acc']
        trans = task_results['Transformer']['acc']
        lstm = task_results['LSTM']['acc']

        best = max(psi, trans, lstm)
        margin = 0.03  # 3% margin for winner

        if psi >= best - margin and trans < best - margin and lstm < best - margin:
            winner = 'PSI'
            psi_wins += 1
        elif trans >= best - margin and psi < best - margin and lstm < best - margin:
            winner = 'Trans'
            trans_wins += 1
        elif lstm >= best - margin and psi < best - margin and trans < best - margin:
            winner = 'LSTM'
            lstm_wins += 1
        elif psi >= best - margin and lstm >= best - margin and trans < best - margin:
            winner = 'PSI/LSTM'
            ties += 1
        elif psi >= best - margin and trans >= best - margin and lstm < best - margin:
            winner = 'PSI/Trans'
            ties += 1
        else:
            winner = 'Tie'
            ties += 1

        print(f"{task_name:<32} {psi:>7.1%} {trans:>7.1%} {lstm:>7.1%} {winner:>12}")

    print("-" * 75)
    print(f"{'WINS':<32} {psi_wins:>8} {trans_wins:>8} {lstm_wins:>8} Ties: {ties}")
    print()

    # Training time comparison
    print("=" * 80)
    print("TRAINING TIME COMPARISON (seconds)")
    print("=" * 80)
    print()

    print(f"{'Task':<32} {'PSI':>10} {'Trans':>10} {'LSTM':>10} {'Fastest':>10}")
    print("-" * 75)

    total_psi = 0
    total_trans = 0
    total_lstm = 0

    for task_name, task_results in results.items():
        psi_t = task_results['PSI']['time']
        trans_t = task_results['Transformer']['time']
        lstm_t = task_results['LSTM']['time']

        total_psi += psi_t
        total_trans += trans_t
        total_lstm += lstm_t

        fastest = min(psi_t, trans_t, lstm_t)
        if psi_t == fastest:
            fast_name = 'PSI'
        elif trans_t == fastest:
            fast_name = 'Trans'
        else:
            fast_name = 'LSTM'

        print(f"{task_name:<32} {psi_t:>9.1f}s {trans_t:>9.1f}s {lstm_t:>9.1f}s {fast_name:>10}")

    print("-" * 75)
    print(f"{'TOTAL':<32} {total_psi:>9.1f}s {total_trans:>9.1f}s {total_lstm:>9.1f}s")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Accuracy plot
    task_names = list(results.keys())
    x = np.arange(len(task_names))
    width = 0.25

    psi_accs = [results[t]['PSI']['acc'] for t in task_names]
    trans_accs = [results[t]['Transformer']['acc'] for t in task_names]
    lstm_accs = [results[t]['LSTM']['acc'] for t in task_names]

    ax1.bar(x - width, psi_accs, width, label='PSI', color='steelblue')
    ax1.bar(x, trans_accs, width, label='Transformer', color='coral')
    ax1.bar(x + width, lstm_accs, width, label='LSTM', color='seagreen')

    ax1.set_xlabel('Task')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Hard Compositional Tasks: Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')

    # Time plot
    psi_times = [results[t]['PSI']['time'] for t in task_names]
    trans_times = [results[t]['Transformer']['time'] for t in task_names]
    lstm_times = [results[t]['LSTM']['time'] for t in task_names]

    ax2.bar(x - width, psi_times, width, label='PSI', color='steelblue')
    ax2.bar(x, trans_times, width, label='Transformer', color='coral')
    ax2.bar(x + width, lstm_times, width, label='LSTM', color='seagreen')

    ax2.set_xlabel('Task')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Hard Compositional Tasks: Training Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('hard_compositional_tasks.png', dpi=150)
    plt.close()
    print("\nSaved hard_compositional_tasks.png")

    # Key insights
    print()
    print("=" * 80)
    print("ANALYSIS: What These Results Tell Us")
    print("=" * 80)
    print("""
    TASK CATEGORIES:

    1. INTEGRATION STRESS (Long XOR, Cumulative XOR):
       - Tests maintaining cumulative state over long sequences
       - PSI/LSTM should excel, Transformer should struggle

    2. CHAINED OPERATIONS (Chained XOR, Nested Modular Sum):
       - Tests multi-step compositional computation
       - Requires integration + reset + integration

    3. STACK/COUNTER (Deep Brackets, Stack Simulation):
       - Tests unbounded counter tracking
       - Deeper nesting = harder problem

    4. COMPARISON (Sorting, Running Max Diff):
       - Tests tracking running extrema
       - Requires state + comparison

    5. LONG-RANGE (Long-Range XOR):
       - Tests direct long-range access
       - Transformer should win (can attend directly to first/last)

    KEY QUESTION:
    Does PSI offer anything LSTM can't do?
    - If PSI = LSTM in accuracy but faster -> PSI wins on speed
    - If PSI > LSTM -> cumsum is more expressive than LSTM gates
    - If LSTM > PSI -> LSTM's gating is more flexible
    """)


if __name__ == "__main__":
    main()
