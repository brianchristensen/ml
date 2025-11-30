"""
Compositional Tasks: Where Does PSI's Cumsum Integration Shine?

KEY FINDING: PSI solves XOR (100%) while Transformer fails (51%)

WHY? Hypothesis:
- XOR requires INTEGRATING information across the entire sequence
- PSI's cumsum naturally accumulates/integrates over sequence
- Transformer's attention is designed for RETRIEVAL, not integration

COMPOSITIONAL TASKS TO TEST:

1. COUNTING: Count occurrences of a token
   - Requires: Incrementing a counter (integration)
   - cumsum naturally does this

2. RUNNING SUM: Output cumulative sum at each position
   - Requires: Adding each new value to running total
   - cumsum IS this operation

3. MAJORITY VOTE: Which token appears more often?
   - Requires: Tracking counts of multiple tokens
   - Similar to counting but with comparison

4. BRACKET MATCHING: Are brackets balanced?
   - Requires: Tracking nesting depth (increment/decrement)
   - Classic stack operation

5. SORTING: Is sequence sorted? / What's the max?
   - Requires: Comparing with running max/min
   - Monotonic tracking

6. MODULAR ARITHMETIC: Sum mod N
   - Requires: Accumulating and wrapping
   - Tests if cumsum learns modular structure

7. FIRST/LAST OCCURRENCE: Position of first/last X
   - Requires: Tracking position state
   - Tests stateful computation

8. PATTERN COUNTING: How many times does [A,B] appear?
   - Requires: State machine + counting
   - Compositional: pattern detection + counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models (same as before)
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
    """LSTM baseline - also has cumulative state via hidden/cell."""
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
# Task Datasets
# =============================================================================

class CountingDataset(Dataset):
    """Count occurrences of token 1 in sequence of 0s and 1s."""
    def __init__(self, n_examples=2000, seq_len=10, max_count=10):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, 2, size=seq_len).tolist()
            count = sum(seq)
            if count <= max_count:
                self.examples.append({
                    'input': torch.tensor(seq, dtype=torch.long),
                    'target': torch.tensor(count, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RunningSumDataset(Dataset):
    """Predict cumulative sum at each position (mod vocab_size)."""
    def __init__(self, n_examples=2000, seq_len=10, max_val=5, mod=20):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, max_val, size=seq_len).tolist()
            cumsum = np.cumsum(seq) % mod
            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(cumsum.tolist(), dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MajorityVoteDataset(Dataset):
    """Which of {0, 1} appears more? Output 0 or 1."""
    def __init__(self, n_examples=2000, seq_len=11):  # Odd length to avoid ties
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, 2, size=seq_len).tolist()
            majority = 1 if sum(seq) > seq_len // 2 else 0
            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(majority, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class BracketMatchingDataset(Dataset):
    """
    Are brackets balanced? Simplified: sequence of +1 (open) and -1 (close).
    Valid if: running sum never goes negative AND final sum is 0.

    Use tokens: 0 = open, 1 = close
    Output: 2 = balanced, 3 = unbalanced
    """
    def __init__(self, n_examples=2000, seq_len=10):
        self.examples = []
        for _ in range(n_examples):
            # Generate random bracket sequence
            seq = np.random.randint(0, 2, size=seq_len).tolist()

            # Check if balanced
            depth = 0
            valid = True
            for b in seq:
                depth += 1 if b == 0 else -1
                if depth < 0:
                    valid = False
                    break
            if depth != 0:
                valid = False

            target = 2 if valid else 3  # 2=balanced, 3=unbalanced

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ModularSumDataset(Dataset):
    """Sum of sequence mod N."""
    def __init__(self, n_examples=2000, seq_len=8, max_val=5, mod=7):
        self.examples = []
        self.mod = mod
        for _ in range(n_examples):
            seq = np.random.randint(0, max_val, size=seq_len).tolist()
            total = sum(seq) % mod
            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(total, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MaxValueDataset(Dataset):
    """Find maximum value in sequence."""
    def __init__(self, n_examples=2000, seq_len=8, max_val=10):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, max_val, size=seq_len).tolist()
            max_v = max(seq)
            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(max_v, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class FirstOccurrenceDataset(Dataset):
    """Find position of first occurrence of token 1 (or seq_len if not present)."""
    def __init__(self, n_examples=2000, seq_len=8):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, 2, size=seq_len).tolist()
            try:
                first_pos = seq.index(1)
            except ValueError:
                first_pos = seq_len  # Not found

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(first_pos, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PatternCountDataset(Dataset):
    """Count occurrences of pattern [1, 0] in sequence."""
    def __init__(self, n_examples=2000, seq_len=12):
        self.examples = []
        for _ in range(n_examples):
            seq = np.random.randint(0, 2, size=seq_len).tolist()

            # Count [1, 0] patterns
            count = 0
            for i in range(len(seq) - 1):
                if seq[i] == 1 and seq[i + 1] == 0:
                    count += 1

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(count, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class XORDataset(Dataset):
    """XOR of all bits (baseline from before)."""
    def __init__(self, n_examples=2000, seq_len=8):
        self.examples = []
        for _ in range(n_examples):
            bits = np.random.randint(0, 2, size=seq_len).tolist()
            target = sum(bits) % 2
            self.examples.append({
                'input': torch.tensor(bits, dtype=torch.long),
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

    targets = torch.stack(targets)
    return {'input': torch.stack(inputs), 'target': targets}


def train_model(model, train_loader, epochs=50, lr=1e-3, task_type='final'):
    """
    task_type: 'final' = predict from last position, 'sequence' = predict at each position
    """
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

            if task_type == 'final':
                loss = F.cross_entropy(logits[:, -1], targets)
            else:  # sequence
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    return total_loss / len(train_loader)


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
    print("COMPOSITIONAL TASKS: Where PSI's Integration Shines")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    dim = 128
    num_layers = 4

    # Define tasks
    tasks = [
        ('XOR (baseline)', XORDataset, {'seq_len': 8}, 2, 'final', 60),
        ('Counting', CountingDataset, {'seq_len': 10, 'max_count': 10}, 11, 'final', 60),
        ('Majority Vote', MajorityVoteDataset, {'seq_len': 11}, 2, 'final', 60),
        ('Bracket Matching', BracketMatchingDataset, {'seq_len': 10}, 4, 'final', 80),
        ('Modular Sum (mod 7)', ModularSumDataset, {'seq_len': 8, 'max_val': 5, 'mod': 7}, 7, 'final', 60),
        ('Max Value', MaxValueDataset, {'seq_len': 8, 'max_val': 10}, 10, 'final', 60),
        ('First Occurrence', FirstOccurrenceDataset, {'seq_len': 8}, 9, 'final', 60),
        ('Pattern Count [1,0]', PatternCountDataset, {'seq_len': 12}, 7, 'final', 60),
        ('Running Sum', RunningSumDataset, {'seq_len': 10, 'max_val': 5, 'mod': 20}, 20, 'sequence', 60),
    ]

    results = {}

    for task_name, dataset_cls, dataset_kwargs, vocab_size, task_type, epochs in tasks:
        print("=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        train_data = dataset_cls(n_examples=4000, **dataset_kwargs)
        test_data = dataset_cls(n_examples=500, **dataset_kwargs)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

        task_results = {}

        for name, model_fn in [
            ('PSI', lambda vs=vocab_size: PSIModel(vs, dim, num_layers)),
            ('Transformer', lambda vs=vocab_size: TransformerModel(vs, dim, num_layers)),
            ('LSTM', lambda vs=vocab_size: LSTMModel(vs, dim, num_layers)),
        ]:
            model = model_fn().to(device)
            n_params = sum(p.numel() for p in model.parameters())

            train_model(model, train_loader, epochs=epochs, task_type=task_type)
            acc = evaluate_model(model, test_loader, task_type=task_type)

            task_results[name] = acc
            print(f"  {name}: {acc:.1%}")

        results[task_name] = task_results
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Compositional Task Performance")
    print("=" * 80)
    print()

    print(f"{'Task':<25} {'PSI':>10} {'Transformer':>12} {'LSTM':>10} {'Winner':>12}")
    print("-" * 75)

    psi_wins = 0
    trans_wins = 0
    lstm_wins = 0
    ties = 0

    for task_name, task_results in results.items():
        psi = task_results.get('PSI', 0)
        trans = task_results.get('Transformer', 0)
        lstm = task_results.get('LSTM', 0)

        best = max(psi, trans, lstm)
        if psi == best and trans < best - 0.02 and lstm < best - 0.02:
            winner = 'PSI'
            psi_wins += 1
        elif trans == best and psi < best - 0.02 and lstm < best - 0.02:
            winner = 'Transformer'
            trans_wins += 1
        elif lstm == best and psi < best - 0.02 and trans < best - 0.02:
            winner = 'LSTM'
            lstm_wins += 1
        else:
            winner = 'Tie'
            ties += 1

        print(f"{task_name:<25} {psi:>9.1%} {trans:>11.1%} {lstm:>9.1%} {winner:>12}")

    print("-" * 75)
    print(f"{'WINS':<25} {psi_wins:>10} {trans_wins:>12} {lstm_wins:>10}")
    print()

    # Categorize tasks
    print("=" * 80)
    print("ANALYSIS: Task Categories")
    print("=" * 80)

    integration_tasks = ['XOR (baseline)', 'Counting', 'Majority Vote', 'Modular Sum (mod 7)', 'Running Sum']
    state_machine_tasks = ['Bracket Matching', 'First Occurrence', 'Pattern Count [1,0]']
    comparison_tasks = ['Max Value']

    print("\nINTEGRATION TASKS (accumulate over sequence):")
    for task in integration_tasks:
        if task in results:
            r = results[task]
            print(f"  {task}: PSI {r['PSI']:.1%}, Trans {r['Transformer']:.1%}, LSTM {r['LSTM']:.1%}")

    print("\nSTATE MACHINE TASKS (track discrete state):")
    for task in state_machine_tasks:
        if task in results:
            r = results[task]
            print(f"  {task}: PSI {r['PSI']:.1%}, Trans {r['Transformer']:.1%}, LSTM {r['LSTM']:.1%}")

    print("\nCOMPARISON TASKS (compare values):")
    for task in comparison_tasks:
        if task in results:
            r = results[task]
            print(f"  {task}: PSI {r['PSI']:.1%}, Trans {r['Transformer']:.1%}, LSTM {r['LSTM']:.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    task_names = list(results.keys())
    x = np.arange(len(task_names))
    width = 0.25

    psi_accs = [results[t]['PSI'] for t in task_names]
    trans_accs = [results[t]['Transformer'] for t in task_names]
    lstm_accs = [results[t]['LSTM'] for t in task_names]

    ax.bar(x - width, psi_accs, width, label='PSI', color='steelblue')
    ax.bar(x, trans_accs, width, label='Transformer', color='coral')
    ax.bar(x + width, lstm_accs, width, label='LSTM', color='seagreen')

    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('Compositional Tasks: PSI vs Transformer vs LSTM')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('compositional_tasks.png', dpi=150)
    plt.close()
    print("\nSaved compositional_tasks.png")

    # Key insight
    print()
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
    PSI's cumsum operation naturally performs INTEGRATION:
    - cumsum(gate * value) / cumsum(gate)

    This is mathematically similar to:
    - Running average (with learned weights)
    - Exponential moving average
    - Numerical integration

    Tasks where this helps:
    - XOR: Parity is sum mod 2 (integration + mod)
    - Counting: Direct summation
    - Modular Sum: Integration with wraparound
    - Running Sum: cumsum IS the operation

    Tasks where attention helps:
    - Lookup/retrieval tasks (associative recall)
    - Pattern matching (specific token positions)
    - Long-range dependencies (attention can skip)

    LSTM also has cumulative state (hidden/cell), which may explain
    why it sometimes matches PSI on integration tasks.
    """)


if __name__ == "__main__":
    main()
