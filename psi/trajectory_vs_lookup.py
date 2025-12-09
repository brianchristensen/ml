"""
Trajectory vs Lookup: Testing PSI's Fundamental Difference

HYPOTHESIS: PSI learns dynamical trajectories, Transformers learn content-addressable lookup.

TEST DESIGN:
We create tasks where these two strategies diverge:

1. TRAJECTORY TASK (PSI should win):
   - Input: A sequence that follows a hidden dynamical rule
   - Output: Predict continuation based on the TRAJECTORY pattern
   - Key: The answer depends on HOW you got there, not WHERE you are

   Example: "A B C D E" → next depends on the trajectory (incrementing)
            vs "E D C B A" → same tokens, different trajectory (decrementing)

2. LOOKUP TASK (Transformer should win):
   - Input: Key-value pairs, then query
   - Output: Retrieve the value for queried key
   - Key: The answer depends on WHAT specific token appeared earlier

3. INTERFERENCE TASK (tests robustness):
   - Trajectory task with distractor tokens inserted
   - PSI should ignore distractors (they don't change trajectory)
   - Transformer may get confused (distractors appear in attention)

4. COMPOSITIONAL TRAJECTORY (the killer test):
   - Learn multiple trajectory rules independently
   - Test on compositions never seen in training
   - PSI's continuous dynamics should generalize
   - Transformer's discrete lookup should fail
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
# PSI Model
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

        # Cumsum memory
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

        h = self.norm(h)
        return self.head(h)


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

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)

        h = self.norm(h)
        return self.head(h)


# =============================================================================
# Task 1: Trajectory Prediction (Direction Matters)
# =============================================================================

class TrajectoryDataset(Dataset):
    """
    Sequences follow hidden trajectories. Same destination, different paths.

    ASCENDING:  A → B → C → D → E → F (predict F after seeing trajectory)
    DESCENDING: F → E → D → C → B → A (predict A after seeing trajectory)

    Key insight: If you only see "...C → D → E", you can't know direction!
    But if you see the full trajectory, the pattern is clear.

    PSI should learn the trajectory dynamics.
    Transformer might memorize (A,B,C,D,E) → F without understanding direction.
    """

    def __init__(self, n_examples=2000, seq_len=10, vocab_size=26):
        self.examples = []
        self.vocab_size = vocab_size

        for _ in range(n_examples):
            # Random starting point and direction
            direction = np.random.choice(['asc', 'desc'])

            # Choose a range that fits
            range_size = seq_len
            if direction == 'asc':
                start = np.random.randint(0, vocab_size - range_size)
                seq = list(range(start, start + range_size))
            else:
                start = np.random.randint(range_size - 1, vocab_size)
                seq = list(range(start, start - range_size, -1))

            # Input: all but last, Target: predict each next token
            input_seq = seq[:-1]
            target_seq = seq[1:]

            self.examples.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target_seq, dtype=torch.long),
                'direction': direction
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Task 2: Ambiguous Trajectory (Critical Test)
# =============================================================================

class AmbiguousTrajectoryDataset(Dataset):
    """
    THE CRITICAL TEST: Same recent tokens, different history → different answer.

    Example:
    - History A: "0 1 2 3 4" → next is 5 (ascending)
    - History B: "8 7 6 5 4" → next is 3 (descending)

    Both end in "4", but the trajectory determines the answer!

    We test: given partial context that's ambiguous, can model use trajectory?

    Training: Full trajectories (unambiguous)
    Testing:
      - Full trajectory (should be easy for both)
      - Partial trajectory from middle (ambiguous - tests trajectory memory)
    """

    def __init__(self, n_examples=2000, full_len=8, vocab_size=20, mode='train'):
        self.examples = []
        self.vocab_size = vocab_size
        self.mode = mode
        self.full_len = full_len

        for _ in range(n_examples):
            direction = np.random.choice(['asc', 'desc'])

            if direction == 'asc':
                start = np.random.randint(0, vocab_size - full_len)
                seq = list(range(start, start + full_len))
            else:
                start = np.random.randint(full_len - 1, vocab_size)
                seq = list(range(start, start - full_len, -1))

            self.examples.append({
                'full_input': torch.tensor(seq[:-1], dtype=torch.long),
                'full_target': torch.tensor(seq[-1], dtype=torch.long),
                'direction': direction,
                'seq': seq
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Task 3: Trajectory with Distractors
# =============================================================================

class DistractorTrajectoryDataset(Dataset):
    """
    Trajectory with random distractor tokens inserted.

    Example:
    - Clean: "0 1 2 3 4" → 5
    - With distractors: "0 [X] 1 [Y] 2 [Z] 3 4" → 5

    The distractors are from a separate vocab range and should be ignored.

    PSI hypothesis: Cumsum naturally smooths over distractors
    Transformer hypothesis: Attention may attend to distractors
    """

    def __init__(self, n_examples=2000, traj_len=6, n_distractors=3,
                 traj_vocab=20, distractor_vocab_start=20, distractor_vocab_size=10):
        self.examples = []
        self.traj_vocab = traj_vocab
        self.total_vocab = distractor_vocab_start + distractor_vocab_size

        for _ in range(n_examples):
            direction = np.random.choice(['asc', 'desc'])

            if direction == 'asc':
                start = np.random.randint(0, traj_vocab - traj_len)
                traj = list(range(start, start + traj_len))
            else:
                start = np.random.randint(traj_len - 1, traj_vocab)
                traj = list(range(start, start - traj_len, -1))

            # Insert distractors at random positions
            seq = []
            distractor_positions = sorted(np.random.choice(
                range(1, traj_len - 1),
                size=min(n_distractors, traj_len - 2),
                replace=False
            ))

            dist_idx = 0
            for i, t in enumerate(traj[:-1]):  # Don't include target in input
                seq.append(t)
                if dist_idx < len(distractor_positions) and i == distractor_positions[dist_idx]:
                    # Add distractor
                    d = np.random.randint(distractor_vocab_start, self.total_vocab)
                    seq.append(d)
                    dist_idx += 1

            target = traj[-1]

            self.examples.append({
                'input': torch.tensor(seq, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long),
                'clean_traj': torch.tensor(traj[:-1], dtype=torch.long),
                'direction': direction
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Task 4: Compositional Trajectories (Generalization Test)
# =============================================================================

class CompositionalTrajectoryDataset(Dataset):
    """
    Learn simple trajectory rules, test on compositions.

    Rules:
    - Rule A: +1 step (ascending)
    - Rule B: -1 step (descending)
    - Rule C: +2 step (skip ascending)
    - Rule D: -2 step (skip descending)

    Training: Single rules applied consistently
    Testing:
      - Held-out starting points (interpolation)
      - Rule switches mid-sequence (composition - extrapolation)

    Example composition (NOT in training):
    "0 1 2 3 [SWITCH] 3 1" → next is -1 (switched from +1 to -2)
    """

    def __init__(self, n_examples=2000, seq_len=8, vocab_size=30,
                 mode='train', include_compositions=False):
        self.examples = []
        self.vocab_size = vocab_size
        self.SWITCH_TOKEN = vocab_size  # Special token
        self.total_vocab = vocab_size + 1

        rules = {
            'asc1': lambda x: x + 1,
            'desc1': lambda x: x - 1,
            'asc2': lambda x: x + 2,
            'desc2': lambda x: x - 2,
        }

        for _ in range(n_examples):
            if include_compositions and np.random.random() < 0.3:
                # Compositional: switch rules mid-sequence
                rule1_name = np.random.choice(list(rules.keys()))
                rule2_name = np.random.choice([r for r in rules.keys() if r != rule1_name])
                rule1 = rules[rule1_name]
                rule2 = rules[rule2_name]

                switch_point = seq_len // 2

                # Find valid starting point
                start = np.random.randint(5, vocab_size - 5)
                seq = [start]

                # Apply rule 1
                for _ in range(switch_point - 1):
                    next_val = rule1(seq[-1])
                    if 0 <= next_val < vocab_size:
                        seq.append(next_val)
                    else:
                        break

                # Add switch token
                seq.append(self.SWITCH_TOKEN)

                # Apply rule 2
                for _ in range(seq_len - len(seq)):
                    next_val = rule2(seq[-2] if seq[-1] == self.SWITCH_TOKEN else seq[-1])
                    if 0 <= next_val < vocab_size:
                        seq.append(next_val)
                    else:
                        break

                if len(seq) < 4:
                    continue

            else:
                # Single rule
                rule_name = np.random.choice(list(rules.keys()))
                rule = rules[rule_name]

                # Find valid starting point based on rule
                if 'asc' in rule_name:
                    step = 1 if '1' in rule_name else 2
                    start = np.random.randint(0, vocab_size - seq_len * step)
                else:
                    step = 1 if '1' in rule_name else 2
                    start = np.random.randint(seq_len * step, vocab_size)

                seq = [start]
                for _ in range(seq_len - 1):
                    next_val = rule(seq[-1])
                    if 0 <= next_val < vocab_size:
                        seq.append(next_val)
                    else:
                        break

            if len(seq) >= 3:
                self.examples.append({
                    'input': torch.tensor(seq[:-1], dtype=torch.long),
                    'target': torch.tensor(seq[-1], dtype=torch.long),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Training and Evaluation
# =============================================================================

def collate_fn(batch):
    """Pad sequences."""
    max_input_len = max(item['input'].shape[0] for item in batch)
    max_target_len = max(item['target'].shape[0] for item in batch) if batch[0]['target'].dim() > 0 else 0

    inputs = []
    targets = []

    for item in batch:
        inp = item['input']
        if inp.shape[0] < max_input_len:
            inp = F.pad(inp, (0, max_input_len - inp.shape[0]))
        inputs.append(inp)

        tgt = item['target']
        if tgt.dim() == 0:
            # Single target value
            targets.append(tgt)
        else:
            # Sequence target - pad to match input length
            if tgt.shape[0] < max_input_len:
                tgt = F.pad(tgt, (0, max_input_len - tgt.shape[0]))
            targets.append(tgt)

    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets)
    }


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, task_type='sequence'):
    """Train model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            if task_type == 'sequence':
                # Predict each next token - align shapes properly
                # logits: [batch, seq_len, vocab], targets: [batch, seq_len]
                # We predict position i from position i (autoregressive would be different)
                batch_size, seq_len, vocab_size = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=0
                )
            else:
                # Predict single final token
                loss = F.cross_entropy(logits[:, -1], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)

                logits = model(inputs)

                if task_type == 'sequence':
                    preds = logits.argmax(dim=-1)
                    mask = targets != 0
                    correct += (preds[mask] == targets[mask]).sum().item()
                    total += mask.sum().item()
                else:
                    preds = logits[:, -1].argmax(dim=-1)
                    correct += (preds == targets).sum().item()
                    total += len(targets)

        val_acc = correct / total if total > 0 else 0
        best_val_acc = max(best_val_acc, val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.1%}")

    return best_val_acc


def test_ambiguous(model, dataset, context_sizes):
    """Test on ambiguous trajectories with varying context."""
    model.eval()
    results = {}

    for ctx_size in context_sizes:
        correct = 0
        total = 0

        with torch.no_grad():
            for item in dataset.examples:
                full_input = item['full_input']
                target = item['full_target']

                # Take only last ctx_size tokens
                if ctx_size >= len(full_input):
                    partial_input = full_input
                else:
                    partial_input = full_input[-ctx_size:]

                # Predict
                logits = model(partial_input.unsqueeze(0).to(device))
                pred = logits[0, -1].argmax().item()

                if pred == target.item():
                    correct += 1
                total += 1

        results[ctx_size] = correct / total

    return results


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("TRAJECTORY VS LOOKUP: Testing PSI's Fundamental Difference")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Model configs (matched parameters)
    dim = 128
    num_layers = 4
    vocab_size = 32  # Small vocab for trajectory tasks

    results = {}

    # =========================================================================
    # TEST 1: Basic Trajectory (Direction Matters)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: Basic Trajectory Prediction")
    print("Can the model learn ascending vs descending patterns?")
    print("=" * 70)
    print()

    train_data = TrajectoryDataset(n_examples=2000, seq_len=8, vocab_size=vocab_size)
    val_data = TrajectoryDataset(n_examples=500, seq_len=8, vocab_size=vocab_size)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(vocab_size, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(vocab_size, dim, num_layers))
    ]:
        print(f"Training {name}...")
        model = model_fn().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        acc = train_model(model, train_loader, val_loader, epochs=30, task_type='sequence')
        results[f'basic_traj_{name}'] = acc
        print(f"  Final accuracy: {acc:.1%}")
        print()

    # =========================================================================
    # TEST 2: Ambiguous Trajectory (Context Dependency)
    # =========================================================================
    print("=" * 70)
    print("TEST 2: Ambiguous Trajectory (THE CRITICAL TEST)")
    print("Same recent tokens, different history -> different answer")
    print("Tests: Does model remember trajectory or just recent tokens?")
    print("=" * 70)
    print()

    train_data = AmbiguousTrajectoryDataset(n_examples=3000, full_len=8, vocab_size=20)
    val_data = AmbiguousTrajectoryDataset(n_examples=500, full_len=8, vocab_size=20)

    # Custom collate for this task
    def ambig_collate(batch):
        return {
            'input': torch.stack([item['full_input'] for item in batch]),
            'target': torch.stack([item['full_target'] for item in batch])
        }

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=ambig_collate)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=ambig_collate)

    context_sizes = [2, 4, 6, 8]  # Test with varying context

    for name, model_fn in [
        ('PSI', lambda: PSIModel(20, dim, num_layers)),
        ('Transformer', lambda: TransformerModel(20, dim, num_layers))
    ]:
        print(f"Training {name}...")
        model = model_fn().to(device)

        acc = train_model(model, train_loader, val_loader, epochs=50, task_type='single')
        print(f"  Full context accuracy: {acc:.1%}")

        # Test with partial context
        print(f"  Testing with partial context:")
        partial_results = test_ambiguous(model, val_data, context_sizes)
        for ctx, pacc in partial_results.items():
            print(f"    Context {ctx}: {pacc:.1%}")
            results[f'ambig_{name}_ctx{ctx}'] = pacc
        print()

    # =========================================================================
    # TEST 3: Trajectory with Distractors
    # =========================================================================
    print("=" * 70)
    print("TEST 3: Trajectory with Distractors")
    print("Can model ignore irrelevant tokens in the sequence?")
    print("=" * 70)
    print()

    # Train on clean, test on distracted
    train_clean = TrajectoryDataset(n_examples=2000, seq_len=6, vocab_size=20)
    val_clean = TrajectoryDataset(n_examples=500, seq_len=6, vocab_size=20)
    val_distracted = DistractorTrajectoryDataset(n_examples=500, traj_len=6, n_distractors=2)

    train_loader = DataLoader(train_clean, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader_clean = DataLoader(val_clean, batch_size=64, shuffle=False, collate_fn=collate_fn)

    def distractor_collate(batch):
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

    val_loader_dist = DataLoader(val_distracted, batch_size=64, shuffle=False,
                                  collate_fn=distractor_collate)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(30, dim, num_layers)),  # Larger vocab for distractors
        ('Transformer', lambda: TransformerModel(30, dim, num_layers))
    ]:
        print(f"Training {name} on clean data...")
        model = model_fn().to(device)

        # Train on clean
        acc_clean = train_model(model, train_loader, val_loader_clean, epochs=30, task_type='sequence')
        print(f"  Clean accuracy: {acc_clean:.1%}")

        # Test on distracted
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader_dist:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                logits = model(inputs)
                preds = logits[:, -1].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        acc_dist = correct / total
        results[f'distractor_{name}_clean'] = acc_clean
        results[f'distractor_{name}_noisy'] = acc_dist
        print(f"  Distracted accuracy: {acc_dist:.1%}")
        print(f"  Degradation: {(acc_clean - acc_dist) / acc_clean * 100:.1f}%")
        print()

    # =========================================================================
    # TEST 4: Compositional Trajectories
    # =========================================================================
    print("=" * 70)
    print("TEST 4: Compositional Trajectories (Generalization)")
    print("Train on single rules, test on rule compositions")
    print("=" * 70)
    print()

    train_data = CompositionalTrajectoryDataset(n_examples=3000, seq_len=8, vocab_size=30,
                                                 include_compositions=False)
    val_simple = CompositionalTrajectoryDataset(n_examples=500, seq_len=8, vocab_size=30,
                                                 include_compositions=False)
    val_comp = CompositionalTrajectoryDataset(n_examples=500, seq_len=8, vocab_size=30,
                                               include_compositions=True)

    def comp_collate(batch):
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

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=comp_collate)
    val_loader_simple = DataLoader(val_simple, batch_size=64, shuffle=False, collate_fn=comp_collate)
    val_loader_comp = DataLoader(val_comp, batch_size=64, shuffle=False, collate_fn=comp_collate)

    for name, model_fn in [
        ('PSI', lambda: PSIModel(31, dim, num_layers)),  # +1 for SWITCH token
        ('Transformer', lambda: TransformerModel(31, dim, num_layers))
    ]:
        print(f"Training {name} on simple rules...")
        model = model_fn().to(device)

        # Train on simple rules
        acc_simple = train_model(model, train_loader, val_loader_simple, epochs=40, task_type='single')
        print(f"  Simple rules accuracy: {acc_simple:.1%}")

        # Test on compositions
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader_comp:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                logits = model(inputs)
                preds = logits[:, -1].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        acc_comp = correct / total
        results[f'comp_{name}_simple'] = acc_simple
        results[f'comp_{name}_composed'] = acc_comp
        print(f"  Composition accuracy: {acc_comp:.1%}")
        print(f"  Generalization gap: {(acc_simple - acc_comp) / acc_simple * 100:.1f}%")
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY: Trajectory vs Lookup")
    print("=" * 80)
    print()

    print(f"{'Test':<40} {'PSI':>12} {'Transformer':>12} {'Winner':>12}")
    print("-" * 80)

    comparisons = [
        ('Basic Trajectory', 'basic_traj_PSI', 'basic_traj_Transformer'),
        ('Ambiguous (ctx=2)', 'ambig_PSI_ctx2', 'ambig_Transformer_ctx2'),
        ('Ambiguous (ctx=4)', 'ambig_PSI_ctx4', 'ambig_Transformer_ctx4'),
        ('Ambiguous (ctx=8)', 'ambig_PSI_ctx8', 'ambig_Transformer_ctx8'),
        ('Distractor (clean)', 'distractor_PSI_clean', 'distractor_Transformer_clean'),
        ('Distractor (noisy)', 'distractor_PSI_noisy', 'distractor_Transformer_noisy'),
        ('Compositional (simple)', 'comp_PSI_simple', 'comp_Transformer_simple'),
        ('Compositional (composed)', 'comp_PSI_composed', 'comp_Transformer_composed'),
    ]

    psi_wins = 0
    trans_wins = 0

    for test_name, psi_key, trans_key in comparisons:
        psi_val = results.get(psi_key, 0)
        trans_val = results.get(trans_key, 0)

        if psi_val > trans_val + 0.02:
            winner = "PSI"
            psi_wins += 1
        elif trans_val > psi_val + 0.02:
            winner = "Transformer"
            trans_wins += 1
        else:
            winner = "Tie"

        print(f"{test_name:<40} {psi_val:>11.1%} {trans_val:>11.1%} {winner:>12}")

    print("-" * 80)
    print(f"{'TOTAL WINS':<40} {psi_wins:>12} {trans_wins:>12}")
    print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Check ambiguous trajectory hypothesis
    psi_ctx2 = results.get('ambig_PSI_ctx2', 0)
    psi_ctx8 = results.get('ambig_PSI_ctx8', 0)
    trans_ctx2 = results.get('ambig_Transformer_ctx2', 0)
    trans_ctx8 = results.get('ambig_Transformer_ctx8', 0)

    print(f"""
HYPOTHESIS TEST: Does PSI learn trajectory dynamics?

1. Ambiguous Context Test:
   - With minimal context (2 tokens): PSI {psi_ctx2:.1%} vs Trans {trans_ctx2:.1%}
   - With full context (8 tokens):    PSI {psi_ctx8:.1%} vs Trans {trans_ctx8:.1%}

   If PSI learns trajectories, it should:
   - Perform well even with minimal context (trajectory momentum)
   - Show smaller gap between minimal and full context

   PSI context gap: {(psi_ctx8 - psi_ctx2) / psi_ctx8 * 100:.1f}%
   Transformer context gap: {(trans_ctx8 - trans_ctx2) / trans_ctx8 * 100:.1f}%

2. Distractor Robustness:
   - PSI degradation: {(results.get('distractor_PSI_clean', 0) - results.get('distractor_PSI_noisy', 0)) / max(results.get('distractor_PSI_clean', 0.01), 0.01) * 100:.1f}%
   - Transformer degradation: {(results.get('distractor_Transformer_clean', 0) - results.get('distractor_Transformer_noisy', 0)) / max(results.get('distractor_Transformer_clean', 0.01), 0.01) * 100:.1f}%

   If PSI learns smooth dynamics, it should be MORE robust to distractors.

3. Compositional Generalization:
   - PSI generalization gap: {(results.get('comp_PSI_simple', 0) - results.get('comp_PSI_composed', 0)) / max(results.get('comp_PSI_simple', 0.01), 0.01) * 100:.1f}%
   - Transformer generalization gap: {(results.get('comp_Transformer_simple', 0) - results.get('comp_Transformer_composed', 0)) / max(results.get('comp_Transformer_simple', 0.01), 0.01) * 100:.1f}%

   If PSI learns continuous dynamics, it should generalize better to compositions.
""")

    # Save results
    plt.figure(figsize=(12, 8))

    # Bar chart comparison
    tests = ['Basic', 'Ambig(2)', 'Ambig(4)', 'Ambig(8)', 'Dist(clean)', 'Dist(noisy)', 'Comp(simple)', 'Comp(composed)']
    psi_vals = [results.get(k, 0) for k in ['basic_traj_PSI', 'ambig_PSI_ctx2', 'ambig_PSI_ctx4',
                                             'ambig_PSI_ctx8', 'distractor_PSI_clean', 'distractor_PSI_noisy',
                                             'comp_PSI_simple', 'comp_PSI_composed']]
    trans_vals = [results.get(k, 0) for k in ['basic_traj_Transformer', 'ambig_Transformer_ctx2', 'ambig_Transformer_ctx4',
                                               'ambig_Transformer_ctx8', 'distractor_Transformer_clean', 'distractor_Transformer_noisy',
                                               'comp_Transformer_simple', 'comp_Transformer_composed']]

    x = np.arange(len(tests))
    width = 0.35

    plt.bar(x - width/2, psi_vals, width, label='PSI', color='steelblue')
    plt.bar(x + width/2, trans_vals, width, label='Transformer', color='coral')

    plt.xlabel('Test')
    plt.ylabel('Accuracy')
    plt.title('Trajectory vs Lookup: PSI vs Transformer')
    plt.xticks(x, tests, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('trajectory_vs_lookup.png', dpi=150)
    plt.close()
    print("\nSaved trajectory_vs_lookup.png")


if __name__ == "__main__":
    main()
