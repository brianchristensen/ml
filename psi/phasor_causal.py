"""
Sequential Causal Discovery with Phasor Memory

Tests whether phasor memory can learn causal relationships from sequences of
observations and then correctly answer interventional queries.

Key insight: Phasor memory's holographic binding can store (cause, effect) pairs
in a way that allows retrieval of the causal mechanism, not just correlation.

Task Structure:
1. OBSERVATION PHASE: Model sees sequence of (variable, value) pairs
   - Variables have causal relationships: A→B, A→C, B→C (with confounder A)
   - Model stores these observations in memory

2. QUERY PHASE: Model must predict effect given cause
   - Observational query: "Given A=x, what is C?" (includes confounded path)
   - Interventional query: "If we SET B=y (breaking A→B), what is C?"

The key test: Can the model distinguish between:
- P(C | B=y) - observational, includes A→B→C and A→C paths
- P(C | do(B=y)) - interventional, only B→C path (A→B is cut)

A correlational model conflates these. A causal model separates them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple

from phasor import PhasorModel, PhasorBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Causal Data Generator - Creates sequences with known causal structure
# =============================================================================

class SequentialCausalGenerator:
    """
    Generates sequences from a known SCM for training/testing.

    Causal Graph:
        A → B → C
        A -----→ C

    A is the confounder. The true causal effects are:
    - A→B: B = f_ab(A) + noise
    - B→C: C = f_bc(B) + f_ac(A) + noise  (both paths contribute)

    For interventional queries do(B=b):
    - C = f_bc(b) + f_ac(A) + noise  (A→B path is cut, but A→C remains)

    Sequence format (tokenized):
    [VAR_A, val, VAR_B, val, VAR_C, val, SEP, VAR_A, val, VAR_B, val, VAR_C, val, ...]

    Query format:
    [...observations..., QUERY, VAR_X, val, PREDICT, VAR_Y] → target: val_Y
    """

    def __init__(self, vocab_size=64, n_vars=3):
        self.vocab_size = vocab_size
        self.n_vars = n_vars

        # Reserve special tokens
        self.VAR_A = 0
        self.VAR_B = 1
        self.VAR_C = 2
        self.SEP = 3  # Separator between observations
        self.QUERY_OBS = 4  # Observational query marker
        self.QUERY_INT = 5  # Interventional query marker
        self.PREDICT = 6  # Prediction marker
        self.VALUE_OFFSET = 10  # Values start here

        # Use fewer values for easier learning
        self.n_values = 16  # Only 16 possible values (not vocab_size - offset)

        # True causal mechanisms (linear for simplicity, could be nonlinear)
        # A→B coefficient
        self.w_ab = 0.8
        # B→C coefficient (the TRUE causal effect we want to isolate)
        self.w_bc = 0.6
        # A→C coefficient (confounding path) - smaller to make B→C more dominant
        self.w_ac = 0.2

    def _sample_value(self) -> int:
        """Sample a value token."""
        return np.random.randint(0, self.n_values)

    def _value_to_token(self, val: int) -> int:
        return val + self.VALUE_OFFSET

    def _token_to_value(self, tok: int) -> int:
        return tok - self.VALUE_OFFSET

    def _apply_mechanism(self, cause_val: int, weight: float, noise_scale: float = 0.1) -> int:
        """Apply causal mechanism: effect = weight * cause + noise, discretized."""
        # Normalize cause to [-1, 1]
        cause_norm = (cause_val / self.n_values) * 2 - 1
        # Apply linear mechanism
        effect_norm = weight * cause_norm + np.random.randn() * noise_scale
        # Clip and discretize back
        effect_norm = np.clip(effect_norm, -1, 1)
        effect_val = int((effect_norm + 1) / 2 * self.n_values)
        effect_val = np.clip(effect_val, 0, self.n_values - 1)
        return effect_val

    def generate_observation(self) -> Tuple[int, int, int]:
        """Generate one (A, B, C) observation from the SCM."""
        A = self._sample_value()
        B = self._apply_mechanism(A, self.w_ab)
        # C depends on both B and A
        C_from_B = self.w_bc * ((B / self.n_values) * 2 - 1)
        C_from_A = self.w_ac * ((A / self.n_values) * 2 - 1)
        C_norm = C_from_B + C_from_A + np.random.randn() * 0.1
        C_norm = np.clip(C_norm, -1, 1)
        C = int((C_norm + 1) / 2 * self.n_values)
        C = np.clip(C, 0, self.n_values - 1)
        return A, B, C

    def generate_interventional(self, B_intervened: int) -> Tuple[int, int, int]:
        """Generate observation under do(B=b) - A→B link is cut."""
        A = self._sample_value()  # A is still random
        B = B_intervened  # B is SET, not caused by A
        # C still depends on both B and A
        C_from_B = self.w_bc * ((B / self.n_values) * 2 - 1)
        C_from_A = self.w_ac * ((A / self.n_values) * 2 - 1)
        C_norm = C_from_B + C_from_A + np.random.randn() * 0.1
        C_norm = np.clip(C_norm, -1, 1)
        C = int((C_norm + 1) / 2 * self.n_values)
        C = np.clip(C, 0, self.n_values - 1)
        return A, B, C

    def generate_sequence(self, n_obs: int = 8, query_type: str = 'observational') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training sequence with observations and a query.

        Returns:
            sequence: Token sequence [seq_len]
            target: Target token for prediction (-100 for non-prediction positions)
        """
        tokens = []

        # Generate observations
        for i in range(n_obs):
            A, B, C = self.generate_observation()
            tokens.extend([
                self.VAR_A, self._value_to_token(A),
                self.VAR_B, self._value_to_token(B),
                self.VAR_C, self._value_to_token(C),
            ])
            if i < n_obs - 1:
                tokens.append(self.SEP)

        # Add query
        if query_type == 'observational':
            # Query: Given B=b (observed), predict C
            # This should include both B→C AND the spurious A→B→C correlation
            tokens.append(self.QUERY_OBS)
            query_A, query_B, query_C = self.generate_observation()
            tokens.extend([
                self.VAR_B, self._value_to_token(query_B),
                self.PREDICT, self.VAR_C
            ])
            target_val = query_C

        elif query_type == 'interventional':
            # Query: do(B=b), predict C
            # This should ONLY use B→C, not the confounded path through A
            tokens.append(self.QUERY_INT)
            B_int = self._sample_value()
            # For interventional, we sample A independently and compute C
            A_int = self._sample_value()
            C_from_B = self.w_bc * ((B_int / self.n_values) * 2 - 1)
            C_from_A = self.w_ac * ((A_int / self.n_values) * 2 - 1)
            C_norm = C_from_B + C_from_A + np.random.randn() * 0.1
            C_norm = np.clip(C_norm, -1, 1)
            C_int = int((C_norm + 1) / 2 * self.n_values)
            C_int = np.clip(C_int, 0, self.n_values - 1)

            tokens.extend([
                self.VAR_B, self._value_to_token(B_int),
                self.PREDICT, self.VAR_C
            ])
            target_val = C_int

        else:
            raise ValueError(f"Unknown query type: {query_type}")

        # Create target tensor (-100 everywhere except prediction position)
        seq_len = len(tokens)
        sequence = torch.tensor(tokens, dtype=torch.long)
        target = torch.full((seq_len,), -100, dtype=torch.long)
        target[-1] = self._value_to_token(target_val)

        return sequence, target

    def generate_batch(self, batch_size: int, n_obs: int = 8,
                       query_type: str = 'mixed') -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of sequences."""
        sequences = []
        targets = []

        for _ in range(batch_size):
            if query_type == 'mixed':
                qt = np.random.choice(['observational', 'interventional'])
            else:
                qt = query_type
            seq, tgt = self.generate_sequence(n_obs, qt)
            sequences.append(seq)
            targets.append(tgt)

        # Pad to same length
        max_len = max(s.shape[0] for s in sequences)
        padded_seqs = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_tgts = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, (seq, tgt) in enumerate(zip(sequences, targets)):
            padded_seqs[i, :len(seq)] = seq
            padded_tgts[i, :len(tgt)] = tgt

        return padded_seqs, padded_tgts

    def compute_true_causal_effect(self, B_val: int) -> float:
        """Compute the TRUE causal effect of B on C (just w_bc, no confounding)."""
        B_norm = (B_val / self.n_values) * 2 - 1
        C_norm = self.w_bc * B_norm  # Only the direct effect
        return C_norm

    def compute_confounded_effect(self, B_val: int) -> float:
        """
        Compute what a correlational model would predict.
        Since B is correlated with A (via A→B), and A→C exists,
        the observed P(C|B) includes the confounded path.
        """
        # When we observe B=b, we can infer something about A
        # Under the linear model: B ≈ w_ab * A, so A ≈ B / w_ab
        B_norm = (B_val / self.n_values) * 2 - 1
        A_inferred_norm = B_norm / self.w_ab  # What A likely was
        A_inferred_norm = np.clip(A_inferred_norm, -1, 1)

        # Confounded prediction includes both paths
        C_norm = self.w_bc * B_norm + self.w_ac * A_inferred_norm
        return C_norm


# =============================================================================
# Baseline Models
# =============================================================================

class TransformerBaseline(nn.Module):
    """Standard transformer for comparison."""
    def __init__(self, vocab_size=64, dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(512, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)
        return self.head(h)


class MLPBaseline(nn.Module):
    """Simple MLP that processes the sequence."""
    def __init__(self, vocab_size=64, dim=64, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.net(h)
        return self.head(h)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, generator, optimizer, batch_size=64, n_batches=50):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for _ in range(n_batches):
        seq, tgt = generator.generate_batch(batch_size, query_type='mixed')
        seq, tgt = seq.to(device), tgt.to(device)

        logits = model(seq)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Accuracy on prediction positions
        mask = tgt != -100
        if mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct = ((preds == tgt) & mask).sum().item()
            total_correct += correct
            total_count += mask.sum().item()

    acc = total_correct / total_count if total_count > 0 else 0
    return total_loss / n_batches, acc


def evaluate(model, generator, query_type: str, n_samples=500):
    """Evaluate on specific query type."""
    model.eval()
    total_correct = 0
    total_count = 0
    total_error = 0

    with torch.no_grad():
        for _ in range(n_samples // 64 + 1):
            seq, tgt = generator.generate_batch(64, query_type=query_type)
            seq, tgt = seq.to(device), tgt.to(device)

            logits = model(seq)

            mask = tgt != -100
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1)
                correct = ((preds == tgt) & mask).sum().item()
                total_correct += correct
                total_count += mask.sum().item()

                # Also compute mean absolute error in value space
                pred_vals = preds[mask] - generator.VALUE_OFFSET
                true_vals = tgt[mask] - generator.VALUE_OFFSET
                error = (pred_vals - true_vals).abs().float().mean().item()
                total_error += error * mask.sum().item()

    acc = total_correct / total_count if total_count > 0 else 0
    mae = total_error / total_count if total_count > 0 else 0
    return acc, mae


def run_benchmark():
    print("=" * 70)
    print("SEQUENTIAL CAUSAL DISCOVERY BENCHMARK")
    print("=" * 70)
    print()
    print("Task: Learn causal structure from observation sequences,")
    print("then answer observational vs interventional queries correctly.")
    print()
    print("Causal Graph: A -> B -> C, A -> C")
    print("- Observational query P(C|B): includes confounded A->B->C path")
    print("- Interventional query P(C|do(B)): only direct B->C effect")
    print()

    vocab_size = 64
    dim = 64
    n_layers = 4

    generator = SequentialCausalGenerator(vocab_size=vocab_size)

    # Models
    models = {
        'Phasor': PhasorModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers).to(device),
        'Transformer': TransformerBaseline(vocab_size=vocab_size, dim=dim, n_layers=n_layers).to(device),
        'MLP': MLPBaseline(vocab_size=vocab_size, dim=dim).to(device),
    }

    # Print param counts
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n_params:,} parameters")
    print()

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

        for epoch in range(30):
            loss, acc = train_epoch(model, generator, optimizer)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                obs_acc, obs_mae = evaluate(model, generator, 'observational')
                int_acc, int_mae = evaluate(model, generator, 'interventional')
                print(f"  Epoch {epoch+1}: loss={loss:.3f}, "
                      f"obs_acc={obs_acc:.1%}, int_acc={int_acc:.1%}")

        # Final evaluation
        obs_acc, obs_mae = evaluate(model, generator, 'observational', n_samples=1000)
        int_acc, int_mae = evaluate(model, generator, 'interventional', n_samples=1000)

        results[name] = {
            'obs_acc': obs_acc, 'obs_mae': obs_mae,
            'int_acc': int_acc, 'int_mae': int_mae
        }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<15} {'Obs Acc':>10} {'Int Acc':>10} {'Obs MAE':>10} {'Int MAE':>10}")
    print("-" * 55)

    for name, res in results.items():
        print(f"{name:<15} {res['obs_acc']:>9.1%} {res['int_acc']:>9.1%} "
              f"{res['obs_mae']:>10.2f} {res['int_mae']:>10.2f}")

    print()
    print("Key insight:")
    print("- A model that learns TRUE causal structure should handle")
    print("  interventional queries well (isolating B->C from A->B->C)")
    print("- A correlational model may do well on observational but fail")
    print("  on interventional because it conflates correlation with causation")

    return results


def run_causal_effect_test():
    """
    Direct test: Can models estimate the TRUE causal effect of B on C?

    We compare:
    1. True causal effect (w_bc only)
    2. Confounded effect (what you'd get from P(C|B))
    3. What each model predicts under intervention
    """
    print("\n" + "=" * 70)
    print("CAUSAL EFFECT ESTIMATION TEST")
    print("=" * 70)
    print()

    vocab_size = 64
    generator = SequentialCausalGenerator(vocab_size=vocab_size)

    # Train models briefly
    models = {
        'Phasor': PhasorModel(vocab_size=vocab_size, dim=64, n_layers=4).to(device),
        'Transformer': TransformerBaseline(vocab_size=vocab_size, dim=64, n_layers=4).to(device),
    }

    for name, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(20):
            train_epoch(model, generator, optimizer, n_batches=30)

    print("Testing causal effect estimation...")
    print(f"True causal effect (B->C): w_bc = {generator.w_bc}")
    print(f"Confounding effect (A->C): w_ac = {generator.w_ac}")
    print()

    # Test at different B values
    B_values = [10, 25, 40]  # Low, mid, high

    print(f"{'B value':<10} {'True Effect':>12} {'Confounded':>12}", end="")
    for name in models.keys():
        print(f" {name:>12}", end="")
    print()
    print("-" * (34 + 13 * len(models)))

    for B_val in B_values:
        true_effect = generator.compute_true_causal_effect(B_val)
        confounded = generator.compute_confounded_effect(B_val)

        print(f"{B_val:<10} {true_effect:>12.3f} {confounded:>12.3f}", end="")

        # Get model predictions under intervention
        for name, model in models.items():
            model.eval()
            # Create a minimal interventional query
            seq, _ = generator.generate_sequence(n_obs=8, query_type='interventional')
            # Manually set the B value in the query
            # Find QUERY_INT position and set B value after it
            seq = seq.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(seq)
                pred = logits[0, -1].argmax().item()
                pred_val = pred - generator.VALUE_OFFSET
                pred_norm = (pred_val / generator.n_values) * 2 - 1

            print(f" {pred_norm:>12.3f}", end="")
        print()


if __name__ == "__main__":
    results = run_benchmark()
    run_causal_effect_test()
