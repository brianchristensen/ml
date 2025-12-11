"""
Language Learning Diagnostics for PHI and Phasor

Analyzes what these models are actually learning (or failing to learn)
on small language tasks, with detailed probing of internal representations.

Key questions:
1. Are the models learning position-independent token statistics?
2. Are they learning local n-gram patterns?
3. Are they learning longer-range dependencies?
4. What do the memory/phase representations look like?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

from phi import PHI, ParallelHolographicIntegrator
from phasor import PhasorModel, PhasorBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# =============================================================================
# Simple Language Dataset
# =============================================================================

class TinyLanguageDataset:
    """
    Small synthetic language with clear structure for diagnostics.

    Patterns:
    1. Simple bigrams: "ab", "cd", "ef" (local patterns)
    2. Trigrams with gap: "a_a" -> next is 'b' (short-range)
    3. Bracket matching: "(x)" or "[y]" (medium-range)
    4. Repetition: "xyz...xyz" (long-range copy)
    """

    def __init__(self):
        # Simple vocab for diagnostics
        self.chars = list("abcdefghijklmnopqrstuvwxyz ()[]0123456789")
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Define patterns
        self.patterns = [
            # Bigram completions
            ("ab", "Pattern: ab"),
            ("cd", "Pattern: cd"),
            ("ef", "Pattern: ef"),
            # Trigram with context
            ("the", "Pattern: the"),
            ("and", "Pattern: and"),
            # Bracket matching
            ("(a)", "Pattern: brackets"),
            ("[b]", "Pattern: brackets"),
            # Repetition
            ("abc abc", "Pattern: repeat"),
            ("xyz xyz", "Pattern: repeat"),
        ]

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices):
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)

    def generate_sample(self, length=64):
        """Generate a sample with mixed patterns."""
        text = ""
        while len(text) < length:
            pattern_type = np.random.choice(['bigram', 'word', 'bracket', 'repeat', 'random'])

            if pattern_type == 'bigram':
                text += np.random.choice(['ab', 'cd', 'ef', 'gh']) + " "
            elif pattern_type == 'word':
                text += np.random.choice(['the', 'and', 'for', 'not']) + " "
            elif pattern_type == 'bracket':
                inner = np.random.choice(list('abcxyz'))
                bracket = np.random.choice(['()', '[]'])
                text += bracket[0] + inner + bracket[1] + " "
            elif pattern_type == 'repeat':
                chunk = ''.join(np.random.choice(list('abc'), 3))
                text += chunk + " " + chunk + " "
            else:
                text += ''.join(np.random.choice(list('abcdefgh'), 3)) + " "

        return text[:length]

    def generate_batch(self, batch_size, length=64):
        texts = [self.generate_sample(length) for _ in range(batch_size)]
        encoded = [self.encode(t) for t in texts]
        return torch.tensor(encoded, dtype=torch.long)


# =============================================================================
# Diagnostic Hooks
# =============================================================================

class DiagnosticPHI(PHI):
    """PHI with hooks to capture internal states."""

    def forward(self, x, return_diagnostics=False):
        omega = self.to_omega(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        phi_init = self.to_phase_init(x)
        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled
        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x
        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        output = self.to_out(context)

        if return_diagnostics:
            return x + output, {
                'phi': phi,
                'magnitude': magnitude,
                'gate': gate,
                'memory_real': memory_real_normalized,
                'memory_imag': memory_imag_normalized,
                'retrieved_real': retrieved_real,
                'retrieved_imag': retrieved_imag,
                'query_offset': query_offset,
            }
        return x + output


class DiagnosticPhasorBlock(PhasorBlock):
    """PhasorBlock with hooks to capture internal states."""

    def forward(self, x, return_diagnostics=False):
        B, L, D = x.shape
        P = self.n_phases

        pos_phases = self.pos_phases[:L]
        pos_cos = torch.cos(pos_phases)
        pos_sin = torch.sin(pos_phases)

        v1 = self.mem1_value(x)
        magnitude = torch.sigmoid(self.to_magnitude(x)) * self.magnitude_scale.abs()
        weighted_v1 = magnitude * v1

        mem1_cos = torch.cumsum(pos_cos.unsqueeze(0) * weighted_v1, dim=1)
        mem1_sin = torch.cumsum(pos_sin.unsqueeze(0) * weighted_v1, dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)

        mem1_cos = mem1_cos / sqrt_magnitude
        mem1_sin = mem1_sin / sqrt_magnitude

        offset_input = torch.cat([x, pos_phases.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        offset = torch.tanh(self.offset_predictor(offset_input)) * 3.14159

        offset_cos = torch.cos(offset)
        offset_sin = torch.sin(offset)
        query_cos = pos_cos.unsqueeze(0) * offset_cos - pos_sin.unsqueeze(0) * offset_sin
        query_sin = pos_sin.unsqueeze(0) * offset_cos + pos_cos.unsqueeze(0) * offset_sin

        pos_ret = (mem1_cos * query_cos + mem1_sin * query_sin) / (P ** 0.5)
        pos_out = self.mem1_out(pos_ret)

        # Memory 2
        key_phases = torch.tanh(self.key_encoder(x)) * 3.14159
        key_cos = torch.cos(key_phases)
        key_sin = torch.sin(key_phases)

        values = self.value_encoder(x)
        store_gate = self.store_gate(x)

        context = torch.cumsum(x, dim=1)
        positions = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        context_avg = context / positions

        storage_phases = torch.tanh(self.storage_key(torch.cat([x, context_avg], dim=-1))) * 3.14159
        store_cos = torch.cos(storage_phases)
        store_sin = torch.sin(storage_phases)

        kv_bound_cos = store_cos.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)
        kv_bound_sin = store_sin.unsqueeze(-1) * values.unsqueeze(2) * store_gate.unsqueeze(-1)

        kv_mem_cos = torch.cumsum(kv_bound_cos, dim=1)
        kv_mem_sin = torch.cumsum(kv_bound_sin, dim=1)

        gate_cumsum = torch.cumsum(store_gate, dim=1).clamp(min=1)

        query_cos_kv = key_cos.unsqueeze(-1)
        query_sin_kv = key_sin.unsqueeze(-1)

        kv_retrieved = (kv_mem_cos * query_cos_kv + kv_mem_sin * query_sin_kv).sum(dim=2)
        kv_retrieved = kv_retrieved / (torch.sqrt(gate_cumsum) * (P ** 0.5))

        trajectory_real = x * key_cos.mean(dim=-1, keepdim=True)
        trajectory_imag = x * key_sin.mean(dim=-1, keepdim=True)

        combined = torch.cat([pos_out, kv_retrieved, trajectory_real, trajectory_imag], dim=-1)
        output = self.to_out(combined)

        if return_diagnostics:
            return x + output, {
                'magnitude': magnitude,
                'offset': offset,
                'pos_memory_cos': mem1_cos,
                'pos_memory_sin': mem1_sin,
                'pos_retrieved': pos_out,
                'key_phases': key_phases,
                'storage_phases': storage_phases,
                'store_gate': store_gate,
                'kv_retrieved': kv_retrieved,
            }
        return x + output


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(model, dataset, n_epochs=50, batch_size=32, seq_len=64):
    """Train model and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            x = dataset.generate_batch(batch_size, seq_len).to(device)

            # Predict next token
            logits = model(x[:, :-1])
            targets = x[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            bpc = avg_loss / np.log(2)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.3f}, BPC={bpc:.3f}")

    return loss_history


def evaluate_patterns(model, dataset):
    """Evaluate model on specific patterns."""
    model.eval()

    test_sequences = [
        # Bigram completion
        ("a", "b", "After 'a', predict 'b'"),
        ("c", "d", "After 'c', predict 'd'"),
        ("e", "f", "After 'e', predict 'f'"),
        # Word completion
        ("th", "e", "After 'th', predict 'e'"),
        ("an", "d", "After 'an', predict 'd'"),
        # Bracket matching
        ("(a", ")", "After '(a', predict ')'"),
        ("[b", "]", "After '[b', predict ']'"),
        # Repetition
        ("abc ab", "c", "After 'abc ab', predict 'c'"),
    ]

    results = []

    with torch.no_grad():
        for context, expected, description in test_sequences:
            encoded = dataset.encode(context)
            x = torch.tensor([encoded], dtype=torch.long, device=device)

            logits = model(x)
            probs = F.softmax(logits[0, -1], dim=-1)

            expected_idx = dataset.char_to_idx.get(expected, 0)
            expected_prob = probs[expected_idx].item()

            top5_probs, top5_idx = torch.topk(probs, 5)
            top5_chars = [dataset.idx_to_char[i.item()] for i in top5_idx]

            results.append({
                'description': description,
                'context': context,
                'expected': expected,
                'expected_prob': expected_prob,
                'top5': list(zip(top5_chars, top5_probs.tolist())),
                'rank': (probs.argsort(descending=True) == expected_idx).nonzero().item() + 1
            })

    return results


def analyze_representations(model, dataset, model_name):
    """Analyze internal representations."""
    model.eval()

    # Generate a sample
    text = "ab cd the (x) abc abc "
    encoded = dataset.encode(text)
    x = torch.tensor([encoded], dtype=torch.long, device=device)

    # Get embeddings
    if hasattr(model, 'embed'):
        h = model.embed(x)
    else:
        h = model.token_embedding(x)

    # Get first block's diagnostics
    if model_name == 'PHI':
        block = model.blocks[0].integration
        if hasattr(block, 'forward'):
            # Wrap in diagnostic version
            diag_block = DiagnosticPHI(block.dim).to(device)
            diag_block.load_state_dict(block.state_dict())
            with torch.no_grad():
                _, diagnostics = diag_block(h, return_diagnostics=True)
    else:  # Phasor
        block = model.blocks[0]
        diag_block = DiagnosticPhasorBlock(block.dim, max_seq_len=128).to(device)
        diag_block.load_state_dict(block.state_dict())
        with torch.no_grad():
            _, diagnostics = diag_block(model.norms[0](h), return_diagnostics=True)

    return text, diagnostics


def plot_diagnostics(text, diagnostics, model_name, save_path):
    """Plot diagnostic visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{model_name} Internal Representations', fontsize=14)

    seq_len = len(text)

    # Magnitude over sequence
    ax = axes[0, 0]
    mag = diagnostics['magnitude'][0].cpu().numpy()
    ax.imshow(mag.T[:32, :], aspect='auto', cmap='viridis')
    ax.set_title('Magnitude (first 32 dims)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')

    # Add character labels
    ax.set_xticks(range(0, seq_len, 4))
    ax.set_xticklabels([text[i] if i < len(text) else '' for i in range(0, seq_len, 4)], fontsize=8)

    if model_name == 'PHI':
        # Phase evolution
        ax = axes[0, 1]
        phi = diagnostics['phi'][0].cpu().numpy()
        ax.imshow(phi.T[:32, :], aspect='auto', cmap='twilight')
        ax.set_title('Phase (first 32 dims)')
        ax.set_xlabel('Position')

        # Gate values
        ax = axes[0, 2]
        gate = diagnostics['gate'][0].cpu().numpy()
        ax.imshow(gate.T[:32, :], aspect='auto', cmap='RdYlGn')
        ax.set_title('Gate (first 32 dims)')
        ax.set_xlabel('Position')

        # Memory content
        ax = axes[1, 0]
        mem_real = diagnostics['memory_real'][0].cpu().numpy()
        ax.imshow(mem_real.T[:32, :], aspect='auto', cmap='RdBu')
        ax.set_title('Memory Real (first 32 dims)')
        ax.set_xlabel('Position')

        # Retrieved content
        ax = axes[1, 1]
        ret_real = diagnostics['retrieved_real'][0].cpu().numpy()
        ax.imshow(ret_real.T[:32, :], aspect='auto', cmap='RdBu')
        ax.set_title('Retrieved Real (first 32 dims)')
        ax.set_xlabel('Position')

        # Query offset
        ax = axes[1, 2]
        offset = diagnostics['query_offset'][0].cpu().numpy()
        ax.imshow(offset.T[:32, :], aspect='auto', cmap='coolwarm')
        ax.set_title('Query Offset (first 32 dims)')
        ax.set_xlabel('Position')

    else:  # Phasor
        # Positional memory
        ax = axes[0, 1]
        pos_mem = diagnostics['pos_memory_cos'][0].cpu().numpy()
        ax.imshow(pos_mem.T[:32, :], aspect='auto', cmap='RdBu')
        ax.set_title('Pos Memory Cos (first 32 dims)')
        ax.set_xlabel('Position')

        # Offset learned
        ax = axes[0, 2]
        offset = diagnostics['offset'][0].cpu().numpy()
        ax.imshow(offset.T[:32, :], aspect='auto', cmap='coolwarm')
        ax.set_title('Learned Offset (first 32 dims)')
        ax.set_xlabel('Position')

        # KV retrieved
        ax = axes[1, 0]
        kv_ret = diagnostics['kv_retrieved'][0].cpu().numpy()
        ax.imshow(kv_ret.T[:32, :], aspect='auto', cmap='RdBu')
        ax.set_title('KV Retrieved (first 32 dims)')
        ax.set_xlabel('Position')

        # Store gate
        ax = axes[1, 1]
        store_gate = diagnostics['store_gate'][0, :, 0].cpu().numpy()
        ax.bar(range(len(store_gate)), store_gate)
        ax.set_title('Store Gate')
        ax.set_xlabel('Position')
        ax.set_xticks(range(0, seq_len, 4))
        ax.set_xticklabels([text[i] if i < len(text) else '' for i in range(0, seq_len, 4)], fontsize=8)

        # Key phases (average)
        ax = axes[1, 2]
        key_phases = diagnostics['key_phases'][0].cpu().numpy()
        ax.imshow(key_phases.T[:32, :], aspect='auto', cmap='twilight')
        ax.set_title('Key Phases (first 32 dims)')
        ax.set_xlabel('Position')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved diagnostics to {save_path}")


def compare_ngram_predictions(model, dataset, model_name):
    """Compare model predictions to simple n-gram baselines."""
    model.eval()

    # Generate test data
    test_texts = [dataset.generate_sample(64) for _ in range(100)]

    # Build bigram counts from training-like data
    bigram_counts = defaultdict(lambda: defaultdict(int))
    for _ in range(500):
        text = dataset.generate_sample(64)
        for i in range(len(text) - 1):
            bigram_counts[text[i]][text[i+1]] += 1

    # Normalize to probabilities
    bigram_probs = {}
    for c1, counts in bigram_counts.items():
        total = sum(counts.values())
        bigram_probs[c1] = {c2: count/total for c2, count in counts.items()}

    # Compare predictions
    model_correct = 0
    bigram_correct = 0
    model_better = 0
    bigram_better = 0
    total = 0

    with torch.no_grad():
        for text in test_texts:
            encoded = dataset.encode(text)
            x = torch.tensor([encoded], dtype=torch.long, device=device)

            logits = model(x[:, :-1])
            preds = logits.argmax(dim=-1)[0]

            for i in range(len(text) - 1):
                true_next = text[i + 1]

                # Model prediction
                model_pred = dataset.idx_to_char.get(preds[i].item(), '?')

                # Bigram prediction
                prev_char = text[i]
                if prev_char in bigram_probs:
                    bigram_pred = max(bigram_probs[prev_char], key=bigram_probs[prev_char].get)
                else:
                    bigram_pred = ' '  # Default to space

                model_correct += (model_pred == true_next)
                bigram_correct += (bigram_pred == true_next)

                if model_pred == true_next and bigram_pred != true_next:
                    model_better += 1
                elif bigram_pred == true_next and model_pred != true_next:
                    bigram_better += 1

                total += 1

    return {
        'model_acc': model_correct / total,
        'bigram_acc': bigram_correct / total,
        'model_wins': model_better / total,
        'bigram_wins': bigram_better / total,
    }


# =============================================================================
# Main
# =============================================================================

def run_diagnostics():
    print("=" * 70)
    print("LANGUAGE LEARNING DIAGNOSTICS")
    print("=" * 70)
    print()

    dataset = TinyLanguageDataset()
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Sample: {dataset.generate_sample(50)}")
    print()

    # Create models
    dim = 64
    n_layers = 4

    phi_model = ParallelHolographicIntegrator(
        vocab_size=dataset.vocab_size,
        dim=dim,
        num_layers=n_layers,
        max_len=128,
        device=device
    ).to(device)

    phasor_model = PhasorModel(
        vocab_size=dataset.vocab_size,
        dim=dim,
        n_layers=n_layers,
        max_seq_len=128
    ).to(device)

    phi_params = sum(p.numel() for p in phi_model.parameters())
    phasor_params = sum(p.numel() for p in phasor_model.parameters())
    print(f"PHI parameters: {phi_params:,}")
    print(f"Phasor parameters: {phasor_params:,}")
    print()

    # Train both models
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print("\nTraining PHI...")
    phi_history = train_model(phi_model, dataset, n_epochs=50)

    print("\nTraining Phasor...")
    phasor_history = train_model(phasor_model, dataset, n_epochs=50)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(phi_history, label='PHI')
    plt.plot(phasor_history, label='Phasor')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([l/np.log(2) for l in phi_history], label='PHI')
    plt.plot([l/np.log(2) for l in phasor_history], label='Phasor')
    plt.xlabel('Epoch')
    plt.ylabel('BPC')
    plt.title('Bits Per Character')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    print("\nSaved training curves to training_curves.png")

    # Evaluate patterns
    print("\n" + "=" * 70)
    print("PATTERN EVALUATION")
    print("=" * 70)

    for model, name in [(phi_model, 'PHI'), (phasor_model, 'Phasor')]:
        print(f"\n{name}:")
        results = evaluate_patterns(model, dataset)
        for r in results:
            top3 = ', '.join([f"'{c}':{p:.2f}" for c, p in r['top5'][:3]])
            status = "OK" if r['rank'] <= 3 else "MISS"
            print(f"  [{status}] {r['description']}")
            print(f"       Expected '{r['expected']}' (prob={r['expected_prob']:.3f}, rank={r['rank']})")
            print(f"       Top 3: {top3}")

    # N-gram comparison
    print("\n" + "=" * 70)
    print("N-GRAM COMPARISON")
    print("=" * 70)

    for model, name in [(phi_model, 'PHI'), (phasor_model, 'Phasor')]:
        ngram_results = compare_ngram_predictions(model, dataset, name)
        print(f"\n{name}:")
        print(f"  Model accuracy:  {ngram_results['model_acc']:.1%}")
        print(f"  Bigram accuracy: {ngram_results['bigram_acc']:.1%}")
        print(f"  Model wins (model right, bigram wrong): {ngram_results['model_wins']:.1%}")
        print(f"  Bigram wins (bigram right, model wrong): {ngram_results['bigram_wins']:.1%}")

        if ngram_results['model_acc'] > ngram_results['bigram_acc']:
            print(f"  -> Model is learning BEYOND simple bigrams!")
        else:
            print(f"  -> Model is NOT exceeding bigram baseline")

    # Analyze representations
    print("\n" + "=" * 70)
    print("REPRESENTATION ANALYSIS")
    print("=" * 70)

    for model, name in [(phi_model, 'PHI'), (phasor_model, 'Phasor')]:
        print(f"\nAnalyzing {name}...")
        text, diagnostics = analyze_representations(model, dataset, name)
        plot_diagnostics(text, diagnostics, name, f'{name.lower()}_diagnostics.png')

    # Generation samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES")
    print("=" * 70)

    for model, name in [(phi_model, 'PHI'), (phasor_model, 'Phasor')]:
        model.eval()
        print(f"\n{name} generations:")

        seeds = ["ab ", "the ", "(a"]
        for seed in seeds:
            generated = list(dataset.encode(seed))

            with torch.no_grad():
                for _ in range(30):
                    x = torch.tensor([generated], dtype=torch.long, device=device)
                    logits = model(x)
                    probs = F.softmax(logits[0, -1] / 0.8, dim=-1)  # temperature=0.8
                    next_idx = torch.multinomial(probs, 1).item()
                    generated.append(next_idx)

            generated_text = dataset.decode(generated)
            print(f"  '{seed}' -> '{generated_text}'")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFinal BPC - PHI: {phi_history[-1]/np.log(2):.3f}, Phasor: {phasor_history[-1]/np.log(2):.3f}")
    print("\nKey observations to check:")
    print("1. Are models learning beyond bigram statistics?")
    print("2. Do they handle bracket matching (medium-range dependency)?")
    print("3. Do they handle repetition (long-range copy)?")
    print("4. What do the internal representations look like?")


if __name__ == "__main__":
    run_diagnostics()
