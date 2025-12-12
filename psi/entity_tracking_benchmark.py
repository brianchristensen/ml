"""
Entity Tracking Benchmark

Tests the ability to maintain entity references across distance.
This is what TPI lacks despite good BPC - coherent discourse requires
remembering "John is a doctor" and later saying "the doctor examined..."

Tasks:
1. Entity-Attribute Binding: Learn entity→attribute mappings, query later
2. Entity Coreference: Track which entity is being referred to across context
3. Multi-hop: Entity A relates to B, B has attribute X, query A's indirect attribute

These tasks require EXACT retrieval, not fuzzy clustering.
Success here should transfer to coherent language generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Task 1: Entity-Attribute Binding with Distractor Context
# =============================================================================

def generate_entity_attribute_task(batch_size, seq_len, vocab_size, n_entities=4, context_noise=20):
    """
    Format: [E1 has A1] [E2 has A2] ... [noise tokens] ... [E3 has ?] -> A3

    Tests: Can you remember that Entity 3 was bound to Attribute 3,
           even after seeing unrelated context?

    vocab layout:
    - 0: padding/delimiter
    - 1-15: entity tokens (E1, E2, ...)
    - 16-31: attribute tokens (A1, A2, ...)
    - 32+: noise/context tokens
    """
    entity_start = 1
    attr_start = 16
    noise_start = 32
    has_token = vocab_size - 2  # "has" marker
    query_token = vocab_size - 1  # "?" marker

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        pos = 0

        # Generate entity-attribute pairs
        entities = torch.randperm(15)[:n_entities] + entity_start
        attributes = torch.randperm(15)[:n_entities] + attr_start

        # Store bindings: [E1 has A1] [E2 has A2] ...
        for i in range(n_entities):
            if pos + 3 < seq_len:
                sequences[b, pos] = entities[i]
                sequences[b, pos + 1] = has_token
                sequences[b, pos + 2] = attributes[i]
                pos += 3

        # Add delimiter
        sequences[b, pos] = 0
        pos += 1

        # Add distractor context (noise)
        noise_len = min(context_noise, seq_len - pos - 4)
        if noise_len > 0:
            noise = torch.randint(noise_start, vocab_size - 2, (noise_len,))
            sequences[b, pos:pos + noise_len] = noise
            pos += noise_len

        # Add delimiter
        sequences[b, pos] = 0
        pos += 1

        # Query: [E_i has ?] -> should predict A_i
        query_order = torch.randperm(n_entities)
        for idx in query_order:
            if pos + 3 < seq_len:
                sequences[b, pos] = entities[idx]
                sequences[b, pos + 1] = has_token
                sequences[b, pos + 2] = query_token
                # Target: predict the correct attribute
                targets[b, pos + 2] = attributes[idx]
                pos += 3

    return sequences, targets


# =============================================================================
# Task 2: Entity Coreference Resolution
# =============================================================================

def generate_coreference_task(batch_size, seq_len, vocab_size, n_entities=3):
    """
    Format: [E1 is type T1] [E2 is type T2] ... [context] ... [the T1 did X] -> E1

    Tests: Can you resolve "the doctor" back to the entity that was a doctor?

    vocab layout:
    - 0: padding
    - 1-10: entity names (John, Mary, ...)
    - 11-20: types (doctor, teacher, ...)
    - 21-30: actions (examined, taught, ...)
    - 31: "is" marker
    - 32: "the" marker
    - 33: "did" marker
    """
    name_start = 1
    type_start = 11
    action_start = 21
    is_token = 31
    the_token = 32
    did_token = 33

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        pos = 0

        # Generate entity-type pairs
        names = torch.randperm(10)[:n_entities] + name_start
        types = torch.randperm(10)[:n_entities] + type_start

        # Introduce entities: [John is doctor] [Mary is teacher] ...
        for i in range(n_entities):
            if pos + 3 < seq_len:
                sequences[b, pos] = names[i]
                sequences[b, pos + 1] = is_token
                sequences[b, pos + 2] = types[i]
                pos += 3

        # Delimiter
        sequences[b, pos] = 0
        pos += 1

        # Some filler
        filler_len = min(10, seq_len - pos - 10)
        pos += filler_len

        # Coreference queries: [the doctor did] -> John
        query_order = torch.randperm(n_entities)
        for idx in query_order:
            if pos + 4 < seq_len:
                action = torch.randint(action_start, action_start + 10, (1,)).item()
                sequences[b, pos] = the_token
                sequences[b, pos + 1] = types[idx]  # "the doctor"
                sequences[b, pos + 2] = did_token
                # Target: predict which entity this refers to
                targets[b, pos + 2] = names[idx]
                pos += 3

    return sequences, targets


# =============================================================================
# Task 3: Multi-hop Entity Reasoning
# =============================================================================

def generate_multihop_task(batch_size, seq_len, vocab_size, n_chains=3, chain_len=2, noise_between=10):
    """
    Format: Multiple entity chains with distractors, then queries.

    [A1 friend-of B1] [B1 lives-in L1]
    [A2 friend-of B2] [B2 lives-in L2]
    [noise...]
    [A1 lives-in ?] -> L1
    [A2 lives-in ?] -> L2

    Tests: Two-hop reasoning through entity relationships with interference.

    Made harder by:
    - Multiple overlapping chains (can't just memorize one)
    - Noise tokens between definition and query
    - Shuffled query order
    """
    entity_start = 1
    location_start = 30
    friend_token = vocab_size - 4
    lives_token = vocab_size - 3
    query_token = vocab_size - 2
    noise_start = 40

    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        pos = 0

        # Generate multiple chains
        all_entities = torch.randperm(25)[:n_chains * 2] + entity_start
        locations = torch.randperm(10)[:n_chains] + location_start

        chains = []  # Store (A, B, location) for each chain

        for i in range(n_chains):
            A = all_entities[i * 2].item()
            B = all_entities[i * 2 + 1].item()
            loc = locations[i].item()
            chains.append((A, B, loc))

            # [A friend-of B]
            if pos + 3 < seq_len:
                sequences[b, pos:pos+3] = torch.tensor([A, friend_token, B])
                pos += 3

            # [B lives-in location]
            if pos + 3 < seq_len:
                sequences[b, pos:pos+3] = torch.tensor([B, lives_token, loc])
                pos += 3

        # Delimiter
        if pos < seq_len:
            sequences[b, pos] = 0
            pos += 1

        # Add noise
        noise_len = min(noise_between, seq_len - pos - n_chains * 4)
        if noise_len > 0:
            noise = torch.randint(noise_start, vocab_size - 4, (noise_len,))
            sequences[b, pos:pos + noise_len] = noise
            pos += noise_len

        # Delimiter
        if pos < seq_len:
            sequences[b, pos] = 0
            pos += 1

        # Queries in shuffled order
        query_order = torch.randperm(n_chains)
        for idx in query_order:
            A, B, loc = chains[idx]
            if pos + 3 < seq_len:
                sequences[b, pos] = A
                sequences[b, pos + 1] = lives_token
                sequences[b, pos + 2] = query_token
                targets[b, pos + 2] = loc
                pos += 3

    return sequences, targets


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_and_eval(model, task_fn, task_name, n_epochs=20, batch_size=128):
    """Train model on task and return best validation accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    vocab_size = 64
    best_val = 0

    print(f"\n{task_name}:")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0

        for _ in range(50):  # 50 batches per epoch
            seq, tgt = task_fn(batch_size, 128, vocab_size)
            seq, tgt = seq.to(device), tgt.to(device)

            logits = model(seq)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                seq, tgt = task_fn(batch_size, 128, vocab_size)
                seq, tgt = seq.to(device), tgt.to(device)

                logits = model(seq)
                mask = tgt != -100

                if mask.sum() > 0:
                    preds = logits.argmax(dim=-1)
                    correct += ((preds == tgt) & mask).sum().item()
                    total += mask.sum().item()

        val_acc = correct / total if total > 0 else 0
        best_val = max(best_val, val_acc)

        if epoch % 5 == 0 or val_acc > 0.9:
            print(f"  Epoch {epoch:2d}: loss={total_loss/50:.3f}, val_acc={val_acc:.1%}")

        if val_acc >= 0.99:
            print(f"  Converged at epoch {epoch}!")
            break

    return best_val


def run_benchmark(models_dict):
    """Run all entity tracking tasks on provided models."""

    print("=" * 70)
    print("ENTITY TRACKING BENCHMARK")
    print("=" * 70)
    print("Tests ability to maintain entity references across context.")
    print("Success here -> coherent discourse in real language.")
    print()

    tasks = {
        'Entity-Attribute': lambda bs, sl, vs: generate_entity_attribute_task(bs, sl, vs, n_entities=4, context_noise=30),
        'Coreference': lambda bs, sl, vs: generate_coreference_task(bs, sl, vs, n_entities=3),
        'Multi-hop': generate_multihop_task,
    }

    results = {}

    for model_name, model_fn in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print('='*60)

        results[model_name] = {}

        for task_name, task_fn in tasks.items():
            # Fresh model for each task
            model = model_fn().to(device)
            n_params = sum(p.numel() for p in model.parameters())

            if task_name == 'Entity-Attribute':
                print(f"  Parameters: {n_params:,}")

            acc = train_and_eval(model, task_fn, task_name)
            results[model_name][task_name] = acc

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Header
    task_names = list(tasks.keys())
    header = f"{'Model':<20}"
    for tn in task_names:
        header += f"{tn:>15}"
    header += f"{'Average':>12}"
    print(header)
    print("-" * len(header))

    # Results
    for model_name in results:
        row = f"{model_name:<20}"
        accs = []
        for tn in task_names:
            acc = results[model_name][tn]
            accs.append(acc)
            row += f"{acc:>14.1%}"
        avg = sum(accs) / len(accs)
        row += f"{avg:>11.1%}"
        print(row)

    return results


if __name__ == "__main__":
    from phasor import PhasorModel
    from tpi import NovelAttentionLM

    print(f"Device: {device}")

    # Define model constructors (same vocab, similar params)
    vocab_size = 64

    models = {
        'Phasor': lambda: PhasorModel(
            vocab_size=vocab_size,
            dim=64,
            n_layers=4,
            n_phases=64,
            value_dim=8
        ),
        'TPI': lambda: NovelAttentionLM(
            vocab_size=vocab_size,
            dim=64,
            num_layers=4,
            num_heads=4,
            max_len=256
        ),
    }

    # Optionally add transformer baseline
    try:
        from associative_recall_benchmark import TransformerBaseline
        models['Transformer'] = lambda: TransformerBaseline(
            vocab_size=vocab_size,
            dim=64,
            n_layers=4,
            n_heads=4
        )
    except ImportError:
        pass

    results = run_benchmark(models)
