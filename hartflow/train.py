"""
Training script for HRR + Learned Execution on SCAN dataset.

Downloads and trains on the official SCAN benchmark for compositional generalization.
"""

import torch
import random
import json
import os
from model_general import GeneralCompositionalModel


def load_scan_dataset(split='simple'):
    """
    Load SCAN dataset.

    Args:
        split: Which SCAN split to use ('simple', 'length', 'add_prim_jump', etc.)

    Returns:
        train_data, test_data: Lists of (command_tokens, action_sequence) tuples
    """
    # Try to load from local file first
    data_dir = 'data/scan'
    train_file = f'{data_dir}/tasks_train_{split}.txt'
    test_file = f'{data_dir}/tasks_test_{split}.txt'

    if not os.path.exists(train_file):
        print(f"SCAN dataset not found at {train_file}")
        print("Please download SCAN dataset from: https://github.com/brendenlake/SCAN")
        print("Or place it in: data/scan/")
        print()
        print("Using synthetic dataset instead...")
        return None, None

    def parse_scan_file(filepath):
        """Parse SCAN format: IN: command OUT: action1 action2 ..."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('IN:'):
                    continue

                parts = line.split('OUT:')
                if len(parts) != 2:
                    continue

                command = parts[0].replace('IN:', '').strip()
                actions = parts[1].strip()

                # Parse command into tokens
                cmd_tokens = command.split()

                # Parse actions into list
                action_list = actions.split()

                data.append((cmd_tokens, action_list))

        return data

    train_data = parse_scan_file(train_file)
    test_data = parse_scan_file(test_file)

    return train_data, test_data


def train_on_scan(split='simple', num_epochs=200, batch_size=32):
    """
    Train model on SCAN dataset.

    Args:
        split: Which SCAN split ('simple', 'length', 'add_prim_jump', etc.)
        num_epochs: Number of training epochs
        batch_size: Batch size (currently processes one at a time)
    """
    print("="*70)
    print(f"TRAINING ON SCAN DATASET (split: {split})")
    print("="*70)
    print()

    # Load dataset
    train_data, test_data = load_scan_dataset(split)

    if train_data is None:
        # Fallback to synthetic data
        print("Using synthetic SCAN-like dataset...")
        from model import generate_scan_dataset
        full_dataset = generate_scan_dataset(num_examples=1000)
        split_idx = int(len(full_dataset) * 0.8)
        train_data = full_dataset[:split_idx]
        test_data = full_dataset[split_idx:]

    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print()

    # Subsample training data for faster iteration (use 25%)
    original_size = len(train_data)
    train_data = train_data[:len(train_data)//4]
    print(f"Subsampled training data: {len(train_data)}/{original_size} examples (25%)")

    # Analyze dataset composition
    atomic_count = sum(1 for tokens, _ in train_data
                       if 'and' not in tokens and 'after' not in tokens)
    print(f"  Atomic commands: {atomic_count}/{len(train_data)} ({100*atomic_count/len(train_data):.1f}%)")
    print(f"  Compound commands: {len(train_data)-atomic_count}/{len(train_data)} ({100*(len(train_data)-atomic_count)/len(train_data):.1f}%)")
    print()

    # Analyze vocabulary from training data
    all_tokens = set()
    for cmd_tokens, action_seq in train_data:
        all_tokens.update(cmd_tokens)

    # Categorize tokens (heuristic based on SCAN structure)
    actions_set = {'jump', 'walk', 'run', 'look', 'turn'}
    modifiers_set = {'twice', 'thrice', 'around', 'opposite', 'after', 'and'}
    directions_set = {'left', 'right'}

    # Add any unknown tokens to modifiers (safest assumption)
    for token in all_tokens:
        if token not in actions_set and token not in modifiers_set and token not in directions_set:
            modifiers_set.add(token)

    # Define vocabulary
    primitives = {
        'actions': list(actions_set),
        'modifiers': list(modifiers_set),
        'directions': list(directions_set),
    }

    # Get unique output actions
    output_actions_set = set()
    for _, action_seq in train_data:
        for action in action_seq:
            output_actions_set.add(action)

    outputs = sorted(list(output_actions_set))

    print(f"Vocabulary: {len(primitives['actions'])} actions, {len(primitives['modifiers'])} modifiers")
    print(f"Output actions: {outputs}")
    print()

    # Create model
    model = GeneralCompositionalModel(
        primitives_dict=primitives,
        output_vocab=outputs,
        hrr_dim=1024,
        hidden_dim=256
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with gradient accumulation
    print("TRAINING:")
    print("-"*70)
    print(f"Using batch size {batch_size} for gradient accumulation")
    print()

    best_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        random.shuffle(train_data)

        optimizer.zero_grad()
        batch_loss = 0
        batch_count = 0

        for i, (cmd_tokens, target_actions) in enumerate(train_data):
            try:
                # Compute loss
                loss = model.compute_loss(cmd_tokens, target_actions)

                if loss.item() > 0:  # Only backprop if there's actual loss
                    loss.backward()
                    batch_loss += loss.item()
                    batch_count += 1

                # Update weights every batch_size examples
                if (i + 1) % batch_size == 0 and batch_count > 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += batch_loss
                    num_batches += 1
                    batch_loss = 0
                    batch_count = 0

            except Exception as e:
                # Skip examples that cause errors
                print(f"Error on example: {e}")
                continue

        # Final update for remaining examples in incomplete batch
        if batch_count > 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            model.eval()

            # Track accuracy by command length
            results_by_length = {}

            with torch.no_grad():
                for cmd_tokens, expected in test_data:
                    cmd_len = len(cmd_tokens)
                    if cmd_len not in results_by_length:
                        results_by_length[cmd_len] = {'correct': 0, 'total': 0}

                    try:
                        predicted, _ = model(cmd_tokens)
                        if predicted == expected:
                            results_by_length[cmd_len]['correct'] += 1
                        results_by_length[cmd_len]['total'] += 1
                    except:
                        results_by_length[cmd_len]['total'] += 1

            # Overall accuracy
            total_correct = sum(r['correct'] for r in results_by_length.values())
            total_examples = sum(r['total'] for r in results_by_length.values())
            overall_acc = 100 * total_correct / total_examples if total_examples > 0 else 0

            # Print breakdown
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Overall Acc = {overall_acc:.1f}% ({total_correct}/{total_examples})")

            # Show accuracy by length
            for length in sorted(results_by_length.keys())[:5]:  # Show first 5 lengths
                r = results_by_length[length]
                acc = 100 * r['correct'] / r['total'] if r['total'] > 0 else 0
                print(f"           Len {length}: {acc:.1f}% ({r['correct']}/{r['total']})")

            # Show sample predictions for debugging
            if epoch == 0:
                print("\n           Sample predictions:")
                sample_count = 0
                for cmd_tokens, expected in test_data[:100]:
                    if sample_count >= 5:
                        break
                    if len(cmd_tokens) <= 3:  # Focus on simple ones first
                        try:
                            predicted, decomposed = model(cmd_tokens)
                            cmd_str = ' '.join(cmd_tokens)
                            print(f"           '{cmd_str}' -> decomposed: {decomposed}")
                            print(f"             Expected: {expected[:4]}, Got: {predicted[:4]}")
                            sample_count += 1
                        except Exception as e:
                            pass

            test_acc = overall_acc
            correct = total_correct
            total = total_examples

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Save best model
                torch.save(model.state_dict(), 'best_model.pt')

            # Early stopping at 95% accuracy
            if test_acc >= 95.0:
                print(f"\nEarly stopping: Reached {test_acc:.1f}% accuracy!")
                break
        else:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")

    print()
    print("="*70)
    print(f"BEST TEST ACCURACY: {best_test_acc:.1f}%")
    print("="*70)
    print()

    # Final evaluation on test set
    print("FINAL EVALUATION:")
    print("-"*70)

    model.eval()
    correct = 0
    total = 0
    errors = []

    with torch.no_grad():
        for cmd_tokens, expected in test_data[:50]:  # Show first 50
            try:
                predicted, decomposed = model(cmd_tokens)
                match = "[OK]" if predicted == expected else "[FAIL]"

                if predicted == expected:
                    correct += 1
                else:
                    errors.append((cmd_tokens, expected, predicted))

                total += 1

                cmd_str = ' '.join(cmd_tokens)
                if total <= 20:  # Only print first 20
                    print(f"{cmd_str:30s} -> {str(predicted[:8]):50s} {match}")

            except Exception as e:
                total += 1
                errors.append((cmd_tokens, expected, str(e)))

    print()
    print(f"Final Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print()

    # Show some errors
    if errors and len(errors) > 0:
        print("SAMPLE ERRORS:")
        print("-"*70)
        for cmd_tokens, expected, predicted in errors[:10]:
            cmd_str = ' '.join(cmd_tokens)
            print(f"{cmd_str:30s}")
            print(f"  Expected:  {expected[:8]}")
            print(f"  Predicted: {str(predicted)[:60]}")
            print()


if __name__ == '__main__':
    # Train on SCAN
    # Available splits: 'simple', 'length', 'add_prim_jump', 'add_prim_turn_left', etc.
    train_on_scan(split='simple', num_epochs=50, batch_size=32)
