"""Test memory-augmented compositional model with learned operations."""

from model_learned_ops import OperationLearningModel
import os


def load_scan(split='simple'):
    """Load SCAN dataset."""
    data_dir = 'data/scan'
    train_file = f'{data_dir}/tasks_train_{split}.txt'
    test_file = f'{data_dir}/tasks_test_{split}.txt'

    def parse_file(filepath):
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

                cmd_tokens = command.split()
                action_list = actions.split()

                data.append((cmd_tokens, action_list))

        return data

    return parse_file(train_file), parse_file(test_file)


def test_memory_model():
    """Test memory-augmented model with learned operations."""
    print("="*70)
    print("MEMORY-AUGMENTED MODEL WITH LEARNED OPERATIONS")
    print("="*70)
    print()

    train_data, test_data = load_scan('simple')
    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print()

    primitives = {
        'actions': ['jump', 'walk', 'run', 'look', 'turn'],
        'modifiers': ['twice', 'thrice', 'around', 'opposite', 'after', 'and'],
        'directions': ['left', 'right'],
    }

    outputs = ['I_JUMP', 'I_WALK', 'I_RUN', 'I_LOOK', 'I_TURN_LEFT', 'I_TURN_RIGHT']

    # Create model
    model = OperationLearningModel(
        primitives_dict=primitives,
        output_vocab=outputs,
        hrr_dim=2048
    )

    # Store training examples
    model.train_on_dataset(train_data)
    print()

    # Test
    print("TESTING:")
    print("-"*70)

    correct = 0
    total = 0

    # Track by length
    results_by_length = {}

    for cmd_tokens, expected in test_data:
        cmd_len = len(cmd_tokens)
        if cmd_len not in results_by_length:
            results_by_length[cmd_len] = {'correct': 0, 'total': 0}

        try:
            predicted, _ = model.forward(cmd_tokens)

            if predicted == expected:
                correct += 1
                results_by_length[cmd_len]['correct'] += 1

            results_by_length[cmd_len]['total'] += 1
            total += 1

            # Show first 20
            if total <= 20:
                match = "[OK]" if predicted == expected else "[FAIL]"
                cmd_str = ' '.join(cmd_tokens)
                print(f"{cmd_str[:40]:42s} -> {str(predicted[:4]):40s} {match}")

        except Exception as e:
            total += 1
            results_by_length[cmd_len]['total'] += 1

    print()
    print(f"Overall Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print()

    print("Accuracy by command length:")
    for length in sorted(results_by_length.keys())[:10]:
        r = results_by_length[length]
        acc = 100 * r['correct'] / r['total'] if r['total'] > 0 else 0
        print(f"  Length {length}: {acc:.1f}% ({r['correct']}/{r['total']})")

    print()
    print("="*70)


if __name__ == '__main__':
    test_memory_model()
