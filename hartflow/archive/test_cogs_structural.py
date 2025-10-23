"""
Test structural HRR model on COGS dataset.

Goal: Prove HRR can learn compositional predicate-logic structures.
"""

from model_structural_hrr import StructuralHRRModel
import time


def load_cogs(split='train', limit=None):
    """Load COGS dataset."""
    data_dir = 'data/cogs_repo/data'
    filepath = f'{data_dir}/{split}.tsv'

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            sentence = parts[0].strip()
            logical_form = parts[1].strip()

            # Tokenize sentence
            tokens = sentence.lower().replace('.', '').split()

            data.append((tokens, logical_form))

            if limit and len(data) >= limit:
                break

    return data


def structure_to_string(structure, indent=0):
    """Convert structure back to readable string for debugging."""
    if structure is None:
        return "None"

    ind = "  " * indent
    struct_type = structure[0]

    if struct_type == 'atom':
        _, pred, arg = structure
        return f"{ind}{pred}({arg})"

    elif struct_type == 'role':
        _, pred, role, arg1, arg2 = structure
        return f"{ind}{pred}.{role}({arg1}, {arg2})"

    elif struct_type == 'and':
        _, left, right = structure
        left_str = structure_to_string(left, indent+1)
        right_str = structure_to_string(right, indent+1)
        return f"{ind}AND(\n{left_str}\n{right_str}\n{ind})"

    elif struct_type == 'seq':
        _, left, right = structure
        left_str = structure_to_string(left, indent+1)
        right_str = structure_to_string(right, indent+1)
        return f"{ind}SEQ(\n{left_str}\n{right_str}\n{ind})"

    return str(structure)


def test_parsing():
    """Test if we can parse COGS logical forms."""
    print("="*70)
    print("TESTING COGS LOGICAL FORM PARSING")
    print("="*70)
    print()

    model = StructuralHRRModel()

    # Test cases from COGS
    test_cases = [
        "dog ( x _ 1 )",
        "love . agent ( x _ 1 , Emma )",
        "dog ( x _ 1 ) AND cat ( x _ 2 )",
        "* dog ( x _ 1 ) ; love . agent ( x _ 2 , x _ 1 )",
        "rose ( x _ 1 ) AND help . theme ( x _ 3 , x _ 1 ) AND help . agent ( x _ 3 , x _ 6 ) AND dog ( x _ 6 )",
    ]

    for i, test in enumerate(test_cases):
        print(f"Test {i+1}: {test[:60]}")
        try:
            structure = model.parse_cogs_output(test)
            print(f"  Parsed: {structure}")
            print(f"  Readable:")
            print(structure_to_string(structure, indent=2))
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()


def test_retrieval():
    """Test retrieval-based structural matching."""
    print("="*70)
    print("TESTING STRUCTURAL HRR RETRIEVAL")
    print("="*70)
    print()

    # Load small subset
    train_data = load_cogs('train', limit=100)
    test_data = load_cogs('test', limit=20)

    print(f"Loaded {len(train_data)} train, {len(test_data)} test examples")
    print()

    # Create model
    model = StructuralHRRModel(hrr_dim=2048)

    # Train
    start = time.time()
    model.train_on_dataset(train_data)
    print(f"Training took {time.time()-start:.2f}s")
    print()

    # Test retrieval
    print("Testing retrieval on first 10 examples:")
    print("-"*70)

    correct_structure = 0
    correct_tokens = 0

    for i, (sent_tokens, expected_str) in enumerate(test_data[:10]):
        expected_struct = model.parse_cogs_output(expected_str)
        predicted_struct = model.forward(sent_tokens)

        # Check if structures match
        match = (predicted_struct == expected_struct)
        if match:
            correct_structure += 1

        # Also check token-level match (more lenient)
        if predicted_struct is not None and expected_struct is not None:
            pred_str = str(predicted_struct)
            exp_str = str(expected_struct)
            if pred_str == exp_str:
                correct_tokens += 1

        status = "[OK]" if match else "[FAIL]"
        sent_str = ' '.join(sent_tokens[:8])
        print(f"{i+1:2}. {status} {sent_str[:40]}")
        if not match and i < 3:
            print(f"     Expected: {expected_str[:60]}")
            if predicted_struct:
                print(f"     Got:      {str(predicted_struct)[:60]}")

    print()
    print(f"Exact structure match: {correct_structure}/10")
    print(f"Token-level match: {correct_tokens}/10")
    print()
    print("="*70)


if __name__ == '__main__':
    # First test parsing
    test_parsing()

    print("\n" * 2)

    # Then test retrieval
    test_retrieval()
