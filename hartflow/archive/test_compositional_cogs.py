"""Test compositional COGS model."""

from model_compositional_cogs import CompositionalCOGSModel
from test_cogs_structural import load_cogs
import time


def test_compositional():
    print("="*70)
    print("TESTING COMPOSITIONAL COGS MODEL")
    print("="*70)
    print()

    # Load data
    train_data = load_cogs('train', limit=500)
    test_data = load_cogs('test', limit=50)

    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print()

    # Create model
    model = CompositionalCOGSModel(hrr_dim=2048)

    # Train
    start = time.time()
    model.train_on_dataset(train_data)
    print(f"Training took {time.time()-start:.2f}s")
    print()

    # Test
    print("TESTING:")
    print("-"*70)

    exact_match = 0
    pred_match = 0  # Matches at least one predicate
    total = 0

    for i, (sent_tokens, expected_str) in enumerate(test_data):
        expected_struct = model.parse_cogs_output(expected_str)
        predicted_struct = model.forward(sent_tokens)

        # Exact structure match
        if str(predicted_struct) == str(expected_struct):
            exact_match += 1

        # Check if any predicates match
        if predicted_struct and expected_struct:
            pred_preds = model._extract_predicates(predicted_struct)
            exp_preds = model._extract_predicates(expected_struct)
            if pred_preds & exp_preds:  # Any overlap
                pred_match += 1

        total += 1

        # Show first 20
        if i < 20:
            match_str = "[OK]" if str(predicted_struct) == str(expected_struct) else "[FAIL]"
            sent_str = ' '.join(sent_tokens[:8])
            print(f"{i+1:2}. {match_str} {sent_str[:50]}")

            if i < 5 and str(predicted_struct) != str(expected_struct):
                print(f"     Expected predicates: {model._extract_predicates(expected_struct)}")
                if predicted_struct:
                    print(f"     Got predicates:      {model._extract_predicates(predicted_struct)}")

    print()
    print(f"Exact structure match: {exact_match}/{total} = {100*exact_match/total:.1f}%")
    print(f"Predicate overlap:     {pred_match}/{total} = {100*pred_match/total:.1f}%")
    print()
    print("="*70)


if __name__ == '__main__':
    test_compositional()
