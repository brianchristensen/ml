"""Test how HRR scales with data on COGS."""

from model_compositional_cogs import CompositionalCOGSModel
from test_cogs_structural import load_cogs
import time


def test_at_scale(train_limit, test_limit=100):
    """Test model with different amounts of training data."""
    print("="*70)
    print(f"TESTING WITH {train_limit} TRAINING EXAMPLES")
    print("="*70)
    print()

    # Load data
    train_data = load_cogs('train', limit=train_limit)
    test_data = load_cogs('test', limit=test_limit)

    print(f"Dataset: {len(train_data)} train, {len(test_data)} test")
    print()

    # Create model
    model = CompositionalCOGSModel(hrr_dim=2048)

    # Train
    start = time.time()
    model.train_on_dataset(train_data)
    train_time = time.time() - start
    print(f"Training took {train_time:.2f}s")
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

        # Show first 10
        if i < 10:
            match_str = "[OK]" if str(predicted_struct) == str(expected_struct) else "[FAIL]"
            sent_str = ' '.join(sent_tokens[:8])
            print(f"{i+1:2}. {match_str} {sent_str[:50]}")

    print()
    print(f"Exact structure match: {exact_match}/{total} = {100*exact_match/total:.1f}%")
    print(f"Predicate overlap:     {pred_match}/{total} = {100*pred_match/total:.1f}%")
    print(f"Training time:         {train_time:.2f}s")
    print(f"Examples learned:      {len(model.examples)}")
    print(f"Lexicon size:          {len(model.lexicon)}")
    print()
    print("="*70)
    print()

    return {
        'train_size': len(train_data),
        'exact': exact_match / total,
        'overlap': pred_match / total,
        'train_time': train_time,
        'lexicon_size': len(model.lexicon)
    }


if __name__ == '__main__':
    # Test at different scales
    results = []

    scales = [500, 2000, 5000, 10000]

    for scale in scales:
        result = test_at_scale(scale, test_limit=100)
        results.append(result)

    # Summary
    print("\n")
    print("="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    print()
    print(f"{'Train Size':>12} | {'Exact %':>8} | {'Overlap %':>10} | {'Time (s)':>9} | {'Lexicon':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['train_size']:>12,} | {r['exact']*100:>7.1f}% | {r['overlap']*100:>9.1f}% | {r['train_time']:>9.1f} | {r['lexicon_size']:>8,}")
    print()
