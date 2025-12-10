"""
Hyperparameter search: How do dim, n_sets, and planes_per_set interact?

Key questions:
1. What gives best associative recall scaling?
2. What maintains copy/positional performance?
3. Where is the sweet spot for both?

Theory:
- Address space ~ 2^(n_sets * planes_per_set) for cross-bank binding
- Memory capacity ~ planes_per_set * dim
- More sets = more combinatorial addresses
- More planes = more parallel retrievals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from clifford_memory import OrthogonalModel
from associative_recall_benchmark import (
    generate_copy_task, generate_induction_task,
    train_epoch, evaluate, TransformerBaseline
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def test_config(dim, n_sets, planes_per_set, n_kv_pairs=8, n_epochs=10, verbose=True):
    """Test a single Clifford configuration on both tasks."""

    vocab_size = 64
    n_layers = 4
    seq_len = 128
    n_train = 1500
    n_test = 400

    # Create model
    model = OrthogonalModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_orthogonal_sets=n_sets,
        planes_per_set=planes_per_set,
        pos_planes=planes_per_set  # Match for consistency
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # Theoretical address space
    address_space = 2 ** (n_sets * planes_per_set)

    results = {
        'dim': dim,
        'n_sets': n_sets,
        'planes_per_set': planes_per_set,
        'params': n_params,
        'address_space_log2': n_sets * planes_per_set,
    }

    # === Test Copy Task ===
    copy_train_seq, copy_train_tgt = generate_copy_task(n_train, seq_len, vocab_size)
    copy_test_seq, copy_test_tgt = generate_copy_task(n_test, seq_len, vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(n_epochs):
        train_epoch(model, copy_train_seq, copy_train_tgt, optimizer)

    _, copy_acc = evaluate(model, copy_test_seq, copy_test_tgt)
    results['copy_acc'] = copy_acc

    # === Reset and test Associative Task ===
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    assoc_train_seq, assoc_train_tgt = generate_induction_task(n_train, seq_len, vocab_size, n_pairs=n_kv_pairs)
    assoc_test_seq, assoc_test_tgt = generate_induction_task(n_test, seq_len, vocab_size, n_pairs=n_kv_pairs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(n_epochs):
        train_epoch(model, assoc_train_seq, assoc_train_tgt, optimizer)

    _, assoc_acc = evaluate(model, assoc_test_seq, assoc_test_tgt)
    results['assoc_acc'] = assoc_acc

    if verbose:
        print(f"  dim={dim:3d}, n_sets={n_sets}, planes={planes_per_set:2d} | "
              f"addr_space=2^{n_sets*planes_per_set:2d} | "
              f"params={n_params:6,d} | "
              f"copy={copy_acc:.1%} assoc={assoc_acc:.1%}")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


def run_hyperparameter_search():
    print("=" * 80)
    print("HYPERPARAMETER SEARCH: dim × n_sets × planes_per_set")
    print("=" * 80)
    print()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Search space
    dims = [32, 64, 128]
    n_sets_options = [2, 4, 6, 8]
    planes_options = [4, 8, 16]

    all_results = []

    # Test all combinations
    total = len(dims) * len(n_sets_options) * len(planes_options)
    count = 0

    for dim in dims:
        print(f"\n--- dim={dim} ---")
        for n_sets in n_sets_options:
            for planes in planes_options:
                count += 1
                print(f"[{count}/{total}]", end=" ")

                try:
                    result = test_config(dim, n_sets, planes, n_kv_pairs=8, n_epochs=10)
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")

    # === Analysis ===
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Sort by combined score (copy + assoc)
    all_results.sort(key=lambda r: r['copy_acc'] + r['assoc_acc'], reverse=True)

    print("\nTop 10 configurations (by combined accuracy):")
    print(f"{'dim':>4} {'sets':>4} {'planes':>6} {'addr':>6} {'params':>8} {'copy':>8} {'assoc':>8} {'total':>8}")
    print("-" * 60)

    for r in all_results[:10]:
        total = r['copy_acc'] + r['assoc_acc']
        print(f"{r['dim']:>4} {r['n_sets']:>4} {r['planes_per_set']:>6} "
              f"2^{r['address_space_log2']:>3} {r['params']:>8,} "
              f"{r['copy_acc']:>7.1%} {r['assoc_acc']:>7.1%} {total:>7.1%}")

    # === Analyze trends ===
    print("\n" + "=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)

    # Effect of dim
    print("\nEffect of dim (averaged across other params):")
    for dim in dims:
        subset = [r for r in all_results if r['dim'] == dim]
        avg_copy = np.mean([r['copy_acc'] for r in subset])
        avg_assoc = np.mean([r['assoc_acc'] for r in subset])
        print(f"  dim={dim:3d}: copy={avg_copy:.1%}, assoc={avg_assoc:.1%}")

    # Effect of n_sets
    print("\nEffect of n_sets (averaged across other params):")
    for n_sets in n_sets_options:
        subset = [r for r in all_results if r['n_sets'] == n_sets]
        avg_copy = np.mean([r['copy_acc'] for r in subset])
        avg_assoc = np.mean([r['assoc_acc'] for r in subset])
        print(f"  n_sets={n_sets}: copy={avg_copy:.1%}, assoc={avg_assoc:.1%}")

    # Effect of planes_per_set
    print("\nEffect of planes_per_set (averaged across other params):")
    for planes in planes_options:
        subset = [r for r in all_results if r['planes_per_set'] == planes]
        avg_copy = np.mean([r['copy_acc'] for r in subset])
        avg_assoc = np.mean([r['assoc_acc'] for r in subset])
        print(f"  planes={planes:2d}: copy={avg_copy:.1%}, assoc={avg_assoc:.1%}")

    # Effect of address space size
    print("\nEffect of address space (n_sets × planes_per_set):")
    addr_groups = {}
    for r in all_results:
        addr = r['address_space_log2']
        if addr not in addr_groups:
            addr_groups[addr] = []
        addr_groups[addr].append(r)

    for addr in sorted(addr_groups.keys()):
        subset = addr_groups[addr]
        avg_copy = np.mean([r['copy_acc'] for r in subset])
        avg_assoc = np.mean([r['assoc_acc'] for r in subset])
        print(f"  2^{addr:2d}: copy={avg_copy:.1%}, assoc={avg_assoc:.1%}")

    return all_results


def test_scaling_with_best_config(best_dim, best_n_sets, best_planes):
    """Test the best config against increasing KV pairs."""
    print("\n" + "=" * 80)
    print(f"SCALING TEST: dim={best_dim}, n_sets={best_n_sets}, planes={best_planes}")
    print("=" * 80)

    np.random.seed(42)
    torch.manual_seed(42)

    vocab_size = 64
    kv_counts = [8, 16, 24, 32]

    print(f"\n{'KV pairs':>10} {'Transformer':>15} {'Clifford':>15} {'Winner':>12}")
    print("-" * 55)

    for n_kv in kv_counts:
        seq_len = max(128, n_kv * 4 + 32)
        n_train = 2000
        n_test = 500
        n_epochs = 15

        # Transformer
        tf_model = TransformerBaseline(vocab_size=vocab_size, dim=64, n_layers=4).to(device)
        train_seq, train_tgt = generate_induction_task(n_train, seq_len, vocab_size, n_pairs=n_kv)
        test_seq, test_tgt = generate_induction_task(n_test, seq_len, vocab_size, n_pairs=n_kv)

        optimizer = torch.optim.AdamW(tf_model.parameters(), lr=1e-3)
        for _ in range(n_epochs):
            train_epoch(tf_model, train_seq, train_tgt, optimizer)
        _, tf_acc = evaluate(tf_model, test_seq, test_tgt)

        del tf_model
        torch.cuda.empty_cache() if device == 'cuda' else None

        # Clifford with best config
        cl_model = OrthogonalModel(
            vocab_size=vocab_size,
            dim=best_dim,
            n_layers=4,
            n_orthogonal_sets=best_n_sets,
            planes_per_set=best_planes
        ).to(device)

        optimizer = torch.optim.AdamW(cl_model.parameters(), lr=1e-3)
        for _ in range(n_epochs):
            train_epoch(cl_model, train_seq, train_tgt, optimizer)
        _, cl_acc = evaluate(cl_model, test_seq, test_tgt)

        del cl_model
        torch.cuda.empty_cache() if device == 'cuda' else None

        winner = "Clifford" if cl_acc > tf_acc else "Transformer" if tf_acc > cl_acc else "Tie"
        print(f"{n_kv:>10} {tf_acc:>14.1%} {cl_acc:>14.1%} {winner:>12}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        # Quick scaling test with specific config
        dim = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        n_sets = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        planes = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        test_scaling_with_best_config(dim, n_sets, planes)
    else:
        results = run_hyperparameter_search()

        # Find best config
        best = max(results, key=lambda r: r['copy_acc'] + r['assoc_acc'])
        print(f"\nBest config: dim={best['dim']}, n_sets={best['n_sets']}, planes={best['planes_per_set']}")

        # Also find best for associative
        best_assoc = max(results, key=lambda r: r['assoc_acc'])
        print(f"Best for associative: dim={best_assoc['dim']}, n_sets={best_assoc['n_sets']}, planes={best_assoc['planes_per_set']} ({best_assoc['assoc_acc']:.1%})")

        # Test scaling with best config
        test_scaling_with_best_config(best['dim'], best['n_sets'], best['planes_per_set'])
