"""
Test: Are h2o weights stable after fixing double-decay?
Also track class separation during training.
"""

import numpy as np
from mnist_benchmark import PSIClassifier, load_mnist_subset


def test_weight_stability():
    print("=" * 70)
    print("WEIGHT STABILITY TEST: After removing double-decay")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=500, n_test=200)

    model = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    # Track initial weights
    initial_h2o = np.mean(np.abs(model.hidden_to_output))
    initial_proj = np.mean(np.abs(model.input_proj))
    initial_node_W = np.mean(np.abs(model.node_W))

    print(f"\nInitial weights:")
    print(f"  h2o: {initial_h2o:.4f}")
    print(f"  input_proj: {initial_proj:.4f}")
    print(f"  node_W: {initial_node_W:.4f}")

    # Train for a few epochs
    for epoch in range(5):
        correct = 0
        for i in range(len(X_train)):
            model.train_step(X_train[i], y_train[i])

        # Test
        for i in range(len(X_test)):
            pred = model.predict(X_test[i])
            if pred == y_test[i]:
                correct += 1

        acc = correct / len(X_test) * 100

        # Track weights
        h2o_ratio = np.mean(np.abs(model.hidden_to_output)) / initial_h2o
        proj_ratio = np.mean(np.abs(model.input_proj)) / initial_proj
        node_W_ratio = np.mean(np.abs(model.node_W)) / initial_node_W

        print(f"\nEpoch {epoch+1}: {acc:.1f}% acc")
        print(f"  h2o: {h2o_ratio:.3f}x  proj: {proj_ratio:.3f}x  node_W: {node_W_ratio:.3f}x")
        print(f"  ho_importance mean: {np.mean(model.ho_importance):.4f}")

        # Also check class separation
        hidden_reps = []
        labels = []
        hidden_start = model.n_input
        hidden_end = model.n_input + model.n_hidden

        for i in range(min(100, len(X_test))):
            model.free_phase(X_test[i])
            hidden = model.states[hidden_start:hidden_end].flatten()
            hidden_reps.append(hidden)
            labels.append(y_test[i])

        hidden_reps = np.array(hidden_reps)
        labels = np.array(labels)

        # Compute separation
        class_means = []
        within_vars = []
        for c in range(10):
            mask = labels == c
            if mask.sum() > 0:
                class_features = hidden_reps[mask]
                class_mean = class_features.mean(axis=0)
                class_means.append(class_mean)
                within_var = np.mean(np.var(class_features, axis=0))
                within_vars.append(within_var)

        if class_means:
            class_means = np.array(class_means)
            between_var = np.mean(np.var(class_means, axis=0))
            avg_within = np.mean(within_vars)
            ratio = between_var / (avg_within + 1e-8)
            print(f"  Class separation ratio: {ratio:.4f}")


def analyze_why_separation_fails():
    """Deeper analysis: Why doesn't learning transfer target-phase separation to free phase?"""
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS: Why doesn't target-phase separation transfer?")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=200, n_test=100)

    model = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    hidden_start = model.n_input
    hidden_end = model.n_input + model.n_hidden

    # Before training: analyze what drives free vs target phase separation
    print("\n1. BEFORE TRAINING - Analyzing one sample per class:")

    free_hidden_by_class = {}
    target_hidden_by_class = {}

    for c in range(10):
        # Find a sample of this class
        idx = np.where(y_train == c)[0][0]
        img = X_train[idx]

        # Free phase
        model.states = np.random.randn(model.n_nodes, model.dim) * 0.01
        model.clamped = np.full((model.n_nodes, model.dim), np.nan)
        projected = model.project_input(img)
        model.clamped[:model.n_input] = projected
        model.settle(max_iters=5, min_steps=2)
        free_hidden = model.states[hidden_start:hidden_end].flatten()
        free_hidden_by_class[c] = free_hidden

        # Target phase (from same state)
        target = np.full((model.n_output, model.dim), -1.0)
        target[c] = 1.0
        model.clamped[-model.n_output:] = target
        model.settle(max_iters=15, min_steps=6, target_mode=True)
        target_hidden = model.states[hidden_start:hidden_end].flatten()
        target_hidden_by_class[c] = target_hidden

    # Compute pairwise distances
    print("\n  Free phase - pairwise cosine similarity between classes:")
    free_sims = []
    for i in range(10):
        for j in range(i+1, 10):
            sim = np.dot(free_hidden_by_class[i], free_hidden_by_class[j]) / (
                np.linalg.norm(free_hidden_by_class[i]) * np.linalg.norm(free_hidden_by_class[j]) + 1e-8)
            free_sims.append(sim)
    print(f"    Mean: {np.mean(free_sims):.4f} (should be low for good separation)")

    print("\n  Target phase - pairwise cosine similarity between classes:")
    target_sims = []
    for i in range(10):
        for j in range(i+1, 10):
            sim = np.dot(target_hidden_by_class[i], target_hidden_by_class[j]) / (
                np.linalg.norm(target_hidden_by_class[i]) * np.linalg.norm(target_hidden_by_class[j]) + 1e-8)
            target_sims.append(sim)
    print(f"    Mean: {np.mean(target_sims):.4f}")

    # The delta: what learning signal would be generated
    print("\n  Contrastive delta (target - free) magnitude per class:")
    for c in range(10):
        delta = target_hidden_by_class[c] - free_hidden_by_class[c]
        print(f"    Class {c}: |delta| = {np.mean(np.abs(delta)):.4f}")

    # Key insight: The contrastive delta tells free phase how to change
    # But if free phases are too similar, learning just amplifies uniform directions
    print("\n2. THE PROBLEM:")
    print("   - Free phase hidden states are determined by input + random settling")
    print("   - Different inputs all settle to similar hidden states (high similarity)")
    print("   - Target phase forces class-specific patterns via output clamping")
    print("   - But learning signal (target - free) doesn't make FREE phase class-specific")
    print("   - Because the contrastive delta is applied to weights that affect both phases!")

    # What would help?
    print("\n3. POTENTIAL SOLUTIONS:")
    print("   a) Stronger input -> hidden connections (input_proj and node_W)")
    print("   b) More diverse input projections per class")
    print("   c) Input-dependent hidden biases")
    print("   d) Lateral inhibition to force winner-take-all in hidden layer")


if __name__ == "__main__":
    test_weight_stability()
    analyze_why_separation_fails()
