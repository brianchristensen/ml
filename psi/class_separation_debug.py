"""
Debug: Why is class separation ratio so low?

The ratio (between-class variance / within-class variance) should be >1 for good separation.
Currently it's ~0.09, meaning all classes map to similar hidden representations.

Let's trace the signal flow to find where discriminative information is lost.
"""

import numpy as np
from mnist_benchmark import PSIClassifier, load_mnist_subset


def analyze_signal_flow():
    """Trace where discriminative information is lost in the pipeline."""

    print("=" * 70)
    print("CLASS SEPARATION DEBUG: Where is discriminative info lost?")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=500, n_test=200)

    # Use a fresh model
    model = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    # Collect representations at each stage for a batch of samples
    n_samples = 200

    # Storage for each stage
    raw_inputs = []           # Original 784-dim images
    projected_inputs = []     # After input projection [n_input, dim]
    hidden_states_free = []   # Hidden states after free phase
    hidden_states_target = [] # Hidden states after target phase
    output_logits = []        # Final classification logits
    labels = []

    print("\n1. Collecting representations at each stage...")

    for i in range(n_samples):
        img = X_test[i]
        label = y_test[i]
        labels.append(label)

        # Stage 1: Raw input
        raw_inputs.append(img)

        # Stage 2: Project input
        projected = model.project_input(img)
        projected_inputs.append(projected.flatten())

        # Stage 3: Free phase (get hidden states)
        model.states = np.random.randn(model.n_nodes, model.dim) * 0.01
        model.clamped = np.full((model.n_nodes, model.dim), np.nan)
        model.clamped[:model.n_input] = projected
        model.settle(max_iters=5, min_steps=2)

        hidden_start = model.n_input
        hidden_end = model.n_input + model.n_hidden
        hidden_free = model.states[hidden_start:hidden_end].flatten()
        hidden_states_free.append(hidden_free)

        # Stage 4: Get output logits
        model.free_states = model.states.copy()
        input_flat = model.states[:model.n_input].flatten()
        combined = np.concatenate([input_flat, hidden_free])
        feat_mean = combined.mean()
        feat_std = combined.std() + 1e-8
        combined_norm = (combined - feat_mean) / feat_std
        logits = combined_norm @ model.hidden_to_output
        output_logits.append(logits)

        # Stage 5: Target phase (for comparison)
        target = np.full((model.n_output, model.dim), -1.0)
        target[label] = 1.0
        model.clamped[-model.n_output:] = target
        model.settle(max_iters=15, min_steps=6, target_mode=True)
        hidden_target = model.states[hidden_start:hidden_end].flatten()
        hidden_states_target.append(hidden_target)

    # Convert to arrays
    raw_inputs = np.array(raw_inputs)
    projected_inputs = np.array(projected_inputs)
    hidden_states_free = np.array(hidden_states_free)
    hidden_states_target = np.array(hidden_states_target)
    output_logits = np.array(output_logits)
    labels = np.array(labels)

    print(f"\n2. Computing class separation at each stage...")
    print("-" * 70)

    def compute_class_separation(features, labels, stage_name):
        """Compute within-class and between-class variance."""
        n_classes = 10

        # Compute class means
        class_means = []
        within_vars = []
        class_counts = []

        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_features = features[mask]
                class_mean = class_features.mean(axis=0)
                class_means.append(class_mean)

                # Within-class variance: average distance to class mean
                within_var = np.mean(np.var(class_features, axis=0))
                within_vars.append(within_var)
                class_counts.append(mask.sum())
            else:
                class_means.append(np.zeros(features.shape[1]))
                within_vars.append(0)
                class_counts.append(0)

        class_means = np.array(class_means)

        # Global mean
        global_mean = features.mean(axis=0)

        # Between-class variance: variance of class means around global mean
        between_var = np.mean(np.var(class_means, axis=0))

        # Average within-class variance
        avg_within_var = np.mean(within_vars)

        # Separation ratio
        ratio = between_var / (avg_within_var + 1e-8)

        print(f"\n{stage_name}:")
        print(f"  Feature dim: {features.shape[1]}")
        print(f"  Feature magnitude: {np.mean(np.abs(features)):.4f}")
        print(f"  Within-class variance: {avg_within_var:.6f}")
        print(f"  Between-class variance: {between_var:.6f}")
        print(f"  Separation ratio: {ratio:.4f} {'(GOOD)' if ratio > 1 else '(POOR)'}")

        return ratio, avg_within_var, between_var

    stages = [
        (raw_inputs, "Stage 1: Raw Input (784-dim)"),
        (projected_inputs, "Stage 2: After Input Projection"),
        (hidden_states_free, "Stage 3: Hidden States (Free Phase)"),
        (hidden_states_target, "Stage 4: Hidden States (Target Phase)"),
        (output_logits, "Stage 5: Output Logits"),
    ]

    ratios = []
    for features, name in stages:
        r, _, _ = compute_class_separation(features, labels, name)
        ratios.append(r)

    print("\n" + "=" * 70)
    print("3. DIAGNOSIS: Where does separation break down?")
    print("=" * 70)

    print(f"\nSeparation ratio progression:")
    for i, ((_, name), r) in enumerate(zip(stages, ratios)):
        arrow = " -> " if i > 0 else "    "
        change = ""
        if i > 0:
            if ratios[i] > ratios[i-1]:
                change = f" (+{(ratios[i]/ratios[i-1]-1)*100:.0f}%)"
            else:
                change = f" ({(ratios[i]/ratios[i-1]-1)*100:.0f}%)"
        print(f"{arrow}{name.split(':')[0]}: {r:.4f}{change}")

    # Additional analysis: Check if input projection preserves class info
    print("\n" + "=" * 70)
    print("4. INPUT PROJECTION ANALYSIS")
    print("=" * 70)

    # Check correlation between raw and projected
    from_raw = compute_class_separation(raw_inputs, labels, "Raw")[0]
    from_proj = compute_class_separation(projected_inputs, labels, "Projected")[0]

    print(f"\nProjection preserves {(from_proj/from_raw)*100:.1f}% of class separation")

    # Check if projection weights are too random/uniform
    print(f"\nInput projection stats:")
    print(f"  Shape: {model.input_proj.shape}")
    print(f"  Mean: {np.mean(model.input_proj):.4f}")
    print(f"  Std: {np.std(model.input_proj):.4f}")
    print(f"  Max: {np.max(np.abs(model.input_proj)):.4f}")

    # Check hidden layer dynamics
    print("\n" + "=" * 70)
    print("5. HIDDEN LAYER DYNAMICS")
    print("=" * 70)

    # How much does hidden state vary between inputs?
    hidden_var_across_samples = np.var(hidden_states_free, axis=0).mean()
    hidden_var_within_sample = np.mean([np.var(h) for h in hidden_states_free])

    print(f"\nHidden state variance:")
    print(f"  Across samples (should be high): {hidden_var_across_samples:.6f}")
    print(f"  Within sample (activation spread): {hidden_var_within_sample:.6f}")

    # Check if hidden states are saturated or dead
    hidden_mean = np.mean(np.abs(hidden_states_free))
    hidden_max = np.max(np.abs(hidden_states_free))
    near_zero = np.mean(np.abs(hidden_states_free) < 0.1)
    near_saturation = np.mean(np.abs(hidden_states_free) > 0.9)

    print(f"\nHidden state health:")
    print(f"  Mean |activation|: {hidden_mean:.4f}")
    print(f"  Max |activation|: {hidden_max:.4f}")
    print(f"  % near zero (<0.1): {near_zero*100:.1f}%")
    print(f"  % saturated (>0.9): {near_saturation*100:.1f}%")

    # Compare free vs target phase hidden states
    print("\n" + "=" * 70)
    print("6. FREE vs TARGET PHASE COMPARISON")
    print("=" * 70)

    state_diff = hidden_states_target - hidden_states_free
    diff_magnitude = np.mean(np.abs(state_diff))
    diff_per_class = []

    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            class_diff = np.mean(np.abs(state_diff[mask]))
            diff_per_class.append(class_diff)

    print(f"\nContrastive signal strength:")
    print(f"  Mean |target - free|: {diff_magnitude:.4f}")
    print(f"  Per-class diff magnitude: {np.mean(diff_per_class):.4f} (std: {np.std(diff_per_class):.4f})")

    if diff_magnitude < 0.1:
        print("  WARNING: Contrastive signal is WEAK - target clamping not propagating!")

    # Check the hidden_to_output weights
    print("\n" + "=" * 70)
    print("7. CLASSIFIER WEIGHTS ANALYSIS")
    print("=" * 70)

    h2o = model.hidden_to_output
    print(f"\nHidden-to-output weights:")
    print(f"  Shape: {h2o.shape}")
    print(f"  Mean: {np.mean(h2o):.4f}")
    print(f"  Std: {np.std(h2o):.4f}")
    print(f"  Mean |weight|: {np.mean(np.abs(h2o)):.4f}")

    # Check if weights discriminate classes
    weight_norms = np.linalg.norm(h2o, axis=0)  # Norm per output class
    print(f"  Per-class weight norms: {weight_norms}")

    # Correlation between class weights
    weight_corr = np.corrcoef(h2o.T)
    off_diag = weight_corr[np.triu_indices(10, k=1)]
    print(f"  Mean weight correlation between classes: {np.mean(off_diag):.4f}")
    if np.mean(off_diag) > 0.5:
        print("  WARNING: Class weights are too similar - not discriminating!")


if __name__ == "__main__":
    analyze_signal_flow()
