"""
Component-level diagnostics to understand WHY PSI doesn't scale.
Measures each piece of the model separately.
"""
import numpy as np
from mnist_benchmark import load_mnist_subset, PSIClassifier


def measure_phase_coherence(phases, mask):
    """Compute mean pairwise phase coherence within a group."""
    group_phases = phases[mask]
    if len(group_phases) < 2:
        return 0.0
    n = len(group_phases)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            diff = group_phases[i] - group_phases[j]
            total += np.mean(np.cos(diff))
            count += 1
    return total / count if count > 0 else 0.0


def measure_cross_coherence(phases, mask1, mask2):
    """Phase coherence between two groups."""
    g1 = phases[mask1]
    g2 = phases[mask2]
    total = 0.0
    count = 0
    for p1 in g1:
        for p2 in g2:
            total += np.mean(np.cos(p1 - p2))
            count += 1
    return total / count if count > 0 else 0.0


def run_diagnostics(n_train=500, n_hidden=24, dim=24, n_epochs=10):
    """Run full diagnostics with given parameters."""
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=n_train, n_test=200)

    print(f"=== DIAGNOSTICS: n_train={n_train}, n_hidden={n_hidden}, dim={dim} ===")
    print()

    psi = PSIClassifier(n_input=28, n_hidden=n_hidden, n_output=10, dim=dim, seed=42)

    # Masks for different node groups
    input_mask = np.zeros(psi.n_nodes, dtype=bool)
    input_mask[:psi.n_input] = True
    hidden_mask = np.zeros(psi.n_nodes, dtype=bool)
    hidden_mask[psi.n_input:psi.n_input + psi.n_hidden] = True
    output_mask = np.zeros(psi.n_nodes, dtype=bool)
    output_mask[psi.n_input + psi.n_hidden:] = True

    metrics = []

    for epoch in range(n_epochs):
        epoch_td = []
        epoch_reward = []
        epoch_ho_update = []
        epoch_nodeW_update = []
        epoch_hidden_features = []
        epoch_output_logits = []
        epoch_settle_steps = []

        idx = np.random.permutation(len(X_train))

        for i in idx:
            old_ho = psi.hidden_to_output.copy()
            old_nodeW = psi.node_W.copy()

            image = X_train[i]
            label = y_train[i]

            # Free phase - track settling
            psi.states = np.random.randn(psi.n_nodes, psi.dim) * 0.01
            psi.clamped = np.full((psi.n_nodes, psi.dim), np.nan)
            psi.phases = np.random.uniform(0, 2*np.pi, (psi.n_nodes, psi.dim))

            input_states = psi.project_input(image)
            psi.clamped[:psi.n_input] = input_states
            psi.states[:psi.n_input] = input_states

            # Track convergence
            for step in range(12):
                psi.step(target_mode=False)
                if step >= 3 and psi.has_converged():
                    epoch_settle_steps.append(step + 1)
                    break
            else:
                epoch_settle_steps.append(12)

            free_states = psi.states.copy()

            # Get features
            hidden_start = psi.n_input
            hidden_end = psi.n_input + psi.n_hidden
            input_flat = free_states[:psi.n_input].flatten()
            hidden_flat = free_states[hidden_start:hidden_end].flatten()
            combined = np.concatenate([input_flat, hidden_flat])

            logits = combined @ psi.hidden_to_output
            epoch_output_logits.append(np.var(logits))
            epoch_hidden_features.append(np.var(hidden_flat))

            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            reward = probs[label]
            td_error = reward - psi.expected_reward
            epoch_td.append(td_error)
            epoch_reward.append(reward)

            # Train
            psi.train_step(X_train[i], y_train[i])

            ho_update = np.mean(np.abs(psi.hidden_to_output - old_ho))
            nodeW_update = np.mean(np.abs(psi.node_W - old_nodeW))
            epoch_ho_update.append(ho_update)
            epoch_nodeW_update.append(nodeW_update)

        # Phase coherence on test samples
        coherence_samples = 20
        input_coh, hidden_coh, output_coh, cross_coh = [], [], [], []
        for i in range(coherence_samples):
            psi.free_phase(X_test[i])
            input_coh.append(measure_phase_coherence(psi.phases, input_mask))
            hidden_coh.append(measure_phase_coherence(psi.phases, hidden_mask))
            output_coh.append(measure_phase_coherence(psi.phases, output_mask))
            cross_coh.append(measure_cross_coherence(psi.phases, input_mask, hidden_mask))

        # Accuracy
        correct = sum(1 for i in range(len(X_test)) if psi.predict(X_test[i]) == y_test[i])
        acc = correct / len(X_test)

        m = {
            'epoch': epoch,
            'accuracy': acc,
            'avg_reward': np.mean(epoch_reward),
            'avg_td_error': np.mean(epoch_td),
            'expected_reward': psi.expected_reward,
            'phase_coh_input': np.mean(input_coh),
            'phase_coh_hidden': np.mean(hidden_coh),
            'phase_coh_output': np.mean(output_coh),
            'phase_coh_cross': np.mean(cross_coh),
            'input_proj_norm': np.mean(np.abs(psi.input_proj)),
            'node_W_norm': np.mean(np.abs(psi.node_W)),
            'ho_norm': np.mean(np.abs(psi.hidden_to_output)),
            'W_conn_mean': np.mean(psi.W_conn[psi.adj > 0]),
            'ho_update': np.mean(epoch_ho_update),
            'nodeW_update': np.mean(epoch_nodeW_update),
            'hidden_var': np.mean(epoch_hidden_features),
            'logit_var': np.mean(epoch_output_logits),
            'importance_mean': np.mean(psi.ho_importance),
            'importance_max': np.max(psi.ho_importance),
            'settle_steps': np.mean(epoch_settle_steps),
        }
        metrics.append(m)

        print(f"Epoch {epoch+1}: acc={acc:.1%}, reward={m['avg_reward']:.3f}, settle={m['settle_steps']:.1f}")

    return metrics


def compare_scaling():
    """Compare metrics across different scales."""
    print("=" * 70)
    print("SCALING COMPARISON: Why does PSI get worse with more capacity?")
    print("=" * 70)
    print()

    configs = [
        {'n_train': 500, 'n_hidden': 24, 'dim': 24, 'label': 'baseline'},
        {'n_train': 2000, 'n_hidden': 24, 'dim': 24, 'label': 'more_data'},
        {'n_train': 500, 'n_hidden': 48, 'dim': 24, 'label': 'more_hidden'},
        {'n_train': 500, 'n_hidden': 24, 'dim': 48, 'label': 'higher_dim'},
    ]

    all_metrics = {}
    for cfg in configs:
        label = cfg.pop('label')
        print()
        metrics = run_diagnostics(n_epochs=5, **cfg)
        all_metrics[label] = metrics[-1]  # Final epoch

    print()
    print("=" * 70)
    print("FINAL EPOCH COMPARISON")
    print("=" * 70)

    keys_to_compare = [
        ('accuracy', 'Accuracy'),
        ('avg_reward', 'Avg Reward'),
        ('phase_coh_hidden', 'Hidden Coherence'),
        ('phase_coh_cross', 'Cross Coherence'),
        ('hidden_var', 'Hidden Variance'),
        ('logit_var', 'Logit Variance'),
        ('ho_update', 'H->O Update'),
        ('nodeW_update', 'NodeW Update'),
        ('settle_steps', 'Settle Steps'),
        ('importance_mean', 'Importance Mean'),
    ]

    print()
    print(f"{'Metric':<20} | {'baseline':>10} | {'more_data':>10} | {'more_hidden':>10} | {'higher_dim':>10}")
    print("-" * 75)

    for key, name in keys_to_compare:
        vals = [all_metrics[label][key] for label in ['baseline', 'more_data', 'more_hidden', 'higher_dim']]
        if key == 'accuracy':
            print(f"{name:<20} | {vals[0]:>9.1%} | {vals[1]:>9.1%} | {vals[2]:>9.1%} | {vals[3]:>9.1%}")
        else:
            print(f"{name:<20} | {vals[0]:>10.4f} | {vals[1]:>10.4f} | {vals[2]:>10.4f} | {vals[3]:>10.4f}")


def analyze_gradient_flow():
    """Check if gradients are flowing properly through the network."""
    print()
    print("=" * 70)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=500, n_test=200)

    psi = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    # Train for one epoch and track per-class statistics
    class_rewards = {i: [] for i in range(10)}
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for i in range(len(X_train)):
        label = y_train[i]

        # Get prediction before training
        psi.free_phase(X_train[i])
        hidden_start = psi.n_input
        hidden_end = psi.n_input + psi.n_hidden
        input_flat = psi.states[:psi.n_input].flatten()
        hidden_flat = psi.states[hidden_start:hidden_end].flatten()
        combined = np.concatenate([input_flat, hidden_flat])
        logits = combined @ psi.hidden_to_output
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        class_rewards[label].append(probs[label])
        pred = np.argmax(probs)
        if pred == label:
            class_correct[label] += 1
        class_total[label] += 1

        psi.train_step(X_train[i], y_train[i])

    print()
    print("Per-class performance (after 1 epoch):")
    print(f"{'Class':<8} | {'Count':>6} | {'Correct':>8} | {'Accuracy':>10} | {'Avg Reward':>10}")
    print("-" * 55)

    for c in range(10):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
            avg_r = np.mean(class_rewards[c])
            print(f"  {c:<6} | {class_total[c]:>6} | {class_correct[c]:>8} | {acc:>9.1%} | {avg_r:>10.3f}")

    # Check hidden layer activation patterns
    print()
    print("Hidden layer activation analysis:")
    activations_by_class = {i: [] for i in range(10)}

    for i in range(min(100, len(X_test))):
        psi.free_phase(X_test[i])
        hidden_states = psi.states[psi.n_input:psi.n_input + psi.n_hidden]
        activations_by_class[y_test[i]].append(hidden_states.flatten())

    # Compute within-class vs between-class variance
    class_means = {}
    for c in range(10):
        if activations_by_class[c]:
            class_means[c] = np.mean(activations_by_class[c], axis=0)

    # Within-class variance
    within_var = []
    for c in range(10):
        if len(activations_by_class[c]) > 1:
            var = np.var(activations_by_class[c], axis=0)
            within_var.append(np.mean(var))

    # Between-class variance
    all_means = list(class_means.values())
    between_var = np.var(all_means, axis=0)

    print(f"  Within-class variance (avg): {np.mean(within_var):.4f}")
    print(f"  Between-class variance (avg): {np.mean(between_var):.4f}")
    print(f"  Ratio (higher = better separation): {np.mean(between_var) / (np.mean(within_var) + 1e-8):.4f}")


if __name__ == "__main__":
    compare_scaling()
    analyze_gradient_flow()
