"""
Debug: Is the learning signal actually updating weights?

We know:
- Target phase creates good class separation (ratio 8.48)
- Free phase has poor separation (ratio 0.14)
- The contrastive signal exists (0.148 magnitude)

Question: Why doesn't learning transfer this to free phase?
"""

import numpy as np
from mnist_benchmark import PSIClassifier, load_mnist_subset


def debug_learning_signal():
    print("=" * 70)
    print("LEARNING SIGNAL DEBUG: Is learning actually happening?")
    print("=" * 70)

    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_mnist_subset(n_train=500, n_test=200)

    model = PSIClassifier(n_input=28, n_hidden=24, n_output=10, dim=24, seed=42)

    # Store initial weights
    initial_node_W = model.node_W.copy()
    initial_h2o = model.hidden_to_output.copy()
    initial_input_proj = model.input_proj.copy()
    initial_biases = model.biases.copy()

    print("\n1. Initial weight statistics:")
    print(f"   node_W mean |w|: {np.mean(np.abs(initial_node_W)):.6f}")
    print(f"   h2o mean |w|: {np.mean(np.abs(initial_h2o)):.6f}")
    print(f"   input_proj mean |w|: {np.mean(np.abs(initial_input_proj)):.6f}")

    # Train for a few samples and track weight changes
    print("\n2. Training and tracking weight changes...")

    n_samples = 100
    node_W_changes = []
    h2o_changes = []
    input_proj_changes = []
    bias_changes = []
    td_errors = []
    rewards = []

    for i in range(n_samples):
        # Store pre-update weights
        pre_node_W = model.node_W.copy()
        pre_h2o = model.hidden_to_output.copy()
        pre_input_proj = model.input_proj.copy()
        pre_biases = model.biases.copy()

        # Training step
        model.last_label = y_train[i]
        outputs = model.free_phase(X_train[i])
        model.target_phase(y_train[i])
        model.learn()

        # Compute changes
        dW_node = np.mean(np.abs(model.node_W - pre_node_W))
        dW_h2o = np.mean(np.abs(model.hidden_to_output - pre_h2o))
        dW_proj = np.mean(np.abs(model.input_proj - pre_input_proj))
        dW_bias = np.mean(np.abs(model.biases - pre_biases))

        node_W_changes.append(dW_node)
        h2o_changes.append(dW_h2o)
        input_proj_changes.append(dW_proj)
        bias_changes.append(dW_bias)

        # Get reward/TD error from model
        if hasattr(model, 'expected_reward'):
            rewards.append(model.expected_reward)

    print(f"\n3. Weight change magnitudes over {n_samples} samples:")
    print(f"   node_W: {np.mean(node_W_changes):.8f} (std: {np.std(node_W_changes):.8f})")
    print(f"   h2o: {np.mean(h2o_changes):.8f} (std: {np.std(h2o_changes):.8f})")
    print(f"   input_proj: {np.mean(input_proj_changes):.8f} (std: {np.std(input_proj_changes):.8f})")
    print(f"   biases: {np.mean(bias_changes):.8f} (std: {np.std(bias_changes):.8f})")

    # Total change from initial
    total_node_W = np.mean(np.abs(model.node_W - initial_node_W))
    total_h2o = np.mean(np.abs(model.hidden_to_output - initial_h2o))
    total_input_proj = np.mean(np.abs(model.input_proj - initial_input_proj))

    print(f"\n4. Total weight change from initial:")
    print(f"   node_W: {total_node_W:.6f} ({total_node_W/np.mean(np.abs(initial_node_W))*100:.2f}% of initial)")
    print(f"   h2o: {total_h2o:.6f} ({total_h2o/np.mean(np.abs(initial_h2o))*100:.2f}% of initial)")
    print(f"   input_proj: {total_input_proj:.6f} ({total_input_proj/np.mean(np.abs(initial_input_proj))*100:.2f}% of initial)")

    print(f"\n5. Expected reward trajectory:")
    print(f"   Start: {rewards[0]:.4f}")
    print(f"   End: {rewards[-1]:.4f}")
    print(f"   Change: {rewards[-1] - rewards[0]:.4f}")

    # Check the contrastive signal more carefully
    print("\n" + "=" * 70)
    print("6. CONTRASTIVE SIGNAL ANALYSIS")
    print("=" * 70)

    # Do one more sample and inspect the signal
    img = X_train[0]
    label = y_train[0]

    model.states = np.random.randn(model.n_nodes, model.dim) * 0.01
    model.clamped = np.full((model.n_nodes, model.dim), np.nan)

    # Project input
    projected = model.project_input(img)
    model.clamped[:model.n_input] = projected

    # Free phase
    model.settle(max_iters=5, min_steps=2)
    free_states = model.states.copy()

    # Target phase
    target = np.full((model.n_output, model.dim), -1.0)
    target[label] = 1.0
    model.clamped[-model.n_output:] = target
    model.settle(max_iters=15, min_steps=6, target_mode=True)
    target_states = model.states.copy()

    # Contrastive signal
    state_diff = target_states - free_states
    hidden_start = model.n_input
    hidden_end = model.n_input + model.n_hidden

    print(f"\nContrastive signal by layer:")
    print(f"   Input nodes: {np.mean(np.abs(state_diff[:model.n_input])):.6f}")
    print(f"   Hidden nodes: {np.mean(np.abs(state_diff[hidden_start:hidden_end])):.6f}")
    print(f"   Output nodes: {np.mean(np.abs(state_diff[-model.n_output:])):.6f}")

    # Check the learning rate and scaling factors
    print("\n" + "=" * 70)
    print("7. LEARNING RATE ANALYSIS")
    print("=" * 70)

    print(f"\n   Base learning rate: {model.lr}")
    print(f"   Combined size: {model.combined_size}")
    print(f"   Size scale (1/sqrt): {1.0 / np.sqrt(model.combined_size):.6f}")

    # After warmup
    effective_lr = model.lr * 1.0 / np.sqrt(model.combined_size)
    print(f"   Effective LR (after warmup): {effective_lr:.6f}")

    # Check dopamine gating
    print("\n" + "=" * 70)
    print("8. DOPAMINE SIGNAL ANALYSIS")
    print("=" * 70)

    # Simulate what the dopamine signal would be
    # Get current probs
    model.last_label = label
    model.free_states = free_states
    model.target_states = target_states

    input_flat = free_states[:model.n_input].flatten()
    hidden_flat = free_states[hidden_start:hidden_end].flatten()
    combined = np.concatenate([input_flat, hidden_flat])
    feat_mean = combined.mean()
    feat_std = combined.std() + 1e-8
    combined_norm = (combined - feat_mean) / feat_std
    logits = combined_norm @ model.hidden_to_output
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    correct_prob = probs[label]
    max_wrong_prob = np.max(np.delete(probs, label))
    reward = correct_prob - max_wrong_prob

    print(f"\n   Correct class prob: {correct_prob:.4f}")
    print(f"   Best wrong class prob: {max_wrong_prob:.4f}")
    print(f"   Bipolar reward: {reward:.4f}")
    print(f"   Expected reward: {model.expected_reward:.4f}")
    print(f"   TD error: {reward - model.expected_reward:.4f}")
    print(f"   Dopamine (clipped): {np.clip(reward - model.expected_reward, -0.5, 0.5):.4f}")

    dopamine = np.clip(reward - model.expected_reward, -0.5, 0.5)
    dopamine_mod = 1.0 + dopamine
    print(f"   Dopamine modulation: {dopamine_mod:.4f}")

    # Final effective update magnitude
    effective_update = effective_lr * dopamine_mod
    print(f"\n   Final effective update scale: {effective_update:.6f}")

    # Check if synaptic decay is too strong
    print("\n" + "=" * 70)
    print("9. SYNAPTIC DECAY VS LEARNING")
    print("=" * 70)

    decay = 0.9995
    avg_weight = np.mean(np.abs(model.node_W))
    decay_per_step = avg_weight * (1 - decay)
    print(f"\n   Synaptic decay rate: {1 - decay}")
    print(f"   Average weight magnitude: {avg_weight:.6f}")
    print(f"   Weight lost per step to decay: {decay_per_step:.8f}")
    print(f"   Weight gained per step from learning: {np.mean(node_W_changes):.8f}")

    ratio = np.mean(node_W_changes) / (decay_per_step + 1e-10)
    print(f"\n   Learning/Decay ratio: {ratio:.4f}")
    if ratio < 1:
        print("   WARNING: Decay is STRONGER than learning! Weights are shrinking!")
    else:
        print("   OK: Learning outpaces decay")


if __name__ == "__main__":
    debug_learning_signal()
