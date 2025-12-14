"""
Debug script to compare slot simulation vs real network dynamics.
Goal: Understand why slot simulation fails while manual execution works.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def get_pattern(mind, name):
    """Get pattern for a name."""
    if name in mind.pattern_traces:
        return mind.pattern_traces[name]
    else:
        np.random.seed(hash(name) % 10000)
        n_active = np.random.randint(20, 50)
        units = np.random.choice(mind.n, n_active, replace=False)
        pattern = np.zeros(mind.n)
        pattern[units] = 1.0
        np.random.seed(None)
        return pattern


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def learn_transitions(mind):
    """Learn A->B and B->C transitions."""
    # A + action_0 -> B
    for _ in range(30):
        mind.inject_text('room_A', 0.9)
        mind.set_action('action_0')
        for _ in range(5):
            mind.step(0.1)
        mind.set_target('room_B', 1.0)
        for _ in range(20):
            mind.step(0.1)
        mind.clear_action()
        mind.A_target *= 0
        mind.target_strength = 0

    # B + action_1 -> C
    for _ in range(30):
        mind.inject_text('room_B', 0.9)
        mind.set_action('action_1')
        for _ in range(5):
            mind.step(0.1)
        mind.set_target('room_C', 1.0)
        for _ in range(20):
            mind.step(0.1)
        mind.clear_action()
        mind.A_target *= 0
        mind.target_strength = 0


def debug_manual_execution(mind):
    """
    Execute A -> B -> C manually, printing state at each step.
    """
    print("\n" + "=" * 60)
    print("MANUAL EXECUTION (real network dynamics)")
    print("=" * 60)

    mind.reset()
    mind.inject_text('room_A', 0.9)

    print(f"\nAfter inject room_A:")
    print(f"  A overlap: A={cosine_sim(mind.A, get_pattern(mind, 'room_A')):.3f}, "
          f"B={cosine_sim(mind.A, get_pattern(mind, 'room_B')):.3f}, "
          f"C={cosine_sim(mind.A, get_pattern(mind, 'room_C')):.3f}")
    print(f"  A mean: {mind.A.mean():.4f}, max: {mind.A.max():.4f}")

    # Let it settle
    for _ in range(5):
        mind.step(0.1)

    print(f"\nAfter settling (5 steps):")
    print(f"  A overlap: A={cosine_sim(mind.A, get_pattern(mind, 'room_A')):.3f}, "
          f"B={cosine_sim(mind.A, get_pattern(mind, 'room_B')):.3f}, "
          f"C={cosine_sim(mind.A, get_pattern(mind, 'room_C')):.3f}")
    print(f"  A mean: {mind.A.mean():.4f}, max: {mind.A.max():.4f}")

    # Apply action_0
    print(f"\n--- Applying action_0 ---")
    mind.set_action('action_0')

    for step in range(30):
        mind.step(0.1)
        if step % 10 == 9:
            print(f"  Step {step+1}: A={cosine_sim(mind.A, get_pattern(mind, 'room_A')):.3f}, "
                  f"B={cosine_sim(mind.A, get_pattern(mind, 'room_B')):.3f}, "
                  f"C={cosine_sim(mind.A, get_pattern(mind, 'room_C')):.3f}, "
                  f"mean={mind.A.mean():.4f}")

    state_after_action0 = mind.A.copy()

    # Apply action_1
    print(f"\n--- Applying action_1 ---")
    mind.set_action('action_1')

    for step in range(30):
        mind.step(0.1)
        if step % 10 == 9:
            print(f"  Step {step+1}: A={cosine_sim(mind.A, get_pattern(mind, 'room_A')):.3f}, "
                  f"B={cosine_sim(mind.A, get_pattern(mind, 'room_B')):.3f}, "
                  f"C={cosine_sim(mind.A, get_pattern(mind, 'room_C')):.3f}, "
                  f"mean={mind.A.mean():.4f}")

    final_C = cosine_sim(mind.A, get_pattern(mind, 'room_C'))
    print(f"\n  FINAL: C overlap = {final_C:.3f}")
    return final_C, state_after_action0


def debug_slot_simulation(mind, start_state, steps_per_action=30):
    """
    Simulate the same path in slot simulation mode.
    """
    print("\n" + "=" * 60)
    print(f"SLOT SIMULATION (steps_per_action={steps_per_action})")
    print("=" * 60)

    # Use slot 0
    mind.init_planning()

    # Create action patterns
    action_0 = np.zeros(mind.n_action_units)
    action_0[0:mind.action_units_per_dim] = 1.0

    action_1 = np.zeros(mind.n_action_units)
    action_1[mind.action_units_per_dim:2*mind.action_units_per_dim] = 1.0

    # Set up slot 0
    mind.slot_A[0] = start_state.copy()
    mind.slot_action_sequence[0] = [action_0, action_1]

    print(f"\nInitial slot state:")
    print(f"  A overlap: A={cosine_sim(mind.slot_A[0], get_pattern(mind, 'room_A')):.3f}, "
          f"B={cosine_sim(mind.slot_A[0], get_pattern(mind, 'room_B')):.3f}, "
          f"C={cosine_sim(mind.slot_A[0], get_pattern(mind, 'room_C')):.3f}")
    print(f"  mean: {mind.slot_A[0].mean():.4f}, max: {mind.slot_A[0].max():.4f}")

    # Run simulation
    print(f"\n--- Running slot simulation ---")
    _simulate_slot_debug(mind, 0, steps_per_action)

    final_C = cosine_sim(mind.slot_A[0], get_pattern(mind, 'room_C'))
    print(f"\n  FINAL: C overlap = {final_C:.3f}")
    return final_C


def _simulate_slot_debug(mind, slot_idx, steps_per_action=10):
    """
    Debug version of _simulate_slot that prints intermediate states.
    """
    temp_A = mind.slot_A[slot_idx].copy()
    temp_input_trace = temp_A.copy() * 0.5

    action_sequence = mind.slot_action_sequence[slot_idx]

    for action_idx, action_pattern in enumerate(action_sequence):
        action_dim = mind._get_action_dim_from_pattern(action_pattern)
        print(f"\n  Simulating action_{action_dim}:")

        for step in range(steps_per_action):
            firing = (temp_A > mind.threshold).astype(float)
            rows, cols = mind._conn_rows, mind._conn_cols

            # This is what slot simulation uses
            if action_dim is not None:
                action_weights = mind.W_action_longterm[action_dim]
                effective_weights = 0.9 * action_weights + 0.1 * mind.W_pred_data
            else:
                effective_weights = mind.W_pred_data

            excitatory = np.maximum(effective_weights, 0)
            inhibitory = np.minimum(effective_weights, 0)

            exc_input = excitatory * firing[cols]
            inh_input = inhibitory * firing[cols]

            exc_sum = np.zeros(mind.n)
            np.add.at(exc_sum, rows, exc_input)
            inh_sum = np.zeros(mind.n)
            np.add.at(inh_sum, rows, inh_input)

            in_degree = np.zeros(mind.n)
            np.add.at(in_degree, rows, np.abs(effective_weights))
            in_degree = np.maximum(in_degree, 1)

            total_input = (exc_sum + inh_sum) / in_degree
            trace_contribution = mind.trace_weight * temp_input_trace

            dA = total_input + trace_contribution - mind.decay * temp_A
            dA += np.random.normal(0, mind.noise * 0.3, mind.n)
            temp_A = np.clip(temp_A + 0.1 * dA, 0, 1)

            temp_input_trace *= 0.95

            if step % 10 == 9:
                print(f"    Step {step+1}: A={cosine_sim(temp_A, get_pattern(mind, 'room_A')):.3f}, "
                      f"B={cosine_sim(temp_A, get_pattern(mind, 'room_B')):.3f}, "
                      f"C={cosine_sim(temp_A, get_pattern(mind, 'room_C')):.3f}, "
                      f"mean={temp_A.mean():.4f}, firing={firing.sum():.0f}")

    mind.slot_A[slot_idx] = temp_A


def compare_weights(mind):
    """Compare W_action_longterm vs W_transition to see if learning worked."""
    print("\n" + "=" * 60)
    print("WEIGHT ANALYSIS")
    print("=" * 60)

    print(f"\nW_action_longterm[0] (action_0): mean={mind.W_action_longterm[0].mean():.4f}, "
          f"max={mind.W_action_longterm[0].max():.4f}, nonzero={(mind.W_action_longterm[0] > 0.01).sum()}")
    print(f"W_action_longterm[1] (action_1): mean={mind.W_action_longterm[1].mean():.4f}, "
          f"max={mind.W_action_longterm[1].max():.4f}, nonzero={(mind.W_action_longterm[1] > 0.01).sum()}")

    print(f"\nW_transition[0] (action_0): mean={mind.W_transition[0].mean():.4f}, "
          f"max={mind.W_transition[0].max():.4f}")
    print(f"W_transition[1] (action_1): mean={mind.W_transition[1].mean():.4f}, "
          f"max={mind.W_transition[1].max():.4f}")

    print(f"\nW_pred_data: mean={mind.W_pred_data.mean():.4f}, max={mind.W_pred_data.max():.4f}")
    print(f"W_dyn_data: mean={mind.W_dyn_data.mean():.4f}, max={mind.W_dyn_data.max():.4f}")


def check_threshold_firing(mind, state, pattern_name):
    """Check how many units fire for a given state."""
    firing = (state > mind.threshold).astype(float)
    pattern = get_pattern(mind, pattern_name)
    pattern_units = np.where(pattern > 0.5)[0]

    pattern_firing = firing[pattern_units].sum()
    total_firing = firing.sum()

    print(f"  {pattern_name}: {pattern_firing:.0f}/{len(pattern_units)} pattern units firing, "
          f"{total_firing:.0f} total firing, threshold={mind.threshold:.3f}")


if __name__ == "__main__":
    print("Slot Simulation Debug")
    print("=" * 60)

    np.random.seed(42)  # For reproducibility

    mind = PlanningMind(n_units=200)

    # Learn transitions
    print("\nLearning A->B, B->C...")
    learn_transitions(mind)

    # Analyze weights
    compare_weights(mind)

    # Get the starting state after injection
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(5):
        mind.step(0.1)
    start_state = mind.A.copy()

    print(f"\nStart state analysis:")
    check_threshold_firing(mind, start_state, 'room_A')
    check_threshold_firing(mind, start_state, 'room_B')
    check_threshold_firing(mind, start_state, 'room_C')

    # Run manual execution
    manual_C, state_after_0 = debug_manual_execution(mind)

    # Reset and run slot simulation with same start
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(5):
        mind.step(0.1)
    start_state = mind.A.copy()

    slot_C = debug_slot_simulation(mind, start_state, steps_per_action=30)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Manual execution: C overlap = {manual_C:.3f}")
    print(f"  Slot simulation:  C overlap = {slot_C:.3f}")
    print(f"  Difference: {manual_C - slot_C:.3f}")

    if slot_C < 0.3:
        print("\n  >>> SLOT SIMULATION FAILED <<<")
        print("  Possible causes:")
        print("    1. Weights not properly loaded in simulation")
        print("    2. Input trace handling differs")
        print("    3. Energy (E) not used in simulation")
        print("    4. Fast weights not used in simulation")
        print("    5. Tau (adaptive timescale) not used in simulation")
