"""
Debug script to understand what W_longterm actually learns.
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


def analyze_weights(mind, action_dim, from_pattern, to_pattern, from_name, to_name):
    """Analyze what connections were learned for a transition."""
    rows, cols = mind._conn_rows, mind._conn_cols
    weights = mind.W_longterm[action_dim]

    # Find connections FROM from_pattern units TO to_pattern units
    from_units = set(np.where(from_pattern > 0.5)[0])
    to_units = set(np.where(to_pattern > 0.5)[0])

    # Check all connections
    from_to_weights = []  # Correct: from_pattern -> to_pattern
    other_to_weights = []  # Wrong: other -> to_pattern
    from_other_weights = []  # Wrong: from_pattern -> other
    other_weights = []  # Unrelated

    for i in range(len(rows)):
        pre, post = cols[i], rows[i]
        w = weights[i]

        if w > 0.01:  # Only nonzero
            pre_is_from = pre in from_units
            post_is_to = post in to_units

            if pre_is_from and post_is_to:
                from_to_weights.append(w)
            elif pre_is_from and not post_is_to:
                from_other_weights.append(w)
            elif not pre_is_from and post_is_to:
                other_to_weights.append(w)
            else:
                other_weights.append(w)

    print(f"\n  W_longterm[{action_dim}] ({from_name} -> {to_name}):")
    print(f"    {from_name} -> {to_name} (correct): {len(from_to_weights)} connections, "
          f"mean={np.mean(from_to_weights) if from_to_weights else 0:.3f}")
    print(f"    {from_name} -> other: {len(from_other_weights)} connections, "
          f"mean={np.mean(from_other_weights) if from_other_weights else 0:.3f}")
    print(f"    other -> {to_name}: {len(other_to_weights)} connections, "
          f"mean={np.mean(other_to_weights) if other_to_weights else 0:.3f}")
    print(f"    other -> other: {len(other_weights)} connections, "
          f"mean={np.mean(other_weights) if other_weights else 0:.3f}")


def check_base_transition(mind, from_name, to_name):
    """Check if base weights cause a transition without any action."""
    mind.reset()
    mind.inject_text(from_name, 0.9)
    for _ in range(5):
        mind.step(0.1)

    from_sim = np.dot(mind.A, get_pattern(mind, from_name)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, from_name)) + 1e-8)
    to_sim = np.dot(mind.A, get_pattern(mind, to_name)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_name)) + 1e-8)

    print(f"\n  After injection ({from_name}): {from_name}={from_sim:.3f}, {to_name}={to_sim:.3f}")

    # Run without any action
    mind.clear_action()
    for _ in range(30):
        mind.step(0.1)

    from_sim = np.dot(mind.A, get_pattern(mind, from_name)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, from_name)) + 1e-8)
    to_sim = np.dot(mind.A, get_pattern(mind, to_name)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_name)) + 1e-8)

    print(f"  After 30 steps (no action): {from_name}={from_sim:.3f}, {to_name}={to_sim:.3f}")

    return to_sim


def test_before_learning():
    """Test transitions BEFORE any learning."""
    print("=" * 60)
    print("BEFORE LEARNING")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    print("\nBase transitions (no learning, no actions):")
    check_base_transition(mind, 'room_A', 'room_B')
    check_base_transition(mind, 'room_B', 'room_C')
    check_base_transition(mind, 'room_A', 'room_C')

    print("\nTransitions with action_0 (no learning):")
    for from_room, to_room in [('room_A', 'room_B'), ('room_B', 'room_C')]:
        mind.reset()
        mind.inject_text(from_room, 0.9)
        for _ in range(5):
            mind.step(0.1)
        mind.set_action('action_0')
        for _ in range(30):
            mind.step(0.1)
        to_sim = np.dot(mind.A, get_pattern(mind, to_room)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_room)) + 1e-8)
        print(f"  {from_room} + action_0: {to_room}={to_sim:.3f}")


def test_after_learning():
    """Test transitions AFTER learning."""
    print("\n" + "=" * 60)
    print("AFTER LEARNING")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    # Learn A->B with action_0
    print("\nLearning room_A + action_0 -> room_B...")
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

    # Learn B->C with action_1
    print("Learning room_B + action_1 -> room_C...")
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

    # Analyze weights
    print("\nWeight analysis:")
    pattern_A = get_pattern(mind, 'room_A')
    pattern_B = get_pattern(mind, 'room_B')
    pattern_C = get_pattern(mind, 'room_C')

    analyze_weights(mind, 0, pattern_A, pattern_B, 'room_A', 'room_B')
    analyze_weights(mind, 0, pattern_B, pattern_C, 'room_B', 'room_C')  # Should be empty!
    analyze_weights(mind, 1, pattern_A, pattern_B, 'room_A', 'room_B')  # Should be empty!
    analyze_weights(mind, 1, pattern_B, pattern_C, 'room_B', 'room_C')

    # Check W_longterm[2] and W_longterm[3] (should be zeros)
    print(f"\n  W_longterm[2] (untrained): nonzero={(mind.W_longterm[2] > 0.01).sum()}")
    print(f"  W_longterm[3] (untrained): nonzero={(mind.W_longterm[3] > 0.01).sum()}")

    # Test transitions
    print("\nTest transitions after learning:")

    # With correct actions
    print("\n  With CORRECT actions:")
    for (from_room, action, to_room), expected in [
        (('room_A', 'action_0', 'room_B'), True),
        (('room_B', 'action_1', 'room_C'), True),
    ]:
        mind.reset()
        mind.inject_text(from_room, 0.9)
        for _ in range(5):
            mind.step(0.1)
        mind.set_action(action)
        for _ in range(30):
            mind.step(0.1)
        to_sim = np.dot(mind.A, get_pattern(mind, to_room)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_room)) + 1e-8)
        status = "OK" if (to_sim > 0.5) == expected else "PROBLEM"
        print(f"    {from_room} + {action} -> {to_room}: {to_sim:.3f} [{status}]")

    # With WRONG actions
    print("\n  With WRONG actions (should NOT work):")
    for (from_room, action, to_room), expected in [
        (('room_A', 'action_1', 'room_B'), False),  # Wrong action
        (('room_B', 'action_0', 'room_C'), False),  # Wrong action
        (('room_A', 'action_2', 'room_B'), False),  # Untrained action
        (('room_B', 'action_3', 'room_C'), False),  # Untrained action
    ]:
        mind.reset()
        mind.inject_text(from_room, 0.9)
        for _ in range(5):
            mind.step(0.1)
        mind.set_action(action)
        for _ in range(30):
            mind.step(0.1)
        to_sim = np.dot(mind.A, get_pattern(mind, to_room)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_room)) + 1e-8)
        status = "OK" if (to_sim > 0.5) == expected else "PROBLEM"
        print(f"    {from_room} + {action} -> {to_room}: {to_sim:.3f} [{status}]")

    # Without ANY action (should show base behavior)
    print("\n  WITHOUT action (base dynamics):")
    for from_room, to_room in [('room_A', 'room_B'), ('room_B', 'room_C')]:
        mind.reset()
        mind.inject_text(from_room, 0.9)
        for _ in range(5):
            mind.step(0.1)
        mind.clear_action()
        for _ in range(30):
            mind.step(0.1)
        to_sim = np.dot(mind.A, get_pattern(mind, to_room)) / (np.linalg.norm(mind.A) * np.linalg.norm(get_pattern(mind, to_room)) + 1e-8)
        print(f"    {from_room} (no action) -> {to_room}: {to_sim:.3f}")


if __name__ == "__main__":
    np.random.seed(42)
    test_before_learning()
    test_after_learning()
