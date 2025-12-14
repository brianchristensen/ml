"""
Test that learned transitions are SPECIFIC - only the correct action should work.
If action_0 works for A->B, then action_1, action_2, action_3 should NOT work.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


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


def learn_transitions(mind):
    """Learn ONLY A->B (action_0) and B->C (action_1)."""
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


def test_action(mind, start_room, action_name, target_room, n_steps=30):
    """
    Test if a specific action from a room leads to target.
    Returns the overlap with target after taking the action.
    """
    mind.reset()
    mind.inject_text(start_room, 0.9)
    for _ in range(5):
        mind.step(0.1)

    mind.set_action(action_name)
    for _ in range(n_steps):
        mind.step(0.1)

    return cosine_sim(mind.A, get_pattern(mind, target_room))


def test_specificity():
    """Test that transitions are specific to the learned actions."""
    print("=" * 60)
    print("TRANSITION SPECIFICITY TEST")
    print("=" * 60)
    print("\nLearned transitions:")
    print("  room_A + action_0 -> room_B")
    print("  room_B + action_1 -> room_C")
    print("\nOther actions should NOT cause these transitions.\n")

    mind = PlanningMind(n_units=200)
    learn_transitions(mind)

    # Test A -> B with different actions
    print("Testing room_A -> room_B:")
    print("  (Expect: action_0 SHOULD work, others should NOT)")
    for i in range(4):
        action_name = f'action_{i}'
        B_overlap = test_action(mind, 'room_A', action_name, 'room_B')
        is_correct = (i == 0)
        status = "CORRECT" if is_correct else "WRONG"
        should_work = "should work" if is_correct else "should NOT work"
        works = "WORKS" if B_overlap > 0.5 else "doesn't work"
        ok = "OK" if (works == "WORKS") == is_correct else "PROBLEM"
        print(f"    {action_name}: B={B_overlap:.3f} ({works}) - {ok}")

    # Test B -> C with different actions
    print("\nTesting room_B -> room_C:")
    print("  (Expect: action_1 SHOULD work, others should NOT)")
    for i in range(4):
        action_name = f'action_{i}'
        C_overlap = test_action(mind, 'room_B', action_name, 'room_C')
        is_correct = (i == 1)
        status = "CORRECT" if is_correct else "WRONG"
        should_work = "should work" if is_correct else "should NOT work"
        works = "WORKS" if C_overlap > 0.5 else "doesn't work"
        ok = "OK" if (works == "WORKS") == is_correct else "PROBLEM"
        print(f"    {action_name}: C={C_overlap:.3f} ({works}) - {ok}")

    # Test full path A -> C
    print("\nTesting full paths from room_A to room_C:")
    print("  (Expect: ONLY action_0 then action_1 should work)")

    action_sequences = [
        [0, 1],  # Correct path
        [1, 0],  # Wrong order
        [0, 0],  # Wrong second action
        [1, 1],  # Wrong first action
        [2, 3],  # Totally wrong
    ]

    for seq in action_sequences:
        mind.reset()
        mind.inject_text('room_A', 0.9)
        for _ in range(5):
            mind.step(0.1)

        for action_dim in seq:
            mind.set_action(f'action_{action_dim}')
            for _ in range(30):
                mind.step(0.1)

        C_overlap = cosine_sim(mind.A, get_pattern(mind, 'room_C'))
        seq_str = f"[action_{seq[0]}, action_{seq[1]}]"
        is_correct = (seq == [0, 1])
        should_work = "should work" if is_correct else "should NOT work"
        works = "WORKS" if C_overlap > 0.5 else "doesn't work"
        ok = "OK" if (works == "WORKS") == is_correct else "PROBLEM"
        print(f"    {seq_str}: C={C_overlap:.3f} ({works}) - {ok}")


def test_pattern_orthogonality():
    """Check if room patterns are orthogonal (they should be)."""
    print("\n" + "=" * 60)
    print("PATTERN ORTHOGONALITY")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    patterns = ['room_A', 'room_B', 'room_C', 'room_D']
    print("\nCosine similarities between room patterns:")
    print("  (Should be low for orthogonal patterns)")

    for i, p1 in enumerate(patterns):
        sims = []
        for j, p2 in enumerate(patterns):
            sim = cosine_sim(get_pattern(mind, p1), get_pattern(mind, p2))
            sims.append(f"{sim:.2f}")
        print(f"  {p1}: [{', '.join(sims)}]")


if __name__ == "__main__":
    np.random.seed(42)
    test_specificity()
    test_pattern_orthogonality()
