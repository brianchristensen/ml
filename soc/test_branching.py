"""
Test FMC planning with branching paths.

Scenario:
  room_A --action_0--> room_B --action_1--> room_C
  room_A --action_2--> room_D --action_3--> room_E

FMC should find the correct path depending on which goal is set.
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


def learn_transition(mind, from_room, action_name, to_room, n_reps=30):
    """Learn a single transition."""
    for _ in range(n_reps):
        mind.inject_text(from_room, 0.9)
        mind.set_action(action_name)
        for _ in range(5):
            mind.step(0.1)
        mind.set_target(to_room, 1.0)
        for _ in range(20):
            mind.step(0.1)
        mind.clear_action()
        mind.A_target *= 0
        mind.target_strength = 0


def test_branching():
    """Test planning with branching paths."""
    print("=" * 60)
    print("BRANCHING PATHS TEST")
    print("=" * 60)
    print("\nPath 1: room_A --action_0--> room_B --action_1--> room_C")
    print("Path 2: room_A --action_2--> room_D --action_3--> room_E")
    print()

    mind = PlanningMind(n_units=200)

    # Learn Path 1: A -> B -> C
    print("Learning Path 1 (A -> B -> C)...")
    learn_transition(mind, 'room_A', 'action_0', 'room_B')
    learn_transition(mind, 'room_B', 'action_1', 'room_C')

    # Learn Path 2: A -> D -> E
    print("Learning Path 2 (A -> D -> E)...")
    learn_transition(mind, 'room_A', 'action_2', 'room_D')
    learn_transition(mind, 'room_D', 'action_3', 'room_E')

    # Verify single transitions work
    print("\nVerifying transitions...")

    for (from_room, action, to_room) in [
        ('room_A', 'action_0', 'room_B'),
        ('room_B', 'action_1', 'room_C'),
        ('room_A', 'action_2', 'room_D'),
        ('room_D', 'action_3', 'room_E'),
    ]:
        mind.reset()
        mind.inject_text(from_room, 0.9)
        for _ in range(5):
            mind.step(0.1)
        mind.set_action(action)
        for _ in range(30):
            mind.step(0.1)
        overlap = cosine_sim(mind.A, get_pattern(mind, to_room))
        status = "OK" if overlap > 0.4 else "FAIL"
        print(f"  {from_room} + {action} -> {to_room}: {overlap:.3f} [{status}]")

    # Test planning to room_C
    print("\n" + "-" * 40)
    print("Planning to room_C")
    print("-" * 40)

    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_C = get_pattern(mind, 'room_C')
    action_sequence, best_idx = mind.plan(goal_C, n_iterations=100, verbose=False)

    decoded = [f"action_{mind._get_action_dim_from_pattern(a)}" for a in action_sequence]
    print(f"  Found sequence: {decoded}")

    # Execute FULL plan
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(5):
        mind.step(0.1)

    for action in action_sequence:  # Execute ALL actions
        dim = mind._get_action_dim_from_pattern(action)
        mind.set_action(f'action_{dim}')
        for _ in range(30):
            mind.step(0.1)

    C_overlap = cosine_sim(mind.A, get_pattern(mind, 'room_C'))
    E_overlap = cosine_sim(mind.A, get_pattern(mind, 'room_E'))
    print(f"  After executing full plan: C={C_overlap:.3f}, E={E_overlap:.3f}")

    # Success = reached C with higher overlap than E
    path1_success = C_overlap > 0.4 and C_overlap > E_overlap
    print(f"  Goal reached: {'YES' if path1_success else 'NO'}")

    # Test planning to room_E
    print("\n" + "-" * 40)
    print("Planning to room_E")
    print("-" * 40)

    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_E = get_pattern(mind, 'room_E')
    action_sequence, best_idx = mind.plan(goal_E, n_iterations=100, verbose=False)

    decoded = [f"action_{mind._get_action_dim_from_pattern(a)}" for a in action_sequence]
    print(f"  Found sequence: {decoded}")

    # Execute FULL plan
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(5):
        mind.step(0.1)

    for action in action_sequence:  # Execute ALL actions
        dim = mind._get_action_dim_from_pattern(action)
        mind.set_action(f'action_{dim}')
        for _ in range(30):
            mind.step(0.1)

    C_overlap = cosine_sim(mind.A, get_pattern(mind, 'room_C'))
    E_overlap = cosine_sim(mind.A, get_pattern(mind, 'room_E'))
    print(f"  After executing full plan: C={C_overlap:.3f}, E={E_overlap:.3f}")

    # Success = reached E with higher overlap than C
    path2_success = E_overlap > 0.4 and E_overlap > C_overlap
    print(f"  Goal reached: {'YES' if path2_success else 'NO'}")

    # Summary
    print("\n" + "=" * 60)
    print("BRANCHING TEST SUMMARY")
    print("=" * 60)
    print(f"  Planning to room_C: {'PASS' if path1_success else 'FAIL'}")
    print(f"  Planning to room_E: {'PASS' if path2_success else 'FAIL'}")

    # Note: The network may find alternative paths due to generalization
    # The key metric is whether the goal is reached, not the specific sequence
    print("\nNote: Due to neural network generalization, the network may")
    print("find unexpected paths. Success = reaching the goal state.")

    return path1_success and path2_success


if __name__ == "__main__":
    np.random.seed(42)
    success = test_branching()
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
