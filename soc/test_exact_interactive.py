"""
Test that EXACTLY replicates interactive mode behavior.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def get_pattern(mind, name):
    """EXACTLY how interactive mode gets patterns."""
    if name in mind.pattern_traces:
        return mind.pattern_traces[name]
    else:
        np.random.seed(hash(name) % 10000)
        n_active = max(5, int(mind.n * mind.pattern_sparsity))
        units = np.random.choice(mind.n, n_active, replace=False)
        pattern = np.zeros(mind.n)
        pattern[units] = 1.0
        np.random.seed(None)
        return pattern


def learn_transition_interactive(mind, from_room, action_dim, to_room, steps=25):
    """EXACTLY how interactive mode learns transitions."""
    mind.inject_text(from_room, 0.9)
    mind.set_action(f"action_{action_dim}")
    mind.set_target(to_room, 1.0)

    for _ in range(steps):
        mind.step(0.1, dream_mode=False)

    mind.clear_action()
    mind.A_target *= 0
    mind.target_strength = 0


def execute_goal_step(mind, goal_pattern, goal_start_state, planning_iterations, verbose=False):
    """EXACTLY replicates _execute_goal_step from interactive mode."""

    # Check goal overlap
    A_norm = np.linalg.norm(mind.A)
    goal_norm = np.linalg.norm(goal_pattern)
    if A_norm > 1e-8 and goal_norm > 1e-8:
        goal_overlap = np.dot(mind.A, goal_pattern) / (A_norm * goal_norm)
    else:
        goal_overlap = 0.0

    if goal_overlap > 0.5:  # goal_threshold
        return goal_overlap, True, planning_iterations, None

    # _do_planning_iteration
    if np.mean(mind.A) > 0.05:
        planning_state = mind.A.copy()
    else:
        planning_state = goal_start_state.copy()

    # Simulate slots
    for i in range(mind.n_slots):
        mind.slot_A[i] = planning_state.copy()
    for i in range(mind.n_slots):
        mind._simulate_slot(i, steps_per_action=mind.plan_horizon)

    # Compute rewards
    virtual_rewards, rewards, distances = mind._compute_virtual_rewards(goal_pattern)

    # Clone and perturb
    mind._clone_step(virtual_rewards)
    mind._perturb_actions()

    planning_iterations += 1

    # Get best action
    best_idx = np.argmax(rewards)
    planned_actions = list(mind.slot_action_sequence[best_idx])
    best_reward = rewards[best_idx]

    # Execute best action
    best_action_dim = None
    if planned_actions:
        best_action_pattern = planned_actions[0]
        best_action_dim = mind._get_action_dim_from_pattern(best_action_pattern)
        if best_action_dim is not None:
            mind.set_action(f'action_{best_action_dim}')

    # GOAL REINFORCEMENT: weak continuous pull toward goal
    mind.A = np.clip(mind.A + 0.05 * goal_pattern, 0, 1)

    # Keep network alive
    if np.mean(mind.A) < 0.05:
        mind.A = np.clip(mind.A + 0.5 * goal_start_state, 0, 1)

    # Step with execute_mode=True
    for _ in range(5):
        mind.step(0.1, execute_mode=True)

    return goal_overlap, False, planning_iterations, best_action_dim


def test_exact_interactive():
    print("=" * 60)
    print("EXACT INTERACTIVE MODE TEST")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=200)

    # Learn transitions EXACTLY as interactive mode does
    print("\nLearning A->B (action_0)...")
    learn_transition_interactive(mind, 'room_A', 0, 'room_B')

    print("Learning B->C (action_1)...")
    learn_transition_interactive(mind, 'room_B', 1, 'room_C')

    print("Learning C->D (action_2)...")
    learn_transition_interactive(mind, 'room_C', 2, 'room_D')

    # Check what was learned
    print(f"\nExperience buffer size: {len(mind.experience_buffer)}")
    print(f"W_action_longterm[0] nonzero: {(mind.W_action_longterm[0] > 0.01).sum()}")
    print(f"W_action_longterm[1] nonzero: {(mind.W_action_longterm[1] > 0.01).sum()}")
    print(f"W_action_longterm[2] nonzero: {(mind.W_action_longterm[2] > 0.01).sum()}")

    # Now execute goal EXACTLY as interactive mode does
    print("\n" + "-" * 40)
    print("Executing goal: room_D")
    print("-" * 40)

    # Set up starting state
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_start_state = mind.A.copy()
    goal_pattern = get_pattern(mind, 'room_D')
    mind.init_planning()

    g_history = []
    action_history = []
    planning_iterations = 0

    for frame in range(100):
        g, reached, planning_iterations, action = execute_goal_step(
            mind, goal_pattern, goal_start_state, planning_iterations
        )
        g_history.append(g)
        if action is not None:
            action_history.append(action)

        if frame % 10 == 0:
            recent_actions = action_history[-5:] if len(action_history) >= 5 else action_history
            action_str = ','.join(str(a) for a in recent_actions)
            print(f"  Frame {frame}: g={g:.3f}, A_mean={np.mean(mind.A):.3f}, actions=[{action_str}]")

        if reached:
            print(f"  GOAL REACHED at frame {frame}! g={g:.3f}")
            break

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if len(g_history) >= 20:
        early_g = np.mean(g_history[:10])
        late_g = np.mean(g_history[-10:])
        print(f"  Early g (frames 0-9): {early_g:.3f}")
        print(f"  Late g (frames -10 to end): {late_g:.3f}")
        print(f"  Degradation: {early_g - late_g:.3f}")

        if late_g < early_g - 0.05:
            print("  >>> DEGRADATION DETECTED <<<")
        else:
            print("  No significant degradation")


if __name__ == "__main__":
    test_exact_interactive()
