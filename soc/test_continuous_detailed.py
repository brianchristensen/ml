"""
Test continuous planning that mimics interactive mode more closely.
Track g over many iterations to see if it degrades.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_pattern(mind, name):
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


def simulate_interactive_mode(mind, goal_pattern, goal_name, n_frames=100, verbose=True):
    """
    Simulate what happens in interactive mode's _execute_goal_step.
    """
    # Save starting state
    goal_start_state = mind.A.copy()

    # Initialize planning
    mind.init_planning()

    g_history = []
    r_history = []
    action_history = []

    for frame in range(n_frames):
        # 1. Check goal overlap
        g = cosine_sim(mind.A, goal_pattern)
        g_history.append(g)

        if g > 0.6:  # Higher threshold to see more iterations
            if verbose:
                print(f"  GOAL REACHED at frame {frame}! g={g:.3f}")
            break

        # 2. Do planning iteration (same as _do_planning_iteration)
        if np.mean(mind.A) > 0.05:
            planning_state = mind.A.copy()
        else:
            planning_state = goal_start_state.copy()

        for i in range(mind.n_slots):
            mind.slot_A[i] = planning_state.copy()
        for i in range(mind.n_slots):
            mind._simulate_slot(i, steps_per_action=mind.plan_horizon)

        virtual_rewards, rewards, distances = mind._compute_virtual_rewards(goal_pattern)
        mind._clone_step(virtual_rewards)
        mind._perturb_actions()

        # Update best plan
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        r_history.append(best_reward)
        planned_actions = list(mind.slot_action_sequence[best_idx])

        # 3. Execute best action
        if planned_actions:
            best_action_pattern = planned_actions[0]
            best_action_dim = mind._get_action_dim_from_pattern(best_action_pattern)
            if best_action_dim is not None:
                mind.set_action(f'action_{best_action_dim}')
                action_history.append(best_action_dim)

        # 4. Keep network alive
        if np.mean(mind.A) < 0.05:
            mind.A = np.clip(mind.A + 0.5 * goal_start_state, 0, 1)

        # 5. Step network (WITH learning!)
        for _ in range(5):
            mind.step(0.1, dream_mode=False)

        if verbose and frame % 10 == 0:
            seq = [f"a{mind._get_action_dim_from_pattern(a)}" for a in planned_actions[:3]]
            print(f"  Frame {frame}: g={g:.3f}, r={best_reward:.3f}, seq={seq}, A_mean={np.mean(mind.A):.3f}")

    return g_history, r_history, action_history


def test_degradation():
    """Test if g degrades over time like in interactive mode."""
    print("=" * 60)
    print("CONTINUOUS PLANNING DEGRADATION TEST")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=200)

    # Learn 3-step path: A->B->C->D
    print("\nLearning A->B, B->C, C->D (3-step path)...")
    learn_transition(mind, 'room_A', 'action_0', 'room_B')
    learn_transition(mind, 'room_B', 'action_1', 'room_C')
    learn_transition(mind, 'room_C', 'action_2', 'room_D')

    # Check W_action_longterm has learned
    print(f"\nW_action_longterm[0] nonzero: {(mind.W_action_longterm[0] > 0.01).sum()}")
    print(f"W_action_longterm[1] nonzero: {(mind.W_action_longterm[1] > 0.01).sum()}")
    print(f"W_action_longterm[2] nonzero: {(mind.W_action_longterm[2] > 0.01).sum()}")

    # Set up for goal - target room_D (3 steps away)
    print("\nStarting continuous planning to room_D (3 steps away)...")
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, 'room_D')

    # Run simulation
    g_history, r_history, action_history = simulate_interactive_mode(
        mind, goal_pattern, 'room_D', n_frames=100, verbose=True
    )

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

        if late_g < early_g - 0.1:
            print("  >>> DEGRADATION DETECTED <<<")
        else:
            print("  No significant degradation")

    # Check if weights were corrupted
    print(f"\nW_action_longterm[0] nonzero after: {(mind.W_action_longterm[0] > 0.01).sum()}")
    print(f"W_action_longterm[1] nonzero after: {(mind.W_action_longterm[1] > 0.01).sum()}")
    print(f"W_action_longterm[2] nonzero after: {(mind.W_action_longterm[2] > 0.01).sum()}")


if __name__ == "__main__":
    test_degradation()
