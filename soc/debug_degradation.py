"""
Debug why goal overlap degrades after getting close to goal.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def get_pattern(mind, name):
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


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def learn_one_shot(mind, from_room, action_dim, to_room, steps=25):
    mind.inject_text(from_room, 0.9)
    mind.set_action(f'action_{action_dim}')
    mind.set_target(to_room, 1.0)
    for _ in range(steps):
        mind.step(0.1, dream_mode=False)
    mind.clear_action()
    mind.A_target *= 0
    mind.target_strength = 0


def analyze_planning(mind, goal_pattern, frame):
    """Analyze what the planner sees at this frame."""
    print(f"\n  --- Frame {frame} Planning Analysis ---")

    # Current state overlaps
    for room in ['room_A', 'room_B', 'room_C', 'room_D']:
        overlap = cosine_sim(mind.A, get_pattern(mind, room))
        print(f"    Current state vs {room}: {overlap:.3f}")

    # Simulate each action and see rewards
    print(f"\n    Simulated rewards per action (from current state):")
    for action_dim in range(4):
        # Set up a slot
        mind.slot_A[0] = mind.A.copy()
        action_pattern = np.zeros(mind.n_action_units)
        start = action_dim * mind.action_units_per_dim
        action_pattern[start:start + mind.action_units_per_dim] = 1.0
        mind.slot_action_sequence[0] = [action_pattern]

        # Simulate
        mind._simulate_slot(0, steps_per_action=mind.plan_horizon)

        # Compute reward
        slot_norm = np.linalg.norm(mind.slot_A[0])
        goal_norm = np.linalg.norm(goal_pattern)
        if slot_norm > 1e-8 and goal_norm > 1e-8:
            reward = np.dot(mind.slot_A[0], goal_pattern) / (slot_norm * goal_norm)
        else:
            reward = 0.0

        # Also check what room the slot ends up in
        best_room = None
        best_overlap = 0
        for room in ['room_A', 'room_B', 'room_C', 'room_D']:
            overlap = cosine_sim(mind.slot_A[0], get_pattern(mind, room))
            if overlap > best_overlap:
                best_overlap = overlap
                best_room = room

        print(f"    action_{action_dim}: reward={reward:.3f}, ends at {best_room}({best_overlap:.2f})")


def main():
    np.random.seed(42)
    mind = PlanningMind(n_units=200)

    # Learn
    learn_one_shot(mind, 'room_A', 0, 'room_B')
    learn_one_shot(mind, 'room_B', 1, 'room_C')
    learn_one_shot(mind, 'room_C', 2, 'room_D')

    # Set up goal
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_start_state = mind.A.copy()
    goal_pattern = get_pattern(mind, 'room_D')
    mind.init_planning()

    # Run and analyze at key frames
    for frame in range(80):
        # Check goal overlap
        g = cosine_sim(mind.A, goal_pattern)

        if frame in [0, 10, 30, 50, 60, 70]:
            analyze_planning(mind, goal_pattern, frame)

        # Do planning iteration
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

        best_idx = np.argmax(rewards)
        planned_actions = list(mind.slot_action_sequence[best_idx])

        # Execute best action
        if planned_actions:
            best_action_pattern = planned_actions[0]
            best_action_dim = mind._get_action_dim_from_pattern(best_action_pattern)
            if best_action_dim is not None:
                mind.set_action(f'action_{best_action_dim}')

        if np.mean(mind.A) < 0.05:
            mind.A = np.clip(mind.A + 0.5 * goal_start_state, 0, 1)

        for _ in range(5):
            mind.step(0.1, execute_mode=True)

        if frame in [0, 10, 30, 50, 60, 70]:
            print(f"    After execution: g={cosine_sim(mind.A, goal_pattern):.3f}")


if __name__ == "__main__":
    main()
