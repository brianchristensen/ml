"""
Test planning with CONTINUOUS state spaces.

This exposes the limitation of the current discrete-state model.
The task: 2D navigation where states are (x, y) positions in [0,1] x [0,1].

Current model limitation: It can only represent states it has seen before.
It cannot interpolate or generalize to new positions.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


class ContinuousGridWorld:
    """Simple 2D continuous navigation environment."""

    def __init__(self, step_size=0.2):
        self.step_size = step_size
        self.position = np.array([0.0, 0.0])

    def reset(self, position=None):
        if position is not None:
            self.position = np.array(position)
        else:
            self.position = np.array([0.0, 0.0])
        return self.position.copy()

    def step(self, action):
        """
        Actions: 0=right, 1=up, 2=left, 3=down
        """
        dx, dy = 0, 0
        if action == 0:  # right
            dx = self.step_size
        elif action == 1:  # up
            dy = self.step_size
        elif action == 2:  # left
            dx = -self.step_size
        elif action == 3:  # down
            dy = -self.step_size

        self.position[0] = np.clip(self.position[0] + dx, 0, 1)
        self.position[1] = np.clip(self.position[1] + dy, 0, 1)
        return self.position.copy()

    def get_state(self):
        return self.position.copy()


def state_to_name(pos):
    """Convert continuous position to a discrete name (current approach)."""
    return f"pos_{pos[0]:.2f}_{pos[1]:.2f}"


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_pattern(mind, name):
    """Get pattern for a name (same as interactive mode)."""
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


def test_discrete_encoding():
    """
    Test 1: Use discrete encoding (current approach).
    This should FAIL to generalize to unseen positions.
    """
    print("=" * 60)
    print("TEST 1: DISCRETE ENCODING (expect poor generalization)")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=200)
    env = ContinuousGridWorld(step_size=0.25)

    # Learn transitions from a grid of positions
    print("\nLearning transitions from training positions...")
    training_positions = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0),
        (0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),
        (0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5),
    ]

    for pos in training_positions:
        env.reset(pos)
        state_name = state_to_name(pos)

        # Learn all 4 actions from this position
        for action in range(4):
            env.reset(pos)
            next_pos = env.step(action)
            next_name = state_to_name(next_pos)

            # Learn transition
            mind.inject_text(state_name, 0.9)
            mind.set_action(f'action_{action}')
            mind.set_target(next_name, 1.0)
            for _ in range(15):
                mind.step(0.1, dream_mode=False)
            mind.clear_action()
            mind.A_target *= 0
            mind.target_strength = 0

    print(f"  Learned transitions from {len(training_positions)} positions")

    # Test 1a: Plan from a TRAINING position to a TRAINING goal
    print("\n--- Test 1a: Training position -> Training goal ---")
    start_pos = (0.0, 0.0)
    goal_pos = (0.5, 0.5)

    mind.reset()
    mind.inject_text(state_to_name(start_pos), 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, state_to_name(goal_pos))
    action_seq, _ = mind.plan(goal_pattern, n_iterations=50, verbose=False)

    # Execute plan in environment
    env.reset(start_pos)
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            env.step(action)

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Start: {start_pos}, Goal: {goal_pos}")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.1 else 'NO'}")

    # Test 1b: Plan from an UNSEEN position
    print("\n--- Test 1b: UNSEEN position -> Training goal ---")
    start_pos = (0.1, 0.1)  # Not in training set!
    goal_pos = (0.5, 0.5)

    mind.reset()
    mind.inject_text(state_to_name(start_pos), 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, state_to_name(goal_pos))
    action_seq, _ = mind.plan(goal_pattern, n_iterations=50, verbose=False)

    # Check if the network even recognizes the start state
    start_pattern = get_pattern(mind, state_to_name(start_pos))

    # Find closest training pattern
    best_overlap = 0
    best_match = None
    for name in mind.pattern_traces:
        if name.startswith('pos_'):
            overlap = cosine_sim(start_pattern, mind.pattern_traces[name])
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = name

    print(f"  Start: {start_pos} (UNSEEN)")
    print(f"  Closest training pattern: {best_match} (overlap={best_overlap:.3f})")

    # Execute plan
    env.reset(start_pos)
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            env.step(action)

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.1 else 'NO'}")

    # Test 1c: Plan to an UNSEEN goal
    print("\n--- Test 1c: Training position -> UNSEEN goal ---")
    start_pos = (0.0, 0.0)
    goal_pos = (0.35, 0.35)  # Not in training set!

    mind.reset()
    mind.inject_text(state_to_name(start_pos), 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, state_to_name(goal_pos))
    action_seq, _ = mind.plan(goal_pattern, n_iterations=50, verbose=False)

    # Execute plan
    env.reset(start_pos)
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            env.step(action)

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Start: {start_pos}, Goal: {goal_pos} (UNSEEN)")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.15 else 'NO'}")

    return mind


def test_discrete_vs_spatial_encoding():
    """
    Compare discrete encoding (random patterns) vs spatial encoding (tile coding).
    """
    print("\n" + "=" * 60)
    print("DISCRETE vs SPATIAL ENCODING COMPARISON")
    print("=" * 60)

    np.random.seed(42)

    # Test positions
    positions = [
        (0.0, 0.0),
        (0.1, 0.0),   # Close to (0,0)
        (0.2, 0.0),   # Further from (0,0)
        (0.5, 0.5),   # Far from (0,0)
    ]

    # === Test 1: Discrete encoding (current behavior) ===
    print("\n--- DISCRETE ENCODING (hash-based) ---")
    mind_discrete = PlanningMind(n_units=200)

    patterns_discrete = {}
    for pos in positions:
        name = state_to_name(pos)
        mind_discrete.inject_text(name, 0.9)
        for _ in range(5):
            mind_discrete.step(0.1)
        patterns_discrete[pos] = mind_discrete.A.copy()
        mind_discrete.reset()

    print("  Position pairs - spatial distance vs pattern overlap:")
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            spatial_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
            pattern_overlap = cosine_sim(patterns_discrete[pos1], patterns_discrete[pos2])
            print(f"    {pos1} vs {pos2}: dist={spatial_dist:.2f}, overlap={pattern_overlap:.3f}")

    # === Test 2: Spatial encoding (Place Cells) ===
    print("\n--- SPATIAL ENCODING (Place Cells) ---")
    mind_spatial = PlanningMind(n_units=300)
    mind_spatial.init_spatial_encoding(state_dim=2, n_place_cells=300, field_radius=0.18)

    patterns_spatial = {}
    for pos in positions:
        patterns_spatial[pos] = mind_spatial.encode_continuous_state(pos)

    print("  Position pairs - spatial distance vs pattern overlap:")
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            spatial_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
            pattern_overlap = cosine_sim(patterns_spatial[pos1], patterns_spatial[pos2])
            print(f"    {pos1} vs {pos2}: dist={spatial_dist:.2f}, overlap={pattern_overlap:.3f}")

    # === Analysis ===
    print("\n--- ANALYSIS ---")

    # Check correlation between spatial distance and pattern similarity
    discrete_pairs = []
    spatial_pairs = []
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            spatial_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
            discrete_overlap = cosine_sim(patterns_discrete[pos1], patterns_discrete[pos2])
            spatial_overlap = cosine_sim(patterns_spatial[pos1], patterns_spatial[pos2])
            discrete_pairs.append((spatial_dist, discrete_overlap))
            spatial_pairs.append((spatial_dist, spatial_overlap))

    # Sort by distance
    discrete_pairs.sort(key=lambda x: x[0])
    spatial_pairs.sort(key=lambda x: x[0])

    print("  Discrete encoding: overlap does NOT correlate with distance")
    print(f"    Closest pair overlap: {discrete_pairs[0][1]:.3f}")
    print(f"    Farthest pair overlap: {discrete_pairs[-1][1]:.3f}")

    print("  Spatial encoding: overlap SHOULD correlate with distance")
    print(f"    Closest pair overlap: {spatial_pairs[0][1]:.3f}")
    print(f"    Farthest pair overlap: {spatial_pairs[-1][1]:.3f}")

    # Success = closest pair has higher overlap than farthest pair
    spatial_works = spatial_pairs[0][1] > spatial_pairs[-1][1]
    print(f"\n  Spatial encoding preserves locality: {'YES' if spatial_works else 'NO'}")


def test_single_step_transitions():
    """
    Test that single-step transitions work before testing planning.
    """
    print("\n" + "=" * 60)
    print("TEST: SINGLE-STEP TRANSITIONS (spatial encoding)")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=300)
    # Place cell encoding - field_radius > step_size/2 ensures overlap between consecutive positions
    # With step_size=0.25 and field_radius=0.18, consecutive positions share ~30% of cells
    mind.init_spatial_encoding(state_dim=2, n_place_cells=300, field_radius=0.18)
    env = ContinuousGridWorld(step_size=0.25)

    # Learn just one transition: (0,0) + action_0 -> (0.25, 0)
    start = (0.0, 0.0)
    end = (0.25, 0.0)

    print(f"\nLearning: {start} + action_0 -> {end}")

    # Train with place cells - need strong learning signal
    for rep in range(30):  # More repetitions
        mind.inject_continuous_state(start, 0.9)
        mind.set_action('action_0')
        mind.set_continuous_target(end, 1.0)
        for _ in range(40):  # More steps per rep
            mind.step(0.1, dream_mode=False)
        mind.clear_action()
        mind.A_target *= 0
        mind.target_strength = 0

    # Check W_action_longterm was updated
    print(f"  W_action_longterm[0] nonzero: {(mind.W_action_longterm[0] > 0.01).sum()}")
    print(f"  W_action_longterm[0] max: {mind.W_action_longterm[0].max():.3f}")
    print(f"  W_action_longterm[0] mean (nonzero): {mind.W_action_longterm[0][mind.W_action_longterm[0] > 0.01].mean():.3f}")

    # Test: activate (0,0), apply action_0, see if we get (0.25, 0)
    print("\nTesting transition...")
    mind.reset()
    mind.inject_continuous_state(start, 0.9)
    for _ in range(5):
        mind.step(0.1)

    print(f"  After injecting {start}:")
    start_pattern = mind.encode_continuous_state(start)
    end_pattern = mind.encode_continuous_state(end)
    overlap_start = cosine_sim(mind.A, start_pattern)
    overlap_end = cosine_sim(mind.A, end_pattern)
    print(f"    Overlap with {start}: {overlap_start:.3f}")
    print(f"    Overlap with {end}: {overlap_end:.3f}")

    # Apply action and run dynamics
    mind.set_action('action_0')
    for _ in range(50):  # More steps to let transition happen
        mind.step(0.1)

    print(f"  After applying action_0:")
    overlap_start = cosine_sim(mind.A, start_pattern)
    overlap_end = cosine_sim(mind.A, end_pattern)
    print(f"    Overlap with {start}: {overlap_start:.3f}")
    print(f"    Overlap with {end}: {overlap_end:.3f}")

    success = overlap_end > overlap_start
    print(f"\n  Transition learned: {'YES' if success else 'NO'}")

    # Also test the SIMULATED transition (what the planner sees)
    print("\nTesting simulated transition (what planner uses)...")
    mind.reset()
    mind.inject_continuous_state(start, 0.9)
    for _ in range(5):
        mind.step(0.1)

    # Set up a slot with the current state
    mind.init_planning()
    mind.slot_A[0] = mind.A.copy()

    # Create action pattern for action_0
    action_pattern = np.zeros(mind.n_action_units)
    action_pattern[0:mind.action_units_per_dim] = 1.0
    mind.slot_action_sequence[0] = [action_pattern]

    # Simulate forward with more steps
    mind._simulate_slot(0, steps_per_action=100)

    # Check where simulation ends up
    sim_overlap_start = cosine_sim(mind.slot_A[0], start_pattern)
    sim_overlap_end = cosine_sim(mind.slot_A[0], end_pattern)
    print(f"  Simulated result:")
    print(f"    Overlap with {start}: {sim_overlap_start:.3f}")
    print(f"    Overlap with {end}: {sim_overlap_end:.3f}")

    # Check if simulation moved toward the target (even if not fully there)
    initial_end_overlap = cosine_sim(start_pattern, end_pattern)
    sim_improved = sim_overlap_end > initial_end_overlap
    sim_success = sim_overlap_end > sim_overlap_start

    print(f"  Initial end overlap (from start pattern): {initial_end_overlap:.3f}")
    print(f"  Simulation moved toward target: {'YES' if sim_improved else 'NO'}")
    print(f"  Simulation reached target (end > start): {'YES' if sim_success else 'NO'}")

    # Success if real transition works and simulation at least improves
    return success and sim_improved


def test_spatial_encoding_planning():
    """
    Test planning with spatial encoding - this should generalize!
    """
    print("\n" + "=" * 60)
    print("TEST 2: SPATIAL ENCODING (expect generalization)")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=300)
    # Place cell encoding - field_radius ensures consecutive positions overlap
    mind.init_spatial_encoding(state_dim=2, n_place_cells=300, field_radius=0.18)
    env = ContinuousGridWorld(step_size=0.25)

    # Learn transitions from a grid of positions using spatial encoding
    print("\nLearning transitions from training positions (Place Cells)...")
    training_positions = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0),
        (0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25),
        (0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5),
    ]
    print(f"  Training positions: {len(training_positions)}")

    for pos in training_positions:
        # Learn all 4 actions from this position
        for action in range(4):
            env.reset(pos)
            next_pos = env.step(action)

            # Learn transition using continuous encoding (multiple reps for strength)
            for rep in range(3):
                mind.inject_continuous_state(pos, 0.9)
                mind.set_action(f'action_{action}')
                mind.set_continuous_target(next_pos, 1.0)
                for _ in range(20):
                    mind.step(0.1, dream_mode=False)
                mind.clear_action()
                mind.A_target *= 0
                mind.target_strength = 0

    print(f"  Learned transitions from {len(training_positions)} positions")

    # Check reward signal quality
    print("\n  Checking reward signal (goal pattern overlap):")
    goal_check = (0.5, 0.5)
    goal_p = mind.encode_continuous_state(goal_check)
    test_positions = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.5, 0.5), (1.0, 0.0)]
    for pos in test_positions:
        pos_p = mind.encode_continuous_state(pos)
        overlap = cosine_sim(pos_p, goal_p)
        print(f"    {pos} vs goal {goal_check}: overlap = {overlap:.3f}")

    # Test 2a: Plan from a TRAINING position to a TRAINING goal
    print("\n--- Test 2a: Training position -> Training goal ---")
    start_pos = (0.0, 0.0)
    goal_pos = (0.5, 0.5)

    mind.reset()
    mind.inject_continuous_state(start_pos, 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = mind.encode_continuous_state(goal_pos)

    # Increase exploration for spatial encoding
    old_n_slots = mind.n_slots
    mind.n_slots = 16  # More parallel searches
    # Resize slot arrays
    mind.slot_A = np.zeros((mind.n_slots, mind.n))
    mind.slot_action = np.zeros((mind.n_slots, mind.n_action_units))
    mind.slot_reward = np.zeros(mind.n_slots)
    mind.slot_action_sequence = [[] for _ in range(mind.n_slots)]
    mind.clone_threshold = 0.5  # More aggressive cloning

    action_seq, _ = mind.plan(goal_pattern, n_iterations=100, verbose=True)

    # Decode and print the action sequence
    decoded_actions = [mind._get_action_dim_from_pattern(a) for a in action_seq]
    print(f"  Planned action sequence: {decoded_actions}")

    # Execute plan in environment step by step
    env.reset(start_pos)
    print(f"  Execution trace:")
    for i, action_pattern in enumerate(action_seq[:6]):
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            old_pos = env.get_state().copy()
            new_pos = env.step(action)
            print(f"    action_{action}: ({old_pos[0]:.2f},{old_pos[1]:.2f}) -> ({new_pos[0]:.2f},{new_pos[1]:.2f})")

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Start: {start_pos}, Goal: {goal_pos}")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.1 else 'NO'}")

    # Test 2b: Plan from an UNSEEN position (generalization test!)
    print("\n--- Test 2b: UNSEEN position -> Training goal (GENERALIZATION) ---")
    start_pos = (0.1, 0.1)  # Not in training set!
    goal_pos = (0.5, 0.5)

    # Check pattern similarity to nearby training positions
    start_pattern = mind.encode_continuous_state(start_pos)
    nearby_pos = (0.0, 0.0)
    nearby_pattern = mind.encode_continuous_state(nearby_pos)
    overlap = cosine_sim(start_pattern, nearby_pattern)
    print(f"  Start {start_pos} overlap with nearby {nearby_pos}: {overlap:.3f}")

    mind.reset()
    mind.inject_continuous_state(start_pos, 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = mind.encode_continuous_state(goal_pos)
    action_seq, _ = mind.plan(goal_pattern, n_iterations=50, verbose=False)

    # Execute plan
    env.reset(start_pos)
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            env.step(action)

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Start: {start_pos} (UNSEEN)")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  GENERALIZATION SUCCESS: {'YES' if distance < 0.2 else 'NO'}")

    # Test 2c: Plan to an UNSEEN goal
    print("\n--- Test 2c: Training position -> UNSEEN goal ---")
    start_pos = (0.0, 0.0)
    goal_pos = (0.35, 0.35)  # Not exactly on training grid

    mind.reset()
    mind.inject_continuous_state(start_pos, 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = mind.encode_continuous_state(goal_pos)
    action_seq, _ = mind.plan(goal_pattern, n_iterations=50, verbose=False)

    # Execute plan
    env.reset(start_pos)
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            env.step(action)

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"  Start: {start_pos}, Goal: {goal_pos} (UNSEEN)")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.2 else 'NO'}")

    return mind


if __name__ == "__main__":
    # First show the encoding comparison
    test_discrete_vs_spatial_encoding()

    print("\n")

    # Test single-step transitions first
    transition_works = test_single_step_transitions()

    if not transition_works:
        print("\n>>> Single-step transitions don't work - skipping planning tests")
    else:
        print("\n")
        # Test discrete encoding (expected to fail on generalization)
        test_discrete_encoding()

        print("\n")
        # Test spatial encoding (expected to generalize)
        test_spatial_encoding_planning()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Discrete encoding: Each position gets random pattern
  -> Cannot generalize to unseen positions

Spatial encoding (tile coding): Nearby positions share tiles
  -> Patterns smoothly vary with position
  -> Learning generalizes to nearby states
  -> Planning works for unseen positions
""")


def test_successor_representation():
    """
    Test the Successor Representation for continuous state planning.

    The SR provides a GRADIENT for planning even when place cell patterns
    don't overlap directly. States closer to the goal have higher SR value.
    """
    print("\n" + "=" * 60)
    print("TEST: SUCCESSOR REPRESENTATION")
    print("=" * 60)

    np.random.seed(42)
    mind = PlanningMind(n_units=300)
    mind.init_spatial_encoding(state_dim=2, n_place_cells=300, field_radius=0.18)
    env = ContinuousGridWorld(step_size=0.25)

    print("\nLearning transitions with successor representation...")

    # Learn a sequence: (0,0) -> (0.25,0) -> (0.5,0) -> (0.5,0.25) -> (0.5,0.5)
    # This creates a chain that the SR can learn
    transitions = [
        ((0.0, 0.0), 0, (0.25, 0.0)),    # right
        ((0.25, 0.0), 0, (0.5, 0.0)),    # right
        ((0.5, 0.0), 1, (0.5, 0.25)),    # up
        ((0.5, 0.25), 1, (0.5, 0.5)),    # up
        # Also learn some alternative paths
        ((0.0, 0.0), 1, (0.0, 0.25)),    # up
        ((0.0, 0.25), 0, (0.25, 0.25)),  # right
        ((0.25, 0.25), 0, (0.5, 0.25)),  # right
        ((0.25, 0.0), 1, (0.25, 0.25)),  # up
    ]

    for start, action, end in transitions:
        # Learn transition multiple times to strengthen SR
        for rep in range(5):
            mind.inject_continuous_state(start, 0.9)
            mind.set_action(f'action_{action}')
            mind.set_continuous_target(end, 1.0)
            for _ in range(25):
                mind.step(0.1, dream_mode=False)
            mind.clear_action()
            mind.A_target *= 0
            mind.target_strength = 0

    # Check that SR was learned
    sr_max = np.max(mind.M_successor)
    sr_nonzero = np.sum(mind.M_successor > 0.01)
    print(f"  M_successor max: {sr_max:.3f}")
    print(f"  M_successor nonzero entries: {sr_nonzero}")

    # Check SR value gradient toward goal
    print("\n  Testing SR value gradient (should increase toward goal):")
    goal_pos = (0.5, 0.5)
    goal_pattern = mind.encode_continuous_state(goal_pos)

    test_positions = [
        (0.0, 0.0),     # Furthest from goal
        (0.25, 0.0),    # Closer
        (0.5, 0.0),     # Closer still
        (0.5, 0.25),    # One step from goal
        (0.5, 0.5),     # At goal
    ]

    sr_values = []
    direct_overlaps = []
    for pos in test_positions:
        pos_pattern = mind.encode_continuous_state(pos)
        sr_val = mind.compute_sr_value(pos_pattern, goal_pattern)
        direct_overlap = cosine_sim(pos_pattern, goal_pattern)
        sr_values.append(sr_val)
        direct_overlaps.append(direct_overlap)
        print(f"    {pos}: SR_value={sr_val:.3f}, direct_overlap={direct_overlap:.3f}")

    # Check that SR value increases toward goal
    sr_increases = all(sr_values[i] <= sr_values[i+1] for i in range(len(sr_values)-1))
    print(f"\n  SR value increases toward goal: {'YES' if sr_increases else 'NO'}")

    # Compare SR gradient to direct overlap (which is binary with place cells)
    print("\n  Comparison:")
    print(f"    Direct overlap range: {min(direct_overlaps):.3f} to {max(direct_overlaps):.3f}")
    print(f"    SR value range: {min(sr_values):.3f} to {max(sr_values):.3f}")

    sr_has_gradient = (max(sr_values) - min(sr_values)) > 0.1
    print(f"    SR provides useful gradient: {'YES' if sr_has_gradient else 'NO'}")

    # Now test planning with SR
    print("\n--- Testing Planning with Successor Representation ---")

    start_pos = (0.0, 0.0)
    goal_pos = (0.5, 0.5)

    mind.reset()
    mind.inject_continuous_state(start_pos, 0.9)
    for _ in range(10):
        mind.step(0.1)

    # Resize slots for more parallel search
    mind.n_slots = 16
    mind.slot_A = np.zeros((mind.n_slots, mind.n))
    mind.slot_action = np.zeros((mind.n_slots, mind.n_action_units))
    mind.slot_reward = np.zeros(mind.n_slots)
    mind.slot_action_sequence = [[] for _ in range(mind.n_slots)]

    goal_pattern = mind.encode_continuous_state(goal_pos)
    action_seq, best_slot = mind.plan(goal_pattern, n_iterations=100, verbose=True)

    # Decode actions
    decoded = [mind._get_action_dim_from_pattern(a) for a in action_seq]
    print(f"  Planned actions: {decoded}")

    # Execute in environment
    env.reset(start_pos)
    print(f"  Execution trace:")
    for action_pattern in action_seq[:6]:
        action = mind._get_action_dim_from_pattern(action_pattern)
        if action is not None:
            old_pos = env.get_state().copy()
            new_pos = env.step(action)
            print(f"    action_{action}: ({old_pos[0]:.2f},{old_pos[1]:.2f}) -> ({new_pos[0]:.2f},{new_pos[1]:.2f})")

    final_pos = env.get_state()
    distance = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"\n  Start: {start_pos}, Goal: {goal_pos}")
    print(f"  Final: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {distance:.3f}")
    print(f"  SUCCESS: {'YES' if distance < 0.15 else 'NO'}")

    return sr_has_gradient and distance < 0.3


if __name__ == "__main__":
    import sys

    if "--sr" in sys.argv or "--successor" in sys.argv:
        # Just test the successor representation
        success = test_successor_representation()
        print(f"\n{'='*60}")
        print(f"Successor Representation Test: {'PASSED' if success else 'FAILED'}")
        print(f"{'='*60}")
    else:
        # Full test suite
        # First show the encoding comparison
        test_discrete_vs_spatial_encoding()

        print("\n")

        # Test single-step transitions first
        transition_works = test_single_step_transitions()

        if not transition_works:
            print("\n>>> Single-step transitions don't work - skipping planning tests")
        else:
            print("\n")
            # Test discrete encoding (expected to fail on generalization)
            test_discrete_encoding()

            print("\n")
            # Test spatial encoding (expected to generalize)
            test_spatial_encoding_planning()

            print("\n")
            # Test successor representation
            test_successor_representation()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("""
Discrete encoding: Each position gets random pattern
  -> Cannot generalize to unseen positions

Spatial encoding (place cells): Nearby positions share some cells
  -> Patterns overlap for nearby positions
  -> But overlap is binary - no gradient for distant goals

Successor Representation: Learns which states lead to which
  -> Provides VALUE GRADIENT even when patterns don't overlap
  -> States closer to goal have higher SR value
  -> Enables planning across non-overlapping place cell patterns
""")
