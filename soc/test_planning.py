"""
Debug script for planning_soc.py
Tests basic transitions and planning in non-interactive mode.
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind

def test_single_transition():
    """Test that a single learned transition actually works."""
    print("=" * 60)
    print("TEST 1: Single Transition (A + action_0 -> B)")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    # Learn A -> B with action 0
    print("\nLearning: room_A + action_0 -> room_B")
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

    # Test: inject A, set action_0, does it evolve to B?
    print("\nTesting transition...")
    mind.reset()
    mind.inject_text('room_A', 0.9)
    initial_A_overlap = mind.pattern_overlap('room_A')
    initial_B_overlap = mind.pattern_overlap('room_B')
    print(f"  Initial: A={initial_A_overlap:.3f}, B={initial_B_overlap:.3f}")

    mind.set_action('action_0')
    for step in range(50):
        mind.step(0.1)
        if step % 10 == 9:
            A_overlap = mind.pattern_overlap('room_A')
            B_overlap = mind.pattern_overlap('room_B')
            print(f"  Step {step+1}: A={A_overlap:.3f}, B={B_overlap:.3f}")

    final_B = mind.pattern_overlap('room_B')
    passed = final_B > 0.3
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (B overlap = {final_B:.3f})")
    return passed, mind


def test_two_step_path():
    """Test A -> B -> C path."""
    print("\n" + "=" * 60)
    print("TEST 2: Two-Step Path (A -> B -> C)")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    # Learn transitions
    print("\nLearning transitions...")

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
    print("  Learned: room_A + action_0 -> room_B")

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
    print("  Learned: room_B + action_1 -> room_C")

    # Test manual execution of path
    print("\nManual execution of path A -> B -> C...")
    mind.reset()
    mind.inject_text('room_A', 0.9)
    print(f"  Start: A={mind.pattern_overlap('room_A'):.3f}")

    # Step 1: action_0
    mind.set_action('action_0')
    for _ in range(30):
        mind.step(0.1)
    print(f"  After action_0: A={mind.pattern_overlap('room_A'):.3f}, B={mind.pattern_overlap('room_B'):.3f}")

    # Step 2: action_1
    mind.set_action('action_1')
    for _ in range(30):
        mind.step(0.1)
    print(f"  After action_1: B={mind.pattern_overlap('room_B'):.3f}, C={mind.pattern_overlap('room_C'):.3f}")

    final_C = mind.pattern_overlap('room_C')
    passed = final_C > 0.3
    print(f"\nManual path result: {'PASS' if passed else 'FAIL'} (C overlap = {final_C:.3f})")

    return passed, mind


def get_pattern(mind, name):
    """Get pattern for a name (same logic as pattern_overlap uses)."""
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


def test_slot_simulation():
    """Test that slot simulation correctly predicts transitions."""
    print("\n" + "=" * 60)
    print("TEST 3: Slot Simulation")
    print("=" * 60)

    # Use mind from test 2
    _, mind = test_two_step_path()

    print("\nTesting slot simulation...")
    mind.reset()
    mind.inject_text('room_A', 0.9)

    # Initialize planning
    mind.init_planning()

    # Manually set slot 0 to have action sequence [action_0, action_1]
    action_0 = np.zeros(mind.n_action_units)
    action_0[0:mind.action_units_per_dim] = 1.0

    action_1 = np.zeros(mind.n_action_units)
    action_1[mind.action_units_per_dim:2*mind.action_units_per_dim] = 1.0

    mind.slot_action_sequence[0] = [action_0, action_1]
    mind.slot_A[0] = mind.A.copy()

    print(f"  Slot 0 sequence: [action_0, action_1]")
    print(f"  Initial slot state A overlap: {mind.pattern_overlap('room_A'):.3f}")

    # Simulate slot 0
    mind._simulate_slot(0, steps_per_action=20)

    # Check where slot ended up using cosine similarity
    slot_state = mind.slot_A[0]
    pattern_A = get_pattern(mind, 'room_A')
    pattern_B = get_pattern(mind, 'room_B')
    pattern_C = get_pattern(mind, 'room_C')

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    A_sim = cosine_sim(slot_state, pattern_A)
    B_sim = cosine_sim(slot_state, pattern_B)
    C_sim = cosine_sim(slot_state, pattern_C)

    print(f"  After simulation: A={A_sim:.3f}, B={B_sim:.3f}, C={C_sim:.3f}")

    passed = C_sim > 0.2
    print(f"\nSlot simulation result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_fmc_planning():
    """Test full FMC planning."""
    print("\n" + "=" * 60)
    print("TEST 4: FMC Planning")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    # Learn transitions
    print("\nLearning transitions...")
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
    print("  Learned: A->B, B->C")

    # Set up for planning
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, 'room_C')
    print(f"\nStarting FMC planning from A to C...")
    print(f"  Initial goal overlap: {mind.pattern_overlap('room_C'):.3f}")

    # Run planning
    action_sequence, best_idx = mind.plan(goal_pattern, n_iterations=50, verbose=True)

    print(f"\n  Best slot: {best_idx}")
    print(f"  Action sequence length: {len(action_sequence)}")

    # Decode action sequence
    decoded = []
    for action in action_sequence:
        dim = mind._get_action_dim_from_pattern(action)
        decoded.append(f"action_{dim}")
    print(f"  Decoded sequence: {decoded}")

    # Execute the plan
    print(f"\nExecuting plan...")
    mind.reset()
    mind.inject_text('room_A', 0.9)

    for i, action in enumerate(action_sequence):
        dim = mind._get_action_dim_from_pattern(action)
        mind.set_action(f'action_{dim}')
        for _ in range(30):
            mind.step(0.1)
        print(f"  After action_{dim}: C overlap = {mind.pattern_overlap('room_C'):.3f}")

    final_C = mind.pattern_overlap('room_C')
    passed = final_C > 0.3
    print(f"\nFMC Planning result: {'PASS' if passed else 'FAIL'} (C overlap = {final_C:.3f})")
    return passed


def _get_pattern(mind, name):
    """Helper to get pattern."""
    if name in mind.pattern_traces:
        return mind.pattern_traces[name]
    else:
        np.random.seed(hash(name) % 10000)
        n_active = np.random.randint(20, 50)
        units = np.random.choice(mind.n, n_active, replace=False)
        pattern = np.zeros(mind.n)
        pattern[units] = 1.0
        return pattern


def test_continuous_planning():
    """Test continuous planning like the interactive visualizer does."""
    print("\n" + "=" * 60)
    print("TEST 5: Continuous Planning (simulating interactive mode)")
    print("=" * 60)

    mind = PlanningMind(n_units=200)

    # Learn transitions
    print("\nLearning: A->B, B->C")
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

    # Setup like interactive mode
    mind.reset()
    mind.inject_text('room_A', 0.9)
    for _ in range(10):
        mind.step(0.1)

    goal_pattern = get_pattern(mind, 'room_C')
    goal_start_state = mind.A.copy()

    mind.init_planning()

    print(f"\nStarting continuous planning from A to C...")
    print(f"  Initial goal overlap: {mind.pattern_overlap('room_C'):.3f}")

    # Simulate 100 frames of continuous planning + execution
    best_g = 0
    for frame in range(100):
        # Planning iteration (like _do_planning_iteration)
        planning_state = mind.A.copy() if np.mean(mind.A) > 0.05 else goal_start_state.copy()

        for i in range(mind.n_slots):
            mind.slot_A[i] = planning_state.copy()
        for i in range(mind.n_slots):
            mind._simulate_slot(i, steps_per_action=mind.plan_horizon)

        virtual_rewards, rewards, distances = mind._compute_virtual_rewards(goal_pattern)
        mind._clone_step(virtual_rewards)
        mind._perturb_actions()

        # Get best action
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]

        if mind.slot_action_sequence[best_idx]:
            best_action_pattern = mind.slot_action_sequence[best_idx][0]
            best_action_dim = mind._get_action_dim_from_pattern(best_action_pattern)

            if best_action_dim is not None:
                mind.set_action(f'action_{best_action_dim}')

        # Keep network alive
        if np.mean(mind.A) < 0.05:
            mind.A = np.clip(mind.A + 0.5 * goal_start_state, 0, 1)

        # Step network
        for _ in range(5):
            mind.step(0.1, dream_mode=False)

        g = mind.pattern_overlap('room_C')
        best_g = max(best_g, g)

        if frame % 20 == 0:
            decoded = [f"a{mind._get_action_dim_from_pattern(a)}" for a in mind.slot_action_sequence[best_idx][:3]]
            print(f"  Frame {frame}: g={g:.3f}, r={best_reward:.3f}, best_seq={decoded}")

        if g > 0.4:
            print(f"\n  GOAL REACHED at frame {frame}! g={g:.3f}")
            break

    passed = best_g > 0.4
    print(f"\nContinuous planning result: {'PASS' if passed else 'FAIL'} (best g = {best_g:.3f})")
    return passed


if __name__ == "__main__":
    print("Planning SOC Debug Tests")
    print("=" * 60)

    results = {}

    # Test 1: Single transition
    results['single_transition'], _ = test_single_transition()

    # Test 2: Two-step manual path
    results['two_step_path'], _ = test_two_step_path()

    # Test 3: Slot simulation
    results['slot_simulation'] = test_slot_simulation()

    # Test 4: Full FMC planning
    results['fmc_planning'] = test_fmc_planning()

    # Test 5: Continuous planning
    results['continuous_planning'] = test_continuous_planning()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = sum(results.values())
    print(f"\nTotal: {total}/{len(results)} passed")
