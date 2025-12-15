"""
Interactive testing of Successor Representation with continuous states.

Commands:
  pos <x> <y>       - Set current position (0-1 range)
  goal <x> <y>      - Set goal and plan
  learn <x1> <y1> <action> <x2> <y2>  - Learn a transition
  grid              - Learn a grid of transitions automatically
  sr <x> <y>        - Show SR value for a position relative to current goal
  status            - Show current state
  reset             - Reset the network
  quit              - Exit
"""
import numpy as np
from planning_soc import PlanningSOCMind as PlanningMind


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


class ContinuousInteractive:
    def __init__(self):
        print("Initializing mind with place cell encoding...")
        self.mind = PlanningMind(n_units=300, n_slots=16)
        self.mind.init_spatial_encoding(state_dim=2, n_place_cells=300, field_radius=0.18)

        self.current_pos = None
        self.goal_pos = None
        self.goal_pattern = None
        self.step_size = 0.25

        print(f"  Place cells: 300, field_radius: 0.18")
        print(f"  Step size: {self.step_size}")
        print()

    def set_position(self, x, y):
        """Set current position."""
        self.current_pos = (float(x), float(y))
        self.mind.reset()
        self.mind.inject_continuous_state(self.current_pos, 0.9)
        for _ in range(10):
            self.mind.step(0.1)
        print(f"  Position set to ({x:.2f}, {y:.2f})")
        self._show_sr_values()

    def set_goal(self, x, y):
        """Set goal and run planning."""
        self.goal_pos = (float(x), float(y))
        self.goal_pattern = self.mind.encode_continuous_state(self.goal_pos)

        if self.current_pos is None:
            print("  Error: Set position first with 'pos <x> <y>'")
            return

        print(f"  Goal set to ({x:.2f}, {y:.2f})")
        print()

        # Show SR values along potential paths
        self._show_sr_values()

        # Run planning
        print("\n  Planning...")
        self.mind.init_planning()
        action_seq, best_slot = self.mind.plan(self.goal_pattern, n_iterations=50, verbose=False)

        # Decode actions
        action_names = ['right', 'up', 'left', 'down']
        decoded = []
        for a in action_seq:
            dim = self.mind._get_action_dim_from_pattern(a)
            if dim is not None:
                decoded.append(action_names[dim])

        print(f"  Planned actions: {decoded}")

        # Simulate execution
        print(f"\n  Simulated execution:")
        pos = list(self.current_pos)
        for i, action in enumerate(decoded[:6]):
            old_pos = pos.copy()
            if action == 'right':
                pos[0] = min(1.0, pos[0] + self.step_size)
            elif action == 'left':
                pos[0] = max(0.0, pos[0] - self.step_size)
            elif action == 'up':
                pos[1] = min(1.0, pos[1] + self.step_size)
            elif action == 'down':
                pos[1] = max(0.0, pos[1] - self.step_size)
            print(f"    {action}: ({old_pos[0]:.2f},{old_pos[1]:.2f}) -> ({pos[0]:.2f},{pos[1]:.2f})")

        distance = np.sqrt((pos[0] - self.goal_pos[0])**2 + (pos[1] - self.goal_pos[1])**2)
        print(f"\n  Final position: ({pos[0]:.2f}, {pos[1]:.2f})")
        print(f"  Distance to goal: {distance:.3f}")
        print(f"  Success: {'YES' if distance < 0.1 else 'NO'}")

    def learn_transition(self, x1, y1, action, x2, y2):
        """Learn a single transition."""
        start = (float(x1), float(y1))
        end = (float(x2), float(y2))
        action_dim = int(action)

        for rep in range(5):
            self.mind.inject_continuous_state(start, 0.9)
            self.mind.set_action(f'action_{action_dim}')
            self.mind.set_continuous_target(end, 1.0)
            for _ in range(25):
                self.mind.step(0.1, dream_mode=False)
            self.mind.clear_action()
            self.mind.A_target *= 0
            self.mind.target_strength = 0

        action_names = ['right', 'up', 'left', 'down']
        print(f"  Learned: ({x1:.2f},{y1:.2f}) + {action_names[action_dim]} -> ({x2:.2f},{y2:.2f})")

    def learn_grid(self):
        """Learn transitions for a grid of positions."""
        print("  Learning grid transitions...")
        positions = []
        for x in np.arange(0, 1.01, self.step_size):
            for y in np.arange(0, 1.01, self.step_size):
                positions.append((round(x, 2), round(y, 2)))

        count = 0
        for pos in positions:
            x, y = pos
            # Right (action 0)
            if x + self.step_size <= 1.0:
                self.learn_transition(x, y, 0, x + self.step_size, y)
                count += 1
            # Up (action 1)
            if y + self.step_size <= 1.0:
                self.learn_transition(x, y, 1, x, y + self.step_size)
                count += 1
            # Left (action 2)
            if x - self.step_size >= 0.0:
                self.learn_transition(x, y, 2, x - self.step_size, y)
                count += 1
            # Down (action 3)
            if y - self.step_size >= 0.0:
                self.learn_transition(x, y, 3, x, y - self.step_size)
                count += 1

        print(f"  Learned {count} transitions from {len(positions)} positions")
        print(f"  SR matrix max: {np.max(self.mind.M_successor):.3f}")
        print(f"  SR matrix nonzero: {np.sum(self.mind.M_successor > 0.01)}")

    def _show_sr_values(self):
        """Show SR values for key positions."""
        if self.goal_pattern is None:
            return

        print(f"\n  SR values toward goal {self.goal_pos}:")
        test_positions = [
            (0.0, 0.0), (0.25, 0.0), (0.5, 0.0),
            (0.0, 0.25), (0.25, 0.25), (0.5, 0.25),
            (0.0, 0.5), (0.25, 0.5), (0.5, 0.5),
        ]

        for pos in test_positions:
            pattern = self.mind.encode_continuous_state(pos)
            sr_val = self.mind.compute_sr_value(pattern, self.goal_pattern)
            direct = cosine_sim(pattern, self.goal_pattern)
            marker = " <-- current" if self.current_pos == pos else ""
            marker = " <-- GOAL" if self.goal_pos == pos else marker
            print(f"    ({pos[0]:.2f},{pos[1]:.2f}): SR={sr_val:.3f}, direct={direct:.3f}{marker}")

    def show_sr(self, x, y):
        """Show SR value for a specific position."""
        if self.goal_pattern is None:
            print("  Error: Set goal first with 'goal <x> <y>'")
            return

        pos = (float(x), float(y))
        pattern = self.mind.encode_continuous_state(pos)
        sr_val = self.mind.compute_sr_value(pattern, self.goal_pattern)
        direct = cosine_sim(pattern, self.goal_pattern)
        print(f"  Position ({x:.2f},{y:.2f}):")
        print(f"    SR value: {sr_val:.3f}")
        print(f"    Direct overlap: {direct:.3f}")

    def show_status(self):
        """Show current status."""
        print(f"  Current position: {self.current_pos}")
        print(f"  Goal position: {self.goal_pos}")
        print(f"  SR matrix max: {np.max(self.mind.M_successor):.3f}")
        print(f"  SR matrix nonzero: {np.sum(self.mind.M_successor > 0.01)}")
        print(f"  Experience buffer: {len(self.mind.experience_buffer)}")

    def run(self):
        """Main interactive loop."""
        print("=" * 60)
        print("CONTINUOUS STATE INTERACTIVE MODE")
        print("=" * 60)
        print("""
Commands:
  pos <x> <y>       - Set current position (0-1 range)
  goal <x> <y>      - Set goal and plan
  learn <x1> <y1> <action> <x2> <y2>  - Learn transition (action: 0=right,1=up,2=left,3=down)
  grid              - Learn a grid of transitions automatically
  sr <x> <y>        - Show SR value for a position
  status            - Show current state
  reset             - Reset the network
  quit              - Exit

Example workflow:
  grid              # Learn all transitions
  pos 0 0           # Set starting position
  goal 0.5 0.5      # Set goal and plan
""")

        while True:
            try:
                line = input("> ").strip()
                if not line:
                    continue

                parts = line.split()
                cmd = parts[0].lower()

                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'pos' and len(parts) >= 3:
                    self.set_position(float(parts[1]), float(parts[2]))
                elif cmd == 'goal' and len(parts) >= 3:
                    self.set_goal(float(parts[1]), float(parts[2]))
                elif cmd == 'learn' and len(parts) >= 6:
                    self.learn_transition(float(parts[1]), float(parts[2]),
                                         int(parts[3]),
                                         float(parts[4]), float(parts[5]))
                elif cmd == 'grid':
                    self.learn_grid()
                elif cmd == 'sr' and len(parts) >= 3:
                    self.show_sr(float(parts[1]), float(parts[2]))
                elif cmd == 'status':
                    self.show_status()
                elif cmd == 'reset':
                    self.__init__()
                else:
                    print("  Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\n  Use 'quit' to exit.")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    app = ContinuousInteractive()
    app.run()
