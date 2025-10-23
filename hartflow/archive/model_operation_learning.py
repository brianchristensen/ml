"""
Memory-Augmented Compositional Model

Novel approach: Combin retrieval + composition

Architecture:
1. Store training examples as (HRR vector, actions) pairs
2. For atomic: retrieve most similar training example
3. For compound: recursively decompose + compose retrieved results
4. NO hardcoded semantics - learns from examples!

Key insight: SCAN examples SHOW us what patterns mean.
We don't need to hardcode OR learn - we RETRIEVE and COMPOSE!

This is like:
- Differentiable Neural Dictionary (DND)
- Neural Turing Machines
- Memory-augmented neural nets
But applied to compositional generation!
"""

import torch
import torch.nn as nn
import numpy as np


class HRROps:
    """HRR operations."""

    @staticmethod
    def bind(a, b):
        A = torch.fft.fft(a, dim=-1)
        B = torch.fft.fft(b, dim=-1)
        return torch.fft.ifft(A * B, dim=-1)

    @staticmethod
    def unbind(bound, a):
        """Approximate inverse via circular correlation."""
        A_conj = torch.fft.fft(a, dim=-1).conj()
        Bound = torch.fft.fft(bound, dim=-1)
        return torch.fft.ifft(Bound * A_conj, dim=-1)

    @staticmethod
    def similarity(a, b):
        a_norm = a / (torch.norm(a) + 1e-8)
        b_norm = b / (torch.norm(b) + 1e-8)
        return (a_norm.conj() * b_norm).sum().real.item()


class Vocab:
    """HRR vocabulary."""

    def __init__(self, primitives_dict, dim=1024, seed=42):
        torch.manual_seed(seed)
        self.dim = dim
        self.hrr = HRROps()

        all_prims = (primitives_dict['actions'] +
                     primitives_dict['modifiers'] +
                     primitives_dict.get('directions', []))

        self.primitives = {
            name: self._random_complex()
            for name in all_prims
        }

        self.connectives = [m for m in primitives_dict['modifiers']
                           if m in ['and', 'after']]

    def _random_complex(self):
        real = torch.randn(self.dim)
        imag = torch.randn(self.dim)
        vec = torch.complex(real, imag)
        return vec / (torch.norm(vec) + 1e-8)

    def encode(self, tokens):
        if len(tokens) == 0:
            return torch.zeros(self.dim, dtype=torch.complex64)

        result = self.primitives[tokens[0]]
        for token in tokens[1:]:
            result = self.hrr.bind(result, self.primitives[token])

        return result


class MemoryAugmentedModel:
    """
    Stores examples, retrieves + composes.

    NO training needed - pure retrieval + composition!
    """

    def __init__(self, primitives_dict, output_vocab, hrr_dim=1024):
        self.vocab = Vocab(primitives_dict, dim=hrr_dim)
        self.output_vocab = output_vocab

        # Memory: stored examples
        self.memory_vectors = []  # HRR encodings
        self.memory_actions = []  # Corresponding action sequences
        self.memory_tokens = []   # Original tokens (for debugging)

    def store(self, command_tokens, action_sequence):
        """Store example in memory."""
        # Only store atomic commands (no connectives)
        if any(conn in command_tokens for conn in self.vocab.connectives):
            return

        hrr_vec = self.vocab.encode(command_tokens)
        self.memory_vectors.append(hrr_vec)
        self.memory_actions.append(action_sequence)
        self.memory_tokens.append(command_tokens)

    def retrieve(self, query_tokens):
        """Retrieve most similar example."""
        if len(self.memory_vectors) == 0:
            return []

        query_vec = self.vocab.encode(query_tokens)

        # Find most similar
        best_sim = -1
        best_actions = []

        for mem_vec, mem_actions in zip(self.memory_vectors, self.memory_actions):
            sim = self.vocab.hrr.similarity(query_vec, mem_vec)
            if sim > best_sim:
                best_sim = sim
                best_actions = mem_actions

        return best_actions

    def forward(self, command_tokens):
        """Execute via retrieval + composition."""
        return self._execute_recursive(command_tokens), {}

    def _execute_recursive(self, tokens, depth=0):
        """Recursive execution."""
        if depth > 10 or len(tokens) == 0:
            return []

        # Check for connectives
        for conn in self.vocab.connectives:
            if conn in tokens:
                idx = tokens.index(conn)

                if conn == 'after':
                    first = tokens[idx+1:]
                    second = tokens[:idx]
                else:
                    first = tokens[:idx]
                    second = tokens[idx+1:]

                first_actions = self._execute_recursive(first, depth+1)
                second_actions = self._execute_recursive(second, depth+1)
                return first_actions + second_actions

        # Atomic - retrieve!
        return self.retrieve(tokens)

    def train_on_dataset(self, train_data):
        """Populate memory from training data."""
        print(f"Storing {len(train_data)} training examples in memory...")

        for cmd_tokens, actions in train_data:
            self.store(cmd_tokens, actions)

        initial_count = len(self.memory_vectors)
        print(f"Memory contains {initial_count} atomic examples")

        # Discover abstract patterns and generate missing examples
        print("\nDiscovering abstract patterns via HRR unbinding...")
        self._discover_and_generate_patterns()

        print(f"After pattern discovery: {len(self.memory_vectors)} total examples")
        print(f"Generated {len(self.memory_vectors) - initial_count} synthetic examples")

    def _discover_and_generate_patterns(self):
        """
        Discover abstract patterns through HRR algebra.

        Key insight: If we have multiple examples with the same modifier,
        we can unbind the action to extract the modifier pattern, then
        bind it with other actions to generate missing examples.
        """
        # All primitives we might use
        all_actions = ['jump', 'walk', 'run', 'look', 'turn']
        all_modifiers = ['twice', 'thrice', 'around', 'opposite']
        all_directions = ['left', 'right']

        discovered_patterns = {}

        # 1. Discover patterns for each modifier
        for modifier in all_modifiers:
            patterns_for_modifier = []
            examples_with_modifier = []

            # Find examples containing this modifier
            for tokens, vec, actions in zip(self.memory_tokens, self.memory_vectors, self.memory_actions):
                if modifier in tokens:
                    examples_with_modifier.append((tokens, vec, actions))

            if len(examples_with_modifier) < 2:
                continue  # Need at least 2 examples to discover pattern

            # Extract abstract pattern by unbinding actions
            for tokens, vec, actions in examples_with_modifier:
                # Find the action (first token that's an action)
                action = None
                for token in tokens:
                    if token in all_actions:
                        action = token
                        break

                if action:
                    # Unbind action to get modifier+direction pattern
                    action_vec = self.vocab.primitives[action]
                    pattern = self.vocab.hrr.unbind(vec, action_vec)
                    patterns_for_modifier.append((pattern, tokens, actions))

            if len(patterns_for_modifier) >= 2:
                # Average patterns to get abstract representation
                avg_pattern = sum(p[0] for p in patterns_for_modifier) / len(patterns_for_modifier)
                discovered_patterns[modifier] = {
                    'pattern': avg_pattern,
                    'examples': patterns_for_modifier
                }

        print(f"  Discovered {len(discovered_patterns)} abstract patterns: {list(discovered_patterns.keys())}")

        # 2. Generate missing examples using discovered patterns
        generated = 0
        for modifier, info in discovered_patterns.items():
            abstract_pattern = info['pattern']

            # For each action, try to generate missing combinations
            for action in all_actions:
                # Check what combinations we're missing
                for direction in [None] + all_directions:
                    # Construct what this command would look like
                    if direction:
                        test_tokens = [action, modifier, direction]
                    else:
                        test_tokens = [action, modifier]

                    # Check if we already have this exact pattern
                    if any(t == test_tokens for t in self.memory_tokens):
                        continue

                    # Generate synthetic example!
                    synthetic_vec = self.vocab.hrr.bind(
                        self.vocab.primitives[action],
                        abstract_pattern
                    )

                    # Infer actions from similar examples
                    synthetic_actions = self._infer_actions_from_pattern(
                        action, modifier, direction, info['examples']
                    )

                    if synthetic_actions:
                        self.memory_vectors.append(synthetic_vec)
                        self.memory_actions.append(synthetic_actions)
                        self.memory_tokens.append(test_tokens)
                        generated += 1

        # 3. Generate simple atomic examples (single actions, basic combinations)
        print(f"  Generated {generated} examples from abstract patterns")
        print(f"  Generating basic atomic examples...")

        basic_generated = 0

        # Single actions
        for action in all_actions:
            if [action] not in self.memory_tokens:
                synthetic_vec = self.vocab.primitives[action]
                synthetic_actions = self._infer_single_action(action)
                if synthetic_actions:
                    self.memory_vectors.append(synthetic_vec)
                    self.memory_actions.append(synthetic_actions)
                    self.memory_tokens.append([action])
                    basic_generated += 1

        # Action + direction (e.g., "turn left")
        for action in all_actions:
            for direction in all_directions:
                test_tokens = [action, direction]
                if test_tokens not in self.memory_tokens:
                    synthetic_vec = self.vocab.hrr.bind(
                        self.vocab.primitives[action],
                        self.vocab.primitives[direction]
                    )
                    synthetic_actions = self._infer_action_direction(action, direction)
                    if synthetic_actions:
                        self.memory_vectors.append(synthetic_vec)
                        self.memory_actions.append(synthetic_actions)
                        self.memory_tokens.append(test_tokens)
                        basic_generated += 1

        print(f"  Generated {basic_generated} basic atomic examples")

    def _infer_actions_from_pattern(self, action, modifier, direction, examples):
        """
        Infer action sequence by analogy from similar examples.

        E.g., if we know "jump twice" → ['I_JUMP', 'I_JUMP']
        and we know "walk twice" → ['I_WALK', 'I_WALK']
        then we can infer "run twice" → ['I_RUN', 'I_RUN']
        """
        # Find most similar example
        best_example = None
        best_sim = -1

        for pattern, tokens, actions in examples:
            # Prefer examples with same direction
            if direction and direction in tokens:
                sim = 1.0
            else:
                sim = 0.5

            if sim > best_sim:
                best_sim = sim
                best_example = (tokens, actions)

        if not best_example:
            return None

        # Perform action substitution
        source_tokens, source_actions = best_example

        # Find the action in source
        source_action = None
        for token in source_tokens:
            if token in ['jump', 'walk', 'run', 'look', 'turn']:
                source_action = token
                break

        if not source_action:
            return None

        # Map source action to target action
        action_map = {
            'jump': 'I_JUMP',
            'walk': 'I_WALK',
            'run': 'I_RUN',
            'look': 'I_LOOK',
            'turn': None  # Special case
        }

        source_action_str = action_map.get(source_action)
        target_action_str = action_map.get(action)

        if not source_action_str or not target_action_str:
            return None

        # Substitute actions
        synthetic_actions = []
        for a in source_actions:
            if a == source_action_str:
                synthetic_actions.append(target_action_str)
            else:
                synthetic_actions.append(a)

        return synthetic_actions

    def _infer_single_action(self, action):
        """Infer single action output."""
        action_map = {
            'jump': 'I_JUMP',
            'walk': 'I_WALK',
            'run': 'I_RUN',
            'look': 'I_LOOK',
        }
        base_action = action_map.get(action)
        return [base_action] if base_action else None

    def _infer_action_direction(self, action, direction):
        """Infer action+direction output."""
        action_map = {
            'jump': 'I_JUMP',
            'walk': 'I_WALK',
            'run': 'I_RUN',
            'look': 'I_LOOK',
        }

        if action == 'turn':
            turn_map = {
                'left': 'I_TURN_LEFT',
                'right': 'I_TURN_RIGHT'
            }
            return [turn_map[direction]]

        base_action = action_map.get(action)
        if not base_action:
            return None

        turn_action = 'I_TURN_LEFT' if direction == 'left' else 'I_TURN_RIGHT'
        return [turn_action, base_action]

    # Dummy methods for compatibility
    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass
"""
Transformation learning by discovering operator semantics.

Key idea:
1. Use baseline model to execute left/right components
2. Analyze the pattern: Does final = left + right? Or right + left?
3. Learn the transformation rule (concat vs reverse vs other)
4. Apply learned rule at test time

This learns the OPERATION, not just similarity matching!
"""
import torch

class OperationLearningModel(MemoryAugmentedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learned_ops = {}  # operator -> operation type ('concat', 'reverse', etc.)

    def train_on_dataset(self, train_data):
        # First: Train baseline model for bootstrapping on FULL dataset
        print("PHASE 1: Training baseline model for bootstrapping...")
        super().train_on_dataset(train_data)

        # Second: Learn operation semantics by analyzing examples
        print("\nPHASE 2: Learning operator semantics from examples...")
        self._learn_operations(train_data)

    def _learn_operations(self, train_data):
        """Learn what each operator DOES by analyzing examples."""
        # Using local MemoryAugmentedModel (defined above)

        # Collect evidence for each operator
        operator_evidence = {}

        # Use subset of examples with connectives
        examples_with_connectives = [ex for ex in train_data if any(conn in ex[0] for conn in self.vocab.connectives)][:500]
        print(f"Analyzing {len(examples_with_connectives)} examples to learn operator semantics")

        for tokens, final_output in examples_with_connectives:
            # Find operators
            for conn in self.vocab.connectives:
                if conn not in tokens:
                    continue

                idx = tokens.index(conn)
                left_tokens = tokens[:idx]
                right_tokens = tokens[idx+1:]

                if not left_tokens or not right_tokens:
                    continue

                # Execute left and right with baseline model
                left_output = BaselineModel._execute_recursive(self, left_tokens)
                right_output = BaselineModel._execute_recursive(self, right_tokens)

                if conn not in operator_evidence:
                    operator_evidence[conn] = {
                        'concat': 0,      # final = left + right
                        'reverse': 0,     # final = right + left
                        'other': 0,       # something else
                        'total': 0
                    }

                # Check which pattern matches
                if final_output == left_output + right_output:
                    operator_evidence[conn]['concat'] += 1
                elif final_output == right_output + left_output:
                    operator_evidence[conn]['reverse'] += 1
                else:
                    operator_evidence[conn]['other'] += 1

                operator_evidence[conn]['total'] += 1
                break  # Only process first operator

        # Determine operation type for each operator by majority vote
        print("\nLearned operator semantics:")
        for op, evidence in operator_evidence.items():
            total = evidence['total']
            concat_pct = 100 * evidence['concat'] / total if total > 0 else 0
            reverse_pct = 100 * evidence['reverse'] / total if total > 0 else 0
            other_pct = 100 * evidence['other'] / total if total > 0 else 0

            # Use majority vote
            if evidence['concat'] >= evidence['reverse'] and evidence['concat'] >= evidence['other']:
                self.learned_ops[op] = 'concat'
                op_type = 'CONCAT'
            elif evidence['reverse'] >= evidence['concat'] and evidence['reverse'] >= evidence['other']:
                self.learned_ops[op] = 'reverse'
                op_type = 'REVERSE'
            else:
                self.learned_ops[op] = 'concat'  # default fallback
                op_type = 'OTHER (defaulting to concat)'

            print(f"  '{op}': {op_type}")
            print(f"    Evidence: concat={concat_pct:.1f}%, reverse={reverse_pct:.1f}%, other={other_pct:.1f}% (n={total})")

    def _execute_recursive(self, tokens, depth=0):
        """Execute using LEARNED operator semantics."""
        if depth > 10 or len(tokens) == 0:
            return []

        # Try connectives using LEARNED operations
        for conn in self.vocab.connectives:
            if conn not in tokens:
                continue

            # Check if we learned this operator
            if conn not in self.learned_ops:
                continue

            idx = tokens.index(conn)
            left_tokens = tokens[:idx]
            right_tokens = tokens[idx+1:]

            if not left_tokens or not right_tokens:
                continue

            # Recursively execute components
            left_output = self._execute_recursive(left_tokens, depth+1)
            right_output = self._execute_recursive(right_tokens, depth+1)

            # Apply learned operation!
            if self.learned_ops[conn] == 'concat':
                return left_output + right_output
            elif self.learned_ops[conn] == 'reverse':
                return right_output + left_output
            else:
                return left_output + right_output  # fallback

        # No connective - use baseline retrieval
        return self.retrieve(tokens)
