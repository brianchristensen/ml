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
from model_baseline import MemoryAugmentedModel

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
        from model_baseline import MemoryAugmentedModel as BaselineModel

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
