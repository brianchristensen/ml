"""
Structural HRR Model for Semantic Composition

Key innovation: Use HRR role-filler bindings to represent logical structures,
not just sequences. Learn how to compose predicate structures from examples.

Goal: Prove HRR can learn compositional predicate logic from natural language.

Architecture:
1. Encode logical forms as HRR structures using role bindings
2. Store (sentence_hrr, structure_hrr) pairs
3. Learn structural composition operations
4. Retrieve and compose based on structural similarity

Example encoding:
  "dog(x_1)" → bind(PRED, dog) + bind(ARG1, x_1)
  "love.agent(x, y)" → bind(PRED, love) + bind(ROLE, agent) + bind(ARG1, x) + bind(ARG2, y)
"""

import torch
import torch.nn as nn
import re
from collections import defaultdict


class HRROps:
    """HRR operations for structural composition."""

    @staticmethod
    def bind(a, b):
        """Circular convolution (binding)."""
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
        """Cosine similarity in complex space."""
        a_norm = a / (torch.norm(a) + 1e-8)
        b_norm = b / (torch.norm(b) + 1e-8)
        return (a_norm.conj() * b_norm).sum().real.item()


class StructuralVocab:
    """Vocabulary with role vectors for structural encoding."""

    def __init__(self, dim=2048, seed=42):
        torch.manual_seed(seed)
        self.dim = dim
        self.hrr = HRROps()

        # Core role vectors for structural composition
        self.roles = {
            'PRED': self._random_complex(),      # Predicate
            'ARG1': self._random_complex(),      # First argument
            'ARG2': self._random_complex(),      # Second argument
            'ARG3': self._random_complex(),      # Third argument
            'ROLE': self._random_complex(),      # Thematic role (agent/theme/etc)
            'LEFT': self._random_complex(),      # Left child in tree
            'RIGHT': self._random_complex(),     # Right child in tree
            'OP': self._random_complex(),        # Operator (AND, semicolon, etc)
        }

        # Symbol vectors (learned from data)
        self.symbols = {}  # word/predicate -> vector

    def _random_complex(self):
        """Generate random unit complex vector."""
        real = torch.randn(self.dim)
        imag = torch.randn(self.dim)
        vec = torch.complex(real, imag)
        return vec / (torch.norm(vec) + 1e-8)

    def get_symbol(self, symbol):
        """Get or create vector for a symbol."""
        if symbol not in self.symbols:
            self.symbols[symbol] = self._random_complex()
        return self.symbols[symbol]

    def encode_structure(self, structure):
        """
        Encode a logical form structure as HRR vector.

        Structure types:
        - ('atom', predicate, arg) → simple predicate
        - ('role', predicate, role, arg1, arg2) → predicate with role
        - ('and', left_struct, right_struct) → conjunction
        - ('seq', left_struct, right_struct) → sequence (semicolon)
        """
        if structure is None or structure == []:
            return torch.zeros(self.dim, dtype=torch.complex64)

        struct_type = structure[0]

        if struct_type == 'atom':
            # Simple: predicate(arg)
            _, pred, arg = structure
            return (self.hrr.bind(self.roles['PRED'], self.get_symbol(pred)) +
                   self.hrr.bind(self.roles['ARG1'], self.get_symbol(arg)))

        elif struct_type == 'role':
            # Role-based: predicate.role(arg1, arg2)
            _, pred, role, arg1, arg2 = structure
            return (self.hrr.bind(self.roles['PRED'], self.get_symbol(pred)) +
                   self.hrr.bind(self.roles['ROLE'], self.get_symbol(role)) +
                   self.hrr.bind(self.roles['ARG1'], self.get_symbol(arg1)) +
                   self.hrr.bind(self.roles['ARG2'], self.get_symbol(arg2)))

        elif struct_type == 'and':
            # Conjunction
            _, left, right = structure
            left_vec = self.encode_structure(left)
            right_vec = self.encode_structure(right)
            return (self.hrr.bind(self.roles['OP'], self.get_symbol('AND')) +
                   self.hrr.bind(self.roles['LEFT'], left_vec) +
                   self.hrr.bind(self.roles['RIGHT'], right_vec))

        elif struct_type == 'seq':
            # Sequence (semicolon)
            _, left, right = structure
            left_vec = self.encode_structure(left)
            right_vec = self.encode_structure(right)
            return (self.hrr.bind(self.roles['OP'], self.get_symbol('SEQ')) +
                   self.hrr.bind(self.roles['LEFT'], left_vec) +
                   self.hrr.bind(self.roles['RIGHT'], right_vec))

        else:
            return torch.zeros(self.dim, dtype=torch.complex64)


class StructuralHRRModel:
    """
    HRR model for learning structural composition.

    Tests if HRR can learn to compose predicate-logic structures.
    """

    def __init__(self, hrr_dim=2048):
        self.vocab = StructuralVocab(dim=hrr_dim)

        # Memory: (sentence_tokens, structure) pairs
        self.examples = []

        # Input encoding: sentence tokens -> HRR
        self.sentence_encodings = []
        self.structure_encodings = []

    def parse_cogs_output(self, output_str):
        """
        Parse COGS logical form into structured representation.

        Examples:
        - "dog(x_1)" → ('atom', 'dog', 'x_1')
        - "love.agent(x, y)" → ('role', 'love', 'agent', 'x', 'y')
        - "dog(x) AND cat(y)" → ('and', ('atom', 'dog', 'x'), ('atom', 'cat', 'y'))
        """
        output_str = output_str.strip()

        # Handle prefix operators (* and ;)
        if output_str.startswith('* '):
            # Leading definite marker - parse after marker
            _, rest = output_str.split(' ', 1)
            if '; ' in rest:
                parts = rest.split('; ')
                left_struct = self._parse_atom_or_role(parts[0])
                right_struct = self.parse_cogs_output(parts[1])
                return ('seq', left_struct, right_struct)
            else:
                return self._parse_atom_or_role(rest)

        # Handle AND operator
        if ' AND ' in output_str:
            parts = output_str.split(' AND ', 1)
            left_struct = self._parse_atom_or_role(parts[0])
            right_struct = self.parse_cogs_output(parts[1])
            return ('and', left_struct, right_struct)

        # Handle semicolon operator
        if ' ; ' in output_str:
            parts = output_str.split(' ; ', 1)
            left_struct = self._parse_atom_or_role(parts[0])
            right_struct = self.parse_cogs_output(parts[1])
            return ('seq', left_struct, right_struct)

        # Simple atom or role
        return self._parse_atom_or_role(output_str)

    def _parse_atom_or_role(self, expr):
        """Parse atomic expression or role-based predicate."""
        expr = expr.strip()

        # Match: predicate . role ( arg1 , arg2 ) with optional spaces
        role_match = re.match(r'(\w+)\s*\.\s*(\w+)\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', expr)
        if role_match:
            pred, role, arg1, arg2 = role_match.groups()
            return ('role', pred.strip(), role.strip(), arg1.strip(), arg2.strip())

        # Match: predicate ( arg ) with optional spaces
        atom_match = re.match(r'(\w+)\s*\(\s*([^)]+)\s*\)', expr)
        if atom_match:
            pred, arg = atom_match.groups()
            return ('atom', pred.strip(), arg.strip())

        return None

    def store(self, sentence_tokens, structure):
        """Store example as (sentence, structure) pair."""
        self.examples.append((sentence_tokens, structure))

        # Encode sentence (simple: bind tokens in sequence)
        sent_vec = torch.zeros(self.vocab.dim, dtype=torch.complex64)
        for i, token in enumerate(sentence_tokens):
            token_vec = self.vocab.get_symbol(token)
            # Position-based encoding
            pos_vec = self.vocab.get_symbol(f'POS_{i}')
            sent_vec = sent_vec + self.vocab.hrr.bind(pos_vec, token_vec)

        # Normalize
        norm = torch.norm(sent_vec)
        if norm > 1e-8:
            sent_vec = sent_vec / norm

        # Encode structure
        struct_vec = self.vocab.encode_structure(structure)

        self.sentence_encodings.append(sent_vec)
        self.structure_encodings.append(struct_vec)

    def retrieve(self, query_tokens):
        """Retrieve most similar structure."""
        if len(self.sentence_encodings) == 0:
            return None

        # Encode query
        query_vec = torch.zeros(self.vocab.dim, dtype=torch.complex64)
        for i, token in enumerate(query_tokens):
            token_vec = self.vocab.get_symbol(token)
            pos_vec = self.vocab.get_symbol(f'POS_{i}')
            query_vec = query_vec + self.vocab.hrr.bind(pos_vec, token_vec)

        norm = torch.norm(query_vec)
        if norm > 1e-8:
            query_vec = query_vec / norm

        # Find most similar
        best_sim = -1
        best_idx = 0

        for i, sent_vec in enumerate(self.sentence_encodings):
            sim = self.vocab.hrr.similarity(query_vec, sent_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return self.examples[best_idx][1]  # Return structure

    def train_on_dataset(self, train_data):
        """Train by storing all examples."""
        print(f"Storing {len(train_data)} examples...")

        stored = 0
        failed = 0

        for sentence_tokens, output_str in train_data[:1000]:  # Start with subset
            try:
                structure = self.parse_cogs_output(output_str)
                if structure is not None:
                    self.store(sentence_tokens, structure)
                    stored += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

        print(f"Stored {stored} examples, failed to parse {failed}")

    def forward(self, sentence_tokens):
        """Predict structure for sentence."""
        structure = self.retrieve(sentence_tokens)
        return structure
