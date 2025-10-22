# HRR + Compositional Memory Architecture

## What We've Achieved

### Core Success: HRR Compositional Generalization Works!

**Test Results** (with only 1 stored example: "jump twice"):
- ✅ 'walk twice' → ['I_WALK', 'I_WALK'] [NOVEL - Compositional Generalization!]
- ✅ 'run twice' → ['I_RUN', 'I_RUN'] [NOVEL]
- ✅ 'look twice' → ['I_LOOK', 'I_LOOK'] [NOVEL]
- ✅ 'jump thrice' → ['I_JUMP', 'I_JUMP', 'I_JUMP'] [NOVEL - New modifier!]
- ✅ 'walk thrice' → ['I_WALK', 'I_WALK', 'I_WALK'] [NOVEL - Complete composition!]
- ✅ Single-word commands work ('jump', 'walk', 'run')

**Key Insight**: HRR unbinding enables **true compositional generalization**. By unbinding candidate primitives and checking similarity, we can decompose novel compositions into their constituents WITHOUT seeing them before.

---

## General Components (Domain-Agnostic)

### 1. HRR Operations (100% General)
- `bind(a, b)`: Circular convolution - creates compositional structure
- `unbind(bound, a)`: Approximate inverse - extracts components
- `superpose(vectors)`: Distributed representation
- `similarity(a, b)`: Cosine similarity for retrieval

**Why it's general**: These are mathematical operations that work on ANY vectors. No task-specific logic.

### 2. Compositional Encoding (General Structure)
```python
"walk twice" = bind(primitive['walk'], primitive['twice'])
```

**Why it's general**: The binding operation itself is domain-agnostic. Only the vocabulary is task-specific (which is expected - like word embeddings).

### 3. Similarity-Based Extraction (General Principle)
```python
# To extract action from bind(action, modifier):
for each candidate_modifier:
    unbound = unbind(program, candidate_modifier)
    similarity = cosine_sim(unbound, known_primitives)
```

**Why it's general**: This is a search procedure over primitives using HRR's algebraic properties.

---

## SCAN-Specific Components (Need Generalization)

### 1. Hardcoded Execution Logic ❌
```python
if modifier == "twice":
    return [action, action]
elif modifier == "around":
    return [turn, action] * 4
```

**Problem**: This is a lookup table for SCAN semantics, not a learned program.

**General Alternative**:
- Store programs as HRR compositions of primitive operations
- OR: Store command → action sequence mappings and use HRR for generalization
- OR: Use a learned decoder that maps HRR structures to action sequences

### 2. Hardcoded Primitive Lists ❌
```python
actions_to_try = ['jump', 'walk', 'run', 'look', 'turn']
modifiers_to_try = ['twice', 'thrice', 'around']
```

**Problem**: Not scalable to new tasks.

**General Alternative**:
- Automatically extract primitives from training data
- OR: Use all vocabulary items as candidates
- OR: Learn which primitives to try via a search policy

### 3. SCAN-Specific Parsing ❌
```python
# Special case for "turn left"
if unbind(program, 'left') matches 'turn':
    return "I_TURN_LEFT"
```

**Problem**: Task-specific rules baked into the extractor.

**General Alternative**:
- Hierarchical unbinding that works for ANY nested structure
- Treat all primitives equally (no special "turn" logic)

---

## Path Forward: Truly General Architecture

### Option 1: Pure HRR Memory (Most General)
```python
class GeneralHRR:
    def store(self, input_encoding, output_encoding):
        """Both inputs and outputs are HRR vectors"""
        self.memory.store(input_encoding, output_encoding)

    def retrieve(self, query_encoding):
        """Returns HRR output vector"""
        return self.memory.retrieve(query_encoding)

    def decompose(self, hrr_vector, vocabulary):
        """General decomposition via unbinding search"""
        # Try all possible unbinding combinations
        # Return highest-similarity primitive sequence
```

**Advantages**:
- No task-specific logic
- Fully compositional
- Works for any domain with HRR-encodable structure

**Challenges**:
- How to decode HRR output vectors into sequences?
- Need efficient search over unbinding possibilities

### Option 2: Learned Program Synthesis
```python
class LearnedProgramSynthesizer:
    def encode_program(self, command):
        """Map command to HRR program representation"""
        return hrr_encode(command)

    def execute_program(self, hrr_program):
        """Use learned decoder (small neural net?)"""
        return decoder_net(hrr_program)
```

**Advantages**:
- Separates composition (HRR) from execution (learned)
- Could use small MLP to map HRR → actions
- Still gradient-free for the memory component

**Challenges**:
- Introduces learned parameters (against pure HAM goal?)

### Option 3: Example-Based Execution
```python
class ExampleBasedExecutor:
    def store(self, command, actions):
        """Store command → actions mapping"""
        cmd_encoding = hrr_encode(command)
        self.examples[cmd_encoding] = actions

    def execute(self, command):
        """Find most similar stored example"""
        cmd_encoding = hrr_encode(command)
        most_similar = find_similar(cmd_encoding, self.examples.keys())

        if similarity > threshold:
            return self.examples[most_similar]
        else:
            # Compositional generalization:
            # Decompose command, find examples for parts, compose actions
            return compose_from_parts(command)
```

**Advantages**:
- Simple and interpretable
- No hardcoded rules
- Natural compositional generalization

**Challenges**:
- How to compose action sequences from parts?

---

## Current State

### Working:
- HRR encoding of compositional commands ✓
- Unbinding-based primitive extraction ✓
- Compositional generalization for simple cases (X + twice, X + thrice) ✓

### Needs Work:
- Remove hardcoded SCAN execution logic
- Make unbinding search more general
- Handle nested compositions (around commands) without special cases
- Decide on general execution strategy

---

## Recommendation

For a truly general architecture that achieves the goal of "sparse programs as concepts":

1. **Keep**: HRR binding/unbinding as the compositional substrate
2. **Keep**: Similarity-based primitive extraction
3. **Replace**: Hardcoded interpreter with example-based execution
4. **Add**: General hierarchical decomposition (try unbinding ALL primitives, not just hardcoded lists)
5. **Add**: Compositional action sequence construction from decomposed primitives

This would give us:
- Domain-agnostic compositional operations (HRR)
- Task-specific vocabulary (expected)
- Learned execution from examples (not hardcoded)
- True compositional generalization via algebraic properties

The key insight: **Programs ARE HRR compositions of primitives. Execution should emerge from stored examples + compositional decomposition, not from hardcoded rules.**
