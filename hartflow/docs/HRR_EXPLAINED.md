# HRR Deep Dive: Understanding Holographic Reduced Representations

## What Does "Holographic" Mean?

### Physical Hologram Analogy

Imagine a holographic photograph:
- Stores a 3D scene on a 2D surface
- **Key property**: Every small piece contains the entire image (degraded)
- Cut a hologram in half → both halves show the full scene, just blurrier

### Holographic Vectors

HRR vectors work the same way:
- Information **distributed** across all dimensions
- Each dimension contains fragments of multiple bound concepts
- Truncate the vector → still recovers information (with noise)

**Example:**
```python
# Bind "red" and "square"
red = random_vector(1024)      # Random complex vector
square = random_vector(1024)
red_square = bind(red, square) # All 1024 dims contain info about BOTH

# Even if you only use first 512 dimensions:
truncated = red_square[:512]
unbind(truncated, square) ≈ red  # Still recovers red (with more noise)
```

This is **distributed representation** - like a hologram!

---

## HAM vs HRR: What's the Connection?

They're both in the **VSA (Vector Symbolic Architecture)** family but serve different purposes:

| Feature | HAM | HRR |
|---------|-----|-----|
| **Purpose** | Associative memory | Compositional encoding |
| **Operation** | Store/retrieve patterns | Bind/unbind structures |
| **Math** | Outer product storage | Circular convolution |
| **Analogy** | Neural network weights | Tree serialization |
| **Gradient-free?** | Yes (episodic storage) | Yes (algebraic ops) |

### HAM (Holographic Associative Memory)

**What it does**: Stores associations between patterns

```python
# Storage
Memory = Σ (stimulus_i* ⊗ response_i)

# Retrieval
response = query* @ Memory
```

**Key properties:**
- Stores multiple (stimulus → response) pairs
- Retrieval via correlation (like attention)
- Degrades gracefully with more patterns
- "Holographic" = distributed storage across matrix

### HRR (Holographic Reduced Representations)

**What it does**: Encodes structures as vectors

```python
# Binding (composition)
"red square" = bind(red, square)  # Circular convolution

# Unbinding (decomposition)
unbind("red square", square) ≈ red  # Approximate inverse
```

**Key properties:**
- Encodes trees/graphs/sequences as single vector
- Approximately invertible (can extract components)
- Compositional: bind(A, bind(B, C)) makes nested structures
- "Holographic" = distributed encoding of structure

### The Connection

They often work together:
1. **HRR encodes** structures as vectors
2. **HAM stores** those vectors for retrieval

Example: Store a scene
```python
# Encode scene as HRR structure
scene = bind(
    bind(object1, position1),
    bind(object2, position2)
)

# Store in HAM for later retrieval
HAM.store(scene_name, scene)

# Retrieve and decompose
retrieved = HAM.retrieve(scene_name)
object1 = unbind(unbind(retrieved, position1), ...)
```

---

## Our Architecture: HRR Only (No HAM!)

**What we use:**
- ✅ HRR for encoding commands as compositional structures
- ✅ HRR unbinding for decomposition
- ✅ Learned MLP for execution (not HAM!)

**What we DON'T use:**
- ❌ HAM memory matrix
- ❌ Associative retrieval

### Why No HAM?

Our current approach:
```python
# Training examples "stored" implicitly in MLP weights
model.executor.parameters()  # ← This IS the memory!

# HRR is just for encoding/decoding
program = HRR.encode(['jump', 'twice'])  # Encode
primitives = HRR.decompose(program)      # Decode
actions = MLP(primitives)                # Execute (learned!)
```

The **MLP weights** act as the memory! We don't need a separate HAM matrix.

### Could We Benefit from HAM?

**Yes!** For episodic memory:

```python
class HybridWithHAM:
    def __init__(self):
        self.hrr = HRROperations()
        self.ham = HolographicMemory()  # Episodic storage
        self.executor = LearnedExecutor()  # Learned execution

    def store_example(self, command, actions):
        # Encode command as HRR
        program = self.hrr.encode(command)

        # Store in HAM for retrieval
        self.ham.store(program, program)  # Auto-associative

        # Also learn execution
        self.executor.train(program, actions)

    def execute(self, command):
        # Encode as HRR
        query = self.hrr.encode(command)

        # Retrieve similar programs from HAM
        similar_program = self.ham.retrieve(query)

        # Decompose via HRR
        primitives = self.hrr.decompose(similar_program)

        # Execute via learned MLP
        return self.executor(primitives)
```

**Benefits:**
- Episodic memory for few-shot learning
- Retrieve similar past examples
- Combine retrieved programs with learned execution

**Why we didn't use it:**
- Simpler architecture to start
- MLP already generalizes well
- HAM adds complexity without clear benefit (for now)

---

## Random Vectors for Vocabulary: How Does It Work?

This is a **key insight** of VSAs!

### The Magic of Random High-Dimensional Vectors

**Concentration of Measure**: In high dimensions, random vectors are nearly orthogonal!

```python
# Create random vectors
dim = 1024
red = random_complex_vector(dim)    # Random normal distribution
blue = random_complex_vector(dim)
square = random_complex_vector(dim)

# They're nearly orthogonal!
cosine_similarity(red, blue) ≈ 0.0  # Almost perpendicular
cosine_similarity(red, square) ≈ 0.0
cosine_similarity(blue, square) ≈ 0.0
```

**Why this works:**
1. High dimensionality (1024-4096 dims)
2. Random draws from Gaussian
3. Law of large numbers → dot products cancel out
4. Vectors are ~90° apart (orthogonal)

### Comparison to One-Hot Encoding

**One-hot encoding:**
```python
red    = [1, 0, 0, 0, 0]  # Exactly orthogonal
blue   = [0, 1, 0, 0, 0]
square = [0, 0, 1, 0, 0]

# Perfect orthogonality
dot(red, blue) = 0
```

**Random dense vectors:**
```python
red    = [0.12, -0.54, 0.31, ..., 0.08]  # 1024 random values
blue   = [-0.31, 0.08, -0.12, ..., 0.54]
square = [0.54, 0.31, -0.08, ..., -0.12]

# Nearly orthogonal (with high probability)
dot(red, blue) ≈ 0 ± ε  # Small random error
```

**Advantages over one-hot:**
- ✅ Compositional: `bind(red, square)` creates new vector
- ✅ Continuous: Can interpolate between concepts
- ✅ Distributed: Each dimension contributes to multiple concepts
- ✅ Robust: Noise doesn't completely break it

### Why Use Random Vectors Instead of Learning Them?

**Random vectors (current):**
```python
vocab = {
    'jump': random_complex_vector(1024),  # Fixed random
    'twice': random_complex_vector(1024),
    'left': random_complex_vector(1024),
}
```

**Pros:**
- ✅ No training needed (instant vocabulary)
- ✅ Guaranteed orthogonal (concentration of measure)
- ✅ Gradient-free (no backprop through encoding)
- ✅ Compositional by construction

**Learned embeddings (alternative):**
```python
embeddings = nn.Embedding(vocab_size, 1024)  # Learned
```

**Pros:**
- ✅ Optimized for task
- ✅ Can capture semantic similarity
- ❌ Requires training
- ❌ Might not stay orthogonal
- ❌ Less compositional guarantees

### The Math: Why Random = Orthogonal?

**Law of Large Numbers:**

For two random vectors `a, b` with dimension `d`:
```
E[a · b] = E[Σ aᵢbᵢ] = Σ E[aᵢbᵢ] = Σ E[aᵢ]E[bᵢ] = 0

Var[a · b] = O(1/√d)
```

As `d → ∞`, the dot product → 0 with high probability!

**Example:**
```python
import numpy as np

d = 1024
num_vectors = 100
vectors = [np.random.randn(d) for _ in range(num_vectors)]

# Normalize
vectors = [v / np.linalg.norm(v) for v in vectors]

# Check all pairwise similarities
similarities = []
for i in range(num_vectors):
    for j in range(i+1, num_vectors):
        sim = np.dot(vectors[i], vectors[j])
        similarities.append(sim)

print(f"Mean similarity: {np.mean(similarities):.4f}")  # ≈ 0.0
print(f"Std similarity: {np.std(similarities):.4f}")   # ≈ 0.03
print(f"Max similarity: {np.max(np.abs(similarities)):.4f}")  # ≈ 0.10
```

Output:
```
Mean similarity: 0.0002
Std similarity: 0.0312
Max similarity: 0.0894
```

**Interpretation:**
- 100 random vectors in 1024 dimensions
- Average similarity ≈ 0 (nearly orthogonal!)
- Standard deviation ≈ 0.03 (small noise)
- Even worst case < 0.1 (still very orthogonal)

### Complex vs Real Vectors

**Why use complex vectors?**

```python
# Real vectors
real_vec = np.random.randn(1024)

# Complex vectors
complex_vec = np.random.randn(1024) + 1j * np.random.randn(1024)
```

**Advantages:**
- ✅ Richer structure (magnitude + phase)
- ✅ Better for circular convolution (FFT naturally complex)
- ✅ Can encode more information per dimension
- ✅ Phase can represent relationships/roles

**Example:**
```python
# Real binding: just convolution (lossy)
bind_real(a, b) = circular_conv(a, b)

# Complex binding: preserves more structure
bind_complex(a, b) = FFT^-1(FFT(a) * FFT(b))
# Both magnitude AND phase are preserved!
```

---

## Would We Benefit from Other Storage Mechanisms?

### Current Approach: MLP Weights

```python
# Examples implicitly stored in weights
executor = LearnedExecutor(hidden_dim=128)
# ~50K parameters encode all training examples
```

**Pros:**
- ✅ Generalizes well (interpolates between examples)
- ✅ Compact (50K params for 160 examples)
- ✅ Fast inference (single forward pass)

**Cons:**
- ❌ Can't retrieve exact training examples
- ❌ Catastrophic forgetting if retrained
- ❌ No episodic memory

### Alternative 1: HAM (Holographic Associative Memory)

```python
# Explicit episodic storage
ham = HolographicMemory(dim=1024)
for cmd, actions in training_data:
    program = hrr.encode(cmd)
    ham.store(program, program)  # Auto-associative
```

**Pros:**
- ✅ Exact retrieval of training examples
- ✅ No forgetting (additive storage)
- ✅ Few-shot learning (one example = instant storage)

**Cons:**
- ❌ Degrades with many examples (interference)
- ❌ No generalization (only retrieves stored items)
- ❌ Requires combining with learned executor

**Best use case:** Few-shot learning, episodic recall

### Alternative 2: Attention-Based Retrieval

```python
# Store examples as key-value pairs
class AttentionMemory:
    def __init__(self):
        self.keys = []    # HRR-encoded commands
        self.values = []  # Action sequences

    def store(self, command, actions):
        program = hrr.encode(command)
        self.keys.append(program)
        self.values.append(actions)

    def retrieve(self, query):
        query_vec = hrr.encode(query)
        # Compute attention scores
        scores = [similarity(query_vec, key) for key in self.keys]
        # Weighted sum of values
        return weighted_sum(self.values, scores)
```

**Pros:**
- ✅ Soft retrieval (blends similar examples)
- ✅ Interpretable (can see which examples match)
- ✅ Scales better than HAM

**Cons:**
- ❌ Slower (O(n) search over stored examples)
- ❌ Memory grows with data

**Best use case:** Meta-learning, retrieval-augmented generation

### Alternative 3: Neural Turing Machine / Differentiable Memory

```python
# Learned read/write to external memory
class NTMMemory:
    def __init__(self):
        self.memory = torch.zeros(100, 1024)  # 100 slots
        self.read_head = AttentionHead()
        self.write_head = AttentionHead()
```

**Pros:**
- ✅ Learned memory access patterns
- ✅ Can implement algorithms (sorting, copying)
- ✅ End-to-end differentiable

**Cons:**
- ❌ Complex to train
- ❌ Not gradient-free
- ❌ Loses interpretability

**Best use case:** Learning algorithms, sequential reasoning

### Hybrid Recommendation

**Combine HRR + HAM + Learned Executor:**

```python
class HybridWithMemory:
    def __init__(self):
        self.hrr = HRROperations()
        self.ham = HolographicMemory()     # Episodic storage
        self.executor = LearnedExecutor()  # Generalization

    def forward(self, command):
        # 1. Encode with HRR
        query = self.hrr.encode(command)

        # 2. Retrieve from HAM (if available)
        retrieved, confidence = self.ham.retrieve(query)

        if confidence > 0.9:  # High confidence = exact match
            # Use retrieved program directly
            primitives = self.hrr.decompose(retrieved)
        else:
            # Use query (novel combination)
            primitives = self.hrr.decompose(query)

        # 3. Execute with learned MLP
        actions = self.executor(primitives)

        return actions
```

**Benefits:**
- ✅ Fast retrieval of known examples (HAM)
- ✅ Generalization on novel combinations (MLP)
- ✅ Best of both worlds!

---

## Summary

### Key Insights

1. **"Holographic" = Distributed Representation**
   - Like optical holograms: information spread across entire vector
   - Not specifically about frequency domain (that's just implementation)

2. **HAM ≠ HRR**
   - HAM: Associative memory (storage/retrieval)
   - HRR: Compositional encoding (bind/unbind)
   - Often used together, but serve different purposes

3. **HRR is Encoding, Not Memory**
   - Encodes tree structures as vectors
   - Our model: MLP weights ARE the memory
   - Could add HAM for episodic storage

4. **Random Vectors Are Magic**
   - High dimensions → random = orthogonal
   - No training needed
   - Compositional by construction
   - Better than one-hot for binding operations

5. **Storage Mechanisms Trade-Offs**
   - Current (MLP): Good generalization, no episodic recall
   - HAM: Perfect recall, degrades with interference
   - Attention: Soft retrieval, slower
   - Hybrid: Best of both!

### The Bottom Line

Our architecture uses **HRR for encoding + MLP for execution**. The MLP weights implicitly store training examples through generalization, not explicit memory storage. We could add HAM for episodic memory, but it's not necessary for compositional generalization!
