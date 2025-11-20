# Temporal Phase Integration (TPI): Mathematical Formulation

## Overview

Temporal Phase Integration is a novel attention mechanism that uses **complex-valued phase trajectories** and **holographic memory** to process sequential data. Unlike transformer attention (O(n²)), TPI achieves O(n) complexity through parallel cumulative operations.

## Core Principle

**Key Insight:** Each token induces a rotation in complex phase space. The accumulated phase trajectory encodes both content and temporal context, enabling information retrieval through constructive/destructive interference.

## Mathematical Components

### 1. Phase Trajectory Construction

For input sequence **x** = [x₁, x₂, ..., xₙ] where xᵢ ∈ ℝᵈ:

#### Content-Based Phase Initialization
```
φ_init,t = W_init · xᵗ
```
where W_init ∈ ℝᵈˣᵈ maps content to initial phase regions

#### Trajectory Integration
```
ω_t = W_ω · xᵗ                          (semantic rotation rate)
φ_t = φ_init,t + Σᵢ₌₁ᵗ (α ⊙ ω_i)        (cumulative phase trajectory)
```
where:
- ω_t ∈ ℝᵈ is the learned phase increment
- α ∈ ℝᵈ is a learnable integration scale (initialized to 0.01)
- ⊙ denotes element-wise multiplication
- Σ represents cumulative sum

**Properties:**
- φ_t encodes both content (via φ_init) and temporal position (via cumsum)
- Each dimension can evolve at a different rate (learned α)
- Infinite context: φ_t can grow unbounded with sequence length

### 2. Holographic Memory Storage

#### Magnitude Weighting
```
m_t = σ(W_m · xᵗ) · scale_m
```
where σ is sigmoid, scale_m = 5.0 (importance weighting)

#### Complex Memory Accumulation
```
M_t = Σᵢ₌₁ᵗ (m_i · xᵢ · e^(iφᵢ)) / Σᵢ₌₁ᵗ m_i
```

Expanding e^(iφ) = cos(φ) + i·sin(φ):
```
M_real,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · cos(φᵢ))] / [Σᵢ₌₁ᵗ m_i]
M_imag,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · sin(φᵢ))] / [Σᵢ₌₁ᵗ m_i]
```

#### Accumulated Memories per Position

Each token position stores the accumulated sum of all previous tokens

Token-by-Token Breakdown

Position 1: memory₁ = m₁·x₁·e^(iφ₁)
Position 2: memory₂ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂)
Position 3: memory₃ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂) + m₃·x₃·e^(iφ₃)
...
Position t: memory_t = Σᵢ₌₁ᵗ mᵢ·xᵢ·e^(iφᵢ)

When we query at position t:
retrieved_t = memory_t * e^(-iφ_t)

We're querying the entire accumulated memory up to position t, and interference picks out the relevant content based on phase alignment

TPI doesn't store individual token memories

It stores a single evolving holographic trace that continuously accumulates all previous tokens, weighted by their phases.

**Key Features:**
- **Normalization by Σm_i:** Prevents unbounded memory growth
- **Graceful degradation:** As more patterns stored, each is diluted (holographic property)
- **Magnitude weighting:** Important tokens contribute more to memory
- **Phase encoding:** Each token is bound to its phase via complex multiplication

### 3. Phase-Coherent Retrieval

Retrieve information by multiplying memory by conjugate of query phase:
```
retrieved_t = M_t · e^(-iφ_t)
```

Expanding:
```
retrieved_real,t = M_real,t · cos(φ_t) + M_imag,t · sin(φ_t)
retrieved_imag,t = M_imag,t · cos(φ_t) - M_real,t · sin(φ_t)
```

**Mechanism (Constructive/Destructive Interference):**

For a pattern stored at phase φ_s and queried at phase φ_q:
```
contribution ∝ e^(iφ_s) · e^(-iφ_q) = e^(i(φ_s - φ_q))
```

- If φ_s ≈ φ_q (similar phases): e^(i·0) = 1 → **constructive interference**
- If φ_s ≠ φ_q (different phases): e^(i·Δφ) → **destructive interference**

This provides **automatic content-based filtering** without explicit similarity computation!

### 4. Trajectory Binding

Bind content to trajectory via complex multiplication:
```
x_bound,real,t = xᵗ · cos(φ_t)
x_bound,imag,t = xᵗ · sin(φ_t)
```

This preserves semantic content while modulating it with temporal phase context.

### 5. Context Formation and Output

#### Context Concatenation
```
context_t = [x_bound,real,t; x_bound,imag,t; retrieved_real,t; retrieved_imag,t]
```
where context_t ∈ ℝ⁴ᵈ

This combines:
- **When/where information** (trajectory binding)
- **What information** (retrieved content from phase-coherent memory)

#### Multi-Layer Projection (Pre-Norm)
```
h₁ = LayerNorm(context_t)
h₂ = GELU(W₁ · h₁)
h₃ = LayerNorm(h₂)
h₄ = GELU(W₂ · h₃)
h₅ = Dropout(h₄)
output_t = W₃ · h₅
```

where W₁: 4d→4d, W₂: 4d→2d, W₃: 2d→d

#### Residual Connection
```
y_t = xᵗ + output_t
```

## Complete Forward Pass

Given input sequence **x** ∈ ℝⁿˣᵈ:

1. **Compute phases:**
   - φ_init = W_init · x
   - ω = W_ω · x
   - φ = φ_init + cumsum(α ⊙ ω)

2. **Store in holographic memory:**
   - m = σ(W_m · x) · 5.0
   - M_real = cumsum(m ⊙ x ⊙ cos(φ)) / cumsum(m)
   - M_imag = cumsum(m ⊙ x ⊙ sin(φ)) / cumsum(m)

3. **Retrieve from memory:**
   - r_real = M_real ⊙ cos(φ) + M_imag ⊙ sin(φ)
   - r_imag = M_imag ⊙ cos(φ) - M_real ⊙ sin(φ)

4. **Bind trajectory:**
   - b_real = x ⊙ cos(φ)
   - b_imag = x ⊙ sin(φ)

5. **Project and output:**
   - context = [b_real; b_imag; r_real; r_imag]
   - output = MLP(context)
   - y = x + output

## Why This Works

### Content-Based Phase Initialization
- Maps similar content to similar phase regions
- Prevents phase collapse during generation
- Critical for autoregressive robustness

**Ablation Results:**
- Without φ_init: BPC degrades, generation fails catastrophically
- With φ_init: BPC 2.03, coherent generation

### Holographic Normalization
- Bounded memory: ||M_t|| ≈ O(1) regardless of sequence length
- Graceful degradation: older information blends via interference
- Recency bias: recent tokens have stronger contribution

### Phase-Space Dynamics
- Different content → different φ_init → separated in phase space
- Trajectory integration (cumsum) adds temporal ordering
- Interference-based retrieval provides soft content matching

## Theoretical Connections

### Communication Through Coherence (CTC)
- Neural oscillations synchronize via phase alignment
- TPI: phase coherence → constructive retrieval
- Phase difference → destructive filtering

### Holographic Memory (Gabor, Pribram)
- Information distributed across entire memory
- Interference patterns enable content-addressable storage
- TPI: complex phase space as holographic medium

### State Space Models (S4, Mamba)
- Recurrent dynamics via cumulative operations
- TPI: cumsum(ω) provides implicit recurrence
- O(n) parallel computation via associative scan

### Fourier Neural Operators
- Frequency-based representations
- TPI: ω as learned frequency, φ as accumulated phase
- Content-based frequency allocation

## Complexity Analysis

### Time Complexity
- Phase computation: O(n·d) via cumsum
- Memory storage: O(n·d) via cumsum
- Retrieval: O(n·d) element-wise ops
- Total: **O(n·d)** vs O(n²·d) for attention

### Space Complexity
- Activations: O(n·d)
- No KV cache needed
- Memory-efficient for long sequences

### Parallelization
- All cumsum operations are associative scans
- Fully parallelizable on GPU
- No sequential bottlenecks

## Limitations and Open Questions

### Current Limitations
1. **Arbitrary key-value binding:** Interference-based retrieval assumes similarity structure
2. **Length generalization:** Phase values at unseen positions can be OOD
3. **Exact recall:** Normalization causes graceful degradation, not exact storage

### Theoretical Questions
1. What is the capacity of phase-space holographic memory?
2. Can we characterize the interference patterns mathematically?
3. How does learnable α affect information retention curves?
4. What is the optimal balance between φ_init and cumsum(ω)?

## Comparison with Transformers

| Property | Transformer | TPI |
|----------|------------|-----|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) KV cache | O(n) |
| Context | Fixed window | Unbounded |
| Retrieval | Explicit similarity | Phase interference |
| Degradation | Hard cutoff | Graceful |
| Autoregressive | Stable | Requires φ_init |

## Comparison with Holographic Reduced representations

Classic HRR:
Random vectors → Guaranteed ~orthogonal (high probability)
E[⟨v₁, v₂⟩] ≈ 0 for random v₁, v₂

TPI:
Learned embeddings → NO orthogonality guarantee!
In fact, similar tokens SHOULD be close:
x_cat ≈ x_dog  (semantically similar)
x_king ≈ x_queen

So why does TPI work without guaranteed orthogonality?

The Answer: Orthogonality is in PHASE Space, not Embedding Space!

Key Insight:

# Embeddings can be similar (semantic similarity):
x_cat ≈ x_dog

# But PHASES can be different (learned separation):
phi_cat = phi_init(x_cat) + cumsum(omega_cat)
phi_dog = phi_init(x_dog) + cumsum(omega_dog)

# Even if x_cat ≈ x_dog, the model learns:
phi_init(x_cat) ≠ phi_init(x_dog)  # Separation in phase space!

How TPI Achieves Separation

1. Content-Based Phase Init (phi_init):
phi_init = W_init · x
- W_init learns to map content to distinct phase regions
- Even if x_cat ≈ x_dog, W_init can separate them in phase space
- This is what our ablation proved - phi_init is critical!

2. Trajectory Divergence:
phi = phi_init + cumsum(omega * alpha)
- Small differences in omega compound via cumsum
- Trajectories diverge over time
- Temporal evolution provides additional separation

3. Learned Interference Patterns:
- Model doesn't rely on random orthogonality
- Instead, it learns what should interfere constructively/destructively
- Task-optimized phase structure!

The Beautiful Difference

| Property     | Classic HRR           | TPI                 |
|--------------|-----------------------|---------------------|
| Vector Space | Random orthogonality  | Semantic similarity |
| Phase Space  | N/A                   | Learned separation  |
| Guarantee    | Mathematical (random) | None (learned)      |
| Structure    | Hard-coded            | Optimized for task  |
| Similarity   | All equal             | Task-relevant       |

Classic HRR: "All symbols are equally different" (random)

TPI: "Semantically related content can be close in embedding space but separated in phase space"

This allows:
- ✅ Semantic similarity in embeddings (useful for generalization)
- ✅ Phase separation for interference (useful for retrieval)
- ✅ Task-optimized structure (learned, not random)

## Future Directions

1. **Adaptive phase dynamics:** Learn when to advance phase vs. stay coherent
2. **Multi-scale phases:** Different frequency bands for different timescales
3. **Phase synchronization:** Explicit mechanisms for binding related concepts
4. **Hybrid architectures:** Combine semantic (TPI) + episodic (position-based) memory

---

**Implementation:** See `novel_attention.py` for PyTorch implementation
**Ablation Study:** Content-based phase initialization is critical for generation quality
