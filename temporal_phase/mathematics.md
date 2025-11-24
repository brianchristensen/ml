# Temporal Phase Integration (TPI): Mathematical Formulation

## Overview

Temporal Phase Integration is a universal temporal dynamics learner that uses **complex-valued phase trajectories** and **holographic state integration** to learn arbitrary differential equations from temporal data. TPI achieves O(n) complexity through parallel cumulative operations (Euler integration via cumsum).

## Core Principle

**Key Insight:** Each token induces a learned velocity in complex phase space. The model learns per-dimension differential equations ω(x) and integrates them via cumsum to compute evolving phase trajectories. TPI learns the **dynamics** of temporal sequences, not content-based retrieval.

## Mathematical Components

### 1. Phase Trajectory Construction

For input sequence **x** = [x₁, x₂, ..., xₙ] where xᵢ ∈ ℝᵈ:

#### Content-Based Phase Initialization
```
φ_init,t = W_init · xᵗ
```
where W_init ∈ ℝᵈˣᵈ maps content to initial phase regions

#### Trajectory Integration (Learned Euler Integration)
```
ω_t = W_ω · xᵗ                          (learned phase velocity / dφ/dt)
φ_t = φ_init,t + Σᵢ₌₁ᵗ (α ⊙ ω_i)        (Euler integration of phase ODE)
```
where:
- ω_t ∈ ℝᵈ is the learned phase velocity (differential equation: dφ/dt = ω(x))
- α ∈ ℝᵈ is a learnable integration scale / step size (initialized to 0.01)
- ⊙ denotes element-wise multiplication
- Σ represents cumulative sum (parallel Euler integration)

**Properties:**
- φ_t integrates a learned velocity field ω(x) over the sequence
- Each dimension evolves according to its own learned differential equation
- No solver needed: cumsum implements parallelizable Euler integration
- Infinite context: φ_t can grow unbounded with sequence length

### 2. Complex State Accumulation

#### Magnitude Weighting
```
m_t = σ(W_m · xᵗ) · scale_m
```
where σ is sigmoid, scale_m = 5.0 (learned importance scaling for state evolution)

#### Complex State Integration
```
S_t = Σᵢ₌₁ᵗ (m_i · xᵢ · e^(iφᵢ)) / Σᵢ₌₁ᵗ m_i
```

Expanding e^(iφ) = cos(φ) + i·sin(φ):
```
S_real,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · cos(φᵢ))] / [Σᵢ₌₁ᵗ m_i]
S_imag,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · sin(φᵢ))] / [Σᵢ₌₁ᵗ m_i]
```

#### Accumulated State per Position

Each position represents the integrated state trajectory up to time t

Token-by-Token State Evolution

Position 1: state₁ = m₁·x₁·e^(iφ₁)
Position 2: state₂ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂)
Position 3: state₃ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂) + m₃·x₃·e^(iφ₃)
...
Position t: state_t = Σᵢ₌₁ᵗ mᵢ·xᵢ·e^(iφᵢ)

At position t, we compute trajectory-modulated features:
features_t = state_t * e^(-iφ_t)

This integrates the accumulated phase-modulated state trajectory up to time t. The phase multiplication couples content evolution with learned temporal dynamics.

TPI maintains a single evolving state trajectory that continuously integrates all previous inputs weighted by learned phase dynamics.

**Key Features:**
- **Holographic binding:** Complex multiplication e^(iφ) binds content to phase state
- **Distributed superposition:** All previous states accumulated in shared representation
- **Normalization by Σm_i:** Prevents unbounded state growth (bounded dynamical system)
- **Lossy integration:** Finite state capacity; older information naturally blends/fades
- **Magnitude weighting:** Learned importance scaling for state contributions

### 3. Trajectory-Modulated Features

Compute features by modulating accumulated state with current phase:
```
features_t = S_t · e^(-iφ_t)
```

Expanding:
```
features_real,t = S_real,t · cos(φ_t) + S_imag,t · sin(φ_t)
features_imag,t = S_imag,t · cos(φ_t) - S_real,t · sin(φ_t)
```

**Mechanism (Phase-Space Dynamics):**

The multiplication by e^(-iφ_t) creates phase-dependent feature modulation:
```
contribution ∝ e^(iφ_s) · e^(-iφ_t) = e^(i(φ_s - φ_t))
```

- Different phase offsets (φ_s - φ_t) produce different feature activations
- This couples the integrated state with the current trajectory position
- Not retrieval: the model cannot recall arbitrary past content

**What TPI Cannot Do:**
- No content-based retrieval (cannot look up arbitrary facts)
- No associative recall (cannot bind arbitrary key-value pairs)
- No episodic memory (will drift over long contexts)

**What TPI Does:**
- Learns temporal dynamics (how sequences evolve over time)
- Integrates state trajectories in phase space
- Generates via learned flow fields, not memory lookup

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
context_t = [x_bound,real,t; x_bound,imag,t; features_real,t; features_imag,t]
```
where context_t ∈ ℝ⁴ᵈ

This combines:
- **Current trajectory position** (x modulated by current phase)
- **Integrated state dynamics** (accumulated trajectory-modulated features)

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

1. **Compute phase trajectories (Euler integration):**
   - φ_init = W_init · x
   - ω = W_ω · x  (learned velocity field)
   - φ = φ_init + cumsum(α ⊙ ω)  (integrate dφ/dt = ω)

2. **Integrate state dynamics:**
   - m = σ(W_m · x) · 5.0
   - S_real = cumsum(m ⊙ x ⊙ cos(φ)) / cumsum(m)
   - S_imag = cumsum(m ⊙ x ⊙ sin(φ)) / cumsum(m)

3. **Compute trajectory-modulated features:**
   - f_real = S_real ⊙ cos(φ) + S_imag ⊙ sin(φ)
   - f_imag = S_imag ⊙ cos(φ) - S_real ⊙ sin(φ)

4. **Bind to trajectory:**
   - b_real = x ⊙ cos(φ)
   - b_imag = x ⊙ sin(φ)

5. **Project and output:**
   - context = [b_real; b_imag; f_real; f_imag]
   - output = MLP(context)
   - y = x + output

#### Dimensionality

Each dimension in TPI evolves independently:

omega = self.to_omega(x)  # [batch, seq, dim]
phi = phi_init + cumsum(omega * self.integration_scale.abs())

- integration_scale is per-dimension: torch.ones(dim) * 0.01
- Each dimension i has its own trajectory: phi[:, :, i]
- Each dimension stores/retrieves independently in its own phase subspace

So with dim=512, you effectively have 512 independent channels, each with its own:
- Phase velocity field (omega)
- Integration rate / step size (scale)
- Phase trajectory (phi)
- Learned differential equation dynamics

## Why This Works for Dynamics Learning

### Content-Based Phase Initialization
- Maps input content to initial phase states
- Prevents phase collapse during autoregressive generation
- Critical for maintaining diverse phase trajectories

**Ablation Results:**
- Without φ_init: BPC degrades, generation fails catastrophically
- With φ_init: BPC 2.03, coherent (local) generation

### State Normalization
- Bounded dynamics: ||S_t|| ≈ O(1) regardless of sequence length
- Natural forgetting: older information blends into integrated state
- Lossy compression: finite capacity like real dynamical systems

### Learning Temporal Dynamics
- Model learns velocity fields ω(x) that govern sequence evolution
- Euler integration via cumsum computes trajectories in parallel
- Works for: language dynamics, video dynamics, chaotic systems (Lorenz)
- Does NOT work for: fact recall, associative memory, long-range grounding

**Why Language Works Without Retrieval:**
- Language has local statistical structure (grammar, syntax)
- Short-term dynamics captured by phase trajectories
- Long-range coherence drifts (no episodic memory)
- Model learns "how text flows" not "what facts are true"

## Theoretical Connections

### Communication Through Coherence (CTC)
- Neural oscillations synchronize via phase alignment
- TPI: learned phase trajectories couple content and temporal dynamics
- Phase modulation creates temporal binding

### Holographic Memory / HRR (Holographic Reduced Representations)
- Distributed storage via binding operations (complex multiplication)
- Superposition of bound patterns in shared representation space
- TPI: uses holographic binding (x · e^(iφ)) to accumulate state trajectories
- NOT for retrieval - used to integrate dynamics in distributed phase-space representation
- Interference patterns drive state evolution, not content lookup

### Dynamical Systems Theory
- State evolution governed by differential equations
- TPI: learns dφ/dt = ω(x) per dimension
- Euler integration via cumsum (no solver needed)
- Universal approximation of temporal dynamics

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
- State integration: O(n·d) via cumsum
- Feature computation: O(n·d) element-wise ops
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
1. **No content-based retrieval:** Cannot recall arbitrary facts or do associative memory
2. **Long-range drift:** No episodic memory; generations drift over long contexts
3. **Length generalization:** Phase values at unseen positions can be OOD
4. **Lossy dynamics:** Finite state capacity causes information blending/forgetting

### Theoretical Questions
1. What class of differential equations can TPI learn?
2. What is the capacity of phase-space state integration?
3. How does learnable α affect trajectory stability and information retention?
4. What is the optimal balance between φ_init and cumsum(ω)?
5. Can we add episodic memory without losing O(n) complexity?

## Comparison with Transformers

| Property | Transformer | TPI |
|----------|------------|-----|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) KV cache | O(n) |
| Context | Fixed window | Unbounded |
| Mechanism | Content retrieval | Dynamics learning |
| Degradation | Hard cutoff | Graceful drift |
| Autoregressive | Stable | Requires φ_init |
| Fact recall | Yes | No |
| Temporal dynamics | Implicit | Explicit |

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

## How TPI Achieves Separation

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
- ✅ Phase separation in trajectory space (useful for dynamics)
- ✅ Task-optimized structure (learned, not random)

## Future Directions

1. **Adaptive phase dynamics:** Learn when to advance phase vs. stay coherent
2. **Multi-scale dynamics:** Different frequency bands for different timescales
3. **Hierarchical integration:** Multi-level temporal dynamics
4. **Hybrid architectures:** Combine dynamics (TPI) + episodic memory (attention/retrieval)
5. **Higher-order integrators:** Beyond Euler (RK4, adaptive step size)
6. **Symbolic dynamics:** Learn discrete state transitions in continuous phase space

---

**Implementation:** See `tempo.py` for PyTorch implementation
**Ablation Study:** Content-based phase initialization is critical for generation quality
