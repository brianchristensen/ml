# Phase-Space Integration (PSI Ψ): Mathematical Formulation

## Overview

A parallel sequence processor that uses cumsum-based memory with phase/frequency encoding. It's essentially a learned Euler integrator with trigonometric basis functions.

- Integration learnable (ω is learned, not specified)
- Parallelizable (cumsum, not sequential solver)
- Phase-bound superposition (content bound to learned phases, accumulated via addition)
- Multi-scale (per-dimension rates)

## Core Principle

**Key Insight:** Each token induces a learned velocity in complex phase space. The model learns per-dimension differential equations ω(x) and integrates them via cumsum to compute evolving phase trajectories. PSI learns the **dynamics** of temporal sequences, not content-based retrieval.

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

### 2. Phase-Bound State Accumulation

#### Magnitude Weighting
```
m_t = σ(W_m · xᵗ) · scale_m
```
where σ is sigmoid, scale_m = 5.0 (learned importance scaling for state evolution)

#### Phase Binding

Each content vector is bound to its phase via element-wise complex multiplication:
```
bound_t = m_t · x_t · e^(iφ_t)
        = m_t · x_t · (cos(φ_t) + i·sin(φ_t))
```

This is **not** circular convolution (as in HRR). It's element-wise multiplication:
- Each dimension binds independently
- No cross-dimensional mixing
- Simpler and faster than FFT-based binding

#### Superposition via Cumulative Sum
```
S_t = Σᵢ₌₁ᵗ (m_i · xᵢ · e^(iφᵢ))
```

The superposition is simply addition in complex space:
```
Position 1: S₁ = m₁·x₁·e^(iφ₁)
Position 2: S₂ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂)
Position 3: S₃ = m₁·x₁·e^(iφ₁) + m₂·x₂·e^(iφ₂) + m₃·x₃·e^(iφ₃)
...
Position t: S_t = Σᵢ₌₁ᵗ mᵢ·xᵢ·e^(iφᵢ)
```

At each position, you have access to a superposition of **all previous positions** bound to their respective phases.

#### Normalization (Graceful Degradation)

The state is normalized to prevent unbounded growth:
```
S_real,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · cos(φᵢ))] / √[Σᵢ₌₁ᵗ m_i]
S_imag,t = [Σᵢ₌₁ᵗ (m_i · xᵢ · sin(φᵢ))] / √[Σᵢ₌₁ᵗ m_i]
```

**Graceful degradation** comes from this normalization, not from holographic properties:
- As more terms are added, accumulated_magnitude grows
- Division by √(accumulated_magnitude) keeps the state bounded
- Strongly weighted items persist longer
- Weakly weighted items fade faster
- Acts like a weighted running average in complex space

#### Interference Patterns

When terms are added in complex space:
- **Similar phases** → constructive interference (reinforce)
- **Different phases** → destructive interference (cancel/blur)

The model learns:
- What phases to assign (via φ_init and ω)
- What to weight strongly (via magnitude)
- How to query (via query_offset)

### 3. Query-Based Feature Extraction

Extract features by querying the accumulated state with a learned phase offset:
```
φ_query = φ_t + W_query · x_t
features_t = S_t · e^(-iφ_query)
```

Expanding:
```
features_real,t = S_real,t · cos(φ_query) + S_imag,t · sin(φ_query)
features_imag,t = S_imag,t · cos(φ_query) - S_real,t · sin(φ_query)
```

**Mechanism (Phase-Space Dynamics):**

The multiplication by e^(-iφ_query) creates phase-dependent feature modulation:
```
contribution ∝ e^(iφ_s) · e^(-iφ_query) = e^(i(φ_s - φ_query))
```

- Different phase offsets (φ_s - φ_query) produce different feature activations
- This couples the integrated state with the current trajectory position
- Not retrieval: the model cannot recall arbitrary past content

**What PSI Cannot Do:**
- No content-based retrieval (cannot look up arbitrary facts)
- No associative recall (cannot bind arbitrary key-value pairs)
- No episodic memory (will drift over long contexts)

**What PSI Does:**
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

2. **Accumulate phase-bound state:**
   - m = σ(W_m · x) · 5.0
   - S_real = cumsum(m ⊙ x ⊙ cos(φ)) / √cumsum(m)
   - S_imag = cumsum(m ⊙ x ⊙ sin(φ)) / √cumsum(m)

3. **Query accumulated state:**
   - φ_query = φ + W_query · x
   - f_real = S_real ⊙ cos(φ_query) + S_imag ⊙ sin(φ_query)
   - f_imag = S_imag ⊙ cos(φ_query) - S_real ⊙ sin(φ_query)

4. **Bind to trajectory:**
   - b_real = x ⊙ cos(φ)
   - b_imag = x ⊙ sin(φ)

5. **Project and output:**
   - context = [b_real; b_imag; f_real; f_imag]
   - output = MLP(context)
   - y = x + output

#### Dimensionality

Each dimension in PSI evolves independently:

omega = self.to_omega(x)  # [batch, seq, dim]
phi = phi_init + cumsum(omega * self.integration_scale.abs())

- integration_scale is per-dimension: torch.ones(dim) * 0.01
- Each dimension i has its own trajectory: phi[:, :, i]
- Each dimension accumulates/queries independently in its own phase subspace

So with dim=512, you effectively have 512 independent channels, each with its own:
- Phase velocity field (omega)
- Integration rate / step size (scale)
- Phase trajectory (phi)
- Learned differential equation dynamics

## Why This Works for Dynamics Learning

### The Architecture IS an ODE Integrator

The core mechanism:
```
φ = φ_init + cumsum(ω)
```

This is literally Euler integration of dφ/dt = ω(x). The model learns:
- ω(x): the differential equation governing phase evolution
- φ_init(x): initial conditions in phase space

Other architectures (MLP, RNN, CNN, Transformer) would have to **discover** that they need to:
1. Track phase relationships
2. Integrate over time
3. Handle multi-scale dynamics

PSI has this **built into the math**. The cumsum isn't learned behavior—it's the architecture. The model just learns ω.

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
- Works for: language dynamics, video dynamics, chaotic systems (Lorenz, n-body, turbulence)
- Does NOT work for: fact recall, associative memory, long-range grounding

**Why Language Works Without Retrieval:**
- Language has local statistical structure (grammar, syntax)
- Short-term dynamics captured by phase trajectories
- Long-range coherence drifts (no episodic memory)
- Model learns "how text flows" not "what facts are true"

## Theoretical Connections

### Dynamical Systems Theory
- State evolution governed by differential equations
- PSI: learns dφ/dt = ω(x) per dimension
- Euler integration via cumsum (no solver needed)
- Universal approximation of temporal dynamics

### Neural ODEs (Chen et al., 2018)
- Learn continuous dynamics dz/dt = f(z)
- But: Requires ODE solver at inference (slow, sequential)
- PSI: Parallel integration via cumsum, no solver needed

### State Space Models (S4, Mamba)
- Recurrent dynamics via cumulative operations
- PSI: cumsum(ω) provides implicit recurrence
- O(n) parallel computation via associative scan
- Key difference: SSMs have linear dynamics; PSI has nonlinear phase dynamics

### Communication Through Coherence (CTC)
- Neural oscillations synchronize via phase alignment
- PSI: learned phase trajectories couple content and temporal dynamics
- Phase modulation creates temporal binding

### Relation to HRR (Holographic Reduced Representations)

PSI was inspired by HRR but differs significantly:

| Property | Classic HRR | PSI |
|----------|-------------|-----|
| Binding | Circular convolution (FFT) | Element-wise complex multiplication |
| Cross-dimension | Yes (convolution mixes) | No (independent per dimension) |
| Phases | Random | Learned |
| Superposition | Sum of bound vectors | Same (cumsum) |
| Retrieval | Correlation (unbinding) | Phase-offset query |

PSI uses **phase-bound superposition**, not true holographic binding:
- Binding via x · e^(iφ) (element-wise, not convolution)
- Superposition via cumsum (same as HRR)
- Normalization for bounded state (not holographic graceful degradation)

### Why Phase Separation Works Without Orthogonality

Classic HRR requires random orthogonal vectors. PSI learns phase separation:

```
# Embeddings can be similar (semantic similarity):
x_cat ≈ x_dog

# But PHASES can be different (learned separation):
φ_cat = φ_init(x_cat) + cumsum(ω_cat)
φ_dog = φ_init(x_dog) + cumsum(ω_dog)

# Even if x_cat ≈ x_dog, the model learns:
φ_init(x_cat) ≠ φ_init(x_dog)  # Separation in phase space!
```

| Property | Classic HRR | PSI |
|----------|-------------|-----|
| Vector Space | Random orthogonality | Semantic similarity |
| Phase Space | N/A | Learned separation |
| Guarantee | Mathematical (random) | None (learned) |
| Structure | Hard-coded | Optimized for task |

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
1. What class of differential equations can PSI learn?
2. What is the capacity of phase-bound state accumulation?
3. How does learnable α affect trajectory stability and information retention?
4. What is the optimal balance between φ_init and cumsum(ω)?
5. Can we add episodic memory without losing O(n) complexity?

## Comparison with Transformers

| Property | Transformer | PSI |
|----------|------------|-----|
| Complexity | O(n²) | O(n) |
| Memory | O(n²) KV cache | O(n) |
| Context | Fixed window | Unbounded |
| Mechanism | Content retrieval | Dynamics learning |
| Degradation | Hard cutoff | Graceful drift |
| Autoregressive | Stable | Requires φ_init |
| Fact recall | Yes | No |
| Temporal dynamics | Implicit | Explicit |

## Future Directions

1. **Adaptive phase dynamics:** Learn when to advance phase vs. stay coherent
2. **Hybrid architectures:** Combine dynamics (PSI) + episodic memory (attention/retrieval)
3. **Higher-order integrators:** Beyond Euler (RK4, adaptive step size)
4. **Symbolic dynamics:** Learn discrete state transitions in continuous phase space

---

**Implementation:** See `psi.py` for PyTorch implementation
**Ablation Study:** Content-based phase initialization is critical for generation quality
