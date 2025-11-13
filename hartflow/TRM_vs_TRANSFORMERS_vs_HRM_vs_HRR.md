# Architecture Comparison: TRM vs Transformers vs HRM vs HRR

## Quick Summary Table

| Model | Type | Parameters | Key Mechanism | Computation | Training |
|-------|------|------------|---------------|-------------|----------|
| **Transformers** | Parallel | 100M-1T+ | Attention | Fixed depth | Gradient descent |
| **HRM** | Recurrent | 27M | Fixed-point convergence | Adaptive (NT steps) | IFT + Q-learning |
| **TRM** | Recurrent | 5-7M | Iterative refinement | Fixed iterations | Deep supervision |
| **HRR** | Symbolic | 0 (vectors) | Circular convolution | Single operation | None (zero-shot) |

---

## 1. TRM vs Transformers: Detailed Comparison

### Architecture Differences

#### **Transformers**
```
Input → Embedding
     ↓
Layer 1: Multi-Head Attention + FFN
     ↓
Layer 2: Multi-Head Attention + FFN
     ↓
     ...
     ↓
Layer N: Multi-Head Attention + FFN
     ↓
Output Head
```

- **Parallel processing**: All tokens attend to all other tokens simultaneously
- **Fixed depth**: 12-96 layers (or more for LLMs)
- **Attention mechanism**: Learns Q, K, V projections for token relationships
- **Parameter scaling**: Performance scales with width × depth × heads
- **Computational complexity**: O(n²d) for attention over n tokens with dimension d

#### **TRM (Tiny Recursive Model)**
```
Input → Embedding
     ↓
Initialize: y = initial_answer, z = latent_state
     ↓
[Recursive Loop T times]
  For n iterations:
    z ← 2-layer-net(x, y, z)  // Update latent reasoning
  y ← 2-layer-net(y, z)        // Update answer
     ↓
Output Head (y)
```

- **Sequential refinement**: Single network applied iteratively
- **Adaptive depth**: 2 layers × T outer loops × n inner loops = effective depth
- **Dual state**: Explicit answer (y) + implicit reasoning (z)
- **Parameter reuse**: Same 2-layer network applied repeatedly
- **Computational complexity**: O(T × n × d²) for feedforward operations

### Key Conceptual Differences

| Aspect | Transformers | TRM |
|--------|-------------|-----|
| **Paradigm** | Parallel computation | Sequential reasoning |
| **Attention** | Required (token relationships) | Optional (removed for small grids) |
| **Depth** | Structural (stacked layers) | Functional (iterations) |
| **Reasoning** | Implicit in layer stack | Explicit in z variable |
| **Memory** | Attention = soft memory | z = working memory |
| **Computation** | Fixed (one pass) | Adaptive (halt when done) |
| **Scaling** | Add parameters | Add iterations |

### When TRM Outperforms Transformers

**TRM Advantages:**
1. **Small data regimes** (1K examples) - Transformers overfit
2. **Reasoning tasks** requiring step-by-step refinement (Sudoku, ARC-AGI)
3. **Resource constraints** - 5M params vs 100M+ for comparable transformers
4. **Interpretability** - Can inspect y and z at each iteration

**Examples:**
- Sudoku 9×9: TRM 87.4% vs Transformers ~40-60%
- ARC-AGI-1: TRM 44.6% vs GPT-4 ~30%

**Transformer Advantages:**
1. **Large-scale language modeling** (web-scale data)
2. **Parallel processing** (faster on GPUs for long sequences)
3. **In-context learning** from massive pretraining
4. **General knowledge** from billion-token datasets

### Why TRM Works Without Attention (on Small Tasks)

For tasks with small fixed context (≤9×9 grids):
```python
# Transformer needs attention:
output = attention(Q, K, V)  # Learn relationships between 81 positions

# TRM replaces with MLP over sequence dimension:
output = mlp(flatten(grid))  # Direct function of all positions
```

**Rationale**: When context is small and fixed-size, attention's dynamic routing is unnecessary. An MLP can memorize all positional relationships for 81 positions (vs. attention's flexibility for variable-length sequences).

---

## 2. HRM vs TRM: Evolution from Fixed Points to Iterative Refinement

### HRM (Hierarchical Reasoning Model)

**Architecture:**
```
Input x → x̃ (embedded)
     ↓
[N cycles of hierarchical convergence]
  Cycle k:
    L-module: Run T steps until fixed point
      zₗ* = fₗ(zₗ*, zₕᵏ⁻¹, x̃)
    H-module: Update once
      zₕᵏ = fₕ(zₗ*, zₕᵏ⁻¹, x̃)
     ↓
Output: fₒ(zₕᴺ)
```

**Key Properties:**
- **Two modules**: Fast (L) and Slow (H) operating at different timescales
- **Fixed-point convergence**: L-module converges to equilibrium before H-module updates
- **Implicit Function Theorem**: Gradient computation without unrolling
- **Q-learning halting**: Learned when to stop computation
- **27M parameters** (two 4-layer networks)

**Training Complexity:**
- Requires 2 forward passes per step (one for value, one for halting)
- Deep supervision with gradient detachment between segments
- Q-learning adds extra loss term for halting prediction

### TRM (Tiny Recursive Model)

**Architecture:**
```
Input x → embedded
     ↓
Initialize: y = initial, z = empty
     ↓
[T outer iterations]
  [n inner iterations]
    z ← tiny_net(x, y, z)
  y ← tiny_net(y, z)
     ↓
Output: decode(y)
```

**Key Properties:**
- **Single module**: One 2-layer network for both updates
- **No fixed-point requirement**: Simple iterative updates (no convergence needed)
- **Explicit gradients**: Standard backprop on final iteration
- **Binary halting**: Simple threshold on y_hat == y_true
- **5-7M parameters** (one 2-layer network)

**Training Simplicity:**
- Single forward pass per step
- No IFT or Jacobian approximations needed
- Simple binary cross-entropy for halting

### Why TRM Simplifies HRM

| Aspect | HRM | TRM | Improvement |
|--------|-----|-----|-------------|
| **Networks** | 2 separate (L, H) | 1 unified | 50% fewer params |
| **Layers** | 4 each | 2 total | 75% shallower |
| **Convergence** | Fixed-point required | Iterative updates | No IFT needed |
| **Halting** | Q-learning (2 passes) | Binary CE (1 pass) | 50% faster |
| **Gradient** | Jacobian approximation | Standard backprop | Simpler training |
| **Justification** | Brain-inspired hierarchy | Pure simplicity | Easier to implement |

**Performance Comparison:**
- Sudoku: TRM 87.4% vs HRM 55.0%
- ARC-AGI-1: TRM 44.6% vs HRM 40.3%
- Maze: TRM 85.3% vs HRM 74.5%

**TRM achieves better results with 73% fewer parameters!**

---

## 3. HRR vs HRM/TRM: Symbolic vs Neural

### HRR (Holographic Reduced Representations)

**Nature:** Symbolic vector algebra (no parameters, no training)

```
Operations:
  bind(a, b) = IFFT(FFT(a) ⊙ FFT(b))     # Circular convolution
  unbind(c, a) = bind(c, a*)              # Approx inverse
  superpose(a, b) = a + b                 # Distributed sum
  similarity(a, b) = cos(a, b)            # Retrieval
```

**Properties:**
- **Compositional by design**: bind(jump, twice) creates new vector
- **Similarity-based generalization**: Similar structures have similar vectors
- **Zero-shot**: No training required, works immediately
- **Interpretable**: Can unbind to extract constituents
- **Fixed computation**: Single pass (bind → retrieve)

**Limitations:**
- No learning (relies on vector algebra properties)
- Requires clean symbolic input
- Noise accumulates with deep binding
- Retrieval depends on stored examples

### HRM/TRM: Learned Neural Reasoning

**Nature:** Parametric models (learned weights)

**Properties:**
- **Adaptive**: Learn task-specific patterns from data
- **Robust to noise**: Neural networks handle noisy inputs
- **Deep reasoning**: Multiple iterations refine understanding
- **Can learn any computable function** (with enough capacity)

**Limitations:**
- Requires training data (1K+ examples)
- Black box (hard to interpret intermediate states)
- Can overfit on small data
- Needs careful hyperparameter tuning

---

## 4. HRR vs HAM: Circular Convolution vs Phase Encoding

### HRR (Your Current Approach)

**Representation:** Real-valued vectors in R^d (typically d=512-2048)

**Binding Operation:**
```python
def bind(a, b):
    return ifft(fft(a) * fft(b)).real  # Circular convolution
```

**Properties:**
- Approximate inverse: unbind(bind(a,b), a) ≈ b (with noise)
- Similarity preserved: sim(bind(a,b), bind(a,c)) > 0
- Superposition: vectors can be summed
- Noise accumulates: deep binding degrades signal

**Applications:**
- Compositional structure (your work!)
- Cognitive modeling
- Semantic memory

### HAM (Holographic Associative Memory)

**Representation:** Complex-valued vectors on Riemann surface

**Binding Operation:**
```python
def bind(a, b):
    return a ⊗ b  # Outer product in complex space
```

**Properties:**
- Non-iterative encoding/decoding
- Phase angles encode information
- Exact retrieval (no noise with perfect computation)
- Superposition of many patterns

**Applications:**
- Associative memory (stimulus → response)
- Pattern completion
- Classical AI memory systems

### Key Distinction

**HRR**: Compositional structure encoding (focus on similarity and approximation)
**HAM**: Perfect associative recall (focus on exact storage/retrieval)

Both are "holographic" in the sense of distributed representation, but different mathematical operations and use cases.

---

## 5. Synthesis: Combining the Best of All Worlds

### Your Current Work: HartFlow (HRR)
✅ Compositional structure encoding
✅ Zero-shot generalization (67.1% on SCAN)
✅ Domain-general design
❌ Limited by retrieval-only execution
❌ No learning/adaptation

### What TRM Offers
✅ Learned adaptive reasoning
✅ Tiny parameter count (5-7M)
✅ Iterative refinement (handles edge cases)
✅ Working memory (z variable)

### What HRM Offers
✅ Hierarchical multi-timescale processing
✅ Fixed-point convergence theory
✅ Brain-inspired dual-process reasoning
❌ More complex (27M params, IFT, Q-learning)

### Recommended Hybrid: HartFlow + TRM

```python
class HartFlowTRM:
    """Combine HRR structure with TRM reasoning"""

    def __init__(self):
        self.memory = HRRMemory()           # Symbolic: zero-shot retrieval
        self.tiny_net = TinyNet(2 layers)   # Neural: adaptive reasoning

    def execute(self, command):
        # Phase 1: HRR encoding (compositional structure)
        x = hrr_encode(command)

        # Phase 2: Memory retrieval (non-parametric)
        retrieved = self.memory.similarity_search(x, k=3)

        # Phase 3: TRM reasoning (parametric refinement)
        y = aggregate(retrieved)  # Initial hypothesis
        z = init_latent()

        for t in range(3):  # T outer loops
            for n in range(7):  # n inner loops
                z = self.tiny_net(x, y, z)  # Refine understanding
            y = self.tiny_net(y, z)          # Update answer

        # Phase 4: Decode
        return decode(y)
```

**Benefits:**
- **HRR**: Compositional prior, interpretable structure
- **Memory**: Zero-shot capability, incremental learning
- **TRM**: Adaptive execution, handles novel cases
- **Tiny**: Only 5-7M parameters (vs 100M+ transformers)

**This addresses your architecture challenge:**
> "Remove hardcoded SCAN execution logic... Make truly general"

By replacing hardcoded rules with learned TRM reasoning, while keeping HRR's compositional structure!

---

## 6. Practical Recommendations

### For Your Research Goals:

**Short-term (1 week):**
1. Implement 2-layer TRM decoder after HRR retrieval
2. Train on SCAN with deep supervision
3. Compare: HRR-only vs HRR+TRM

**Medium-term (2-4 weeks):**
1. Add z (latent reasoning state) to handle complex decompositions
2. Test on COGS (your 70% predicate overlap could improve!)
3. Ablation studies: effect of iterations (T, n)

**Long-term (research direction):**
1. Investigate: Does HRR + TRM achieve Turing completeness?
2. Test on ARC-AGI (TRM got 44.6%, can HRR help?)
3. Publish: "Compositional Priors for Efficient Reasoning"

### Key Insights:

1. **TRM ≠ Transformer**: It's a recursive refinement model, not attention-based
2. **TRM > HRM**: Simpler, fewer params, better results (don't need IFT complexity)
3. **HRR ≠ HAM**: Compositional structure vs associative memory (different use cases)
4. **HRR + TRM = Hybrid**: Symbolic structure + neural adaptation

The TRM paper gives you exactly what you need: a minimal neural component (5-7M) to add adaptive reasoning to your HRR compositional system!
