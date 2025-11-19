# Frontier Language Model Architecture - Core Objectives

I'm trying to experiment on the frontier of language modeling and general cognition, using ideas from cognitive science,
physics research, extremely recent papers in CS, and esoteric mathematics, because i want to come up with a paradigm-shifting replacement for standard attention, and hopefully achieve human-brain-like cognition patterns, so that I can be the person who invents AGI.

## Must Have (Non-Negotiable)

1. **Sub-O(n²) Complexity**: Not O(n²) like transformers. Target O(n log n) or O(n).

2. **Parallelizable Training**: NO sequential loops (`for t in range(seq_len)`). Must train on GPU efficiently.

3. **Works on Real Tasks**: Must pass ALL three benchmarks at least as well as a single layer transformer:
   - Copy task
   - Associative recall
   - Character LM: BPC < 2

4. **Infinite Context Window**: No hard-coded max_len limits. Should handle arbitrary sequence lengths, and degrade gracefully near infinity.

5. **Faster than Transformer+Attention**: Must be faster to train and infer than a typical single layer transformer+attention.

6. **No Hard Coded Rules**: No hard-coded rules specific to datasets/tasks, or data-specific modules (like use this NN/Gate for this task, but not this one)

7. **No Exploding Parameters**: Do not parameterize the model in such a way that task to task the parameters may explode in size, and make sure parameters are less than a typical single layer transformer.

8. **No Using Existing Methods - PROTECT THE NOVEL IDEA**:
   - The point of this task is to innovate on the frontier! We are trying to break through to the unknown, explore the state space of possible language models, to find something orders of magnitude better than what already exists (attention).
   - We shouldn't be falling back to existing solutions (like linear attention, state space models, convolution layers, standard RNNs) with known limitations.
   - We can take inspiration from existing methods and tweak them, but we shouldn't just mash them together or use them directly.

   **CRITICAL: When Facing Challenges, ITERATE on the Novel Idea, Don't Replace It:**
   - ❌ **WRONG**: "This novel approach is slow" → Replace with standard convolution/attention
   - ✅ **RIGHT**: "This novel approach is slow" → Vectorize/optimize the NOVEL approach itself
   - ❌ **WRONG**: "Getting poor results" → Fall back to known working methods
   - ✅ **RIGHT**: "Getting poor results" → Debug and improve the novel mechanism
   - ❌ **WRONG**: "Too complex" → Simplify by using standard components
   - ✅ **RIGHT**: "Too complex" → Simplify while preserving the core novelty

   **The Novel Idea is Sacred:**
   - Accept that novel approaches will be slower/messier initially - that's OK for research
   - "Slow but novel" >> "Fast but derivative"
   - Prove the idea works FIRST, optimize LATER
   - If we can't make the novel approach work, that's valuable negative results
   - But we should NEVER abandon novelty just because it's harder

   **Examples of Falling Back (DO NOT DO THIS):**
   - Starting with "hyperbolic geometry" → ending with "just L2 normalization"
   - Starting with "Möbius addition" → ending with "standard convolution"
   - Starting with "content-based binding" → ending with "position-based decay"
   - Starting with novel idea X → ending with "simplified version of existing method Y"

   **What IS allowed:**
   - Using standard components (LayerNorm, Linear layers, etc.) as building blocks
   - Using efficient implementations (GPU kernels, vectorization) of novel operations
   - Taking mathematical inspiration from existing work to BUILD something new
   - Combining ideas in truly new ways (not just "multi-head X" when X exists)

9. **What language (and thinking) needs**:
  - Asymmetric relationships: key → value (not value → key)
  - Sequential structure: "the cat" ≠ "cat the"
  - Compositional semantics: meaning from word order 

## Guiding Principles
- **The Frontier Question**: How do we get learnable bilinear-like expressiveness without computing all O(n²) pairs?  How does the brain do this? Achieving both selective content-based retrieval AND sub-O(n²) complexity requires a fundamentally different mechanism than similarity-based (or even factorized causal) attention. Semantic binding emerges from temporal dynamics, not static patterns! The temporal SEQUENCE of activations = the representation. The brain doesn't have perfect retrieval:
  - Humans can't perfectly recall arbitrary facts from conversation
  - We compress, summarize, lose details
  - We use hierarchies, chunking, and imperfect heuristics

  Perfect O(n²) attention is overkill, and we should embrace:
  - Lossy compression (like trajectory integration)
  - Hierarchical structure
  - Imperfect but efficient retrieval
- **Selectivity**: Must retrieve specific relevant tokens, not uniform/decaying weights
- **Content-based binding**: Tokens bind based on semantic content (Communication Through Coherence)
- **Simplicity**: If adding complexity doesn't help language modeling, remove it

## Key Constraint

**NO Q@K^T attention matrix** - This is O(n²). We need a fundamentally different approach.

## Open Question

Can we achieve strong selectivity (needed for language) without O(n²) similarity computation?

## Inspiration

Use the human brain as inspiration. It does not have full perfect attention, and it does not have a limited context window.  Instead it has limited attention, and a gracefully degrading context window.

If we can't find a good mechanism from biology, use something with mathematical elegance.

Active areas of research we should use as inspiration:
Free energy Principle
Hierarchical Temporal memory
Predictive coding
Fractal AI
Hyperbolic Machine learning
Complex Valued neural networks
Adaptive Resonance Theory
Closed-form differential computation graphs/approximate gradients/gradient-free learning
Particle Swarm optimization
Lattice Regression
Connectionism vs Computationalism as a computing paradigm
Variational Inference
Resonance Coupling
Neurosymbolic AI
Kolmogorov-Arnold networks
Tsetlin Machines
Cellular Automata