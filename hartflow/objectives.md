# Frontier Language Model Architecture - Core Objectives

I'm trying to experiment on the frontier of language modeling and general cognition, using ideas from cognitive science,
physics research, extremely recent papers in CS, and esoteric mathematics, because i want to come up with a new way to implement
attention that outperforms transformers at scale, and hopefully achieves human-brain-like cognition patterns, so that I can be the person who invents AGI.

## Must Have (Non-Negotiable)

1. **Sub-O(n²) Complexity**: Not O(n²) like transformers. Target O(n log n) or O(n).

2. **Parallelizable Training**: NO sequential loops (`for t in range(seq_len)`). Must train on GPU efficiently.

3. **Works on Real Tasks**: Must pass ALL three benchmarks at least as well as a single layer transformer:
   - Copy task
   - Associative recall
   - Character LM: BPC < 2.5

4. **Infinite Context Capability**: No hard-coded max_len limits. Should handle arbitrary sequence lengths.

5. **Faster than Transformer+Attention**: Must be faster to train and infer than a typical single layer transformer+attention.

6. **No Hard Coded Rules**: No hard-coded rules specific to datasets/tasks, or data-specific modules (like use this NN/Gate for this task, but not this one)

7. **No Exploding Parameters**: Do not parameterize the model in such a way that task to task the parameters may explode in size, and make sure parameters are less than a typical single layer transformer.

8. **No Using Existing Methods**: The point of this task is to innovate on the frontier!  We shouldn't be falling back to existing solutions (like linear attention) with known limitations.  We are trying to break through to the unknown, explore the state space of possible language models, to find something orders of magnitude better than what already exists (attention).  We can take inspiration from existing methods and tweak them, but we shouldn't just mash them together or use them directly.

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