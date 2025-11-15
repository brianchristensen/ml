# Frontier Language Model Architecture - Core Objectives

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

## Guiding Principles

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