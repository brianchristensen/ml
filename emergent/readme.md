### 🔧 Architecture Philosophy (V1 Manifesto)

1. Algebraic Core.
The model is a computation engine based on structured rewrites in latent space, not a feedforward stack.

2. Emergent Complexity.
No baked-in hierarchies. No ResNets.
Any hierarchy or abstraction must emerge through training or latent self-organization.

3. Interpretability First.
Every module, every rewrite, should be traceable and explorable.
Not post-hoc explanations — native interpretability.

4. Minimality.
Each component must justify its existence — in function, clarity, and necessity.
If it can be learned instead of hardcoded, we learn it.

5. Task-Agnostic Design.
Nothing should be tied to CIFAR-10, classification, or images.
The architecture is general-purpose symbolic computation.
The dataset is just the substrate.

