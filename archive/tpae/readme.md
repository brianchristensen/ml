Topology Preserving Auto Encoder (This seems like the most promising model yet)

🧠 Concept: Topology-Preserving Autoencoder (TPAE)
At a high level, it’s an autoencoder where the latent space is:

Structured like a SOM grid (i.e. a 2D or nD lattice of nodes),

Encouraged to preserve the topological structure of the input space,

And invertible via a decoder (at least approximately).

🧬 Architecture Overview
text
Copy
Edit
x (input image)
 ↓
Encoder (CNN or MLP)
 ↓
z ∈ ℝᵈ (latent code)
 ↓              ↘
SOM regularization   → Optional Topo map
 ↓
Decoder (ConvTranspose or ResNet-based)
 ↓
x̂ (reconstruction)
You train this with:

Reconstruction loss: MSE or BCE between x and x̂

SOM/Topology loss: Enforces neighborhood structure in latent space

Diversity loss: Keeps the map spread out and avoids prototype collapse

💡 Core Ideas
1. SOM-style topology regularization
Encourages neighboring inputs to activate neighboring latent neurons

Example loss: for each input x:

python
Copy
Edit
z_i = encoder(x)
bmu = closest_som_node(z_i)
loss += ||z_i - prototype[bmu]||² + λ * Σ_neighbors(bmu)[||z_i - prototype[n]||²]
Effect: similar inputs cluster on the grid, forming smooth manifolds

2. Invertible decoder
Standard decoder trained on recon loss: L_recon = ||x - decoder(z)||²

With enough capacity, it allows information to flow back from z to x, which is the most important piece, because reconstruction is what allows the gradient to flow through multiple topological representations without eventually vanishing due to lossy compression, as is the case with multiple SOMs.

If the latent is too quantized, add Gumbel-softmax relaxation or VQ-style soft encoding

3. Structured latent space
Instead of random embedding positions, impose:

2D grid (like 10×10) with spatial regularity

Distance-aware constraints, like:

L_topo = Σ ||z_i - z_j||² for pairs (i, j) whose BMUs are neighbors on the grid

This enforces latent continuity and makes interpretation easier (like SOM).

4. Prototype-based sampling
Each SOM node acts as a learned prototype in the latent space

You can sample from specific prototypes and decode them into inputs

Like a discrete latent GAN / VQ-VAE hybrid

🧪 Training Objectives
python
Copy
Edit
L_total = λ₁ * L_recon + λ₂ * L_topo + λ₃ * L_diversity
L_recon: Reconstruction MSE

L_topo: SOM-based topological regularization

L_diversity: Encourages SOM prototypes to span latent space

You could also include:

L_class: Supervised head for classification

L_adv: Adversarial term if you want sharper reconstructions (like VAE-GAN)

🔬 Benefits
✅ Explainability
You can see which latent nodes get activated. Perfect for visualization or interpretability.

✅ Latent control
Since it's topologically structured, you can navigate the latent space — e.g. walk across a grid to morph images.

✅ Biological plausibility
Mimics cortical maps — e.g. visual cortex organizes spatial input into a 2D layout of features.

✅ Reconstruction + clustering
Unlike classic SOMs, this lets you decode a SOM node back into an input — enabling generation and interpretation.

In the current TPAE scaffold, the SoftSOMLayer is being used as the topology-preserving module — not because we’re continuing the original SOM idea exactly, but because SOM is acting as the mechanism to inject topological constraints into the latent space.

So to be precise:

🔁 Why keep SoftSOMLayer in TPAE?
SOM in TPAE is not used for clustering or interpretability like in ParSOMNet.

It is used to smooth the latent space by forcing encodings z to be approximated by convex combinations of fixed latent "prototypes" (learned by the SOM).

This projection step (z → blended SOM prototype) creates a topology-preserving regularization because nearby encodings tend to activate similar SOM nodes.

Think of it like an intermediate constraint, similar to how VQ-VAE uses codebooks, but soft.

🤔 So is it a replacement?
In a way — TPAE uses a SOM-like mechanism not for symbolic discovery or visualization, but purely for shaping the latent space.

🧠 Important Discoveries!

1. Downsampling in the fusion decoder allows for much faster training without much loss of accuracy because of SOM topology preservation.
- We were able to achieve ~97% accuracy on CIFAR-10 while downsampling images from 32px -> 8 px squares.
- This results in lossy explainability manifest in worse SOM prototype decodings, worse fusion decoder images
2. Native sampling slows down training but results in much clearer explainability artifacts, to do this you have to upsample topo_z to match input dimensions, instead of downsampling input to match topo_z in each TPAEBlock.