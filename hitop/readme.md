# HiTop: Hierarchical Topological Autoencoder for Structured Representation Learning

HiTop is a neural architecture designed to learn *topologically structured latent representations* using a cascade of Self-Organizing Maps (SOMs). This model blends the spatial intuition of CNNs with the interpretability of topology-driven learning.

## 🧠 Key Concepts

- **Shared Encoder**: A CNN backbone processes the input image into a rich latent embedding.
- **Progressive SOM Chain**: A sequence of `SoftSOM` modules (e.g., 40×40 → 20×20 → 14×14 → 10×10 → 5×5), each capturing the structure of the data at a different resolution.
- **Constant Latent Dimension**: All SOMs operate in the same latent space, maintaining consistent Euclidean geometry for stable and meaningful prototype learning.
- **Residual Topology**: Each SOM refines the current representation via a residual connection that mixes its topological embedding with the prior feature map.
- **Topological Attention**: Learned class embeddings query SOM representations via multi-head attention for final classification.

## ✅ Design Principles

- **Downsample topology, not embedding**: We decrease SOM grid size to simulate CNN-like spatial downsampling, while preserving the same latent dimension across layers.
- **Interpretability-first**: The model explicitly encodes structure through spatially organized prototypes.
- **Flexible modularity**: Each `Node` (SOM + encoder + classifier) operates independently and can be probed or visualized in isolation.

## 🧪 Training Objectives

- **Cross-Entropy Loss** on final attention-based classification.
- **Node Classification Loss** at each SOM layer.
- **Node Diversity Loss** to encourage spread-out prototypes within each SOM.
- **Graph Diversity Loss** to decorrelate prototypes across SOM layers.
- **Usage Penalty** to ensure full utilization of SOM units.

## 📦 Results

HiTop consistently converges to >85% accuracy on CIFAR-10 with clear topological structure in latent space, and meaningful attention maps over SOM chains.

---

