# 🧠 CLEAR: Conceptual Latent Encoding with Attentive Reconstructions

**CLEAR** is a modular, interpretable deep learning architecture designed to learn and explain semantic representations in high-dimensional data.

CLEAR stands for:

> **C**onceptual  
> **L**atent  
> **E**ncoding with  
> **A**ttentive  
> **R**econstructions  

At its core, CLEAR is a **semantic latent autoencoder** — a model that doesn’t just compress data, but organizes it into **conceptual units** using self-organizing maps (SOMs), then reconstructs meaningful representations through gated attention.

---

## 🔍 What Makes CLEAR Unique

- 🧩 **Self-Organizing Maps (SOMs)** per node to enforce topological structure on latent prototypes  
- 🔁 **Attentional gating** to blend concepts in a compositional, interpretable way  
- 🌱 **Modular node growth**, allowing the model to expand as needed  
- 🖼️ **Explainable reconstructions** from clean targets — not pixel supervision, but latent denoising  
- 📈 **Growth-driven training** using entropy, variance, and diversity diagnostics  
- 🧠 **Clusterable, visualizable prototype grids** per node for deep explainability  

---

## 💡 What It's For

CLEAR is built to bridge the gap between **deep performance** and **deep understanding**:

- Interpretable image classification
- Self-supervised representation learning
- Regression or generation via modular heads (coming soon)
- Prototype-based explanations
- Curriculum learning with growable modules

---

## 📸 Example Outputs

- ✅ Node-wise prototype clusters  
- ✅ Blended reconstructions (per node)  
- ✅ Gating dynamics and entropy over time  
- ✅ Visual concept audit trails

---

## 🚀 Coming Soon

- Top-k cluster visualizers
- Latent traversal tools
- Interactive dashboards for inspecting node behavior
- Support for regression and generation heads

---

## 🧪 Get Started

```bash
python train_cifar.py
python visualize_cifar.py
