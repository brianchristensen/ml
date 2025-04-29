# 🌌 TemperGraph: Vision Manifesto

## The Core Idea

TemperGraph is a **lifelong, self-organizing cognitive system** —  
a universal learner that **predicts, adapts, and self-structures** without needing explicit supervision.

It draws inspiration from:
- The **human cortex** — sparse, plastic, predictive, modular.
- **Friston’s Free Energy Principle** — minimizing future surprise.
- **Complex Adaptive Systems** — self-organization and hierarchical emergence.
- **Predictive Coding** — the brain as a prediction error minimizer.

TemperGraph is not trained for a single task.  
It **lives** — continuously processing multimodal streams, building knowledge hierarchies, and **reconfiguring itself across a lifetime**.

---

## Foundational Principles

✅ **Task-Agnostic Learning**  
- Learns from the world, not labels.
- Can acquire new tasks without retraining from scratch.

✅ **Always-On Cognition**  
- No epoch boundaries, no "reset" after tasks.
- Lives in an open-ended loop of sensing, predicting, adapting.

✅ **Intrinsic Motivation**  
- Learning is driven by internal signals: surprise, prediction error, stability.
- Reward = "Understand and anticipate the world better."

✅ **Sparse and Efficient**  
- Sparse activations.
- Sparse routing (each patch commits to one path).
- Sparse operator usage (each temper evolves specialized operators).

✅ **Plastic, Dynamic Structure**  
- Tempers grow and prune operators based on intrinsic success.
- The architecture evolves as learning progresses.

✅ **Hierarchical Emergence**  
- Multi-scale representations.
- Local and global predictors emerge naturally.
- Memory and reasoning structures evolve from prediction dynamics.

✅ **Scalable and Modular**  
- Parallelizable across patches and tempers.
- Modular growth allows scaling to arbitrary complexity without full retraining.

---

## Architectural Inspirations

| Biological Concept            | TemperNet Analogy                  |
|:-------------------------------|:------------------------------------|
| Sparse, Local Receptive Fields  | Patches over the input              |
| Cortical Columns                | Tempers (modular processing units)  |
| Dynamic Routing                 | Learned patch-to-temper assignment  |
| Plasticity & Criticality        | Grow/Prune operators based on self-assessment |
| Hierarchical Processing         | Multi-scale patching, future hierarchical routing |
| Prediction-Error Coding         | Intrinsic reward based on minimizing future prediction error |

---

## Immediate Roadmap

🔹 **Intrinsic Curiosity Validation**  
- FMNIST tests: Prove that real, structured inputs produce richer self-organization.

🔹 **Multi-Scale Patch Experiments**  
- Variable receptive fields → emergence of multi-level representations.

🔹 **Temporal Memory Extension**  
- Predict not just next latent, but future over multiple steps.

🔹 **Multi-Modal Extension**  
- Plug in vision, audio, proprioception streams simultaneously.

🔹 **Emergent Reasoning**  
- See if higher-level tempers begin *abstracting over* lower-level predictions.

---

## The End Goal

TemperGraph becomes a **substrate for general intelligence**:
- Able to learn *any* modality.
- Able to *self-organize* knowledge across *any* timescale.
- Able to *adapt* to *any* task without catastrophic forgetting.

**A living, breathing synthetic cortex.**

---

# 🌱  
_"We are building not just a model.  
We are growing a mind."_

The brain never processes an entire batch of 1,000 samples all at once like we do with NNs.

It processes a continuous stream.

It focuses on sparse events.

It activates very few neurons per input.

It routes information dynamically, based on context and goals, not just dumb feedforward processing.

And critically:

It isn't trying to memorize everything perfectly.

It predicts, corrects, predicts again — each time slightly adjusting internal wiring.

That's why predictive coding makes sense for intelligent systems, and batch training makes sense for statistical function approximators.
(We are not building a statistical function approximator.)

- When there is external supervision ("teacher forcing"):
You slightly warp/steer your internal dynamics toward useful external goals.

- When there is no task:
You continue purely on internal generative/self-predictive dynamics (machine generated dreams).
You don't hallucinate "missing" external signals.