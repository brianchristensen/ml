# SOC World Modeling Architecture Summary

This document summarizes the core principles and design decisions for the Self-Organized Critical (SOC) world modeling architecture.

## Core Philosophy

1. **No SGD/Backprop** - All learning is local, using delta rules and Hebbian updates
2. **Always-On Brain** - The system runs continuously like a game engine, not in batch epochs
3. **Self-Organized Criticality** - System self-tunes to the edge of chaos for optimal computation
4. **Prediction is Computation** - Each unit predicts its next state; learning minimizes prediction error
5. **Complementary Learning** - Dual memory systems prevent catastrophic interference

---

## Complementary Learning System (CRITICAL)

**This is the key mechanism that prevents catastrophic interference (forgetting earlier learning when acquiring new information).**

Inspired by the hippocampus (fast, specific) / neocortex (slow, generalizing) division in biological brains.

### The Problem

Error-driven contrastive learning (delta rule) can push weights **NEGATIVE**, actively erasing earlier associations:
```
# When learning B->C, the error signal can weaken A->B connections
dW = lr * pre * (target - actual)  # Can be negative!
```

### The Solution: Dual Memory with Additive Hebbian

| Weight Array | Learning Type | Can Go Negative? | Purpose |
|--------------|--------------|------------------|---------|
| `W_dyn` / `W_transition` | Error-driven contrastive | YES | Fast, volatile, recent learning |
| `W_longterm` | **Purely additive Hebbian** | **NO** | Slow, stable, accumulated memory |

**The learning rule for W_longterm:**
```python
# PURELY ADDITIVE - only strengthens, NEVER weakens
pre_active = (input_trace > 0.3)   # Which inputs were active
post_active = (A_target > 0.5)     # Which targets should activate

# Hebbian: strengthen where BOTH pre AND post are active
# NO error term = NO negative updates = NO interference
dW_longterm = lr * (pre_active & post_active)  # Always >= 0
W_longterm = clip(W_longterm + dt * dW_longterm, 0, 1)
```

### How They're Used

```python
# In dynamics, blend the two memories
effective_weights = (1 - longterm_blend) * W_dyn + longterm_blend * W_longterm

# For action-conditioned transitions (planning_soc.py):
# longterm_blend = 1.0 (rely ONLY on stable memory for predictions)
```

### Experience Replay During Dreams

Stored experiences are replayed during dream mode to continuously reinforce W_longterm:
```python
def _dream_learning(self, dt):
    # ... temporal Hebbian ...

    # Replay stored experiences to consolidate long-term memory
    if random() < 0.3:  # 30% of dream steps
        self._replay_to_longterm(dt)
```

### Results

| Before Complementary Learning | After |
|------------------------------|-------|
| 66.7% action prediction (room_A transitions failing) | **100% action prediction** |
| Dream mode degraded learning | Dream mode improves learning |
| Catastrophic interference | **No interference** |

---

## Key Files

- `predictive_soc.py` - Base architecture with prediction-based learning
- `planning_soc.py` - Extended with action-conditioning and FMC planning

## Two-Stage Learning (Separation of Concerns)

The architecture uses **parallel weight networks** for stability:

| Weight Array | Purpose | Learning Rate | Used For |
|--------------|---------|---------------|----------|
| `W_pred` | Internal model / imagination | Fast (0.01) | Predictions, planning |
| `W_dyn` | Actual dynamics / attractors | Slow consolidation | Real state evolution |
| `W_action_pred` | Action gating (internal) | Fast (0.08) | Imagination |
| `W_action_dyn` | Action gating (stable) | Slow consolidation | Real dynamics |

**Why separation?**
- Fast learning on W_pred allows rapid adaptation without destabilizing attractors
- W_dyn consolidates only when prediction confidence is high
- This prevents catastrophic forgetting while enabling exploration

## State Variables Per Unit

```python
A          # Activation [0, 1]
theta      # Phase (for binding/grouping)
E          # Energy (metabolic constraint)
tau        # Adaptive timescale (emergent hierarchy)
A_pred     # Predicted next activation
A_target   # Target activation (for contrastive learning)
A_action   # Current action activation (for world modeling)
```

## Core Learning Rules

### 1. Delta Rule (Prediction Learning)
```
dW_pred = lr * error_post * A_pre
# "Blame the active inputs for prediction errors"
```

### 2. Contrastive Learning (Three-Factor)
```
dW = lr * pre * post * local_error
# Forward: input_trace -> target
# Backward: target -> input_trace (bidirectional)
```

### 3. Temporal Hebbian (Dream Mode)
```
dW = lr * A_prev[source] * A[target] * (1 + surprise)
# Strengthen FROM previously-active TO currently-active
```

### 4. Action Gating Learning
```
dW_action[k, conn] = lr * A_action[k] * pred_error[post] * A[pre]
# Learn which actions enable which transitions
```

## SOC Mechanisms

1. **Homeostatic Threshold** - Adjusts firing threshold to maintain target activity level
2. **Synaptic Scaling** - Units that fire too much have incoming weights scaled down
3. **Avalanche Tracking** - Power-law avalanche sizes indicate criticality

## Dream Mode (Always-On Consolidation)

When no external input:
- Network runs on spontaneous noise
- Temporal Hebbian learning strengthens sequential associations
- Actions can be randomly sampled to practice world model
- This consolidates memories without disrupting attractor dynamics

## Action-Conditioned World Model

Actions **gate** dynamics rather than adding activation:
```python
action_gate = 1 + sum(A_action[k] * W_action[k, connection])
effective_weight = W_dyn * action_gate
```

This creates: "If I do action A in state S, I predict next state S'"

## Fractal Monte Carlo (FMC) Planning

From Fractal AI paper, integrated with SOC:

| FMC Concept | SOC Implementation |
|-------------|-------------------|
| Walker | Parallel activation pattern (slot) |
| Simulator | **W_longterm** dynamics (MUST use stable memory!) |
| Distance | L2 norm between slot activations |
| Reward | Cosine similarity to goal + prediction confidence |
| Virtual Reward | Reward^alpha * Distance |
| Cloning | Copy high-VR slot over low-VR slot |

**CRITICAL**: Slot simulation must use `W_longterm` (stable learned model), NOT `W_transition` (corrupted volatile memory). This was a key bug that prevented planning from working.

**Key formula:**
```
VR = Reward^alpha * Distance
# alpha=0: pure exploration ("common sense" mode)
# alpha=1: balanced exploitation/exploration
# alpha->inf: pure exploitation
```

**Common Sense Mode (alpha=0):**
At alpha=0, VR = Distance, which maximizes entropy of reachable futures.
This maps to SOC criticality - the system explores all possible futures.

## Adaptive Timescales (Emergent Hierarchy)

Each unit has its own time constant tau:
- Small tau = fast response = low-level features
- Large tau = slow integration = high-level patterns

Tau adapts based on input variance:
```
target_tau = k / (input_variance + epsilon)
d_tau = lr * pred_error * (target_tau - current_tau)
```

## Interface Design (Game Engine Style)

The system runs as a continuous loop:
```python
while True:
    # Check for input (sensory)
    if external_input_available:
        mind.inject_text(input, strength)
        mind.set_action(action)
        mind.set_target(outcome)  # For learning

    # Always step (dream mode when no input)
    dream_mode = not external_input_available
    mind.step(dt, dream_mode=dream_mode)

    # Planning can run in parallel (imagination)
    if need_to_plan:
        action_sequence = mind.plan(goal_pattern)
```

## Benchmarks

1. **Action Prediction**: Can network learn action->outcome?
2. **Planning**: Can FMC find action sequences to goals?
3. **Dream Consolidation**: Does dreaming improve world model?

## Current Status (December 2025)

The basic architecture is implemented in `planning_soc.py`. Current benchmark results:

### Base Architecture (predictive_soc.py)
- **Contrastive Prediction**: PASS (0.25 overlap trained vs 0.12 untrained = 2.14x signal/noise)
- **Sequence Learning**: Working (cat->meow->purr chains learned)
- **Dream Consolidation**: Working (temporal Hebbian strengthens sequences)

### Action-Conditioned World Model (planning_soc.py)
- **Action Prediction**: PASS 100% accuracy (6/6 transitions learned)
  - ALL transitions work reliably (0.25-0.55 overlap)
  - Catastrophic interference SOLVED via complementary learning system
- **FMC Planning**: PASS (goal overlap 0.57, finds action sequences to goals)
  - Slot simulation must use W_longterm (stable memory), not W_transition
- **Dream Consolidation**: PASS (0% -> 66.7% after dreaming)

### Key Implementation Details
1. **Action-Specific Transition Weights**: `W_transition[action_dim]` - each action has its OWN weight matrix
2. **Phase Binding**: Actions set distinctive phases making (state, action_0) orthogonal to (state, action_1)
3. **Complementary Learning System**: Dual memory (see below)

### Solved: Catastrophic Interference

The key insight: **W_transition uses error-driven contrastive learning which can PUSH WEIGHTS NEGATIVE, destroying earlier learning. W_longterm uses PURELY ADDITIVE Hebbian learning which only ACCUMULATES associations.**

Solution implemented:
1. **W_transition (volatile, short-term)**: Contrastive learning, can go negative
2. **W_longterm (stable, long-term)**: Purely additive Hebbian (only strengthens, never weakens)
3. **longterm_blend = 1.0**: Predictions use ONLY W_longterm (ignoring corrupted W_transition)
4. **action_weight_contribution = 0.9**: Action-specific weights dominate base dynamics

The learning rule for W_longterm:
```python
pre_active = (input_trace > 0.3)  # Which inputs were active
post_active = (A_target > 0.5)     # Which targets should activate
# Hebbian: strengthen where BOTH pre AND post are active
# NO error term = NO unlearning = NO interference
dW_longterm = lr * pre_active * post_active  # Always >= 0
```

### Working Components
- Action-specific transition weights (no action interference!)
- Phase binding for state+action differentiation
- Continuous game-engine style operation
- Complementary learning system (prevents catastrophic forgetting)
- Dream consolidation via experience replay
- FMC planning infrastructure

## Next Steps

**All core benchmarks now pass (100%)!** Remaining work for practical use:

1. **Sensory Interface**: Connect to continuous inputs (images, sensors)
2. **Persistence**: Save/load W_longterm weights between sessions
3. **Scalability**: Optimize for larger networks (10k+ units)
4. **Add Visualization**: Port interactive visualization from predictive_soc.py
5. **Slot-Based Objects**: For compositional generalization
6. **Information Bottleneck**: For learning abstractions

## Known Architecture Principles (for future sessions)

1. **No epochs, always-on**: Run like a game engine, not batch training
2. **Complementary Learning**:
   - Fast system (W_transition): Error-driven, volatile, can go negative
   - Slow system (W_longterm): Additive Hebbian only, never weakens, stable memory
   - Use longterm_blend=1.0 to rely on stable memory for predictions
3. **Purely additive Hebbian prevents interference**: Only accumulate, never subtract
4. **Dream mode consolidates via replay**: Replays stored experiences to W_longterm
5. **Actions gate, not add**: Actions modulate which connections are active
6. **FMC for planning**: Virtual Reward = Reward^alpha * Distance
7. **FMC simulation MUST use W_longterm**: Imagination uses stable memory, not volatile
8. **SOC criticality = FMC common sense**: Both maximize entropy of futures
