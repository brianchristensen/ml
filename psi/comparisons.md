# PSI Architecture Comparison Results

## Overview

PSI (Phase-Space Integration) is a parallel sequence processing architecture that uses cumsum-based memory with phase/frequency encoding. This document summarizes benchmark results comparing PSI against baseline architectures.

**Important Note:** All LSTM comparisons use a manual PyTorch implementation (no cuDNN optimization) for fair comparison, since PSI also lacks hardware-specific optimization.

---

## Key Findings

### Speed Advantages

| Comparison | PSI Speedup | Notes |
|------------|-------------|-------|
| PSI vs Manual LSTM | **27x faster** | Sequence processing (800 steps) |
| PSI vs Neural ODE | **20x faster** | Same accuracy on trajectory prediction |
| PSI vs Manual LSTM | **2.7x faster** | Training time on 3-body problem |

### Memory Efficiency

| Architecture | Memory Scaling | Notes |
|--------------|----------------|-------|
| PSI | O(n) | Linear with sequence length |
| Transformer | O(n²) | Quadratic - 757 MB at 800 steps |
| Manual LSTM | O(n) | Linear, but sequential |

At 800 steps: PSI uses 187 MB vs Transformer's 757 MB (**4x more efficient**).

---

## Benchmark 1: Comprehensive (Lorenz System)

**Setup:** Lorenz attractor prediction, hidden_dim=64, 4 layers

### Sample Efficiency
| Data % | PSI | Transformer | Manual LSTM |
|--------|-----|-------------|-------------|
| 5% | 0.055 | 0.054 | 0.234 |
| 10% | 0.044 | 0.027 | 0.142 |
| 25% | 0.013 | 0.015 | 0.071 |
| 50% | 0.008 | 0.009 | 0.055 |
| 100% | 0.004 | 0.005 | 0.032 |

**Winner:** PSI/Transformer tied, both significantly better than Manual LSTM

### Computational Efficiency
| Seq Length | PSI (ms) | Transformer (ms) | Manual LSTM (ms) |
|------------|----------|------------------|------------------|
| 50 | 3.8 | 1.8 | 27.7 |
| 100 | 2.6 | 1.7 | 60.6 |
| 200 | 3.1 | 2.7 | 120.6 |
| 400 | 5.6 | 6.3 | 235.5 |
| 800 | 16.2 | 19.7 | 444.2 |

**Winner:** PSI (27x faster than Manual LSTM, comparable to Transformer but O(n) memory)

### Scaling Behavior
- PSI: 4.24x slowdown for 16x longer sequences (sub-linear)
- Transformer: 11.1x slowdown (approaching O(n²))
- Manual LSTM: 16.0x slowdown (linear, as expected)

---

## Benchmark 2: 3-Body Problem

**Setup:** Gravitational 3-body dynamics, 500 trajectories, 50 timesteps

| Metric | PSI | Manual LSTM | Winner |
|--------|-----|-------------|--------|
| Parameters | 933K | 826K | LSTM |
| Training Time | **61.5s** | 167.7s | **PSI (2.7x)** |
| Val Loss (1-step) | **0.038** | 0.096 | **PSI (2.5x)** |
| Rollout MSE (30-step) | **1.34** | 1.39 | **PSI (3.8%)** |

**Winner:** PSI wins on all performance metrics

---

## Benchmark 3: Neural ODE Replacement

**Setup:** Lorenz trajectory prediction from initial condition

| Model | 100-step MSE | 200-step MSE | Time (ms) |
|-------|--------------|--------------|-----------|
| Neural ODE | 65.30 | 75.94 | 26.28 |
| PSI Multi-Step | 65.22 | 76.11 | **1.33** |

**Result:** PSI is **20x faster** with equivalent accuracy

### With History Context
When given 50 steps of history to predict 100 future steps:

| Metric | Value |
|--------|-------|
| Test MSE | 39.5 (vs 76 without history) |
| Time | 1.57 ms for 150 total steps |
| vs Baseline | 3.83x better than repeat-last |

**Key insight:** PSI's cumsum memory naturally accumulates trajectory history, improving prediction accuracy.

---

## Benchmark 4: Integration Tasks

Tasks where cumsum is the ground truth operation.

| Task | PSI | Transformer | Manual LSTM | Winner |
|------|-----|-------------|-------------|--------|
| Running Sum | 4.34 | 4.31 | 9.41 | Transformer |
| Running Average | 0.000046 | 0.000046 | 0.000358 | Tie |
| Counting | **1321** | 1379 | 1352 | **PSI** |
| Integration | **0.00053** | 0.00187 | 0.0049 | **PSI (3.5x)** |

**Winner:** PSI wins on integration of smooth functions (3.5x better than Transformer)

---

## Architecture Summary

### What PSI Is
- A parallel sequence processor using cumsum-based memory
- Essentially a learned Euler integrator with per-dimension integration rates
- Uses trigonometric (sin/cos) basis for phase-modulated addressing

### Genuine Advantages
1. **20-27x faster** than baseline sequential architectures
2. **O(n) memory** vs Transformer's O(n²)
3. **Parallel by design** - cumsum has O(log n) parallel depth
4. **Multi-step prediction** - naturally outputs K future steps via cumsum integration
5. **Competitive accuracy** - matches or beats baselines on dynamics tasks

### Limitations
1. Not magic for accuracy - comparable to other architectures
2. cuDNN LSTM is faster in practice (but that's decades of optimization)
3. The "phase space" interpretation is loose - it's not discovering diffeomorphisms

### Best Use Cases
- Neural ODE replacement (same accuracy, 20x faster)
- Long sequence dynamics prediction
- Settings without optimized RNN kernels (TPU, custom hardware)
- Memory-constrained scenarios where Transformer won't fit

---

## Future Directions

1. **Custom CUDA kernel** - Could achieve speedups similar to cuDNN LSTM
2. **Selective gating** - Mamba-style input-dependent selectivity
3. **Wider hidden dimensions** - Width helps more than depth for PSI
4. **Multi-step output** - Predict K future steps in one pass, integrate via cumsum
