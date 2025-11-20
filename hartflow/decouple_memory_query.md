# Decouple first attempt summary - TPI Associative Recall Experiments

## What I Attempted

### 1. Fully Decoupled Key/Query Phases
**Concept:** Separate omega_key and omega_query to enable associative recall
```python
phi_key = cumsum(omega_key)      # Storage trajectory
phi_query = cumsum(omega_query)  # Retrieval trajectory
memory = cumsum(x * exp(i*phi_key))
retrieved = memory * exp(-i*phi_query)
```

**Results:** ❌ **FAILED BADLY**
- Val BPC: 2.43 (worse than baseline ~2.1-2.2)
- Test BPC: 2.51
- Generation: Completely broken (HTML artifacts, gibberish)
- Training curve: Worse than baseline

**Why it failed:**
- Too much freedom: Two independent trajectories learned incoherent patterns
- Gradient imbalance: query_grad ≈ 2x key_grad
- No incentive for alignment between key and query phases
- Model couldn't learn coherent semantic space

### 2. Query Offset Architecture
**Concept:** Single semantic trajectory + learned content-based offset
```python
phi = cumsum(omega)                    # Coherent semantic trajectory
query_offset = learned_offset(x)       # Content-based "look elsewhere"
phi_query = phi + query_offset         # Query from offset position
```

**Results:** ⏸️ **TRAINING ISSUES**
- Process hung/stuck during initialization
- No training output produced
- Had to kill process

**Theory was sound:**
- Maintains single coherent semantic space (phi)
- Adds content-based query capability (offset)
- Gradient test showed offset_grad 100x larger than omega_grad (strong learning signal)
- But couldn't get actual training to complete

### 3. Reverted to Simple Single-Omega TPI
**Current state:** Back to baseline working version
```python
phi = cumsum(omega)
memory = cumsum(x * exp(i*phi))
retrieved = memory * exp(-i*phi)
```

**Status:** 🔄 **Training in progress** (was attempting when I had issues)

## Previous Baseline Results (for comparison)

**Before decoupling experiments:**
- Hard window (64 tokens): Val BPC 2.13, Test BPC 2.21
- Learned exponential decay (13-17 tokens): Val BPC 2.09, Test BPC 2.17
- Generation: Poor quality but better than decoupled version

**Recent runs I observed:**
- Run 8c6da8: Val BPC 2.74, Test BPC 2.80
- Run 3bec94 (with diversity loss): Val BPC 2.73, Test BPC 2.81

**Target:** BPC < 2.0 (not achieved yet)

## Key Insights from Theory Work

### Why TPI Works
- **Path integral attention:** Retrieves based on accumulated semantic distance, not position
- **Holographic interference:** Phase coherence determines what's retrieved
- **Automatic filtering:** Different phases → destructive interference

### The Associative Recall Problem
**What TPI does well:**
- Semantic trajectory: "capital of France" → accumulates meaning through sequence

**What TPI struggles with:**
- Associative recall: Query "capital of France" should retrieve "Paris"
- Problem: Paris was stored at phi_Paris, but query arrives at phi_query ≠ phi_Paris
- Same phase for storage and retrieval = limited associative capability

### The Decoupling Idea (and why it failed)
**Concept:** Learned phase seeking
- Model learns omega_query to produce phases that match where relevant content was stored
- Gradient descent learns: "capital of France" → phi_query ≈ phi_key(Paris)

**Why it didn't work:**
- Too much freedom without structure
- Lost coherent semantic space
- Model optimized for local prediction (BPC) not associative recall

## Architecture Evolution

```
1. Original TPI (working baseline)
   └─> Single omega, single phi
   └─> BPC ~2.1-2.2, poor generation

2. Added 4x expansion + GELU
   └─> BPC 2.22 (best!), generation broke

3. Added windowing (prevent error accumulation)
   └─> Hard window: BPC 2.13-2.21
   └─> Learned window: Model chose 13-17 tokens
   └─> Generation still poor

4. Added dropout (reduce overfitting)
   └─> Currently at 0.1-0.3 dropout rate

5. Tried decoupled key/query ❌
   └─> BPC 2.43, generation worse

6. Tried query offset ⏸️
   └─> Training hung

7. Back to simple TPI (current)
   └─> Waiting for results...
```

## Current Model Configuration

```python
# novel_attention.py (as of night work)
- Single omega (semantic trajectory)
- Unbounded cumsum (no windowing)
- 4x expansion + GELU + Dropout(0.1) in to_out
- Holographic memory with magnitude weighting
- Phase-coherent retrieval

# Hyperparameters (test_char_lm.py)
- dim: 128
- num_layers: 8
- lr: 3e-3
- dropout: 0.1
- epochs: 5
```

## Recommendations for Morning

### Immediate Actions

1. **Check if simple TPI training completed**
   - Look at `training_simple.log`
   - Check `novel_attention_charlm.pt` and `novel_attention_charlm_final.pt`
   - Run `python generate_text.py` to test generation quality

2. **If BPC still > 2.5:**
   - **Try removing dropout** (might be hurting more than helping)
   - **Try removing 4x expansion** (go back to simple linear projection)
   - **Add back FFN** (we know TPI + FFN worked at ~2.4 BPC)

3. **If generation still poor:**
   - The unbounded cumsum might be causing distribution shift
   - Consider fixed windowing (64 or 128 tokens) during both train and generation
   - Or try scheduled sampling (train on own predictions sometimes)

### Longer-term Directions

#### Option A: Accept TPI limitations
- TPI is good at semantic trajectory, not associative recall
- Focus on getting BPC < 2.0 with good generation
- Don't force associative recall if it breaks the mechanism

#### Option B: Try constrained decoupling
Instead of fully independent trajectories, try:
```python
omega_base = self.to_omega(x)
omega_key = omega_base  # Storage uses base trajectory
query_adjustment = self.to_query_adj(x) * 0.1  # Small learned adjustment
omega_query = omega_base + query_adjustment  # Query is slight variation
```
This keeps them coupled but allows small learned differences.

#### Option C: Hybrid with explicit associative memory
Add a separate small associative memory module alongside TPI:
- TPI handles semantic trajectory (what it's good at)
- Associative module handles content-based retrieval
- Combine their outputs

## Files Modified

- `novel_attention.py` - Main TPI implementation
  - Currently: Simple single-omega version
  - History: Had decoupled phases, query offset (reverted)

- `test_char_lm.py` - Training script (unchanged)
- `generate_text.py` - Generation test (unchanged)
- `check_window.py` - Diagnostic (updated for decay params, but not relevant now)

## Key Learnings

1. **Simplicity matters:** More complex ≠ better. Single-omega baseline was working.

2. **Training objective misalignment:** Model optimizes for next-char prediction (BPC), not generation quality or associative recall.

3. **Holographic memory principle:** Decoupling violated the classical holography principle (same reference for store/retrieve). Extensions possible but require care.

4. **Gradient signal strength:** Query offset showed 100x stronger gradients than omega, suggesting it could learn if training completed.

5. **Distribution shift:** Unbounded cumsum during training (256 tokens) vs generation (500+ tokens) causes phi magnitude mismatch.

## What Didn't Get Tested

Due to training/time issues:
- Query offset architecture (training hung)
- Fixed moderate windowing (64-128 tokens) on simple TPI
- Removing dropout from current config
- Smaller expansion (2x instead of 4x)
- Adding FFN back to simple TPI

## Status at End of Night

- ✅ Identified why decoupled phases failed
- ✅ Designed better alternative (query offset)
- ✅ Reverted to working baseline
- ⏸️ Simple TPI training status unknown (process issues)
- ❌ Did not achieve BPC < 2.0
- ❌ Did not fix generation quality
- ❌ Did not enable associative recall

## Next Steps for User

1. Check training logs and results
2. Test generation quality
3. Decide: iterate on TPI or try different approach
4. Consider whether associative recall is achievable within TPI constraints or if it needs a different mechanism

---

Good luck! The theory work was solid even if the practical results weren't what we hoped for.
