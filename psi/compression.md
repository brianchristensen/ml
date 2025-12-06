Both PSI and Mamba compress history into fixed-size state, so why would
  PSI retrieve better?

  Mamba's compression:
  - State size: d_state × d_model = 16 × 64 = 1,024 floats
  - Everything goes through selective gating (content-dependent)
  - Information gets "blended" into this matrix via SSM dynamics

  PSI's compression:
  - Position block: cumsum over d_model = 64 floats (real+imag = 128)
  - Content block: cumsum over n_oscillators × d_model = 64 × 64 = 4,096
  complex floats

  Actually wait - PSI's content block has a much larger effective state
  size. Each oscillator maintains a separate cumsum of value vectors. So
  PSI has ~4K complex values vs Mamba's ~1K floats.

  But that's not the whole story. The key difference is how retrieval
  works:

  Mamba retrieval:
  - Output = linear function of compressed state
  - No explicit "query" mechanism
  - Must learn to route information through time-invariant structure

  PSI retrieval:
  - Query phase is computed from current input
  - retrieved = Σ memory_k * conj(query_k)
  - Phase matching acts like content-addressable lookup
  - Similar phases constructively interfere, different phases cancel

  So PSI has:
  1. Larger state (4K complex vs 1K float) - more capacity
  2. Explicit query mechanism - can "ask" for specific content
  3. Interference-based retrieval - phase matching provides selectivity

  The multi-query task reveals this: PSI can query multiple times without
  the state changing, while Mamba's output is just a linear readout of
  whatever state it has.

  -----

  Matched State Capacity Test Results:

  | Model            | State Size    | Parameters | Accuracy |
  |------------------|---------------|------------|----------|
  | PSI              | ~8,320 floats | 42,337     | 16.2%    |
  | Mamba d_state=16 | ~4,096 floats | 69,793     | 3.5%     |
  | Mamba d_state=32 | ~8,192 floats | 82,081     | 3.5%     |

  Random baseline: 3.1%

  Key finding: Even when Mamba has matched state capacity (d_state=32 gives
   ~8,192 floats vs PSI's ~8,320 floats), PSI still dramatically
  outperforms it on multi-query recall:
  - PSI: 16.2%
  - Mamba (matched): 3.5%

  The verdict is clear: It's not about the compression ratio - it's about
  the retrieval mechanism.

  PSI's phase-based addressing enables explicit content-addressable lookup:
  1. Keys encode as phases → stored via exp(i*θ_key)
  2. Query phases → retrieve via exp(-i*θ_query) (conjugate multiplication)
  3. Phase matching → when θ_query ≈ θ_key, the value is reconstructed

  Mamba's SSM compresses state into a linear recurrence, but retrieval is a
   single linear readout - there's no mechanism for "looking up" specific
  stored information by content. Adding more state (d_state=32) just gives
  it more compression capacity, but doesn't enable selective retrieval.

  This is strong evidence that PSI's niche is content-addressable memory
  tasks, where the explicit phase binding mechanism provides a structural
  advantage over SSM-style compression.