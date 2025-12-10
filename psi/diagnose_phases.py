"""
Diagnose: Are we getting good phase separation between different keys?

For associative recall to work, we need:
1. Same key -> same phase (consistency)
2. Different keys -> different phases (separation)
3. Random phases should destructively interfere when summed (orthogonality)
"""

import torch
import torch.nn as nn
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a simple phase encoder like in clifford_memory
dim = 64
n_tokens = 64  # vocab size

# Embedding + phase encoder
embed = nn.Embedding(n_tokens, dim).to(device)
phase_encoder = nn.Linear(dim, dim).to(device)
nn.init.orthogonal_(phase_encoder.weight)

print("=" * 70)
print("PHASE SEPARATION DIAGNOSTIC")
print("=" * 70)

# Test 1: Same token -> same phase?
print("\n1. CONSISTENCY: Same token produces same phase?")
token = torch.tensor([5], device=device)
emb = embed(token)
phase1 = torch.tanh(phase_encoder(emb)) * math.pi
phase2 = torch.tanh(phase_encoder(emb)) * math.pi
diff = (phase1 - phase2).abs().max().item()
print(f"   Max phase difference for same token: {diff:.6f}")
print(f"   [OK] Consistent" if diff < 1e-5 else "   [FAIL] Inconsistent!")

# Test 2: Different tokens -> different phases?
print("\n2. SEPARATION: Different tokens produce different phases?")
all_tokens = torch.arange(n_tokens, device=device)
all_emb = embed(all_tokens)  # [n_tokens, dim]
all_phases = torch.tanh(phase_encoder(all_emb)) * math.pi  # [n_tokens, dim]

# Compute pairwise phase differences
# cos(phase_i - phase_j) = cos(phase_i)cos(phase_j) + sin(phase_i)sin(phase_j)
cos_phases = torch.cos(all_phases)  # [n_tokens, dim]
sin_phases = torch.sin(all_phases)

# Pairwise cosine similarity of phasors
# For each pair (i,j), compute mean(cos(phase_i - phase_j)) across dimensions
similarity_matrix = cos_phases @ cos_phases.T + sin_phases @ sin_phases.T  # [n_tokens, n_tokens]
similarity_matrix = similarity_matrix / dim  # Average across dimensions

# Mask diagonal (self-similarity = 1)
mask = ~torch.eye(n_tokens, dtype=torch.bool, device=device)
off_diag = similarity_matrix[mask]

print(f"   Phase similarity stats (excluding self):")
print(f"   Mean: {off_diag.mean().item():.4f} (want ~0 for orthogonality)")
print(f"   Std:  {off_diag.std().item():.4f}")
print(f"   Max:  {off_diag.max().item():.4f} (high = collision risk)")
print(f"   Min:  {off_diag.min().item():.4f}")

# Test 3: Do random phasors cancel when summed?
print("\n3. INTERFERENCE: Random phasors cancel when summed?")
n_samples = 1000
n_phasors = 8  # Like storing 8 key-value pairs

total_interference = []
for _ in range(n_samples):
    # Pick random tokens
    random_tokens = torch.randint(0, n_tokens, (n_phasors,), device=device)
    embs = embed(random_tokens)
    phases = torch.tanh(phase_encoder(embs)) * math.pi  # [n_phasors, dim]

    # Sum the phasors (like cumsum does)
    phasors = torch.exp(1j * phases)  # [n_phasors, dim]
    summed = phasors.sum(dim=0)  # [dim]

    # Magnitude of sum (should be ~sqrt(n_phasors) for random, ~n_phasors if aligned)
    magnitude = summed.abs().mean().item()
    total_interference.append(magnitude)

mean_mag = np.mean(total_interference)
expected_random = np.sqrt(n_phasors)  # Random walk
expected_aligned = n_phasors  # Perfect alignment

print(f"   Mean magnitude of sum of {n_phasors} phasors: {mean_mag:.4f}")
print(f"   Expected if random (sqrt(n)): {expected_random:.4f}")
print(f"   Expected if aligned (n): {expected_aligned:.4f}")
print(f"   Ratio to random: {mean_mag/expected_random:.2f}x")

# Test 4: Retrieval signal vs noise
print("\n4. RETRIEVAL: Signal vs noise in associative recall")
n_trials = 100
snr_list = []

for _ in range(n_trials):
    # Store 8 key-value pairs
    keys = torch.randint(0, n_tokens, (8,), device=device)
    values = torch.randn(8, dim, device=device)

    # Compute key phases
    key_embs = embed(keys)
    key_phases = torch.tanh(phase_encoder(key_embs)) * math.pi  # [8, dim]
    key_phasors = torch.exp(1j * key_phases)  # [8, dim]

    # Bind and sum (like cumsum final state)
    bound = key_phasors * values.to(torch.complex64)  # [8, dim]
    memory = bound.sum(dim=0)  # [dim] - final memory state

    # Query with one of the stored keys (say index 3)
    query_idx = 3
    query_phase = key_phases[query_idx]  # [dim]
    query_phasor = torch.exp(1j * query_phase)

    # Retrieve
    retrieved = (memory * query_phasor.conj()).real  # [dim]

    # Ground truth
    true_value = values[query_idx]  # [dim]

    # Signal = correlation with true value
    signal = (retrieved * true_value).sum().item()

    # Noise = what we get from other values
    noise_power = ((retrieved - true_value) ** 2).sum().item()
    signal_power = (true_value ** 2).sum().item()

    snr = signal_power / (noise_power + 1e-8)
    snr_list.append(snr)

mean_snr = np.mean(snr_list)
print(f"   Mean SNR (signal/noise power): {mean_snr:.4f}")
print(f"   SNR in dB: {10*np.log10(mean_snr):.2f} dB")
print(f"   " + ("[OK] Good SNR" if mean_snr > 1 else "[FAIL] Poor SNR - noise dominates!"))

# Test 5: How does SNR scale with number of stored items?
print("\n5. SCALING: SNR vs number of stored items")
for n_items in [2, 4, 8, 16, 32]:
    snr_list = []
    for _ in range(100):
        keys = torch.randint(0, n_tokens, (n_items,), device=device)
        values = torch.randn(n_items, dim, device=device)

        key_embs = embed(keys)
        key_phases = torch.tanh(phase_encoder(key_embs)) * math.pi
        key_phasors = torch.exp(1j * key_phases)

        bound = key_phasors * values.to(torch.complex64)
        memory = bound.sum(dim=0)

        query_idx = 0
        query_phasor = torch.exp(1j * key_phases[query_idx])
        retrieved = (memory * query_phasor.conj()).real

        true_value = values[query_idx]
        noise_power = ((retrieved - true_value) ** 2).sum().item()
        signal_power = (true_value ** 2).sum().item()
        snr = signal_power / (noise_power + 1e-8)
        snr_list.append(snr)

    mean_snr = np.mean(snr_list)
    print(f"   n={n_items:2d}: SNR={mean_snr:.4f} ({10*np.log10(mean_snr):.1f} dB)")


# Test 6: HRR-style binding - encode values with their own phasor
print("\n6. HRR-STYLE: Bind value with its own random phasor")
print("   Idea: value_bound = value * random_phasor, then bind with key")
print("   Mismatched retrievals get rotated values that should cancel better")

value_encoder = nn.Linear(dim, dim).to(device)
nn.init.orthogonal_(value_encoder.weight)

for n_items in [2, 4, 8, 16, 32]:
    snr_list = []
    for _ in range(100):
        keys = torch.randint(0, n_tokens, (n_items,), device=device)
        values = torch.randn(n_items, dim, device=device)

        key_embs = embed(keys)
        key_phases = torch.tanh(phase_encoder(key_embs)) * math.pi
        key_phasors = torch.exp(1j * key_phases)

        # NEW: Also encode values with a phasor derived from the value itself
        value_phases = torch.tanh(value_encoder(values)) * math.pi
        value_phasors = torch.exp(1j * value_phases)
        values_rotated = values.to(torch.complex64) * value_phasors

        # Bind with BOTH key phasor and value phasor
        bound = key_phasors * values_rotated
        memory = bound.sum(dim=0)

        query_idx = 0
        query_phasor = torch.exp(1j * key_phases[query_idx])
        retrieved_complex = memory * query_phasor.conj()

        # Unbind the value phasor too
        query_value_phase = torch.tanh(value_encoder(values[query_idx:query_idx+1])) * math.pi
        query_value_phasor = torch.exp(1j * query_value_phase)
        retrieved = (retrieved_complex * query_value_phasor.conj()).real.squeeze(0)

        true_value = values[query_idx]
        noise_power = ((retrieved - true_value) ** 2).sum().item()
        signal_power = (true_value ** 2).sum().item()
        snr = signal_power / (noise_power + 1e-8)
        snr_list.append(snr)

    mean_snr = np.mean(snr_list)
    print(f"   n={n_items:2d}: SNR={mean_snr:.4f} ({10*np.log10(mean_snr):.1f} dB)")


# Test 7: TRUE HRR - circular convolution binding
print("\n7. TRUE HRR: Circular convolution binding")
print("   bind(a, b) = ifft(fft(a) * fft(b))")
print("   unbind(bound, a) = circular correlation")

def hrr_bind(a, b):
    """Circular convolution via FFT - keep real since inputs are real"""
    return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=a.shape[-1])

def hrr_unbind(bound, a):
    """Circular correlation = convolution with time-reversed a"""
    # For real signals, correlation is ifft(fft(x) * conj(fft(y)))
    # But we need to use the "involution" of a for proper unbinding
    # In HRR, unbind uses the approximate inverse which is just the flip
    a_inv = torch.flip(a, dims=[-1])  # Approximate inverse for random vectors
    # Shift to align
    a_inv = torch.roll(a_inv, 1, dims=-1)
    return hrr_bind(bound, a_inv)

for n_items in [2, 4, 8, 16, 32]:
    snr_list = []
    for _ in range(100):
        # Random keys and values (just random vectors for HRR)
        # Keys should be normalized for HRR to work well
        keys = torch.randn(n_items, dim, device=device)
        keys = keys / keys.norm(dim=-1, keepdim=True)  # Normalize
        values = torch.randn(n_items, dim, device=device)

        # Bind each key-value pair and sum
        memory = torch.zeros(dim, device=device)
        for i in range(n_items):
            bound = hrr_bind(keys[i], values[i])
            memory = memory + bound

        # Query with first key
        query_idx = 0
        retrieved = hrr_unbind(memory, keys[query_idx])

        true_value = values[query_idx]
        # For HRR, we look at correlation/similarity not exact match
        similarity = (retrieved * true_value).sum().item() / (retrieved.norm().item() * true_value.norm().item() + 1e-8)

        noise_power = ((retrieved - true_value) ** 2).sum().item()
        signal_power = (true_value ** 2).sum().item()
        snr = signal_power / (noise_power + 1e-8)
        snr_list.append(snr)

    mean_snr = np.mean(snr_list)
    print(f"   n={n_items:2d}: SNR={mean_snr:.4f} ({10*np.log10(mean_snr):.1f} dB)")

# Test 8: Multiple independent memory banks (Bloom filter style)
print("\n8. MULTI-BANK: K independent memories, vote on retrieval")
print("   Each bank uses different random projection for phases")
print("   Retrieval requires majority agreement across banks")

n_memory_banks = 16

# Create multiple independent phase encoders
bank_encoders = [nn.Linear(dim, dim).to(device) for _ in range(n_memory_banks)]
for enc in bank_encoders:
    nn.init.orthogonal_(enc.weight)

for n_items in [2, 4, 8, 16, 32]:
    snr_list = []
    for _ in range(100):
        keys = torch.randint(0, n_tokens, (n_items,), device=device)
        values = torch.randn(n_items, dim, device=device)
        key_embs = embed(keys)

        # Store in all banks
        bank_memories = []
        bank_key_phases = []
        for enc in bank_encoders:
            key_phases = torch.tanh(enc(key_embs)) * math.pi
            key_phasors = torch.exp(1j * key_phases)
            bound = key_phasors * values.to(torch.complex64)
            memory = bound.sum(dim=0)
            bank_memories.append(memory)
            bank_key_phases.append(key_phases)

        # Query from all banks
        query_idx = 0
        bank_retrievals = []
        for i, (mem, phases) in enumerate(zip(bank_memories, bank_key_phases)):
            query_phasor = torch.exp(1j * phases[query_idx])
            retrieved = (mem * query_phasor.conj()).real
            bank_retrievals.append(retrieved)

        # Average across banks (could also use median or voting)
        retrieved = torch.stack(bank_retrievals).mean(dim=0)

        true_value = values[query_idx]
        noise_power = ((retrieved - true_value) ** 2).sum().item()
        signal_power = (true_value ** 2).sum().item()
        snr = signal_power / (noise_power + 1e-8)
        snr_list.append(snr)

    mean_snr = np.mean(snr_list)
    print(f"   n={n_items:2d}: SNR={mean_snr:.4f} ({10*np.log10(mean_snr):.1f} dB)")


# Test 9: What if we project values to be orthogonal?
print("\n9. ORTHOGONAL VALUES: Project values to be mutually orthogonal")
print("   If values are orthogonal, mismatched retrievals should cancel!")

for n_items in [2, 4, 8, 16, 32]:
    snr_list = []
    for _ in range(100):
        keys = torch.randint(0, n_tokens, (n_items,), device=device)

        # Create orthogonal values using QR decomposition
        random_matrix = torch.randn(dim, n_items, device=device)
        Q, _ = torch.linalg.qr(random_matrix)
        values = Q[:, :n_items].T  # [n_items, dim], orthonormal rows

        key_embs = embed(keys)
        key_phases = torch.tanh(phase_encoder(key_embs)) * math.pi
        key_phasors = torch.exp(1j * key_phases)

        bound = key_phasors * values.to(torch.complex64)
        memory = bound.sum(dim=0)

        query_idx = 0
        query_phasor = torch.exp(1j * key_phases[query_idx])
        retrieved = (memory * query_phasor.conj()).real

        true_value = values[query_idx]
        noise_power = ((retrieved - true_value) ** 2).sum().item()
        signal_power = (true_value ** 2).sum().item()
        snr = signal_power / (noise_power + 1e-8)
        snr_list.append(snr)

    mean_snr = np.mean(snr_list)
    print(f"   n={n_items:2d}: SNR={mean_snr:.4f} ({10*np.log10(mean_snr):.1f} dB)")

print("\n" + "=" * 70)
