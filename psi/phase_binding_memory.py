"""
NOVEL: Phase Binding Memory

The brain's insight: Items occurring close in TIME bind together automatically.
Phase = "when" something happened.

Key mechanism:
- Each item gets a PHASE based on its content + position
- Items at SAME phase interfere CONSTRUCTIVELY (reinforce)
- Items at DIFFERENT phases interfere DESTRUCTIVELY (cancel)

Retrieval:
- Query generates a phase
- Only items at matching phase contribute
- NO softmax, NO attention weights, NO explicit lookup

The magic: Phase interference IS the selection mechanism.
We don't COMPUTE which items match - matching items NATURALLY reinforce.

This is NOT:
- Attention (no softmax, no explicit comparison)
- Linear attention (no cumsum pattern)
- Modern Hopfield (no exponential energy)

Complexity: O(n) - single pass through sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


class PhaseEncoder(nn.Module):
    """
    Encode content to a phase angle.

    The key insight: Similar content should get similar phases.
    This creates "phase clusters" - related items reinforce.
    """
    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Map content to phase for each oscillator
        self.to_phase = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, n_oscillators)
        )

        # Map content to amplitude (how strongly to store)
        self.to_amp = nn.Sequential(
            nn.Linear(dim, n_oscillators),
            nn.Softplus()
        )

    def forward(self, x):
        """
        x: [..., D]
        Returns: complex phasor [..., K]
        """
        # Phase: content-dependent, bounded to [-pi, pi]
        phase = torch.tanh(self.to_phase(x)) * math.pi  # [..., K]

        # Amplitude: how strongly to store
        amp = self.to_amp(x) + 0.1  # [..., K]

        # Complex phasor
        return amp * torch.exp(1j * phase)


class PhaseBindingMemory(nn.Module):
    """
    Memory via phase binding and interference.

    Write:
    - Each key gets a phasor (amplitude * e^(i*phase))
    - Each value gets modulated by this phasor
    - Values are SUMMED (superposition) - no explicit storage

    Read:
    - Query generates its own phasor
    - Conjugate multiply with memory (demodulation)
    - Items at matching phase add constructively
    - Items at mismatched phase CANCEL

    The genius: No attention computation needed.
    Phase matching is IMPLICIT in the interference.
    """
    def __init__(self, dim, n_oscillators=32):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Encode keys and queries to phases
        self.key_phase = PhaseEncoder(dim, n_oscillators)
        self.query_phase = PhaseEncoder(dim, n_oscillators)

        # Value projection (maps to each oscillator channel)
        self.to_value = nn.Linear(dim, dim)

        # Output projection
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, keys, values, query):
        """
        keys: [B, n, D]
        values: [B, n, D]
        query: [B, D]

        Returns: [B, D]
        """
        B, n, D = keys.shape

        # Encode keys to phasors
        key_phasors = self.key_phase(keys)  # [B, n, K]

        # Project values
        V = self.to_value(values)  # [B, n, D]

        # WRITE: Superimpose all values, each modulated by its key's phase
        # For each oscillator k, memory[k] = sum_i (V[i] * phasor[i,k])
        # This is a "holographic" write - all values overlap in phase space

        # Convert V to complex for modulation
        V_complex = V.to(torch.complex64)  # [B, n, D]

        # Modulate: V * phasor (broadcast over D)
        # key_phasors: [B, n, K] -> [B, n, K, 1]
        # V_complex: [B, n, D] -> [B, n, 1, D]
        modulated = key_phasors.unsqueeze(-1) * V_complex.unsqueeze(-2)  # [B, n, K, D]

        # Sum over sequence to create memory trace
        memory = modulated.sum(dim=1)  # [B, K, D]

        # READ: Query with conjugate phasor (demodulation)
        query_phasors = self.query_phase(query)  # [B, K]

        # Demodulate: memory * conj(query_phasor)
        # query_phasors.conj(): [B, K] -> [B, K, 1]
        demodulated = memory * query_phasors.conj().unsqueeze(-1)  # [B, K, D]

        # Sum over oscillators (real part only - imaginary cancels for matched phases)
        retrieved = demodulated.sum(dim=1).real  # [B, D]

        # Normalize by number of items and oscillators
        retrieved = retrieved / math.sqrt(n * self.n_oscillators)

        return self.out(retrieved)


class TemporalPhaseBinding(nn.Module):
    """
    Add temporal phase offset based on position.

    Key insight: Items that occur TOGETHER should have same phase.
    In the associative recall task, keys and values occur adjacently.
    We can learn to give them the same phase offset.
    """
    def __init__(self, dim, n_oscillators=32, max_len=100):
        super().__init__()
        self.dim = dim
        self.n_oscillators = n_oscillators

        # Content-based phase
        self.content_phase = PhaseEncoder(dim, n_oscillators)

        # Temporal phase: each position gets a learnable offset
        self.temporal_offsets = nn.Embedding(max_len, n_oscillators)

        # Value projection
        self.to_value = nn.Linear(dim, dim)

        # Output
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, keys, values, query, key_positions=None, query_position=None):
        """
        keys: [B, n, D]
        values: [B, n, D]
        query: [B, D]
        key_positions: [B, n] - optional, defaults to 0,1,2,...
        query_position: [B] - optional

        Returns: [B, D]
        """
        B, n, D = keys.shape

        # Default positions
        if key_positions is None:
            key_positions = torch.arange(n, device=keys.device).unsqueeze(0).expand(B, -1)

        # Content phase + temporal phase
        content_phasors = self.content_phase(keys)  # [B, n, K]
        temporal_offsets = self.temporal_offsets(key_positions)  # [B, n, K]

        # Total phase = content + temporal
        # Convert offsets to phasors and multiply
        temporal_phasors = torch.exp(1j * temporal_offsets)
        key_phasors = content_phasors * temporal_phasors  # [B, n, K]

        # Values
        V = self.to_value(values).to(torch.complex64)  # [B, n, D]

        # Write: modulate and sum
        modulated = key_phasors.unsqueeze(-1) * V.unsqueeze(-2)  # [B, n, K, D]
        memory = modulated.sum(dim=1)  # [B, K, D]

        # Query phase
        query_content = self.content_phase(query)  # [B, K]

        # Read: demodulate
        demodulated = memory * query_content.conj().unsqueeze(-1)
        retrieved = demodulated.sum(dim=1).real / math.sqrt(n * self.n_oscillators)

        return self.out(retrieved)


class LearnedBindingMemory(nn.Module):
    """
    NOVEL: Learn the binding-unbinding relationship.

    Key insight: The relationship between key and value can be LEARNED.
    We don't assume any particular binding mechanism.

    Instead:
    1. Learn a "binding transform" that combines key with value
    2. Learn an "unbinding transform" that extracts value given query
    3. The transforms must be compatible (unbind(bind(k,v), k) ≈ v)

    This is more general than phase binding but preserves the key property:
    NO attention, NO softmax, just learned transforms.
    """
    def __init__(self, dim, binding_dim=64):
        super().__init__()
        self.dim = dim
        self.binding_dim = binding_dim

        # Binding transform: key + value -> bound representation
        # We use a bilinear form (outer-product-like but learnable)
        self.bind_key = nn.Linear(dim, binding_dim)
        self.bind_val = nn.Linear(dim, binding_dim)
        self.bind_combine = nn.Linear(binding_dim, dim)

        # Unbinding transform: query -> extraction mask
        self.unbind_query = nn.Linear(dim, binding_dim)
        self.unbind_extract = nn.Linear(binding_dim, dim)

        # Output
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def bind(self, key, value):
        """
        Create a bound representation of key-value pair.
        The binding should encode BOTH key and value such that
        we can later extract the value given the key.
        """
        k = self.bind_key(key)        # [..., B]
        v = self.bind_val(value)      # [..., B]

        # Element-wise multiplication (like HRR but learnable transforms)
        bound = k * v  # [..., B]

        # Project to output dim
        return self.bind_combine(bound)  # [..., D]

    def unbind(self, bound, query):
        """
        Extract value from bound representation using query as key.
        """
        q = self.unbind_query(query)  # [B, binding_dim]

        # Element-wise "division" (implemented as learned inverse)
        # The intuition: if bind(k,v) = k*v, then unbind(k*v, k) should give v
        # We approximate this with learned transforms

        # bound: [B, n, D] or [B, D]
        if bound.dim() == 3:
            # Multiple bounds - need to aggregate
            # This is where the magic happens: only matching keys contribute
            q_expanded = q.unsqueeze(1)  # [B, 1, binding_dim]

            # Project bound to binding space
            bound_proj = self.unbind_query(bound)  # [B, n, binding_dim]

            # "Match": similar keys have similar projections
            match = bound_proj * q_expanded  # [B, n, binding_dim]

            # Extract value direction
            extracted = self.unbind_extract(match)  # [B, n, D]

            # Sum (matching keys add, mismatched cancel due to random alignment)
            return extracted.sum(dim=1) / math.sqrt(bound.shape[1])
        else:
            # Single bound
            bound_proj = self.unbind_query(bound)  # [B, binding_dim]
            match = bound_proj * q
            return self.unbind_extract(match)

    def forward(self, keys, values, query):
        """
        keys: [B, n, D]
        values: [B, n, D]
        query: [B, D]
        """
        B, n, D = keys.shape

        # Bind all key-value pairs
        bindings = self.bind(keys, values)  # [B, n, D]

        # Superposition: sum all bindings
        memory = bindings.sum(dim=1)  # [B, D]

        # Unbind with query
        retrieved = self.unbind(bindings, query)  # [B, D]

        return self.out(retrieved)


# =============================================================================
# Associative Recall Models
# =============================================================================

class PhaseBindingRecall(nn.Module):
    def __init__(self, vocab_size, dim=64, n_oscillators=32, use_temporal=False, use_learned=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, dim)

        if use_learned:
            self.memory = LearnedBindingMemory(dim, binding_dim=dim)
        elif use_temporal:
            self.memory = TemporalPhaseBinding(dim, n_oscillators)
        else:
            self.memory = PhaseBindingMemory(dim, n_oscillators)

        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L, _ = x.shape
        indices = x.argmax(dim=-1)
        h = self.embed(indices)

        n_pairs = (L - 2) // 2
        keys = h[:, 0::2][:, :n_pairs]
        values = h[:, 1::2][:, :n_pairs]
        query = h[:, -1]

        retrieved = self.memory(keys, values, query)
        logits = self.output(retrieved)

        out = torch.zeros(B, L, self.vocab_size, device=x.device)
        out[:, -1] = logits

        return out


# =============================================================================
# Test
# =============================================================================

def generate_associative_recall(batch_size, n_pairs, vocab_size):
    seq_len = n_pairs * 2 + 2
    QUERY_TOKEN = vocab_size

    data = torch.zeros(batch_size, seq_len, vocab_size + 1, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        available = list(range(vocab_size))
        np.random.shuffle(available)
        pairs = [(available[2*i], available[2*i+1]) for i in range(n_pairs)]

        pos = 0
        for k, v in pairs:
            data[b, pos, k] = 1.0
            data[b, pos+1, v] = 1.0
            pos += 2

        data[b, pos, QUERY_TOKEN] = 1.0
        pos += 1

        query_idx = np.random.randint(0, n_pairs)
        query_k, query_v = pairs[query_idx]
        data[b, pos, query_k] = 1.0
        targets[b, pos] = query_v

    return data, targets


def test_phase_binding():
    print('=' * 70)
    print('NOVEL: Phase Binding Memory')
    print('=' * 70)
    print()
    print('Mechanism: Items with matching phase reinforce via interference')
    print('NO attention, NO softmax, NO explicit comparison')
    print()

    vocab_size = 16
    n_pairs = 5
    dim = 64

    models = {
        'PhaseBinding (32 osc)': lambda: PhaseBindingRecall(vocab_size, dim, n_oscillators=32),
        'PhaseBinding (64 osc)': lambda: PhaseBindingRecall(vocab_size, dim, n_oscillators=64),
        'TemporalPhase': lambda: PhaseBindingRecall(vocab_size, dim, n_oscillators=32, use_temporal=True),
        'LearnedBinding': lambda: PhaseBindingRecall(vocab_size, dim, use_learned=True),
    }

    for name, model_fn in models.items():
        print(f'\n--- {name} ---')

        model = model_fn().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {params:,}')

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        query_pos = n_pairs * 2 + 1
        best_acc = 0.0

        for epoch in range(1000):
            model.train()
            data, targets = generate_associative_recall(64, n_pairs, vocab_size)

            logits = model(data)
            B, L, V = logits.shape
            loss = criterion(logits.view(B*L, V), targets.view(B*L))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                model.eval()
                with torch.no_grad():
                    data, targets = generate_associative_recall(500, n_pairs, vocab_size)
                    logits = model(data)
                    preds = logits.argmax(dim=-1)
                    acc = (preds[:, query_pos] == targets[:, query_pos]).float().mean().item() * 100
                    if acc > best_acc:
                        best_acc = acc
                print(f'    Epoch {epoch+1}: acc={acc:.1f}%, best={best_acc:.1f}%')

        random_baseline = 100.0 / vocab_size
        status = 'WORKS' if best_acc > 90 else 'PARTIAL' if best_acc > 20 else 'FAILS'
        print(f'  Final: {best_acc:.1f}% (random={random_baseline:.1f}%) [{status}]')


if __name__ == '__main__':
    test_phase_binding()
