import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PSI(nn.Module):
    """
    Phase-Space Integration module.

    Uses chunked normalization for better extrapolation to longer sequences.
    Memory is accumulated within fixed-size chunks, then normalized per chunk.
    This prevents the memory magnitude decay that causes extrapolation failure.
    """
    def __init__(self, dim, init_scale=0.1, chunk_size=64):
        super().__init__()

        self.dim = dim
        self.chunk_size = chunk_size

        # Learned velocity field (the differential equation)
        self.to_omega = nn.Linear(dim, dim)

        # Learned per-dimension integration scale (log-space for stability)
        self.log_scale = nn.Parameter(torch.ones(dim) * math.log(init_scale))

        # Output projection (4x dim: content_real, content_imag, retrieved_real, retrieved_imag)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Compute velocity field
        omega = self.to_omega(x)

        # Pad sequence to multiple of chunk_size for efficient parallel processing
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            omega_padded = F.pad(omega, (0, 0, 0, pad_len))
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            omega_padded = omega
            x_padded = x

        padded_len = omega_padded.shape[1]
        num_chunks = padded_len // self.chunk_size

        # Reshape for parallel chunk processing: (batch, seq, dim) -> (batch, num_chunks, chunk_size, dim)
        omega_chunked = omega_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        x_chunked = x_padded.view(batch_size, num_chunks, self.chunk_size, dim)

        # Position scaling within each chunk (1/sqrt(pos) where pos resets each chunk)
        chunk_pos = torch.arange(1, self.chunk_size + 1, device=x.device, dtype=x.dtype)
        pos_scale = 1.0 / torch.sqrt(chunk_pos).view(1, 1, -1, 1)

        # Learned scale with position decay
        scale = torch.exp(self.log_scale)
        omega_scaled = omega_chunked * scale * pos_scale

        # PARALLEL cumsum across all chunks simultaneously
        # cumsum on dim=2 (chunk_size dimension) processes all chunks in parallel
        phi_chunked = torch.cumsum(omega_scaled, dim=2)

        # Phase trajectory
        cos_phi = torch.cos(phi_chunked)
        sin_phi = torch.sin(phi_chunked)

        # PARALLEL memory accumulation across all chunks
        chunk_mem_real = torch.cumsum(x_chunked * cos_phi, dim=2)
        chunk_mem_imag = torch.cumsum(x_chunked * sin_phi, dim=2)

        # Normalize by chunk position (same pos_scale pattern, but for denominator)
        chunk_pos_norm = chunk_pos.view(1, 1, -1, 1)
        memory_real = chunk_mem_real / chunk_pos_norm
        memory_imag = chunk_mem_imag / chunk_pos_norm

        # Retrieve at current phase
        retrieved_real = memory_real * cos_phi + memory_imag * sin_phi
        retrieved_imag = memory_imag * cos_phi - memory_real * sin_phi

        # Content modulation by trajectory
        content_modulated_real = x_chunked * cos_phi
        content_modulated_imag = x_chunked * sin_phi

        # Reshape back: (batch, num_chunks, chunk_size, dim) -> (batch, padded_len, dim)
        content_modulated_real = content_modulated_real.view(batch_size, padded_len, dim)
        content_modulated_imag = content_modulated_imag.view(batch_size, padded_len, dim)
        retrieved_real = retrieved_real.view(batch_size, padded_len, dim)
        retrieved_imag = retrieved_imag.view(batch_size, padded_len, dim)

        # Remove padding
        if pad_len > 0:
            content_modulated_real = content_modulated_real[:, :seq_len, :]
            content_modulated_imag = content_modulated_imag[:, :seq_len, :]
            retrieved_real = retrieved_real[:, :seq_len, :]
            retrieved_imag = retrieved_imag[:, :seq_len, :]

        # Combine all signals
        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        # Project to output
        phase_contribution = self.to_out(context)

        return x + phase_contribution

class PSIBlock(nn.Module):
    """PSI block with pre-norm residual connection."""

    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.integration = PSI(dim)

    def forward(self, x):
        x = x + self.integration(self.norm(x))

        return x

class PhaseSpaceIntegrator(nn.Module):
    """Full PSI model for language modeling."""

    def __init__(
        self,
        vocab_size,
        dim=128,
        num_layers=4,
        max_len=2048,
        device='cuda'
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_len, dim))

        self.blocks = nn.ModuleList([
            PSIBlock(dim=dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.output_head = None

        self._init_weights()

    def _create_sinusoidal_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pos_encoding = torch.zeros(max_len, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)

        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        x_emb = token_emb + pos_emb

        for block in self.blocks:
            x_emb = block(x_emb)

        x_emb = self.norm(x_emb)

        logits = F.linear(x_emb, self.token_embedding.weight)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50):
        self.eval()

        generated = prompt

        with torch.no_grad():
            for _ in range(max_length - prompt.shape[1]):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
                    next_token_probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(next_token_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated

if __name__ == "__main__":
    print("=" * 80)
    print("Phase-Space Integration (PSI) - Sanity Check")
    print("=" * 80)
    print()

    # Create tiny model
    model = PhaseSpaceIntegrator(
        vocab_size=100,
        dim=32,
        num_layers=2,
        device='cpu'
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    batch_size = 2
    seq_len = 10
    x = torch.randint(0, 100, (batch_size, seq_len))

    print(f"Input shape: {x.shape}")

    logits = model(x)
    print(f"Output shape: {logits.shape}")

    assert logits.shape == (batch_size, seq_len, 100), "Output shape mismatch!"

    print()
    print("[PASS] Sanity check passed!")
