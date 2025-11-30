import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PSI(nn.Module):
    def __init__(self, dim, init_scale=0.1):
        super().__init__()

        self.dim = dim

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

        # Position-dependent scale: 1/sqrt(position) decay prevents unbounded phase accumulation
        # This enables long-range dependencies while maintaining O(n) complexity
        position = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        pos_scale = 1.0 / torch.sqrt(position)

        # Learned scale (exp to ensure positive) with position decay
        scale = torch.exp(self.log_scale)
        omega_scaled = omega * scale * pos_scale

        # Integrate: phi = cumsum(omega) - this IS Euler integration
        phi = torch.cumsum(omega_scaled, dim=1)

        # Phase trajectory
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Phase-bound memory accumulation (uniform weighting)
        memory_real = torch.cumsum(x * cos_phi, dim=1)
        memory_imag = torch.cumsum(x * sin_phi, dim=1)

        # Position-based normalization
        memory_real_normalized = memory_real / position
        memory_imag_normalized = memory_imag / position

        # Retrieve at current phase (no query offset)
        retrieved_real = memory_real_normalized * cos_phi + memory_imag_normalized * sin_phi
        retrieved_imag = memory_imag_normalized * cos_phi - memory_real_normalized * sin_phi

        # Content modulation by trajectory
        content_modulated_real = x * cos_phi
        content_modulated_imag = x * sin_phi

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
