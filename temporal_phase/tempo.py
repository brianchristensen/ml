import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TPI(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.to_omega = nn.Linear(dim, dim)
        self.to_phase_init = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.to_gate = nn.Linear(dim, dim)

        self.integration_scale = nn.Parameter(torch.ones(dim) * 0.001)

        self.to_magnitude = nn.Linear(dim, dim)

        self.to_query_offset = nn.Linear(dim, dim)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        omega = self.to_omega(x)

        magnitude_scale = 5.0
        magnitude = torch.sigmoid(self.to_magnitude(x)) * magnitude_scale

        phi_init = self.to_phase_init(x)

        gate = torch.sigmoid(self.to_gate(x))

        omega_scaled = omega * self.integration_scale.abs()
        gated_omega = gate * omega_scaled

        phi = phi_init + torch.cumsum(gated_omega, dim=1)

        trajectory_real = torch.cos(phi)
        trajectory_imag = torch.sin(phi)

        weighted_content = magnitude * x

        memory_real = torch.cumsum(weighted_content * torch.cos(phi), dim=1)
        memory_imag = torch.cumsum(weighted_content * torch.sin(phi), dim=1)

        accumulated_magnitude = torch.cumsum(magnitude, dim=1)
        sqrt_magnitude = torch.sqrt(accumulated_magnitude + 1e-8)
        memory_real_normalized = memory_real / sqrt_magnitude
        memory_imag_normalized = memory_imag / sqrt_magnitude

        query_offset = self.to_query_offset(x)
        phi_query = phi + query_offset

        cos_phi_q = torch.cos(phi_query)
        sin_phi_q = torch.sin(phi_query)

        retrieved_real = memory_real_normalized * cos_phi_q + memory_imag_normalized * sin_phi_q
        retrieved_imag = memory_imag_normalized * cos_phi_q - memory_real_normalized * sin_phi_q

        content_modulated_real = x * trajectory_real
        content_modulated_imag = x * trajectory_imag

        context = torch.cat([
            content_modulated_real,
            content_modulated_imag,
            retrieved_real,
            retrieved_imag
        ], dim=-1)

        phase_contribution = self.to_out(context)

        output = x + phase_contribution

        return output

class TPIBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.integration = TPI(dim)

    def forward(self, x):
        x = x + self.integration(self.norm(x))

        return x

class Tempo(nn.Module):
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
            TPIBlock(dim=dim)
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
    print("Temporal Phase Integration (TPI) - Sanity Check")
    print("=" * 80)
    print()

    # Create tiny model
    model = Tempo(
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
