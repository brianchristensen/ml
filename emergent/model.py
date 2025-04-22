import torch
import torch.nn as nn
import torch.nn.functional as F

# === MODEL ===
class ALS(nn.Module):
    def __init__(self, input_dim, latent_dim=256, max_steps=8, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_steps = max_steps

        # Core components
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Linear(latent_dim, num_classes)

        # Rewrite modules
        self.rewrite_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        self.rewrite_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Per-step generators and time embeddings
        self.step_generators = nn.Parameter(torch.randn(max_steps, latent_dim))
        self.time_embed = nn.Embedding(max_steps, latent_dim)

        # Final normalization after each step
        self.norm = nn.LayerNorm(latent_dim)

        # Halting head: learns [batch_size, max_steps] halting weights
        self.halting_head = nn.Linear(latent_dim, max_steps)

    def binary_op(self, prev, gen):
        gate = self.rewrite_gate(prev)
        delta = self.rewrite_net(torch.cat([prev, gen], dim=-1))
        return prev + gate * delta

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)  # [B, D]

        # Track all step outputs
        step_outputs = []

        for step in range(self.max_steps):
            g = self.step_generators[step].unsqueeze(0).expand(batch_size, -1)
            t = self.time_embed(torch.tensor(step, device=x.device)).unsqueeze(0).expand(batch_size, -1)
            z = self.binary_op(z, g + t)
            z = self.norm(z)
            step_outputs.append(z.unsqueeze(1))  # [B, 1, D]

        # Stack all step outputs → [B, T, D]
        z_all = torch.cat(step_outputs, dim=1)

        # Halting distribution over max_steps → [B, T]
        halting_scores = torch.sigmoid(self.halting_head(z_all[:, -1]))  # [B, T]
        halting_weights = halting_scores / (halting_scores.sum(dim=-1, keepdim=True) + 1e-8)
        halting_weights = halting_weights.unsqueeze(-1)

        # Weighted sum across max_steps → [B, D]
        z_final = torch.sum(z_all * halting_weights, dim=1)

        return self.decoder(z_final)
