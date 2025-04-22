import torch
import torch.nn as nn
import torch.nn.functional as F

class ALS(nn.Module):
    def __init__(self, input_dim, latent_dim=256, num_generators=16, steps=3, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.num_generators = num_generators
        self.norm = nn.LayerNorm(input_dim)

        # Input projection (flattened image to latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder to logits
        self.decoder = nn.Linear(latent_dim, num_classes)

        # Learnable generators (basis elements)
        self.generators = nn.Parameter(torch.randn(num_generators, latent_dim))

        # Learnable binary operation over latent space
        self.op_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def binary_op(self, x, y):
        return self.op_net(torch.cat([x, y], dim=-1))

    def forward(self, x):
        z = self.encoder(x)

        for _ in range(self.steps):
            g_indices = torch.randint(0, self.num_generators, (x.size(0),), device=x.device)
            g = self.generators[g_indices]
            z = self.binary_op(z, g)
        
        z = self.norm(z)

        return self.decoder(z)
