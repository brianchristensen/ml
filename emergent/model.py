import torch
import torch.nn as nn
import torch.nn.functional as F

class Rewriter(nn.Module):
    def __init__(self, input_dim, latent_dim=256, num_generators=8, steps=4, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.num_generators = num_generators
        self.norm = nn.LayerNorm(latent_dim)

        # Input projection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

        # Latent generators
        self.generators = nn.Parameter(torch.randn(num_generators, latent_dim))

        # Step embedding (temporal signal)
        self.step_embed = nn.Embedding(steps, latent_dim)

        # Binary operator (latent rewrite)
        self.op_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Feedback (top-down modulation)
        self.feedback_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Feedback to generators (modulates generator embedding)
        self.feedback_to_gen = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Class prototypes (e.g., for cosine or distance-based decoding)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))

    def binary_op(self, x, y):
        return self.op_net(torch.cat([x, y], dim=-1))

    def forward(self, x):
        z_orig = self.encoder(x)
        z = z_orig
        z_states = []
        g_indices_list = []

        # 🔁 Forward pass: latent rewrites over time
        for i in range(self.steps):
            step_emb = self.step_embed(torch.tensor(i, device=x.device)).unsqueeze(0).expand_as(z)

            # 🌀 Random generator sampling
            g_indices = torch.randint(0, self.num_generators, (x.size(0),), device=x.device)
            g = self.generators[g_indices]
            g_indices_list.append(g_indices)

            # 🔁 Residual + rewrite + time awareness
            rewrite = self.binary_op(z, g + step_emb + z_orig)
            z = z + rewrite
            z = self.norm(z)
            z_states.append(z)

        # 🔁 Backward pass: feedback refinement + generator update
        context = z_states[-1]
        for i in reversed(range(self.steps - 1)):
            # 💡 Top-down feedback into latent space
            feedback = self.feedback_net(torch.cat([z_states[i], context], dim=-1))
            z_states[i] = z_states[i] + feedback
            context = z_states[i]

            # 🧬 Feedback into generator representation
            g_indices = g_indices_list[i]
            g_feedback = self.feedback_to_gen(torch.cat([z_states[i], context], dim=-1))
            self.generators.data[g_indices] += 0.01 * g_feedback

        # Final refined state
        refined_z = z_states[0]
        logits = -torch.cdist(refined_z, self.class_prototypes)
        return logits
