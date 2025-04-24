import torch
import torch.nn as nn
import torch.nn.functional as F

class Rewrite(nn.Module):
    def __init__(self, input_dim, latent_dim=256, num_generators=8, steps=4, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.num_generators = num_generators
        self.norm = nn.LayerNorm(latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

        self.generators = nn.Parameter(torch.randn(num_generators, latent_dim))

        self.op_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.class_prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))

        # === Diagnostics ===
        self.register_buffer('generator_counts', torch.zeros(num_generators))
        self.latent_norms = []
        self.rewrite_deltas = []

    def binary_op(self, x, y):
        return self.op_net(torch.cat([x, y], dim=-1))

    def forward(self, x, collect_diagnostics=False):
        z_orig = self.encoder(x)
        z = z_orig
        latent_steps = [z]

        for i in range(self.steps):
            g_indices = torch.randint(0, self.num_generators, (x.size(0),), device=x.device)
            g = self.generators[g_indices]
            rewrite = self.binary_op(z, g + z_orig)
            z = z + rewrite
            z = self.norm(z)
            latent_steps.append(z)

            if collect_diagnostics:
                with torch.no_grad():
                    self.generator_counts += torch.bincount(g_indices, minlength=self.num_generators).float().to(self.generator_counts.device)

        if collect_diagnostics:
            with torch.no_grad():
                self.latent_norms.append([z_.norm(dim=-1).mean().item() for z_ in latent_steps])
                deltas = [
                    (latent_steps[i] - latent_steps[i - 1]).norm(dim=-1).mean().item()
                    for i in range(1, len(latent_steps))
                ]
                self.rewrite_deltas.append(deltas)

        logits = -torch.cdist(z, self.class_prototypes)
        return logits

    def report_diagnostics(self):
        print("\n🔍 Generator Usage:")
        usage = self.generator_counts / self.generator_counts.sum()
        for i, p in enumerate(usage):
            print(f"  G[{i:02d}]: {p.item():.3%}")

        if self.latent_norms:
            avg_norms = torch.tensor(self.latent_norms).mean(dim=0)
            print("\n📈 Avg Latent Norms per Step:")
            print("  ", ["{:.2f}".format(v) for v in avg_norms])

        if self.rewrite_deltas:
            avg_deltas = torch.tensor(self.rewrite_deltas).mean(dim=0)
            print("\n🔁 Avg Rewrite Deltas per Step:")
            print("  ", ["{:.2f}".format(v) for v in avg_deltas])

    def reset_diagnostics(self):
        self.generator_counts.zero_()
        self.latent_norms.clear()
        self.rewrite_deltas.clear()
