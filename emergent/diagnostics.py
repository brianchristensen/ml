import torch

class RewritingDiagnostics:
    def __init__(self, num_generators, latent_dim, steps):
        self.num_generators = num_generators
        self.latent_dim = latent_dim
        self.steps = steps
        self.reset()

    def reset(self):
        self.generator_counts = torch.zeros(self.num_generators)
        self.latent_norms = []
        self.rewrite_deltas = []

    def update(self, g_indices, latent_steps):
        # Count generator usage
        with torch.no_grad():
            counts = torch.bincount(g_indices, minlength=self.num_generators)
            self.generator_counts += counts.cpu()

            norms = [z.norm(dim=-1).mean().item() for z in latent_steps]
            self.latent_norms.append(norms)

            # Calculate deltas between steps
            deltas = [
                (latent_steps[i] - latent_steps[i-1]).norm(dim=-1).mean().item()
                for i in range(1, len(latent_steps))
            ]
            self.rewrite_deltas.append(deltas)

    def report(self):
        avg_norms = torch.tensor(self.latent_norms).mean(dim=0)
        avg_deltas = torch.tensor(self.rewrite_deltas).mean(dim=0)
        usage_dist = self.generator_counts / self.generator_counts.sum()

        print("\n🔍 Generator Usage:")
        for i, p in enumerate(usage_dist):
            print(f"  G[{i:02d}]: {p.item():.3%}")

        print("\n📈 Avg Latent Norms per Step:")
        print("  ", ["{:.2f}".format(v) for v in avg_norms])

        print("\n🔁 Avg Rewrite Deltas per Step:")
        print("  ", ["{:.2f}".format(v) for v in avg_deltas])
