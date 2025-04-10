import torch
import torch.nn as nn
import torch.nn.functional as F

# === Decoder ===
class Decoder(nn.Module):
    def __init__(self, input_dim=256, output_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * 8 * 8),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 128, 8, 8)
        return self.deconv(x)

# === SoftSOM ===
class SoftSOM(nn.Module):
    def __init__(self, input_dim, som_dim, init_temperature=0.5):
        super().__init__()
        self.num_nodes = som_dim * som_dim
        self.prototypes = nn.Parameter(torch.randn(self.num_nodes, input_dim))
        self.temp_raw = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    @property
    def temperature(self):
        return torch.sigmoid(self.temp_raw) * (1.0 - 1e-3) + 1e-3

    def forward(self, x):
        dists = torch.cdist(x.unsqueeze(1), self.prototypes.unsqueeze(0))  # [B, 1, D] vs [1, N, D]
        weights = F.softmax(-dists.squeeze(1) / self.temperature, dim=-1)
        blended = weights @ self.prototypes
        return blended, weights

# === Node ===
class Node(nn.Module):
    def __init__(self, latent_dim, som_dim):
        super().__init__()
        self.encoder_fc = nn.Linear(latent_dim, latent_dim)
        self.som = SoftSOM(latent_dim, som_dim)
        self.last_input = None
        self.last_blended = None

    def forward(self, x):
        z = self.encoder_fc(x)
        topo_z, weights = self.som(z)
        self.last_input = z
        self.last_blended = topo_z
        return topo_z, weights

# === TEA ===
class TEA(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256,
                 num_nodes=4, som_dim=[8, 10, 12, 16], num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.shared_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)
        )

        self.decoders = nn.ModuleList([
            Decoder(input_dim=latent_dim) for _ in range(num_nodes)
        ])

        self.nodes = nn.ModuleList([
            Node(latent_dim, som_dim[i])
            for i in range(num_nodes)
        ])

        self.class_embeddings = nn.Parameter(torch.randn(10, latent_dim))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, latent_dim))
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_proj = nn.Linear(latent_dim, latent_dim * num_heads)
        self.classifier = nn.Linear(latent_dim * num_heads, 10)

    def proto_diversity_loss(self):
        return sum(
            (F.normalize(node.som.prototypes, dim=1) @ F.normalize(node.som.prototypes, dim=1).T)
            .fill_diagonal_(0).pow(2).mean()
            for node in self.nodes
        ) / len(self.nodes)

    def node_diversity_loss(self):
        sims = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                p1 = F.normalize(self.nodes[i].som.prototypes, dim=1)
                p2 = F.normalize(self.nodes[j].som.prototypes, dim=1)
                sims.append((p1 @ p2.T).pow(2).mean())
        return sum(sims) / len(sims)

    def usage_penalty(self):
        loss = 0.0
        for node in self.nodes:
            _, weights = node.som(node.last_input)
            usage = weights.mean(dim=0)
            loss += F.mse_loss(usage, torch.full_like(usage, 1.0 / usage.size(0)))
        return loss

    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)

        topo_z_list = []
        for node in self.nodes:
            topo_z, _ = node(z0)
            topo_z_list.append(topo_z)

        topo_stack = torch.stack(topo_z_list, dim=0)

        if y is not None:
            query = self.query_proj(self.class_embeddings[y])
        else:
            query = self.query_proj(self.class_embeddings.mean(dim=0, keepdim=True)).expand(z0.size(0), -1)

        proj = self.attn_proj(self.node_embeddings).view(self.num_nodes, self.num_heads, self.latent_dim)
        outputs = []
        for h in range(self.num_heads):
            keys = proj[:, h]
            attn_logits = torch.einsum('bd,nd->bn', query, keys)
            attn_weights = F.softmax(attn_logits, dim=-1)
            fused = torch.einsum('bn,nbd->bd', attn_weights, topo_stack)
            outputs.append(fused)

        fused_all = torch.cat(outputs, dim=1)
        logits = self.classifier(fused_all)
        return logits
