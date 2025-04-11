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
    def __init__(self, input_dim, max_grid_size=256, init_temperature=0.5):
        super().__init__()
        self.num_cells = max_grid_size
        self.input_dim = input_dim

        # Latent space prototypes
        self.prototypes = nn.Parameter(torch.randn(self.num_cells, input_dim))

        # Learnable position embeddings for grid cells
        self.grid_pos = nn.Parameter(torch.randn(self.num_cells, input_dim))

        # Learnable temp (for feature dist) and gating
        self.temp_raw = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))
        self.gate_logits = nn.Parameter(torch.zeros(self.num_cells))  # For soft gating

    @property
    def temperature(self):
        return torch.sigmoid(self.temp_raw) * (1.0 - 1e-3) + 1e-3

    def forward(self, x):
        B = x.shape[0]

        # Compute soft gate over grid cells (learns usage pattern)
        gate = torch.sigmoid(self.gate_logits).unsqueeze(0)  # [1, N]
        
        # Blend similarity across both prototype AND positional space
        d_proto = torch.cdist(x.unsqueeze(1), self.prototypes.unsqueeze(0))  # [B, N]
        d_pos = torch.cdist(x.unsqueeze(1), self.grid_pos.unsqueeze(0))      # [B, N]

        # Option 1: Weighted sum of both
        d_total = d_proto + d_pos  # You could weight these differently

        weights = F.softmax(-d_total / self.temperature, dim=-1) * gate  # [B, N]

        # Normalize post-gating
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        blended = weights @ self.prototypes  # [B, D]
        return blended, weights


# === Node ===
class Node(nn.Module):
    def __init__(self, latent_dim, max_grid_size=256):
        super().__init__()
        self.encoder_fc = nn.Linear(latent_dim, latent_dim)
        self.som = SoftSOM(latent_dim, max_grid_size)
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
                 num_nodes=4, max_grid_size=256, num_heads=4):
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
            Node(latent_dim, max_grid_size)
            for _ in range(num_nodes)
        ])

        self.class_embeddings = nn.Parameter(torch.randn(10, latent_dim))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, latent_dim))
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_proj = nn.Linear(latent_dim, latent_dim * num_heads)
        self.classifier = nn.Linear(latent_dim * num_heads, 10)

    def proto_diversity_loss(self):
        losses = []
        for node in self.nodes:
            P = F.normalize(node.som.prototypes, dim=1)
            S = P @ P.T
            S.fill_diagonal_(0)

            gate_probs = torch.sigmoid(node.som.gate_logits)
            usage = gate_probs / (gate_probs.sum() + 1e-8)

            # Pairwise usage weights
            usage_weights = usage.unsqueeze(0) * usage.unsqueeze(1)

            # Entropy of gate activations
            gate_dist = gate_probs / (gate_probs.sum() + 1e-8)
            gate_entropy = -(gate_dist * gate_dist.clamp(min=1e-8).log()).sum()

            diversity_penalty = (S.pow(2) * usage_weights).sum() / usage_weights.sum()
            losses.append(gate_entropy * diversity_penalty)  # <- Entropy scaling here

        return sum(losses) / len(losses)


    def node_diversity_loss(self):
        node_reps = []
        for node in self.nodes:
            prototypes = node.som.prototypes
            gate_probs = torch.sigmoid(node.som.gate_logits)
            weights = gate_probs / (gate_probs.sum() + 1e-8)
            weighted_mean = (weights.unsqueeze(1) * prototypes).sum(dim=0)
            node_reps.append(F.normalize(weighted_mean, dim=0))

        node_reps = torch.stack(node_reps, dim=0)
        sim_matrix = node_reps @ node_reps.T

        off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        
        # Don't square it — keep it smooth
        diversity_loss = sim_matrix[off_diag_mask].mean()

        return diversity_loss


    def gate_entropy_loss(self):
        loss = 0.0
        for node in self.nodes:
            #p = torch.sigmoid(node.som.gate_logits)
            p = F.softmax(node.som.gate_logits, dim=0)
            p = p / (p.sum() + 1e-8)  # normalize just in case
            entropy = -(p * (p + 1e-8).log()).sum()
            loss += entropy
        return loss / len(self.nodes)

    def usage_penalty(self):
        loss = 0.0
        for node in self.nodes:
            usage = torch.sigmoid(node.som.gate_logits)
            loss += F.mse_loss(usage, torch.full_like(usage, 1.0 / usage.size(0)))
        return loss / len(self.nodes)
    
    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)

        topo_z_list = []
        for node in self.nodes:
            topo_z, _ = node(z0)
            topo_z_list.append(topo_z)

        topo_z_list = [tz.view(tz.size(0), -1) for tz in topo_z_list]
        topo_stack = torch.stack(topo_z_list, dim=0)
        topo_stack = topo_stack.permute(1, 0, 2)    

        if y is not None:
            query = self.query_proj(self.class_embeddings[y])
        else:
            query = self.query_proj(self.class_embeddings.mean(dim=0, keepdim=True)).expand(z0.size(0), -1)

        proj = self.attn_proj(self.node_embeddings).view(self.num_nodes, self.num_heads, self.latent_dim)
        outputs = []
        for h in range(self.num_heads):
            keys = proj[:, h]  # [num_nodes, latent_dim]
            attn_logits = torch.einsum('bd,nd->bn', query, keys)
            attn_weights = F.softmax(attn_logits, dim=-1)  # [B, num_nodes]
            fused = torch.einsum('bn,bnd->bd', attn_weights, topo_stack)
            outputs.append(fused)

        fused_all = torch.cat(outputs, dim=1)
        logits = self.classifier(fused_all)
        return logits
