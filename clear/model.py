import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# === Encoder ===
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SharedEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResBlock(64, 64),
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128),
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

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
    def __init__(self, input_dim, max_grid_size=256, init_temperature=0.95, temp_anneal_rate=0.98):
        super().__init__()
        self.num_cells = max_grid_size
        self.input_dim = input_dim
        self.prototypes = nn.Parameter(torch.randn(self.num_cells, input_dim))
        self.grid_pos = nn.Parameter(torch.randn(self.num_cells, input_dim))
        self.temp_raw = nn.Parameter(torch.tensor([math.log(init_temperature / (1.0 - init_temperature))], dtype=torch.float32))
        self.gate_logits = nn.Parameter(torch.zeros(self.num_cells))
        self.anneal_rate = temp_anneal_rate

    @property
    def temperature(self):
        return torch.sigmoid(self.temp_raw) * (1.0 - 1e-3) + 1e-3

    def anneal_temp(self):
        with torch.no_grad():
            current_temp = self.temperature.item()
            new_temp = current_temp * self.anneal_rate
            new_raw = math.log((new_temp - 1e-3) / (1.0 - 1e-3 - new_temp))
            self.temp_raw.fill_(new_raw)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_logits).unsqueeze(0)
        d_proto = torch.cdist(x.unsqueeze(1), self.prototypes.unsqueeze(0))
        d_pos = torch.cdist(x.unsqueeze(1), self.grid_pos.unsqueeze(0))
        d_total = d_proto + d_pos
        weights = F.softmax(-d_total / self.temperature, dim=-1) * gate
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        blended = weights @ self.prototypes
        return blended, weights

# === Node ===
class Node(nn.Module):
    def __init__(self, latent_dim, max_grid_size=256):
        super().__init__()
        self.encoder_fc = nn.Linear(latent_dim, latent_dim)
        self.som = SoftSOM(latent_dim, max_grid_size)
        self.last_blended = None
        self.last_weights = None
        self.proto_div_history = []

        with torch.no_grad():
            gate_probs = torch.sigmoid(self.som.gate_logits)
            norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
            self.initial_gate_entropy = -(norm_probs * norm_probs.clamp(min=1e-8).log()).sum().item()

    def forward(self, x):
        z = self.encoder_fc(x)
        topo_z, weights = self.som(z)
        self.last_weights = weights.detach()
        self.last_blended = topo_z
        return topo_z, weights

# === CLEAR ===
class CLEAR(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, init_nodes=1, max_grid_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_grid_size = max_grid_size
        self.node_count = init_nodes
        self.just_grew = False

        self.shared_encoder = SharedEncoder(input_channels=input_channels, latent_dim=latent_dim)

        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.node_embeddings = nn.Parameter(torch.randn(self.node_count, latent_dim))
        self.attn_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_weights = None
        self.classifier = nn.Linear(latent_dim, 10)

        self.nodes = nn.ModuleList([Node(latent_dim, max_grid_size) for _ in range(self.node_count)])
        self.decoders = nn.ModuleList([Decoder(input_dim=latent_dim) for _ in range(self.node_count)])

    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)  # [B, D]
        
        # Get [B, D] proto blends from each node
        topo_z_list = [node(z0)[0].view(z0.size(0), -1) for node in self.nodes]
        topo_stack = torch.stack(topo_z_list, dim=0).permute(1, 0, 2)  # [B, N, D]

        # Compute entropy drops for each node
        entropy_bias = []
        for node in self.nodes:
            gate_probs = torch.sigmoid(node.som.gate_logits)
            norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
            
            # Use entropy as differentiable tensor
            entropy_now = -(norm_probs * (norm_probs + 1e-8).log()).sum()
            # Normalize drop against initial, make sure it's a tensor
            drop = (node.initial_gate_entropy - entropy_now) / (node.initial_gate_entropy + 1e-8)
            entropy_bias.append(drop)

        entropy_bias = torch.stack(entropy_bias)
        
        keys = self.attn_proj(self.node_embeddings)  # [N, D]
        keys = F.normalize(keys, dim=1)
        query = self.query_proj(z0)  # [B, D]
        query = F.normalize(query, dim=1)

        attn_logits = torch.einsum("bd,nd->bn", query, keys) + entropy_bias.unsqueeze(0)  # [B, N]
        self.attn_weights = F.softmax(attn_logits, dim=-1)
        fused = torch.einsum("bn,bnd->bd", self.attn_weights, topo_stack)

        # Final classifier
        logits = self.classifier(fused)
        return logits

    # === Growth, Diversity, Entropy, Usage ===
    def should_grow(self):
        node = self.nodes[-1]
        gate_entropy = self.gate_entropy_loss().item()
        proto_div = self.proto_diversity_loss().item()
        semantic_ratio = proto_div / (gate_entropy + 1e-8)

        node.proto_div_history.append(semantic_ratio)
        if len(node.proto_div_history) > 5:
            node.proto_div_history.pop(0)

        if len(node.proto_div_history) < 2:
            print("📊 Growth Check: Not enough history, skipping trend check.")
            return False

        recent_deltas = np.diff(node.proto_div_history)
        trend = np.mean(recent_deltas[-2:])
        trend_up = trend > 0

        print(f"📊 Growth Check:")
        print(f"    ➤ Diversity: {proto_div:.6f}, Entropy: {gate_entropy:.4f}, Ratio: {semantic_ratio}")
        print(f"    ➤ Trend: {trend}, Trending Up? {trend_up}")

        return trend_up

    def grow_node(self):
        print(f"🌱 Growing CLEAR: Adding Node {self.node_count}")
        self.node_count += 1
        device = self.node_embeddings.device

        new_node = Node(self.latent_dim, self.max_grid_size).to(device)
        self.nodes.append(new_node)
        self.decoders.append(Decoder(self.latent_dim).to(device))

        with torch.no_grad():
            init_temp = 0.95
            raw_val = math.log(init_temp / (1.0 - init_temp))
            new_node.som.temp_raw.fill_(raw_val)

        new_embedding = nn.Parameter(torch.randn(1, self.latent_dim).to(device))
        self.node_embeddings = nn.Parameter(torch.cat([self.node_embeddings, new_embedding], dim=0))

        self.just_grew = True

    def freeze_node(self, node):
        for param in node.parameters():
            param.requires_grad = False

    def anneal_temp_active_node(self):
        self.nodes[-1].som.anneal_temp()

    def proto_diversity_loss(self):
        node = self.nodes[-1]
        P = F.normalize(node.som.prototypes, dim=1)
        S = P @ P.T
        S.fill_diagonal_(0)
        diversity = S.pow(2).mean()
        return diversity

    def node_diversity_loss(self, tau=0.3):
        if self.node_count <= 1:
            return torch.tensor(0.0, device=self.nodes[0].som.prototypes.device)

        new_som = self.nodes[-1].som
        frozen_soms = [node.som for node in self.nodes[:-1]]
        loss = 0.0
        for old_som in frozen_soms:
            sim = F.cosine_similarity(
                new_som.prototypes.unsqueeze(1),  # [P_new, 1, D]
                old_som.prototypes.unsqueeze(0),  # [1, P_old, D]
                dim=-1  # compare along feature dim
            )  # → [P_new, P_old]
            
            penalty = F.relu(sim - tau).mean()
            loss += penalty

        return loss / len(frozen_soms)

    def gate_entropy_loss(self):
        node = self.nodes[-1]
        p = F.softmax(node.som.gate_logits, dim=0)
        p = p / (p.sum() + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum()
        return entropy

    def log_attn_weights(self):
        mean_attn = self.attn_weights.mean(dim=0)  # [N]
        print(f"📡 Mean Attention per Node: {[f'{w.item():.3f}' for w in mean_attn]}")
