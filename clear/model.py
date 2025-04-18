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
        x = self.fc(z).view(-1, 128, 8, 8)
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
        self.proto_sim_history = []

        with torch.no_grad():
            gate_probs = torch.sigmoid(self.som.gate_logits)
            norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
            self.initial_gate_entropy = -(norm_probs * norm_probs.clamp(min=1e-8).log()).sum().item()

    def forward(self, x):
        z = self.encoder_fc(x)
        topo_z, weights = self.som(z)
        topo_z = topo_z.squeeze(1)
        return topo_z, weights

# === CLEAR ===
class CLEAR(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, init_nodes=1, max_grid_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.max_grid_size = max_grid_size
        self.node_count = init_nodes
        self.just_grew = False

        self.shared_encoder = SharedEncoder(input_channels=input_channels, latent_dim=latent_dim)

        self.alpha_param = nn.Parameter(torch.tensor(0.2))
        self.memory_bank = nn.Parameter(torch.empty(0, self.latent_dim), requires_grad=False)  # initially empty
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.node_embeddings = nn.Parameter(torch.randn(self.node_count, latent_dim))
        self.attn_weights = None

        self.nodes = nn.ModuleList([Node(latent_dim, max_grid_size) for _ in range(self.node_count)])
        self.decoders = nn.ModuleList([Decoder(input_dim=latent_dim) for _ in range(self.node_count)])

        #task specific for cifar/classification head
        self.classifier = nn.Linear(latent_dim, 10)  

    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)  # [B, D]

        if self.memory_bank is not None and self.memory_bank.size(0) > 0:
            query = self.query_proj(z0)  # [B, D]
            query = F.normalize(query, dim=-1)
            memory_keys = F.normalize(self.memory_bank, dim=-1)  # [P, D]

            attn_weights = torch.softmax(query @ memory_keys.T, dim=-1)  # [B, P]
            memory_output = attn_weights @ self.memory_bank  # [B, D]

            z_aug = z0 + self.alpha_param * memory_output
        else:
            z_aug = z0

        topo_z, _ = self.nodes[-1](z_aug)
        recon = self.decoders[-1](topo_z)
        proto_recon = self.decoders[-1](self.nodes[-1].som.prototypes)
        logits = self.classifier(topo_z)
        return logits, recon, proto_recon

    # === Growth, Diversity, Entropy ===
    def should_grow(self):
        gate_entropy = self.gate_entropy_loss().item()
        proto_sim = self.proto_similarity_loss().item()
        semantic_ratio = proto_sim / (gate_entropy + 1e-8)

        saturation = proto_sim - semantic_ratio
        # Are we encoding more similarity per unit entropy than we have in raw similarity?
        semantic_reversal = semantic_ratio > proto_sim

        print(f"📊 Growth Check:")
        print(f"    ➤ Diversity: {proto_sim:.6f}, Entropy: {gate_entropy:.4f}, Ratio: {semantic_ratio:.6f}")
        print(f"    ➤ Saturation: {saturation:.6f}, Reversal? {semantic_reversal}")

        return semantic_reversal

    def grow_node(self):
        print(f"🌱 Growing CLEAR: Adding Node {self.node_count}")
        self.node_count += 1
        device = self.node_embeddings.device

        self.memory_bank = nn.Parameter(
            torch.cat([
                self.memory_bank,
                self.nodes[-1].som.prototypes.detach().clone()
            ], dim=0),
            requires_grad=False
        )

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

    def proto_similarity_loss(self):
        node = self.nodes[-1]
        P = F.normalize(node.som.prototypes, dim=1)
        S = P @ P.T
        S.fill_diagonal_(0)
        diversity = S.pow(2).mean()
        return diversity

    def node_similarity_loss(self, sample_size=128):
        if self.memory_bank is None or self.memory_bank.size(0) == 0:
            return torch.tensor(0.0, device=self.nodes[0].som.prototypes.device)

        new_som = self.nodes[-1].som
        P_new = F.normalize(new_som.prototypes, dim=1)
        P_mem = F.normalize(self.memory_bank, dim=1)

        if sample_size < P_mem.size(0):
            idx = torch.randperm(P_mem.size(0))[:sample_size]
            P_mem = P_mem[idx]

        sim = F.cosine_similarity(
            P_new.unsqueeze(1),  # [P_new, 1, D]
            P_mem.unsqueeze(0),  # [1, P_mem, D]
            dim=-1
        )
        return sim.abs().mean()

    def gate_entropy_loss(self):
        node = self.nodes[-1]
        p = F.softmax(node.som.gate_logits, dim=0)
        p = p / (p.sum() + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum()
        return entropy

    def log_epoch_info(self):
        print("")
        # for i, node in enumerate(self.nodes):
        #     gate_probs = torch.softmax(node.som.gate_logits, dim=0)
        #     entropy = -(gate_probs * gate_probs.clamp(min=1e-8).log()).sum().item()
        #     print(f"gate_entropy/node_{i}: {entropy}")
