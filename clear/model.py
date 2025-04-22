import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
import numpy as np
import math
from sklearn.cluster import KMeans

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
        return self.relu(out)

class SharedEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
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
    def __init__(self, latent_dim=256, output_channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
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

# === ProtoGraph ===
class ProtoGraph(nn.Module):
    def __init__(self, latent_dim, max_prototypes=256):
        super().__init__()
        self.max_prototypes = max_prototypes
        self.latent_dim = latent_dim
        self.graph_history = []

        self.prototypes = nn.Parameter(torch.randn(max_prototypes, latent_dim))
        self.adj_logits = nn.Parameter(torch.randn(max_prototypes, max_prototypes) * 0.01)

        self.gate_logits = nn.Parameter(torch.zeros(max_prototypes))

        # Learnable temperature for entmax15 over gates
        self.gate_temp_raw = nn.Parameter(torch.tensor([-2.0]))

    def get_gate_temperature(self):
        return torch.sigmoid(self.gate_temp_raw) * (1.0 - 1e-3) + 1e-3

    def get_active_prototypes(self, threshold=0.01):
        usage = torch.sigmoid(self.gate_logits)
        return self.prototypes[usage > threshold], usage > threshold

    def get_effective_prototypes(self, verbose=False):
        with torch.no_grad():
            gate_temp = self.get_gate_temperature()
            gate = F.softmax(self.gate_logits / gate_temp, dim=0)
            eff_n = 1.0 / (gate ** 2).sum().item()

            self.graph_history.append(eff_n)
            if len(self.graph_history) > 5:
                self.graph_history.pop(0)

            if verbose:
                active_count = (torch.sigmoid(self.gate_logits) > 0.01).sum().item()
                print(f"🔍 Effective Prototypes: {eff_n:.2f} | Active: {active_count}/{self.max_prototypes}")

            return eff_n

    def forward(self, x):
        gate_temp = self.get_gate_temperature()
        gate = F.softmax(self.gate_logits / gate_temp, dim=0).unsqueeze(0)  # [1, P]
        
        # Prototype distances and attention
        dists = torch.cdist(x, self.prototypes)  # [B, P]
        weights = entmax15(-dists, dim=-1)       # [B, P]
        self.last_routing_weights = weights.detach()

        # Graph propagation via learned adjacency
        routed = weights @ entmax15(self.adj_logits, dim=-1)  # [B, P]

        # Apply gating
        gated = routed * gate  # [B, P]
        gated = gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)

        # Blend into prototype space
        blended = gated @ self.prototypes  # [B, D]
        return blended, gated

# === Node ===
class Node(nn.Module):
    def __init__(self, latent_dim, max_prototypes=256):
        super().__init__()
        self.encoder_fc = nn.Linear(latent_dim, latent_dim)
        self.prog = ProtoGraph(latent_dim, max_prototypes)

    def forward(self, x):
        z = self.encoder_fc(x)
        z_structured, weights = self.prog(z)
        return z_structured, weights

# === CLEAR ===
class CLEAR(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256, init_nodes=1, max_prototypes=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_prototypes = max_prototypes
        self.node_count = init_nodes

        self.shared_encoder = SharedEncoder(input_channels, latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.memory_alpha_raw = nn.Parameter(torch.zeros(latent_dim))

        self.memory_nodes = []  # per-node compressed memory via KMeans
        self.nodes = nn.ModuleList([Node(latent_dim, max_prototypes) for _ in range(init_nodes)])
        self.decoders = nn.ModuleList([Decoder(latent_dim) for _ in range(init_nodes)])
        self.classifier = nn.Linear(latent_dim, 10)

    @property
    def memory_alpha(self):
        return torch.sigmoid(self.memory_alpha_raw)

    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)

        if self.memory_nodes:
            mem_keys = torch.cat(self.memory_nodes, dim=0)  # [M, D]
            mem_keys = F.normalize(mem_keys, dim=-1)
            query = F.normalize(self.query_proj(z0), dim=-1)

            sim = query @ mem_keys.T  # [B, M]
            k = min(16, sim.shape[1])
            topk = sim.topk(k=k, dim=-1)
            sparse_weights = torch.zeros_like(sim)
            sparse_weights.scatter_(1, topk.indices, topk.values)
            weights = entmax15(sparse_weights, dim=-1)

            z_aug = z0 + self.memory_alpha * (weights @ mem_keys)
        else:
            z_aug = z0

        z_structured, _ = self.nodes[-1](z_aug)
        recon = self.decoders[-1](z_structured)
        logits = self.classifier(z_structured)
        return logits, recon

    def add_node_to_memory(self, prototypes, gate_logits, k=16):
        usage = torch.sigmoid(gate_logits)
        active = prototypes[usage > 0.01]
        if active.size(0) < k:
            print("⚠️ Not enough active prototypes for KMeans.")
            return
        with torch.no_grad():
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            centroids = km.fit(active.cpu().numpy()).cluster_centers_
            centroids = torch.tensor(centroids, dtype=prototypes.dtype, device=prototypes.device)
            centroids = F.normalize(centroids, dim=-1)
        self.memory_nodes.append(centroids)

    def should_grow(self, flat_threshold=1.5, usage_threshold=0.90):
        node = self.nodes[-1]
        prog = node.prog

        if not prog.graph_history or len(prog.graph_history) < 3:
            return False  # not enough history

        # Check for plateau
        delta_eff = max(prog.graph_history) - min(prog.graph_history)
        plateau = delta_eff < flat_threshold

        # Check if usage is near max
        usage = torch.sigmoid(prog.gate_logits)
        active = (usage > 0.01).sum().item()
        usage_ratio = active / self.max_prototypes

        print("🔍 Node Growth Check (Static Prototype Count)")
        print(f"    ➤ Δeff: {delta_eff:.4f}")
        print(f"    ➤ Active Prototypes: {active}/{self.max_prototypes} ({usage_ratio * 100:.2f}%)")
        print(f"    ➤ Plateau: {plateau}")

        if plateau and usage_ratio > usage_threshold:
            print("🌲 Graph saturated and stable → Growing a new node.")
            return True

        return False

    def grow_node(self):
        print(f"🌱 Growing CLEAR: Adding Node {self.node_count}")
        self.node_count += 1
        device = self.nodes[-1].prog.prototypes.device
        self.add_node_to_memory(self.nodes[-1].prog.prototypes, self.nodes[-1].prog.gate_logits)
        new_node = Node(self.latent_dim, self.max_prototypes).to(device)
        self.nodes.append(new_node)
        self.decoders.append(Decoder(self.latent_dim).to(device))

    def freeze_node(self, node):
        for param in node.parameters():
            param.requires_grad = False
        node.eval()

    def adjacency_l1_loss(self, node_idx=-1):
        return self.nodes[node_idx].prog.adj_logits.abs().mean()

    def epoch_tasks(self):
        self.nodes[-1].prog.get_effective_prototypes(verbose=True)
