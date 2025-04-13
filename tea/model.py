import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
        self.last_input = None
        self.last_blended = None
        with torch.no_grad():
            gate_probs = torch.sigmoid(self.som.gate_logits)
            norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
            self.initial_gate_entropy = -(norm_probs * norm_probs.clamp(min=1e-8).log()).sum().item()

    def forward(self, x):
        z = self.encoder_fc(x)
        topo_z, weights = self.som(z)
        self.last_input = z
        self.last_blended = topo_z
        return topo_z, weights

# === CLEAR ===
class CLEAR(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256,
                 init_nodes=1, max_grid_size=256, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_grid_size = max_grid_size
        self.num_heads = num_heads
        self.node_count = init_nodes
        self.just_grew = False
        self.history_window = 10
        self.div_history = []
        self.var_history = []

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

        self.class_embeddings = nn.Parameter(torch.randn(10, latent_dim))
        self.node_embeddings = nn.Parameter(torch.randn(self.node_count, latent_dim))
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_proj = nn.Linear(latent_dim, latent_dim * num_heads)
        self.classifier = nn.Linear(latent_dim * num_heads, 10)

        self.nodes = nn.ModuleList([Node(latent_dim, max_grid_size) for _ in range(self.node_count)])
        self.decoders = nn.ModuleList([Decoder(input_dim=latent_dim) for _ in range(self.node_count)])

    def should_grow(self):
        node = self.nodes[-1]
        current_div = self.proto_diversity_loss().item()
        gate_probs = torch.sigmoid(node.som.gate_logits)
        norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
        entropy_now = -(norm_probs * norm_probs.clamp(min=1e-8).log()).sum().item()
        entropy_start = node.initial_gate_entropy
        entropy_drop = (entropy_start - entropy_now) / (entropy_start + 1e-8)

        current_var = node.last_blended.var(dim=0).mean().item() if node.last_blended is not None else 0.0
        self.div_history.append(current_div)
        self.var_history.append(current_var)

        if len(self.div_history) > self.history_window:
            self.div_history.pop(0)
            self.var_history.pop(0)

        if len(self.div_history) < self.history_window:
            print("⏳ Growth check skipped: Not enough history yet.")
            return False

        if self.just_grew:
            print("🛑 Growth check skipped: Just grew last epoch.")
            self.just_grew = False
            return False

        div_delta = max(self.div_history) - min(self.div_history)
        entropy_shrunk = entropy_drop > 0.05
        variance_low = current_var < 0.05
        diversity_stalled = div_delta < 1e-3
        should = entropy_shrunk and (variance_low or diversity_stalled)

        print(f"📊 Growth Check:")
        print(f"    ➤ Entropy now: {entropy_now:.4f}, Drop: {entropy_drop:.2%}, Shrunk? {entropy_shrunk}")
        print(f"    ➤ Diversity delta: {div_delta:.6f}, Stalled? {diversity_stalled}")
        print(f"    ➤ Variance now: {current_var:.4f}, Low? {variance_low}")
        print(f"    ➤ Growth Decision: {'✅ Grow' if should else '❌ No Growth'}")

        return should

    def grow_node(self):
        print(f"🌱 Growing TEA: Adding Node {self.node_count}")
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
        gate_probs = torch.sigmoid(node.som.gate_logits)
        usage = gate_probs / (gate_probs.sum() + 1e-8)
        usage_weights = usage.unsqueeze(0) * usage.unsqueeze(1)
        gate_entropy = -(usage * usage.clamp(min=1e-8).log()).sum()
        diversity_penalty = (S.pow(2) * usage_weights).sum() / (usage_weights.sum() + 1e-8)
        return gate_entropy * diversity_penalty

    def node_diversity_loss(self):
        if self.node_count <= 1:
            return torch.tensor(0.0, device=self.nodes[0].som.prototypes.device)

        new_som = self.nodes[-1].som
        frozen_soms = [node.som for node in self.nodes[:-1]]

        loss = 0.0
        for old_som in frozen_soms:
            sim = F.cosine_similarity(
                new_som.prototypes.unsqueeze(1),
                old_som.prototypes.unsqueeze(0),
                dim=-1
            )
            loss += sim.abs().mean()

        return loss / len(frozen_soms)

    def gate_entropy_loss(self):
        node = self.nodes[-1]
        p = F.softmax(node.som.gate_logits, dim=0)
        p = p / (p.sum() + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum()
        return entropy

    def usage_penalty(self):
        loss = 0.0
        for node in self.nodes:
            usage = torch.sigmoid(node.som.gate_logits)
            entropy = -(usage * torch.log(usage + 1e-8)).sum()
            ideal_usage = torch.full_like(usage, 1.0 / usage.size(0))
            scale = entropy / math.log(usage.numel())
            loss += scale * F.mse_loss(usage, ideal_usage)
        return loss / len(self.nodes)

    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)
        topo_z_list = [node(z0)[0] for node in self.nodes]
        topo_z_list = [tz.view(tz.size(0), -1) for tz in topo_z_list]
        topo_stack = torch.stack(topo_z_list, dim=0).permute(1, 0, 2)

        if y is not None:
            query = self.query_proj(self.class_embeddings[y])
        else:
            query = self.query_proj(self.class_embeddings.mean(dim=0, keepdim=True)).expand(z0.size(0), -1)

        proj = self.attn_proj(self.node_embeddings).view(self.node_count, self.num_heads, self.latent_dim)
        outputs = []
        for h in range(self.num_heads):
            keys = proj[:, h]
            attn_logits = torch.einsum('bd,nd->bn', query, keys)
            attn_weights = F.softmax(attn_logits, dim=-1)
            fused = torch.einsum('bn,bnd->bd', attn_weights, topo_stack)
            outputs.append(fused)

        fused_all = torch.cat(outputs, dim=1)
        logits = self.classifier(fused_all)
        return logits
