import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
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
        self.gate_temp_raw = nn.Parameter(torch.tensor([math.log(0.5 / (1 - 0.5))]))
        self.gate_logits = nn.Parameter(torch.full((self.num_cells,), -3.0))
        self.anneal_rate = temp_anneal_rate

        # Laplacian adjacency initialized as Gaussian over grid_pos
        self.lap_adj_sigma = 1.0  # can be tuned
        self.lap_adjacency = None  # computed on-demand in forward pass

    @property
    def temperature(self):
        return torch.sigmoid(self.temp_raw) * (1.0 - 1e-3) + 1e-3
    
    def anneal_temp(self):
        with torch.no_grad():
            current_temp = self.temperature.item()
            new_temp = current_temp * self.anneal_rate
            new_raw = math.log((new_temp - 1e-3) / (1.0 - 1e-3 - new_temp))
            self.temp_raw.fill_(new_raw)

    def get_gate_temperature(self):
        # Learnable, bounded gate temperature in (0.001, 1.0)
        return torch.sigmoid(self.gate_temp_raw) * (1.0 - 1e-3) + 1e-3

    def get_active_prototypes(self, threshold=0.01):
        usage = torch.sigmoid(self.gate_logits)
        active_mask = usage > threshold
        return self.prototypes[active_mask], active_mask

    def compute_laplacian_adjacency(self):
        with torch.no_grad():
            dist = torch.cdist(self.grid_pos, self.grid_pos)  # [K, K]
            sim = torch.exp(-dist ** 2 / (2 * self.lap_adj_sigma ** 2))
            sim.fill_diagonal_(0)
            self.lap_adjacency = sim  # [K, K]

    def neighborhood_loss(self):
        if self.lap_adjacency is None:
            self.compute_laplacian_adjacency()
        A = self.lap_adjacency
        D = torch.diag(A.sum(1))
        L = D - A  # Laplacian
        return torch.trace(self.prototypes.T @ L @ self.prototypes)

    def sinkhorn_weights(self, x, cost_matrix, reg=0.1):
        # Sinkhorn-style approximation using exponentiated negative cost
        weights = torch.exp(-cost_matrix / reg)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

    def forward(self, x):
        gate_temp = self.get_gate_temperature()
        gate = F.softmax(self.gate_logits / gate_temp, dim=0).unsqueeze(0)

        d_proto = torch.cdist(x, self.prototypes)  # [B, K]
        d_pos = torch.cdist(x, self.grid_pos)      # [B, K]
        d_total = d_proto + d_pos                  # [B, K]

        weights = self.sinkhorn_weights(x, d_total, reg=self.temperature)
        weights = weights * gate
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        blended = weights @ self.prototypes
        return blended, weights

# === Node ===
class Node(nn.Module):
    def __init__(self, latent_dim, max_grid_size=256):
        super().__init__()
        self.encoder_fc = nn.Linear(latent_dim, latent_dim)
        self.som = SoftSOM(latent_dim, max_grid_size)
        self.node_history = []

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
        
        self.shared_encoder = SharedEncoder(input_channels=input_channels, latent_dim=latent_dim)

        self.memory_bank = nn.Parameter(torch.empty(0, self.latent_dim), requires_grad=False)
        self.memory_alpha_raw = nn.Parameter(torch.zeros(latent_dim))
        
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_weights = None

        self.nodes = nn.ModuleList([Node(latent_dim, max_grid_size) for _ in range(self.node_count)])
        self.decoders = nn.ModuleList([Decoder(input_dim=latent_dim) for _ in range(self.node_count)])

        #task specific for cifar/classification head
        self.classifier = nn.Linear(latent_dim, 10)  

    @property
    def memory_alpha(self):
        return torch.sigmoid(self.memory_alpha_raw)
    
    def forward(self, x, y=None):
        z0 = self.shared_encoder(x)  # [B, D]

        if self.memory_bank is not None and self.memory_bank.size(0) > 0:
            query = self.query_proj(z0)  # [B, D]
            query = F.normalize(query, dim=-1)
            memory_keys = F.normalize(self.memory_bank, dim=-1)  # [P, D]
            scores = query @ memory_keys.T  # [B, P]
            memory_weights = entmax15(scores, dim=-1)
            memory_output = memory_weights @ self.memory_bank

            z_aug = z0 + self.memory_alpha * memory_output
        else:
            z_aug = z0

        topo_z, _ = self.nodes[-1](z_aug)
        recon = self.decoders[-1](topo_z)
        logits = self.classifier(topo_z)
        return logits, recon

    # === Growth, Diversity, Entropy ===
    def should_grow(self, min_history=4, convergence_delta=0.05):
        node = self.nodes[-1]

        if len(node.node_history) < min_history:
            print(f"📊 Growth Check: Not enough history yet ({len(node.node_history)} entries).")
            return False

        recent = node.node_history[-min_history:]

        # Compute % diffs for convergence check
        diffs = [abs(recent[i+1] - recent[i]) / (recent[i] + 1e-8) for i in range(len(recent)-1)]
        avg_change = sum(diffs) / len(diffs)

        # Compression trend: is it consistently going down?
        compressing = all(recent[i] > recent[i+1] for i in range(len(recent)-1))

        # Usage guard: are we still using a large fraction of SOM?
        usage_fraction = recent[-1] / node.som.num_cells
        still_broad = usage_fraction > 0.85  # not magic, just a relative cap

        print(f"📊 Growth Check:")
        print(f"    ➤ Effective Prototypes: {recent[-1]:.2f}/{self.max_grid_size}")
        print(f"    ➤ Avg Δ over {min_history} epochs: {avg_change:.4f}")
        print(f"    ➤ Usage Fraction: {usage_fraction:.4f}, Still Broad?: {still_broad}")
        print(f"    ➤ Compressing?: {compressing}")

        return compressing and avg_change < convergence_delta and not still_broad

    def grow_node(self):
        print(f"🌱 Growing CLEAR: Adding Node {self.node_count}")
        self.node_count += 1
        device = self.nodes[-1].som.prototypes.device

        self.blend_memory(self.nodes[-1].som.prototypes, self.nodes[-1].som.gate_logits)

        new_node = Node(self.latent_dim, self.max_grid_size).to(device)

        self.nodes.append(new_node)
        self.decoders.append(Decoder(self.latent_dim).to(device))

        with torch.no_grad():
            init_temp = 0.95
            raw_val = math.log(init_temp / (1.0 - init_temp))
            new_node.som.temp_raw.fill_(raw_val)

    def blend_memory(self, new_prototypes, gate_logits=None, blend_ratio=0.5):
        """
        Blends only active prototypes into the memory bank.
        """
        if gate_logits is not None:
            usage = torch.sigmoid(gate_logits)
            active_mask = usage > 0.01
            new_prototypes = new_prototypes[active_mask]
            if new_prototypes.size(0) == 0:
                print("⚠️ No active prototypes to add to memory.")
                return

        if self.memory_bank is None or self.memory_bank.size(0) == 0:
            self.memory_bank = nn.Parameter(new_prototypes.clone(), requires_grad=False)
            return

        mem = self.memory_bank.data
        new = new_prototypes.detach()

        sim = F.cosine_similarity(new.unsqueeze(1), mem.unsqueeze(0), dim=-1)  # [K, M]
        nearest_idx = sim.argmax(dim=1)  # For each new, find closest mem

        for i in range(new.size(0)):
            mem_idx = nearest_idx[i]
            mem[mem_idx] = F.normalize(
                (1 - blend_ratio) * mem[mem_idx] + blend_ratio * new[i],
                dim=0
            )

        self.memory_bank.data = mem

    def freeze_node(self, node):
        for param in node.parameters():
            param.requires_grad = False

    def node_similarity_loss(self, temperature=0.1):
        if self.memory_bank is None or self.memory_bank.size(0) == 0:
            return torch.tensor(0.0, device=self.nodes[0].som.prototypes.device)

        P_new = F.normalize(self.nodes[-1].som.prototypes, dim=1)  # [M, D]
        P_mem = F.normalize(self.memory_bank, dim=1)               # [N, D]

        logits = P_new @ P_mem.T / temperature  # [M, N]
        # Negative log of similarity to closest mem key
        loss = -logits.logsumexp(dim=1).mean()
        return loss

    def anneal_temp_active_node(self):
        self.nodes[-1].som.anneal_temp()

    def log_effective_prototypes(self, log_verbose=False):
        node = self.nodes[-1]
        with torch.no_grad():
            gate_temp = node.som.get_gate_temperature()
            gate = F.softmax(node.som.gate_logits / gate_temp, dim=0)
            eff_n = 1.0 / (gate ** 2).sum().item()
            node.node_history.append(eff_n)

            if log_verbose:
                active_count = (torch.sigmoid(node.som.gate_logits) > 0.01).sum().item()
                print(f"🔍 Effective Prototypes: {eff_n:.2f} | Active: {active_count}/{node.som.num_cells}")

    def epoch_tasks(self):
        self.log_effective_prototypes()
        self.anneal_temp_active_node()
