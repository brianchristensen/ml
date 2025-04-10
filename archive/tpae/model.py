import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSOMLayer(nn.Module):
    def __init__(self, input_dim, som_dim=10, temperature=0.4):
        super().__init__()
        self.num_nodes = som_dim * som_dim
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(self.num_nodes, input_dim))

    def forward(self, x):
        dists = torch.cdist(x.unsqueeze(1), self.prototypes.unsqueeze(0))
        weights = F.softmax(-dists.squeeze(1) / self.temperature, dim=-1)
        blended = weights @ self.prototypes
        return blended, weights

class TPAEBlock(nn.Module):
    def __init__(self, input_channels=3, latent_dim=64, som_dim=10, temperature=0.4, vis_decoder=False, fusion_mode="concat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_mode = fusion_mode

        # CNN encoder for input + recon
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.encoder_fc = nn.Linear(64 * 8 * 8 + latent_dim, latent_dim)

        self.som = SoftSOMLayer(latent_dim, som_dim=som_dim, temperature=temperature)

        self.vis_decoder = (
            nn.Sequential(
                nn.Linear(latent_dim, 64 * 8 * 8),
                nn.Unflatten(1, (64, 8, 8)),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
                nn.Sigmoid()
            ) if vis_decoder else None
        )

        self.vis_aux_classifier = (
            nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=1, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, 10)  # 10 CIFAR-10 classes
            ) if vis_decoder else None
        )
        
        self.fuse_decoder = nn.Sequential(
            nn.Conv2d(input_channels + latent_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def merge_inputs(self, x, recon, topo_z):
        # x and recon are images [B, C, 32, 32]
        merged_img = torch.cat([x, recon], dim=1)  # [B, 2C, 32, 32]
        cnn_feat = self.encoder_cnn(merged_img)    # [B, 64, 8, 8]
        flat_feat = cnn_feat.view(cnn_feat.size(0), -1)

        if topo_z is not None:
            flat_feat = torch.cat([flat_feat, topo_z], dim=1)
        else:
            flat_feat = torch.cat([flat_feat, torch.zeros(flat_feat.size(0), self.latent_dim, device=x.device)], dim=1)

        return flat_feat, cnn_feat

    def decode_som_prototypes(self):
        if not self.vis_decoder:
            raise ValueError("Visualizer decoder not enabled.")
        with torch.no_grad():
            proto = self.som.prototypes
            return self.vis_decoder(proto).clamp(0, 1)
    
    def decode_recon_prototypes(self, x_sample):
        """
        Generate reconstructed prototypes using the fused decoder.

        Args:
            x_sample (Tensor): A single input image (batch) to extract cnn_feat and x_down [B, C, 32, 32]

        Returns:
            Tensor: Reconstructed prototypes [num_prototypes, C, 32, 32]
        """
        if self.fuse_decoder is None:
            raise ValueError("This block has no fuse decoder.")

        with torch.no_grad():
            proto = self.som.prototypes              # [N, latent_dim]
            num_prototypes, latent_dim = proto.shape
            x_rep = x_sample[0].unsqueeze(0).expand(num_prototypes, -1, -1, -1)  # [N, C, 32, 32]

            topo_feat = proto.view(num_prototypes, latent_dim, 1, 1)
            topo_feat_upsampled = F.interpolate(topo_feat, size=(32, 32), mode='bilinear', align_corners=False)

            fused = torch.cat([x_rep, topo_feat_upsampled], dim=1)               # [N, C + latent_dim, 32, 32]
            recon = self.fuse_decoder(fused)

            return recon.clamp(0, 1)

    def forward(self, x, recon, prev_topo_z=None):
        flat_feat, _ = self.merge_inputs(x, recon, prev_topo_z)
        z = self.encoder_fc(flat_feat)
        topo_z, weights = self.som(z)

        vis_recon = self.vis_decoder(topo_z) if self.vis_decoder else None

        topo_feat = topo_z.view(topo_z.size(0), topo_z.size(1), 1, 1)
        topo_feat_upsampled = F.interpolate(topo_feat, size=(32, 32), mode='bilinear', align_corners=False)
        fused = torch.cat([x, topo_feat_upsampled], dim=1)
        recon_out = self.fuse_decoder(fused)

        return recon_out, z, topo_z, weights, vis_recon

class StackedTPAE(nn.Module):
    def __init__(self, num_blocks=3, input_channels=3, latent_dim=64, som_dim=10, temperature=0.4, vis_decoder=False, fusion_mode="concat"):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            TPAEBlock(input_channels, latent_dim, som_dim, temperature, vis_decoder, fusion_mode)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(latent_dim, 10)

    def prototype_diversity_loss(self):
        sims = []
        for i in range(len(self.blocks)):
            for j in range(i + 1, len(self.blocks)):
                p1 = F.normalize(self.blocks[i].som.prototypes, dim=1)
                p2 = F.normalize(self.blocks[j].som.prototypes, dim=1)
                sims.append((p1 @ p2.T).abs().mean())
        return sum(sims) / len(sims) if sims else torch.tensor(0.0, device=next(self.parameters()).device)

    def get_reconstruction_weights(self):
        denom = sum((j + 1) ** 2 for j in range(self.num_blocks))
        return [((i + 1) ** 2) / denom for i in range(self.num_blocks)]
    
    def forward(self, x):
        recons = []
        vis_recons = []
        z = topo_z = None
        recon = x

        for i, block in enumerate(self.blocks):
            recon, z, topo_z, _, vis_recon = block(x, recon, topo_z if i > 0 else None)
            recons.append(recon)
            vis_recons.append(vis_recon)

        logits = self.classifier(topo_z)
        return logits, recons, vis_recons, z, topo_z
