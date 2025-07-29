# model.py
import torch
import torch.nn as nn

class Mamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):  # expects (T, B, D)
        return self.block(x)

class MambaEncoder(nn.Module):
    def __init__(self, d_model, depth):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Mamba(d_model) for _ in range(depth)]
        )

    def forward(self, x, mask=None):
        # x: (B, T, D) → (T, B, D) for transformer-style processing
        x = x.permute(1, 0, 2)
        x = self.blocks(x)
        x = x.permute(1, 0, 2)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)
        else:
            x = x.mean(dim=1)

        return x

class JEPA_Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, z):
        return self.net(z)


class JEPA_EnergyLoss(nn.Module):
    def __init__(self, mode='l2'):
        super().__init__()
        self.mode = mode

    def forward(self, pred, target):
        if self.mode == 'l2':
            return nn.functional.mse_loss(pred, target)
        elif self.mode == 'cosine':
            pred = nn.functional.normalize(pred, dim=-1)
            target = nn.functional.normalize(target, dim=-1)
            return 1 - (pred * target).sum(dim=-1).mean()
        else:
            raise ValueError("Unknown energy mode")

class JEPA_Mamba_Model(nn.Module):
    def __init__(self, dim=128, depth=4):
        super().__init__()
        self.encoder = MambaEncoder(dim, depth)
        self.target_encoder = MambaEncoder(dim, depth)  # identical structure
        self.predictor = JEPA_Predictor(dim)
        self.energy = JEPA_EnergyLoss(mode='l2')

    def forward(self, x, mask=None):
        # x: (B, T, D), mask: (B, T)
        z_context = self.encoder(x, mask)                  # (B, D)
        z_target = self.target_encoder(x, mask).detach()   # stop-gradient
        z_pred = self.predictor(z_context)                 # predict target

        energy_loss = self.energy(z_pred, z_target)

        # Output z_context for classification head
        return z_context, energy_loss
