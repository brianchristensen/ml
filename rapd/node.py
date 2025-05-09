import torch
import torch.nn as nn
import torch.nn.functional as F

class Node(nn.Module):
    def __init__(self, latent_dim, symbolic_dim, num_codes=16):
        super(Node, self).__init__()
        self.latent_dim = latent_dim
        self.symbolic_dim = symbolic_dim
        self.num_codes = num_codes

        self.codebook = nn.Parameter(torch.randn(num_codes, symbolic_dim))
        self.projector = nn.Linear(latent_dim, symbolic_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim * 2 + symbolic_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def emit_operator(self, z):
        topk = min(self.num_codes, self.symbolic_dim)

        z_proj = self.projector(z)  # [batch, symbolic_dim]
        distances = torch.cdist(z_proj.unsqueeze(1), self.codebook.unsqueeze(0))  # [batch, 1, num_codes]
        distances = distances.squeeze(1)

        _, topk_indices = torch.topk(-distances, k=topk, dim=1)
        selected_codes = self.codebook[topk_indices]  # [batch, topk, symbolic_dim]

        mean_codes = selected_codes.mean(dim=1)  # [batch, symbolic_dim]

        def op_fn(z_in, symbolic_in):
            combined_input = torch.cat([z_in, symbolic_in], dim=1)  # [subbatch, latent_dim + symbolic_dim]
            out = self.out_proj(combined_input)  # [subbatch, latent_dim]
            return out

        return op_fn, mean_codes
