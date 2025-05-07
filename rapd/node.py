import torch
import torch.nn as nn

class Node(nn.Module):
    def __init__(self, latent_dim, symbolic_dim):
        super(Node, self).__init__()
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.symbolic_dim = symbolic_dim

        self.symbolic_head = nn.Linear(latent_dim, symbolic_dim)

        # Replace big dynamic matrix prediction with a small nonlinear operator
        self.op = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def emit_operator(self, z):
        symbolic = self.symbolic_head(z)  # (batch, symbolic_dim)
        return self.op, symbolic  # Return the callable function and the symbolic embedding
