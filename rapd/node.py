import torch
import torch.nn as nn

class Node(nn.Module):
    def __init__(self, latent_dim, symbolic_dim):
        super(Node, self).__init__()
        self.latent_dim = latent_dim
        self.symbolic_dim = symbolic_dim

        self.symbolic_head = nn.Linear(latent_dim, symbolic_dim)

        # Combine latent + symbolic for op
        self.op = nn.Sequential(
            nn.Linear(latent_dim + symbolic_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def emit_operator(self, z):
        symbolic = self.symbolic_head(z)
        def op_fn(z_in, symbolic_in):
            combined = torch.cat([z_in, symbolic_in], dim=-1)
            return self.op(combined)
        return op_fn, symbolic
