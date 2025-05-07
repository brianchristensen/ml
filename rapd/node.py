import torch.nn as nn

class Node(nn.Module):
    def __init__(self, input_dim, hidden_dim, symbolic_dim):
        super(Node, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.symbolic_head = nn.Linear(input_dim, symbolic_dim)

    def forward(self, x):
        transform_out = self.mlp(x)  # nonlinear transform
        symbolic_embed = self.symbolic_head(x)  # symbolic descriptor
        return transform_out, symbolic_embed
