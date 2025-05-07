import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, z):
        return self.mlp(z)
