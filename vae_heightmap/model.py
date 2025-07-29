import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 32x32
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 16x16
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 8x8
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 128x128
        x = F.relu(self.conv2(x))  # 64x64
        x = F.relu(self.conv3(x))  # 32x32
        x = F.relu(self.conv4(x))  # 16x16
        x = F.relu(self.conv5(x))  # 8x8
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32x32
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 64x64
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 128x128
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)     # 256x256

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 512, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))  # final output [B, 1, 256, 256]
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
