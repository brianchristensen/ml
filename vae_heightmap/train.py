import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import VAE
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=256).to(device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob(os.path.join(root_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

dataset = CustomDataset(root_dir='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)
reconstruction_loss_fn = nn.MSELoss(reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

for epoch in range(1, 51):
    vae.train()
    total_loss = 0
    for batch, _ in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        loss = loss_function(recon, batch, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f"Epoch [{epoch}/50] - Loss: {total_loss:.2f}")

torch.save(vae.state_dict(), 'vae.pth')
