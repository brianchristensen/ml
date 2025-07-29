import torch
import numpy as np
from model import VAE
from PIL import Image
import os

# Settings
tile_size = 256  # ← must match decoder output
overlap = 32
stride = tile_size - overlap
grid_size = 4
final_size = stride * grid_size + overlap  # ensures full coverage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
vae = VAE().to(device)
vae.load_state_dict(torch.load('vae.pth', map_location=device, weights_only=True))
vae.eval()

# Output folder
os.makedirs('outputs', exist_ok=True)

# Blending mask
def make_blend_mask(size, overlap):
    mask = np.ones((size, size), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    mask[:overlap, :] *= ramp[:, None]
    mask[-overlap:, :] *= ramp[::-1][:, None]
    mask[:, :overlap] *= ramp[None, :]
    mask[:, -overlap:] *= ramp[::-1][None, :]
    return torch.tensor(mask)

blend_mask = make_blend_mask(tile_size, overlap).to(device)

# Accumulated output and weights
canvas = torch.zeros((1, final_size, final_size), device=device)
weights = torch.zeros_like(canvas)

for row in range(grid_size):
    for col in range(grid_size):
        z = torch.randn(1, 256).to(device)
        decoded = vae.decoder(z).cpu().squeeze(0).clamp(0, 1).to(device)  # should be (1, 256, 256)

        if decoded.shape[-2:] != (tile_size, tile_size):
            raise ValueError(f"Decoded tile has shape {decoded.shape}, expected {(tile_size, tile_size)}")

        y = row * stride
        x = col * stride

        canvas[:, y:y+tile_size, x:x+tile_size] += decoded * blend_mask
        weights[:, y:y+tile_size, x:x+tile_size] += blend_mask

# Normalize and save
final = (canvas / (weights + 1e-8)).squeeze().clamp(0, 1).cpu().detach().numpy()
img = (final * 65535).astype(np.uint16)
Image.fromarray(img, mode='I;16').save("outputs/tiled_blended.png")
