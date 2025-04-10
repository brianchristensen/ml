import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from model import StackedTPAE  # adjust if needed
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

# --- Config ---
MODEL_PATH = 'models/model_tpae.pth'
OUTPUT_DIR = 'explain'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load test data ---
transform = transforms.ToTensor()
test_data = datasets.CIFAR10("../data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StackedTPAE(num_blocks=5, vis_decoder=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Visualize SOM Prototypes (only once, doesn't depend on x) ---
for i, block in enumerate(model.blocks):
    try:
        decoded = block.decode_som_prototypes().cpu()
    except Exception as e:
        print(f"[Block {i+1}] SOM visualization failed:", e)
        continue

    grid = vutils.make_grid(decoded, nrow=10, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.title(f"SOM Prototypes - Block {i+1}")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/TPAE_block{i+1}_som.png')
    plt.close()

# --- Pick 3 random samples from the test set ---
sample_indices = random.sample(range(len(test_data)), 3)
sample_batch = torch.stack([test_data[i][0] for i in sample_indices]).to(device)

# --- Visualize fused decoder outputs per sample ---
for idx, x_sample in enumerate(sample_batch):
    x_sample = x_sample.unsqueeze(0)  # add batch dim [1, C, H, W]

    for i, block in enumerate(model.blocks):
        try:
            recon = block.decode_recon_prototypes(x_sample).cpu()
        except Exception as e:
            print(f"[Block {i+1}] Fused recon failed for sample {idx+1}:", e)
            continue

        grid = vutils.make_grid(recon, nrow=10, padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.title(f"Recon Prototypes - Block {i+1} | Sample {idx+1}")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/TPAE_block{i+1}_recon_sample{idx+1}.png')
        plt.close()

print(f"✅ Saved visualizations for 3 samples to {OUTPUT_DIR}")
