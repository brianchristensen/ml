import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from model import TEA
import warnings

warnings.simplefilter("ignore", FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
num_nodes_default = 4
som_dim_default = [8, 10, 8, 5]
latent_dim = 256
cluster_threshold = 100
model_path = "models/model_tea.pth"
explain_dir = "explain"
os.makedirs(explain_dir, exist_ok=True)

# === Visualization Helpers ===
def save_visual_grid(decoded_imgs, filename="visual.png", n_cols=10, title=None):
    n = len(decoded_imgs)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    axs = axs.reshape((n_rows, n_cols)) if isinstance(axs, np.ndarray) else [[axs]]

    for i in range(n_rows * n_cols):
        ax = axs[i // n_cols][i % n_cols]
        if i < n:
            img = decoded_imgs[i].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    filepath = os.path.join(explain_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"📸 Saved: {filepath}")

def cluster_prototypes(prototypes, n_clusters=100):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = km.fit_predict(prototypes.cpu().numpy())
    centers = torch.tensor(km.cluster_centers_, dtype=prototypes.dtype, device=prototypes.device)
    return centers

# === Main ===
def main():
    print("🧠 Loading TEA model...")
    model = TEA(num_nodes=num_nodes_default, som_dim=som_dim_default, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("🔍 Decoding SOM prototypes...")
    for i, node in enumerate(model.nodes):
        print(f"\n=== [Node {i}] ===")
        with torch.no_grad():
            prototypes = node.som.prototypes.detach()
            proto_count = prototypes.size(0)
            decoder = model.decoders[i].eval() if hasattr(model, "decoders") else model.decoder.eval()

            if proto_count > cluster_threshold:
                print(f"📊 Clustering {proto_count} prototypes...")
                cluster_centers = cluster_prototypes(prototypes, n_clusters=100)
                decoded = decoder(cluster_centers)
                save_visual_grid(decoded, filename=f"node{i}_clusters.png", title=f"Node {i} Cluster Centers")
            else:
                print(f"🧩 Decoding all {proto_count} prototypes...")
                decoded = decoder(prototypes)
                save_visual_grid(decoded, filename=f"node{i}_prototypes.png", title=f"Node {i} Prototypes")

    print("\n🎨 Decoding blended SOM inputs from real images...")
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample.to(device)

    with torch.no_grad():
        z0 = model.shared_encoder(x_sample)
        z_in = z0

        for i, node in enumerate(model.nodes):
            decoder = model.decoders[i]
            z_node = node.encoder_fc(z_in)
            dists = torch.cdist(z_node.unsqueeze(1), node.som.prototypes.unsqueeze(0))
            weights = F.softmax(-dists.squeeze(1) / node.som.temperature, dim=-1)
            blended = weights @ node.som.prototypes  # [B, latent_dim]
            decoded = decoder(blended)               # [B, 3, 32, 32]

            save_visual_grid(decoded, filename=f"node{i}_blended_recons.png", title=f"Node {i} Blended Reconstructions")

            z_in = F.layer_norm(z0 + 0.5 * blended, [model.latent_dim])

        # Optionally save the originals too for comparison
        save_visual_grid(x_sample, filename="original_inputs.png", title="Original Input Samples")

if __name__ == "__main__":
    main()
