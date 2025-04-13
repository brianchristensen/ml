import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from model import CLEAR
import warnings

warnings.simplefilter("ignore", FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
latent_dim = 256
max_grid_size = 256
cluster_threshold = 100
model_path = "models/model_clear.pth"
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
    print("🧠 Loading CLEAR model...")

    # === Step 1: Load checkpoint and infer number of saved nodes ===
    state_dict = torch.load(model_path, map_location=device)
    node_keys = [k for k in state_dict.keys() if k.startswith("nodes.")]
    decoder_keys = [k for k in state_dict.keys() if k.startswith("decoders.")]
    node_indices = set(int(k.split('.')[1]) for k in node_keys)
    init_nodes = max(node_indices) + 1 if node_keys else 1
    print(f"📦 Found {init_nodes} saved nodes")

    # === Step 2: Create model with matching size ===
    model = CLEAR(init_nodes=init_nodes, max_grid_size=max_grid_size, latent_dim=latent_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("🔍 Decoding SOM prototypes...")
    for i, node in enumerate(model.nodes):
        print(f"\n=== [Node {i}] ===")
        with torch.no_grad():
            prototypes = node.som.prototypes.detach()
            usage = torch.sigmoid(node.som.gate_logits)
            active_mask = usage > 0.01
            active_protos = prototypes[active_mask]

            proto_count = active_protos.size(0)
            if proto_count == 0:
                print(f"⚠️  Node {i} has no active prototypes. Skipping visualization.")
                continue
            decoder = model.decoders[i].eval()

            if proto_count > cluster_threshold:
                print(f"📊 Clustering {proto_count} active prototypes...")
                cluster_centers = cluster_prototypes(active_protos, n_clusters=100)
                decoded = decoder(cluster_centers)
                save_visual_grid(decoded, filename=f"node{i}_clusters.png", title=f"Node {i} Cluster Centers")
            else:
                print(f"🧩 Decoding {proto_count} active prototypes...")
                decoded = decoder(active_protos)
                save_visual_grid(decoded, filename=f"node{i}_prototypes.png", title=f"Node {i} Prototypes")

    print("\n🎨 Decoding blended SOM inputs from real images...")
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample.to(device)

    with torch.no_grad():
        z0 = model.shared_encoder(x_sample)

        for i, node in enumerate(model.nodes):
            decoder = model.decoders[i]
            _ = node(z0)
            decoded = decoder(node.last_blended)

            save_visual_grid(decoded, filename=f"node{i}_blended_recons.png", title=f"Node {i} Blended Reconstructions")

        save_visual_grid(x_sample, filename="original_inputs.png", title="Original Input Samples")

if __name__ == "__main__":
    main()
