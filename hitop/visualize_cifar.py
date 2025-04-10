import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from model import HiTop
import warnings

warnings.simplefilter("ignore", FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
num_nodes_default = 4
som_dim_default = [10, 20, 30, 40]
latent_dim = 256
cluster_threshold = 100
model_path = "models/model_hitop.pth"
explain_dir = "explain"
os.makedirs(explain_dir, exist_ok=True)

# === Visualization Helpers ===
def save_visual_grid(decoded_imgs, filename="visual.png", n_cols=10, title=None):
    n = len(decoded_imgs)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    for i in range(n_rows * n_cols):
        ax = axs[i // n_cols, i % n_cols]
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
    print("🧠 Loading HiTop model...")
    model = HiTop(num_nodes=num_nodes_default, som_dim=som_dim_default, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    decoder = model.decoder.eval()

    print("🔍 Decoding SOM prototypes directly...")

    for node_index, node in enumerate(model.nodes):
        print(f"\n=== [Node {node_index}] ===")
        with torch.no_grad():
            prototypes = node.som.prototypes.detach()
            proto_count = prototypes.size(0)

            if proto_count > cluster_threshold:
                print(f"📊 Clustering {proto_count} prototypes...")
                cluster_centers = cluster_prototypes(prototypes, n_clusters=100)
                decoded = decoder(cluster_centers)
                save_visual_grid(decoded, filename=f"node{node_index}_clusters.png", title=f"Node {node_index} Cluster Centers")
            else:
                print(f"🧩 Decoding all {proto_count} prototypes...")
                decoded = decoder(prototypes)
                save_visual_grid(decoded, filename=f"node{node_index}_prototypes.png", title=f"Node {node_index} Prototypes")

if __name__ == "__main__":
    main()
