import torch
import matplotlib.pyplot as plt
from model_hierarchical import HierarchicalSOMModel
from decoder import Decoder
from decoder2 import Decoder2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained hierarchical model
model = HierarchicalSOMModel().to(device)
model.load_state_dict(torch.load("models/model_mnist_hierarchical.pth", map_location=device))
model.eval()

# Load separately trained decoder
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("models/decoder_mnist_hierarchical.pth", map_location=device))
decoder.eval()

decoder2 = Decoder2().to(device)
decoder2.load_state_dict(torch.load("models/decoder_mnist_hierarchical_som2.pth"))
decoder2.eval()

# Extract SOM prototypes
with torch.no_grad():
    som1_protos = model.som1.prototypes.detach().to(device)
    decoded1 = decoder(som1_protos).cpu()

    # Pass through model.decoder to simulate real reconstruction flow
    recon = model.decoder(som1_protos)
    recon_flat = recon.view(recon.size(0), -1)
    som2_input = recon_flat.detach()
    som2_protos = model.som2.prototypes.detach().to(device)

    # Decode SOM2 prototypes by treating them as inputs (already flattened)
    decoded2 = decoder2(som2_protos.to(device)).cpu()

# Plotting utility
def plot_prototypes(decoded, title, filename):
    grid_size = 10
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size * grid_size):
        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(decoded[i][0], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"explain/{filename}.png")
    plt.close()

# Visualize
plot_prototypes(decoded1, "Hierarchical SOM1 Decoded Prototypes", "decoded_hierarchical_som1")
plot_prototypes(decoded2, "Hierarchical SOM2 Decoded Prototypes", "decoded_hierarchical_som2")
