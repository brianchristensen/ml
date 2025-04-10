import torch
import matplotlib.pyplot as plt
from model import SOM_Model
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model and decoder
model = SOM_Model().to(device)
model.load_state_dict(torch.load("models/model_mnist_stacked.pth"))
model.eval()

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("models/decoder_mnist.pth"))  # adjust if different
decoder.eval()

# Get prototypes
with torch.no_grad():
    som1_protos, som2_protos = (p.to(device) for p in model.som.get_prototypes())

    decoded1 = decoder(som1_protos).cpu()
    decoded2 = decoder(som2_protos).cpu()

# Plotting helper
def plot_prototypes(decoded, title, filename):
    grid_size = 10
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size * grid_size):
        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(decoded[i][0], cmap="gray")
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"explain/{filename}.png")
    plt.close()

plot_prototypes(decoded1, "Decoded SOM Layer 1 Prototypes", "decoded_som1_stacked_prototypes")
plot_prototypes(decoded2, "Decoded SOM Layer 2 Prototypes", "decoded_som2_stacked_prototypes")
