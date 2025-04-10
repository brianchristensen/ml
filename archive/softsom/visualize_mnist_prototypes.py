import torch
import matplotlib.pyplot as plt
from model import SOM_Model
from decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and decoder
model = SOM_Model().to(device)
model.load_state_dict(torch.load("models/model_mnist.pth", map_location=device))
model.eval()

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("models/decoder_mnist.pth", map_location=device))
decoder.eval()

# Decode SOM prototypes
prototypes = model.som.prototypes.detach().to(device)
decoded_imgs = decoder(prototypes).cpu()  # (num_prototypes, 28, 28)

# Plot
side = int(decoded_imgs.size(0) ** 0.5)
fig, axs = plt.subplots(side, side, figsize=(10, 10))
for i in range(decoded_imgs.size(0)):
    row = i // side
    col = i % side
    axs[row, col].imshow(decoded_imgs[i].squeeze(0).detach().numpy(), cmap='gray')
    axs[row, col].axis('off')

plt.suptitle("Decoded SOM Prototypes (28x28)")
plt.tight_layout()
plt.savefig("explain/decoded_mnist_prototypes.png")
plt.show()
