import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_mnist import MNISTCSV
from model_hierarchical import HierarchicalSOMModel
from decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILE = "models/model_mnist_hierarchical.pth"

# Load trained SOM model
model = HierarchicalSOMModel().to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# Freeze encoder and SOM
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.som1.parameters():
    param.requires_grad = False

# Prepare dataset
dataset = MNISTCSV("data/mnist_train.csv")
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize decoder
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

# Train decoder for 5 epochs
for epoch in range(5):
    decoder.train()
    total_loss = 0
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            z = model.encoder(x)
        recon = decoder(z)
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Decoder Training] Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save trained decoder
torch.save(decoder.state_dict(), "models/decoder_mnist_hierarchical.pth")
print("✅ Decoder saved to models/decoder_mnist_hierarchical.pth")
