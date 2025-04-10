# train_decoder_som2_mnist.py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_mnist import MNISTCSV
from model_hierarchical import HierarchicalSOMModel
from decoder2 import Decoder2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HierarchicalSOMModel().to(device)
model.load_state_dict(torch.load("models/model_mnist_hierarchical.pth", map_location=device))
model.eval()

for param in model.parameters():
    param.requires_grad = False

dataset = MNISTCSV("data/mnist_train.csv")
loader = DataLoader(dataset, batch_size=128, shuffle=True)

decoder = Decoder2().to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            z = model.encoder(x)
            z1, _ = model.som1(z)
            recon = model.decoder(z1)
            recon_flat = recon.view(recon.size(0), -1)
        output = decoder(recon_flat)
        loss = F.mse_loss(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Decoder SOM2 Training] Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(decoder.state_dict(), "models/decoder_mnist_hierarchical_som2.pth")
print("✅ DecoderSOM2 saved")
