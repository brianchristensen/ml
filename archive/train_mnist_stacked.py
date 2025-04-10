import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_mnist import MNISTCSV
from model_stacked import SOM_Model
from stacked_som import StackedSOM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST data
train_data = MNISTCSV("data/mnist_train.csv")
test_data = MNISTCSV("data/mnist_test.csv")
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# Model setup
model = SOM_Model().to(device)
model.som = StackedSOM(input_dim=64, hidden_dim=64, som_dim=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, z_som, _ = model(x)
        # Main classification loss
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[Stacked MNIST] Epoch {epoch}, Loss: {total_loss:.4f}, Test Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), "models/model_mnist_stacked.pth")
