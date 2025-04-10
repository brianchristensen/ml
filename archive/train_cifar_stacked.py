import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_cifar import get_cifar10_datasets
from model_stacked import SOM_Model
from stacked_som import StackedSOM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR-10 datasets (grayscale, 28x28)
train_dataset, test_dataset = get_cifar10_datasets()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# Initialize model
model = SOM_Model().to(device)
model.som = StackedSOM(input_dim=64, hidden_dim=64, som_dim=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train for 10 epochs
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, z_som, _ = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate on test set
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
    print(f"[CIFAR-10] Epoch {epoch+1}, Loss: {total_loss:.4f}, Test Accuracy: {correct / total:.4f}")

# Save the model
torch.save(model.state_dict(), "models/model_cifar_stacked.pth")
print("✅ CIFAR-10 model saved to models/model_cifar_stacked.pth")
