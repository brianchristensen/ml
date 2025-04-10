import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_cifar import get_cifar10_datasets
from model_hierarchical import HierarchicalSOMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 data
train_dataset, test_dataset = get_cifar10_datasets()
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Model
model = HierarchicalSOMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[Hierarchical CIFAR-10] Epoch {epoch}, Loss: {total_loss:.4f}, Test Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "models/model_cifar_hierarchical.pth")
print("✅ CIFAR-10 model saved to models/model_cifar_hierarchical.pth")
