
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_mnist import MNISTCSV
from model_hierarchical import HierarchicalSOMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = MNISTCSV("data/mnist_train.csv")
test_data = MNISTCSV("data/mnist_test.csv")
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

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
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"[Hierarchical MNIST] Epoch {epoch}, Loss: {total_loss:.4f}, Test Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "models/model_mnist_hierarchical.pth")
