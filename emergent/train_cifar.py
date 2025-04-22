import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from model import ALS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters
epochs = 20
max_steps = 10
latent_dim = 256
input_dim = 3 * 32 * 32
label_smoothing = 0.01

# === Data transforms
train_augment = T.Compose([
    T.ToImage(),
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    T.ToDtype(torch.float32, scale=True)
])
test_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

# === Datasets
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToImage())
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# === Model
model = ALS(input_dim=input_dim, latent_dim=latent_dim, max_steps=max_steps).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Training
start_time = time.time()
print(f"🧠 Training Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        x = train_augment(x)

        optimizer.zero_grad()
        out, _ = model(x, return_trace=True)  # ← enable trace logging
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    print(f"📚 Epoch {epoch}: Loss={running_loss/len(trainloader):.4f}, Train Acc={100.0 * correct / total:.2f}%, Duration: {time.time() - epoch_start_time:.2f}s")

# === Eval
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, predicted = out.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

print(f"\n✅ Test Accuracy: {100.0 * correct / total:.2f}%, Total Duration: {int(time.time() - start_time)//60}m {int(time.time() - start_time)%60}s")
