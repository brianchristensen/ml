import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import numpy as np
from model import ALS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
epochs = 20
num_generators = 24
steps = 4
latent_dim = 512
input_dim = 3 * 32 * 32

label_smoothing = 0.5

# CIFAR-10 as flattened inputs
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Model setup
model = ALS(input_dim=input_dim, latent_dim=latent_dim, num_generators=num_generators, steps=steps).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

start_time = time.time()
print(f"🧠 Training Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in trainloader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        task_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        loss = (
            task_loss
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    epoch_duration = time.time() - epoch_start_time
    print(f"📚 Epoch {epoch}: "
      f"Loss={running_loss/len(trainloader):.4f}, "
      f"Train Acc={100.0 * correct / total:.2f}%, "
      f"Duration: {epoch_duration:.2f}s")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        logits = model(x)
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

total_duration = time.time() - start_time
print(f"\n✅ Test Accuracy: {100.0 * correct / total:.2f}%, Total Duration: {int(total_duration // 60)}m {int(total_duration % 60)}s")
