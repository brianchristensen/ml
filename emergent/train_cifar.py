import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import ALS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
epochs = 30
num_generators=16
steps=4
input_dim = 3 * 32 * 32
latent_dim=256

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in trainloader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"📚 Epoch {epoch}: Loss={running_loss/len(trainloader):.4f}, Train Acc={100.0 * correct / total:.2f}%")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"\n✅ Test Accuracy: {100.0 * correct / total:.2f}%")
