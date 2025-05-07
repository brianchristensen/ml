import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from synthesizer import Synthesizer
from node import Node
from router import Router
from heads import ClassifierHead
from gem import GEM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_dim = 784
hidden_dim = 256
symbolic_dim = 16
num_nodes = 8
num_classes = 10
batch_size = 64

print(f"Using batch size: {batch_size}")

# Setup
nodes = [Node(input_dim, hidden_dim, symbolic_dim).to(device) for _ in range(num_nodes)]
router = Router(input_dim, symbolic_dim, num_nodes, transformer_hidden=128, nhead=4, num_layers=2).to(device)
gem = GEM(symbolic_dim, device=device)
synth = Synthesizer(nodes, router, gem, device)
classifier = ClassifierHead(input_dim, num_classes).to(device)

optimizer = optim.Adam(
    list(classifier.parameters()) + list(router.parameters()) + [p for n in nodes for p in n.parameters()],
    lr=1e-3
)

# MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(5):
    epoch_start = time.time()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        latent, selected_idx, _ = synth.forward(data)
        output = classifier(latent)
        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)

        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    epoch_duration = time.time() - epoch_start
    avg_loss = epoch_loss / total
    accuracy = 100. * correct / total

    print(f'\nEpoch {epoch} | Duration: {epoch_duration:.2f}s | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    top_gem = gem.get_top_k(10)
    print("\nTop GEM entries:")
    for i, entry in enumerate(top_gem):
        print(f"  #{i+1}: Reward: {entry['reward']:.4f}, Program: {entry['program']}")
    print("\nRouter selection counts:")
    total = sum(synth.selection_counter.values())
    for node_idx, count in synth.selection_counter.items():
        print(f"  Node {node_idx}: {count} times ({count / total:.2%})")
    synth.selection_counter.clear()
