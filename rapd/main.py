# RAPD - Retrieval Augmented Program Decoder

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
import time

#torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_nodes = 8
max_ops = 4
max_gem = 500
num_classes = 10
input_dim = 784
latent_dim = 128
symbolic_dim = 16
batch_size = 64

nodes = [Node(latent_dim, symbolic_dim).to(device) for _ in range(num_nodes)]
router = Router(symbolic_dim, num_nodes).to(device)
gem = GEM(symbolic_dim, max_gem=max_gem, device=device)
synth = Synthesizer(nodes, router, gem, input_dim, latent_dim, device).to(device)
classifier = ClassifierHead(latent_dim, num_classes).to(device)

optimizer = optim.Adam(
    list(classifier.parameters()) + list(router.parameters()) + [p for n in nodes for p in n.parameters()],
    lr=1e-3
)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        latent, programs, sym_embeds = synth.forward(data, max_ops=max_ops)
        output = classifier(latent)
        probs = torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        #synth.update_gem(probs, target)

        epoch_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        del latent, programs, sym_embeds, output, loss, data, target
        torch.cuda.empty_cache()

        if batch_idx % 50 == 0:
            batch_duration = time.time() - batch_start
            print(f"Epoch {epoch+1}/{num_epochs}, batch {batch_idx+1}/{len(train_loader)} | Duration: {batch_duration:.4f}s")

    duration = time.time() - start_time
    acc = correct / total * 100
    avg_loss = epoch_loss / total
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | Duration: {duration:.2f}s")
