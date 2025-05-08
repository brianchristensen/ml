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

num_epochs = 10
num_nodes = 1
max_ops = 1
max_gem = 10000
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
    window_start = time.time()
    window_batches = 0
    epoch_loss = 0.0
    correct = 0
    total = 0
    reward_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        latent, programs, sym_embeds = synth.forward(data, max_ops=max_ops)
        output = classifier(latent)
        probs = torch.softmax(output, dim=1)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        reward_list.append(probs[range(len(target)), target].detach().cpu())

        del latent, programs, sym_embeds, output, loss, data, target
        torch.cuda.empty_cache()

        window_batches += 1
        if batch_idx % 50 == 0 and batch_idx > 0:
            window_duration = time.time() - window_start
            avg_per_batch = window_duration / window_batches
            print(f"Epoch {epoch+1}/{num_epochs}, batch {batch_idx+1}/{len(train_loader)} | "
                f"Avg duration per batch: {avg_per_batch:.4f}s over {window_batches} batches")
            window_start = time.time()
            window_batches = 0

    all_rewards = torch.cat(reward_list, dim=0)
    synth.update_gem(all_rewards)
    reward_list = []

    duration = time.time() - start_time
    acc = correct / total * 100
    avg_loss = epoch_loss / total
    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | Duration: {duration:.2f}s")
    gem.print_top_n(10)
