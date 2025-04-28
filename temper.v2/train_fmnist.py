import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from model import TemperGraph

def train_tempernet_fmnist(
    model,
    optimizer,
    num_epochs=10,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    task_loss_weight=0.1  # <<< tiny influence
):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten to 784
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        model.reset_epoch()

        start_time = time.time()
        
        total_loss_accum = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predicted_latent, prediction_error, task_loss, logits = model(inputs, targets)

            prediction_loss = prediction_error.mean()
            routing_loss = model.routing_policy.reinforce()

            total_loss = prediction_loss + routing_loss
            if task_loss is not None:
                total_loss = total_loss + task_loss_weight * task_loss

            total_loss.backward()
            optimizer.step()

            total_loss_accum += total_loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time

        avg_loss = total_loss_accum / len(train_loader)

        print(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.6f} | Duration: {epoch_duration:.2f}s")
        model.print_epoch_summary(epoch, avg_loss)

epochs = 20
hidden_dim = 8
num_tempers = 4
max_path_hops = 8
input_dim = 784  # FMNIST flattened

model = TemperGraph(input_dim=input_dim, hidden_dim=hidden_dim, num_tempers=num_tempers, max_path_hops=max_path_hops)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_tempernet_fmnist(model, optimizer, num_epochs=epochs, batch_size=32)