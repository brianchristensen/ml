import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import TemperGraph

def train_tempernet_v2(
    model,
    optimizer,
    num_epochs=10,
    batch_size=32,
    input_dim=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    model.train()

    for epoch in range(1, num_epochs + 1):
        model.reset_epoch()

        start_time = time.time()

        inputs = torch.randn(batch_size, input_dim, device=device)

        _, _, prediction_error = model(inputs)

        loss = prediction_error.mean()
        routing_loss = model.routing_policy.reinforce()
        total_loss = loss + routing_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"\nEpoch {epoch}: Loss = {loss.item():.6f} | Duration: {epoch_duration:.2f}s")
        model.print_epoch_summary(epoch, loss.item())

epochs = 20
hidden_dim = 8
num_tempers = 4
max_path_hops = 8
input_dim = 128

model = TemperGraph(input_dim=input_dim, hidden_dim=hidden_dim, num_tempers=num_tempers, max_path_hops=max_path_hops)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_tempernet_v2(model, optimizer, num_epochs=epochs, batch_size=32, input_dim=128)
