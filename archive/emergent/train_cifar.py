import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
from model import TemperGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
epochs = 20
num_tempers = 4
latent_dim = 128
input_dim = 3 * 32 * 32

label_smoothing = 0.1
aux_loss_weight = 0.1

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
model = TemperGraph(input_dim=input_dim, latent_dim=latent_dim, num_tempers=num_tempers).to(device)
main_optimizer = optim.Adam(model.encoder.parameters(), lr=1e-3)
main_optimizer.add_param_group({'params': model.msg_update.parameters()})
main_optimizer.add_param_group({'params': model.query_proj.parameters()})
main_optimizer.add_param_group({'params': model.key_proj.parameters()})
main_optimizer.add_param_group({'params': model.final_predictor.parameters()})
main_optimizer.add_param_group({'params': model.aux_heads.parameters()})

per_temper_optimizers = [
    optim.Adam(tmpr.joint_policy.parameters(), lr=1e-3)
    for tmpr in model.tempers
]

start_time = time.time()
print(f"\U0001f9e0 Training Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in trainloader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        for opt in per_temper_optimizers:
            opt.zero_grad()
        main_optimizer.zero_grad()

        logits, weighted_instability, tempers = model(x)
        task_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        # Auxiliary classifier loss
        aux_logits = [head(tr) for head, tr in zip(model.aux_heads, tempers)]
        aux_losses = [F.cross_entropy(logit, y, label_smoothing=label_smoothing) for logit in aux_logits]
        aux_loss = torch.stack(aux_losses).mean()

        loss = task_loss + weighted_instability.clamp(max=10) + aux_loss_weight * aux_loss

        loss.backward()

        for opt in per_temper_optimizers:
            opt.step()
        main_optimizer.step()

        preds = logits.argmax(dim=-1)
        running_loss += loss.item()
        total += y.size(0)
        correct += preds.eq(y).sum().item()

    epoch_duration = time.time() - epoch_start_time
    print(f"\U0001f4da Epoch {epoch}: "
          f"Loss={running_loss/len(trainloader):.4f}, "
          f"Train Acc={100.0 * correct / total:.2f}%, "
          f"Instability: {weighted_instability:.2f}, "
          f"Duration: {epoch_duration:.2f}s")
    model.report_log_summary()

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.view(x.size(0), -1).to(device), y.to(device)
        logits, weighted_instability, _ = model(x)
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

total_duration = time.time() - start_time
print(f"\n✅ Test Accuracy: {100.0 * correct / total:.2f}%, Total Duration: {int(total_duration // 60)}m {int(total_duration % 60)}s")
