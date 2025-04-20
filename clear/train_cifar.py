import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
import time
from datetime import datetime
from model import CLEAR
from dashboard import TrainingDashboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

# === Hyperparameters ===
epochs = 40
latent_dim = 512
max_grid_size = 512
# loss coefficients
classifier_λ = 0.1
recon_λ = 1
node_sim_λ = 1
gate_entropy_λ = 0.1
label_smoothing = 0.1

# === Model & Optimizer ===
model = CLEAR(latent_dim=latent_dim, max_grid_size=max_grid_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# === Data ===
transform_train = transforms.ToTensor()
transform_test = transforms.ToTensor()

train_data = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_data = datasets.CIFAR10("data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=128)

gpu_train_aug = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    T.ToDtype(torch.float32, scale=True)
])

#Simpler augments
# gpu_train_aug = T.Compose([
#     T.RandomCrop(32, padding=4),
#     T.RandomHorizontalFlip(),
#     T.ToDtype(torch.float32, scale=True)
# ])

# === Dashboard Setup ===
dashboard = TrainingDashboard()
start_time = time.time()
dashboard.start(start_time)
dashboard.new_node(node_index=0, epoch=1, acc_at_start=0.0)  # Start with node 0

# === Training ===
print(f"🧠 Training CLEAR Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()

    total_loss = total_acc = 0
    current_proto_sim = current_node_sim = current_gate_entropy = 0
    total_usage_penalty = total_recon_loss = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_aug = gpu_train_aug(x)
        logits, recon = model(x_aug)

        # Losses
        loss_cls = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        node_sim = model.node_similarity_loss()
        gate_entropy = model.gate_entropy_loss()
        recon_loss = F.mse_loss(recon, x)

        loss = (
            classifier_λ * loss_cls +
            node_sim_λ * node_sim +
            gate_entropy_λ * gate_entropy +
            recon_λ * recon_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_acc += (pred == y).sum().item()
            total_loss += loss.item()
            current_node_sim = node_sim.item()
            current_gate_entropy = gate_entropy.item()
            total_recon_loss += recon_loss.item()
            total_samples += y.size(0)

    # === Log to Dashboard
    epoch_duration = time.time() - epoch_start_time
    acc = total_acc / total_samples
    loss_avg = total_loss / len(train_loader)
    node = model.nodes[-1]
    dashboard.log_epoch(epoch, acc=acc, loss=loss_avg)
    dashboard.update_node_metrics(
        node_idx=model.node_count - 1,
        acc=acc,
        node_sim=current_node_sim,
        recon=total_recon_loss,
        temp=node.som.temperature.item(),
        gate_entropy=current_gate_entropy
    )
    dashboard.print_dashboard()

    # === Growth Step
    model.anneal_temp_active_node()
    if model.should_grow():
        model.freeze_node(model.nodes[-1])
        model.grow_node()
        optimizer.add_param_group({'params': model.nodes[-1].parameters()})
        optimizer.add_param_group({'params': model.decoders[-1].parameters()})
        dashboard.new_node(node_index=model.node_count - 1, epoch=epoch, acc_at_start=acc)
    model.epoch_tasks()

# === Save Model
torch.save(model.state_dict(), "models/model_clear.pth")
print("\n✅ Saved to models/model_clear.pth")

# === Evaluation
model.eval()
total_acc = 0
total_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits, recon = model(x)
        pred = logits.argmax(dim=1)
        total_acc += (pred == y).sum().item()
        total_samples += y.size(0)

total_duration = time.time() - start_time
dashboard.final_summary(model, total_acc / total_samples, total_duration)
