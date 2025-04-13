import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
import time
from datetime import datetime
from model import TEA
from dashboard import TrainingDashboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

# === Hyperparameters ===
epochs = 60
latent_dim = 256
max_grid_size = 256
# loss coefficients
recon_loss_λ = 1
proto_div_λ = 4
node_div_λ = 0.5
gate_entropy_λ = 0.1
usage_λ = 0.7
label_smoothing = 0.1

# === Model & Optimizer ===
model = TEA(latent_dim=latent_dim, max_grid_size=max_grid_size).to(device)
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

# Simpler augments - less generality of model, easier to invert prototypes with decoder
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
print(f"🧠 Training TEA Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()

    total_loss = total_acc_topo = 0
    total_proto_div = total_node_div = total_gate_entropy = 0
    total_usage_penalty = total_recon_loss = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_aug = gpu_train_aug(x)
        logits = model(x_aug)

        # Losses
        loss_cls = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        proto_div = model.proto_diversity_loss()
        node_div = model.node_diversity_loss()
        gate_entropy = model.gate_entropy_loss()
        usage_penalty = model.usage_penalty()
        recon_loss = F.mse_loss(model.decoders[-1](model.nodes[-1].last_blended), x)

        loss = (
            loss_cls +
            proto_div_λ * proto_div +
            node_div_λ * node_div +
            gate_entropy_λ * gate_entropy +
            usage_λ * usage_penalty +
            recon_loss_λ * recon_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_topo = logits.argmax(dim=1)
            total_acc_topo += (pred_topo == y).sum().item()
            total_loss += loss.item()
            total_proto_div += proto_div.item()
            total_node_div += node_div.item()
            total_gate_entropy += gate_entropy.item()
            total_usage_penalty += usage_penalty.item()
            total_recon_loss += recon_loss.item()
            total_samples += y.size(0)

    # === Compute Metrics
    epoch_duration = time.time() - epoch_start_time
    acc = total_acc_topo / total_samples
    loss_avg = total_loss / len(train_loader)
    recon = total_recon_loss
    usage = total_usage_penalty
    proto_div = total_proto_div
    node_div = total_node_div

    # === Growth Info
    node = model.nodes[-1]
    current_div = proto_div
    gate_probs = torch.sigmoid(node.som.gate_logits)
    norm_probs = gate_probs / (gate_probs.sum() + 1e-8)
    entropy_now = -(norm_probs * norm_probs.clamp(min=1e-8).log()).sum().item()
    entropy_start = node.initial_gate_entropy
    entropy_drop = (entropy_start - entropy_now) / (entropy_start + 1e-8)

    current_var = node.last_blended.var(dim=0).mean().item() if node.last_blended is not None else 0.0
    model.div_history.append(current_div)
    model.var_history.append(current_var)
    if len(model.div_history) > model.history_window:
        model.div_history.pop(0)
        model.var_history.pop(0)

    div_delta = max(model.div_history) - min(model.div_history) if model.div_history else 0.0
    entropy_shrunk = entropy_drop > 0.05
    variance_low = current_var < 0.05
    diversity_stalled = div_delta < 1e-3
    should_grow = entropy_shrunk and (variance_low or diversity_stalled)

    # === Log to Dashboard
    dashboard.log_epoch(epoch, acc=acc, loss=loss_avg)
    dashboard.update_node_metrics(
        node_idx=model.node_count - 1,
        acc=acc,
        entropy=entropy_now,
        entropy_init=entropy_start,
        proto_div=proto_div,
        recon=recon,
        usage=usage,
        variance=current_var,
        temp=node.som.temperature.item()
    )
    dashboard.print_dashboard()

    # === Growth Step
    model.anneal_temp_active_node()
    if should_grow and model.node_count < 8:
        model.freeze_node(model.nodes[-1])
        model.grow_node()
        optimizer.add_param_group({'params': model.nodes[-1].parameters()})
        optimizer.add_param_group({'params': model.decoders[-1].parameters()})
        dashboard.new_node(node_index=model.node_count - 1, epoch=epoch, acc_at_start=acc)

# === Save Model
torch.save(model.state_dict(), "models/model_tea.pth")
print("\n✅ Saved to models/model_tea.pth")

# === Evaluation
model.eval()
total_topo_acc = 0
total_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred_topo = logits.argmax(dim=1)
        total_topo_acc += (pred_topo == y).sum().item()
        total_samples += y.size(0)

total_duration = time.time() - start_time
dashboard.final_summary(model, total_topo_acc / total_samples, total_duration)
