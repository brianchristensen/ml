import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
import time
import math
from datetime import datetime
from model import TEA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# === Hyperparameters ===
epochs = 20
num_nodes_default = 4
latent_dim_default = 256
# loss coefficients
recon_loss_λ = 1
proto_div_λ = 4
node_div_λ = 4
usage_λ = 0.7
label_smoothing = 0.1

# === Model ===
model = TEA(num_nodes=num_nodes_default, latent_dim=latent_dim_default).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# === Data ===
transform_train = transforms.ToTensor()
transform_test = transforms.ToTensor()

train_data = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_data = datasets.CIFAR10("data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=128)

# Intensive augments - better generality of model, harder to invert prototypes with decoder
# gpu_train_aug = T.Compose([
#     T.RandomCrop(32, padding=4),
#     T.RandomHorizontalFlip(),
#     T.ColorJitter(0.2, 0.2, 0.2, 0.1),
#     T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
#     T.ToDtype(torch.float32, scale=True)
# ])

# Simpler augments - less generality of model, easier to invert prototypes with decoder
gpu_train_aug = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToDtype(torch.float32, scale=True)
])

# === Training ===
print(f"🧠 Training TEA Model @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
start_time = time.time()
for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    model.train()
    total_loss = total_acc_topo = 0
    total_proto_div = total_node_div = total_usage_penalty = total_recon_loss = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_aug = gpu_train_aug(x)
        logits = model(x_aug)

        # Core losses
        loss_cls = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        proto_div = model.proto_diversity_loss()
        node_div = model.node_diversity_loss()
        usage_penalty = model.usage_penalty()

        recon_losses = [
            F.mse_loss(model.decoders[i](model.nodes[i].last_blended), x)
            for i in range(model.num_nodes)
        ]
        recon_loss = sum(recon_losses) / model.num_nodes

        # Total loss
        loss = (
            loss_cls +
            proto_div_λ * proto_div +
            node_div_λ * node_div +
            usage_λ * usage_penalty +
            recon_loss_λ * recon_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()  # Optional if you enable a scheduler

        # Logging
        with torch.no_grad():
            pred_topo = logits.argmax(dim=1)
            total_acc_topo += (pred_topo == y).sum().item()
            total_loss += loss.item()
            total_proto_div += proto_div.item()
            total_node_div += node_div.item()
            total_usage_penalty += usage_penalty.item()
            total_recon_loss += recon_loss.item()
            total_samples += y.size(0)

    epoch_duration = time.time() - epoch_start_time

    # === Epoch summary ===
    print(f"[TRAIN] Epoch {epoch:2d} | "
          f"Accuracy: {total_acc_topo / total_samples:.4f} | "
          f"Recon: {total_recon_loss:.4f} | "
          f"Proto Similarity: {total_proto_div:.4f} | "
          f"Node Similarity: {total_node_div:.4f} | "
          f"Usage Penalty: {total_usage_penalty:.4f}", end=" | ")
    for i, node in enumerate(model.nodes):
        print(f"Node{i}Tmp: {node.som.temperature.item():.4f}", end=" | ")
    print(f"Duration: {epoch_duration:.2f}s")

torch.save(model.state_dict(), "models/model_tea.pth")
print("✅ Saved to models/model_tea.pth")

# === Evaluation ===
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

total_duraton = time.time() - start_time
print(f"📊 [TEST] Accuracy: {total_topo_acc / total_samples:.4f}")
print(f'📊 Total Duration: {int(total_duraton // 60):.2f}m {int(total_duraton % 60)}s')
