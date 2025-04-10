import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import StackedTPAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data ===
transform = transforms.ToTensor()
train_data = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# === Model ===
model = StackedTPAE(num_blocks=5, vis_decoder=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
# === Training ===
for epoch in range(1, epochs+1):
    model.train()
    total_loss, total_acc_raw, total_acc_topo, total_rec = 0, 0, 0, 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, recons, vis_recons, z, topo_z = model(x)

        loss_cls = F.cross_entropy(logits, y)
        loss_recon = sum(F.mse_loss(r, x) for r in recons)
        loss_vis = sum(F.mse_loss(v, x) for v in vis_recons if v is not None)
        loss_vis_cls = 0
        for i, v in enumerate(vis_recons):
            if v is not None:
                pred_aux = model.blocks[i].vis_aux_classifier(v)
                loss_vis_cls += F.cross_entropy(pred_aux, y)
        loss = loss_cls + loss_recon + 0.5 * loss_vis + 0.2 * loss_vis_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_topo = logits.argmax(dim=1)
            raw_logits = model.classifier(z)
            pred_raw = raw_logits.argmax(dim=1)
            total_acc_raw += (pred_raw == y).sum().item()
            total_acc_topo += (pred_topo == y).sum().item()
            total_rec += loss_recon.item()
            total_loss += loss.item()
            total_samples += y.size(0)

    # === Eval ===
    model.eval()
    eval_rec_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, recons, _, _, _ = model(x)
            eval_rec_loss += F.mse_loss(recons[-1], x).item()

    print(f"[TPAE] Epoch {epoch:2d} | Raw Acc: {total_acc_raw / total_samples:.4f} | "
          f"Topo Acc: {total_acc_topo / total_samples:.4f} | "
          f"Recon Loss: {total_rec:.2f} | Eval Recon: {eval_rec_loss:.4f}")

torch.save(model.state_dict(), "models/model_tpae.pth")
print("✅ Saved to models/model_tpae.pth")
