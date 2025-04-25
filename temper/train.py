import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from model import TemperNet

# --------- Synthetic Task ---------
class SymbolicRoutingDataset(Dataset):
    def __init__(self, n_samples=1000, input_dim=3, num_classes=4):
        super().__init__()
        self.data = []
        self.labels = []
        for _ in range(n_samples):
            x = torch.randn(input_dim)
            label = random.randint(0, num_classes - 1)
            y = self.generate_label(x, label)
            self.data.append(x)
            self.labels.append(label)

    def generate_label(self, x, label):
        if label == 0:
            return torch.sum(x).unsqueeze(0)  # route through SUM
        elif label == 1:
            return torch.prod(x).unsqueeze(0)  # PROD
        elif label == 2:
            return torch.mean(x ** 2).unsqueeze(0)  # SQUARE MEAN
        else:
            return torch.norm(x).unsqueeze(0)  # L2 norm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --------- Training Loop ---------
def train():
    epochs = 20
    input_dim = 3
    hidden_dim = 8
    output_dim = 1  # regression target
    num_classes = 4

    model = TemperNet(input_dim, hidden_dim, output_dim, num_tempers=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = SymbolicRoutingDataset(n_samples=2000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs + 1):
        total_loss = 0
        model.reset_epoch()
        for x, label in loader:
            y_target = torch.stack([dataset.generate_label(x[i], label[i]) for i in range(len(x))])

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()

            model.update_tempers_with_local_rewards(y_pred, y_target)
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        for diag in model.get_diagnostics():
            print(f" Temper {diag['id']} | plast: {diag['plasticity']:.3f} | "
                f"nov: {diag['novelty']:.3f} | conf: {diag['conflict']:.3f} | "
                f"used: {diag['usage']} | last op: {diag['last_choice']} | "
                f"rewrites: {diag['rewrites']} | usage: {diag['usage_hist']}")

    model.dump_routing_summary("logs/routing_summary.csv")

if __name__ == "__main__":
    train()
