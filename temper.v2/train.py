# TemperGraphTrainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from model import TemperGraph

# --- Simple Dummy Dataset ---
def create_dummy_dataset(input_dim, num_classes, num_samples=1000):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)

# --- Trainer ---
class TemperGraphTrainer:
    def __init__(self, model, dataset, batch_size=32, lr=1e-3, device='cuda'):
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs=10):
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in self.dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Duration{epoch_duration:.2f}s")
            if epoch % 5 == 0: self.model.print_epoch_summary(epoch, avg_loss)

# --- Example Usage ---
if __name__ == "__main__":
    input_dim = 96
    hidden_dim = 8
    output_dim = 10
    num_tempers = 12

    model = TemperGraph(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_tempers=num_tempers)

    dataset = create_dummy_dataset(input_dim=input_dim, num_classes=output_dim, num_samples=5000)

    trainer = TemperGraphTrainer(model, dataset, batch_size=64, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(num_epochs=20)
