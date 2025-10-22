"""
Training script for Pure Holographic Memory on MNIST

No backpropagation - learning via episodic storage only!
Tests: accuracy, speed, continual learning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import HolographicClassifier
import time


def train_holographic(model, dataloader, epochs=1, device='cuda'):
    """
    "Training" = episodic storage in holographic memory.

    No backpropagation!
    No optimizer!
    Just store patterns as we see them.

    Args:
        model: HolographicClassifier
        dataloader: MNIST dataloader
        epochs: Number of passes (1 is often enough!)
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Flatten images
            x = x.view(x.size(0), -1)

            # Forward: stores patterns in holographic memory
            # NO gradient computation!
            with torch.no_grad():
                outputs = model(x, labels=y, learn=True)

            # Compute loss for monitoring only (not for gradients!)
            loss = F.cross_entropy(outputs['logits'], y)

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches

        print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s): "
              f"Loss = {avg_loss:.4f}, "
              f"Memory energy (avg/max) = {outputs['memory_energy_avg']:.1f}/{outputs['memory_energy_max']:.1f}, "
              f"Patterns stored = {outputs['num_stored']}")


def evaluate(model, dataloader, device='cuda', debug=False):
    """
    Evaluate classification accuracy.

    Args:
        model: HolographicClassifier
        dataloader: MNIST dataloader
        device: 'cuda' or 'cpu'
        debug: Print confusion info
    """
    model.eval()
    correct = 0
    total = 0

    if debug:
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)

            # Retrieve from holographic memory (no learning)
            outputs = model(x, labels=None, learn=False)
            pred = outputs['logits'].argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            if debug:
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    accuracy = correct / total

    if debug:
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        print(f"\nPrediction distribution: {np.bincount(all_preds, minlength=10)}")
        print(f"True label distribution: {np.bincount(all_labels, minlength=10)}")

    return accuracy


def create_split_mnist_loaders(batch_size, task_classes):
    """
    Create data loaders for Split-MNIST continual learning.

    Args:
        batch_size: Batch size
        task_classes: List of classes (e.g., [0,1,2,3,4])

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter for task classes
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    # ===== HYPERPARAMETERS =====
    BATCH_SIZE = 128
    EPOCHS = 1          # Often 1 pass is enough for holographic memory!
    MEMORY_DIM = 4096   # High-dimensional holographic space
    # ===========================

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create Split-MNIST tasks
    task1_classes = [0, 1, 2, 3, 4]  # First 5 digits
    task2_classes = [5, 6, 7, 8, 9]  # Last 5 digits

    print("\n" + "="*60)
    print("PURE HOLOGRAPHIC MEMORY - SPLIT-MNIST BENCHMARK")
    print("="*60)
    print(f"Architecture: Random Projection -> Complex HAM")
    print(f"Learning: Episodic storage only (NO backprop!)")
    print(f"Memory dimension: {MEMORY_DIM}")
    print(f"Task 1: Classes {task1_classes}")
    print(f"Task 2: Classes {task2_classes}")
    print("="*60 + "\n")

    # Create model
    model = HolographicClassifier(
        input_dim=784,
        memory_dim=MEMORY_DIM,
        num_classes=10
    )

    print(f"Model buffers (no learnable params!): {sum(p.numel() for p in model.buffers()):,}\n")

    # ========== TASK 1: Train on digits 0-4 ==========
    print("\n" + "="*60)
    print("TASK 1: Training on digits 0-4")
    print("="*60)

    task1_train, task1_test = create_split_mnist_loaders(BATCH_SIZE, task1_classes)
    train_holographic(model, task1_train, epochs=EPOCHS, device=device)

    print("\nEvaluating Task 1...")
    task1_acc = evaluate(model, task1_test, device=device)
    print(f"Task 1 Accuracy (after Task 1 training): {task1_acc:.4f}")

    # ========== TASK 2: Train on digits 5-9 ==========
    print("\n" + "="*60)
    print("TASK 2: Training on digits 5-9")
    print("="*60)

    task2_train, task2_test = create_split_mnist_loaders(BATCH_SIZE, task2_classes)
    train_holographic(model, task2_train, epochs=EPOCHS, device=device)

    print("\nEvaluating Task 2...")
    task2_acc = evaluate(model, task2_test, device=device)
    print(f"Task 2 Accuracy (after Task 2 training): {task2_acc:.4f}")

    # ========== Re-evaluate Task 1 (Catastrophic Forgetting Test) ==========
    print("\n" + "="*60)
    print("CATASTROPHIC FORGETTING TEST")
    print("="*60)

    print("\nRe-evaluating Task 1 (after Task 2 training)...")
    task1_acc_after = evaluate(model, task1_test, device=device, debug=True)
    print(f"Task 1 Accuracy (after Task 2 training): {task1_acc_after:.4f}")

    # ========== Summary ==========
    forgetting = task1_acc - task1_acc_after
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Task 1 Accuracy (initial):  {task1_acc:.4f}")
    print(f"Task 1 Accuracy (final):    {task1_acc_after:.4f}")
    print(f"Task 2 Accuracy (final):    {task2_acc:.4f}")
    print(f"Catastrophic Forgetting:    {forgetting:.4f} ({forgetting*100:.1f}%)")
    print(f"Average Accuracy:           {(task1_acc_after + task2_acc)/2:.4f}")
    print("="*60)

    if forgetting < 0.05:
        print("\n[EXCELLENT] Minimal catastrophic forgetting (<5%)")
    elif forgetting < 0.15:
        print("\n[GOOD] Low catastrophic forgetting (<15%)")
    elif forgetting < 0.30:
        print("\n[WARNING] Moderate catastrophic forgetting (15-30%)")
    else:
        print("\n[FAIL] High catastrophic forgetting (>30%)")

    # Save model
    torch.save(model.state_dict(), 'holographic_mnist.pth')
    print("\nModel saved to holographic_mnist.pth")
