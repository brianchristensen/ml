import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
from model import TemperGraph  # Your model

# === Setup ===

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_dataset_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size

train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = TemperGraph(input_dim=784, hidden_dim=8, num_tempers=4, max_path_hops=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# External task head
task_head = nn.Sequential(
    nn.LayerNorm(8),
    nn.Linear(8, 10)  # FashionMNIST has 10 classes
).to(device)

task_criterion = nn.CrossEntropyLoss()

# === Training Config ===

task_loss_weight = 0.8  # tiny influence vs intrinsic loss
dreaming_enabled = True
dream_after_steps = 500  # if no real data after N steps, start dreaming
dream_noise_scale = 0.1

max_steps = 10000  # total steps (can replace "epochs")

# === Training Loop ===

model.train()
step = 0
last_real_input_step = 0

start_time = time.time()

while step < max_steps:
    for real_inputs, targets in train_loader:
        model.reset_epoch()
        # Normal training step
        real_inputs = real_inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        latent_output, predicted_next_latent, prediction_error, intrinsic_loss = model(real_inputs)

        # External task loss
        logits = task_head(latent_output)
        task_loss = task_criterion(logits, targets)

        # Reward is negative loss
        reward = -task_loss
        routing_loss = model.routing_policy.reinforce(reward, patch_hop_counts=model.latest_patch_hop_counts)

        # Core losses
        prediction_loss = prediction_error.mean()
        total_loss = (
            prediction_loss +
            intrinsic_loss +
            routing_loss +  # this now has gradient from reward
            task_loss_weight * task_loss
        )

        total_loss.backward()
        optimizer.step()

        step += 1
        last_real_input_step = step

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] Loss: {total_loss.item():.6f} | Time: {elapsed:.2f}s")
            model.print_epoch_summary(step, total_loss.item())
            start_time = time.time()

        if step % 1000 == 0:
            model.eval()
            num_batches = 10  # <-- Stream 10 batches (can adjust this)
            all_preds = []
            all_targets = []

            with torch.no_grad():
                val_iter = iter(validation_loader)
                for _ in range(num_batches):
                    try:
                        inputs, targets = next(val_iter)
                    except StopIteration:
                        val_iter = iter(validation_loader)
                        inputs, targets = next(val_iter)

                    inputs, targets = inputs.to(device), targets.to(device)

                    latent_output, _, _, _ = model(inputs)

                    logits = task_head(latent_output)
                    pred_labels = logits.argmax(dim=-1)

                    all_preds.append(pred_labels)
                    all_targets.append(targets)

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            acc = (preds == targets).float().mean().item()

            print(f"✨ Eval Step {step} | Validation Accuracy (avg over {len(preds)} samples): {acc*100:.2f}%\n")
            model.train()

        if step >= max_steps:
            break

    # === DREAMING mode ===
    if dreaming_enabled and (step - last_real_input_step) >= dream_after_steps:
        print("🛌 Dreaming...")
        # Generate random noise input
        dream_input = torch.randn(1, 784, device=device) * dream_noise_scale

        optimizer.zero_grad()

        latent_output, predicted_next_latent, prediction_error, intrinsic_loss = model(dream_input)

        # No external task loss while dreaming!
        prediction_loss = prediction_error.mean()
        routing_loss = model.routing_policy.reinforce(patch_hop_counts=model.latest_patch_hop_counts)

        total_loss = prediction_loss + routing_loss + intrinsic_loss

        total_loss.backward()
        optimizer.step()

        step += 1

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}] (DREAM) Loss: {total_loss.item():.6f} | Time: {elapsed:.2f}s")
            model.print_epoch_summary(step, total_loss.item())
            start_time = time.time()
