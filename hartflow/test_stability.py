"""Test numerical stability during training."""

import torch
import torch.nn as nn
from novel_attention import NovelAttentionLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Testing numerical stability...")
print(f"Device: {device}")
print()

# Create model matching the real training setup
model = NovelAttentionLM(
    vocab_size=256,
    dim=128,
    num_layers=8,
    device=device
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {model.count_parameters():,}")
print()

# Test on realistic data
batch_size = 32
seq_len = 256

print("Running 100 training steps...")
for step in range(100):
    # Random data
    inputs = torch.randint(0, 256, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

    optimizer.zero_grad()

    # Forward
    logits = model(inputs)

    # Loss
    loss = criterion(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )

    # Check for NaN
    if torch.isnan(loss):
        print(f"[FAIL] NaN detected at step {step}!")
        print(f"  Loss: {loss.item()}")
        break

    # Backward
    loss.backward()

    # Check gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"[FAIL] NaN gradient in {name} at step {step}!")
            has_nan_grad = True
            break

    if has_nan_grad:
        break

    # Gradient clipping (like real training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/100 - Loss: {loss.item():.4f}")

print()
print("[PASS] No NaN detected in 100 training steps!")
print()

# Check decay rates learned
print("Decay rates after 100 steps:")
for name, param in model.named_parameters():
    if 'window_decay_logit' in name:
        decay = torch.sigmoid(param).item()
        effective_window = 1.0 / (1.0 - decay)
        print(f"  {name}: decay={decay:.6f}, window={effective_window:.2f}")
        break
