"""Test that gradients flow to window_decay_logit parameter."""

import torch
from novel_attention import NovelAttentionLM

device = 'cpu'

print("Testing gradient flow to window_decay_logit...")
print()

# Create tiny model
model = NovelAttentionLM(
    vocab_size=100,
    dim=32,
    num_layers=2,
    device=device
)

# Create dummy input
batch_size = 2
seq_len = 10
x = torch.randint(0, 100, (batch_size, seq_len))
target = torch.randint(0, 100, (batch_size, seq_len))

# Forward pass
logits = model(x)

# Compute loss
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    target.reshape(-1)
)

# Backward pass
loss.backward()

# Check gradients on window_decay_logit
print("Checking gradients on window_decay_logit parameters:")
print()
for name, param in model.named_parameters():
    if 'window_decay_logit' in name:
        if param.grad is not None:
            grad_norm = param.grad.abs().item()
            print(f"[OK] {name}")
            print(f"  Value: {param.item():.6f}")
            print(f"  Gradient: {param.grad.item():.6e}")
            print(f"  Gradient magnitude: {grad_norm:.6e}")

            if grad_norm > 1e-10:
                print(f"  GRADIENTS ARE FLOWING!")
            else:
                print(f"  WARNING: Gradient is very small or zero")
        else:
            print(f"[FAIL] {name}: NO GRADIENT!")
        print()

print()
print("Also checking integration_scale for comparison:")
for name, param in model.named_parameters():
    if 'integration_scale' in name:
        if param.grad is not None:
            print(f"[OK] {name}: grad norm = {param.grad.abs().mean().item():.6e}")
        break
