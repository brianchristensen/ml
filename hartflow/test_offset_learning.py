"""Test that query offset learns properly."""

import torch
from novel_attention import NovelAttentionLM

device = 'cpu'

print("Testing query offset learning...")
print()

# Create model
model = NovelAttentionLM(
    vocab_size=256,
    dim=32,
    num_layers=2,
    device=device
)

# Test gradients
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 2
seq_len = 10
x = torch.randint(0, 256, (batch_size, seq_len))
target = torch.randint(0, 256, (batch_size, seq_len))

print("Running training steps...")
layer0 = model.blocks[0].integration

for step in range(10):
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256),
        target.reshape(-1)
    )
    loss.backward()

    # Check gradients
    omega_grad = layer0.to_omega.weight.grad.abs().mean().item()
    offset_grad = layer0.to_query_offset.weight.grad.abs().mean().item()
    phase_init_grad = layer0.to_phase_init.weight.grad.abs().mean().item()
    scale_grad = layer0.integration_scale.grad.abs().mean().item()
    scale_value = layer0.integration_scale.abs().mean().item()

    print(f"Step {step}: loss={loss.item():.4f}, omega_grad={omega_grad:.6f}, offset_grad={offset_grad:.6f}, phase_init_grad={phase_init_grad:.6f}, scale_grad={scale_grad:.6f}, scale_val={scale_value:.6f}")

    optimizer.step()

print()
print("Analysis:")
if offset_grad > omega_grad * 0.5:
    print("[OK] Query offset is learning (gradient flowing)")
else:
    print("[WEAK] Query offset has weak gradient signal")

if phase_init_grad > omega_grad * 0.5:
    print("[OK] Phase init is learning (gradient flowing)")
else:
    print("[WEAK] Phase init has weak gradient signal")

# Check offset magnitudes learned
print()
print("Query offset statistics after 10 steps:")
offset_weight = layer0.to_query_offset.weight
print(f"  Mean: {offset_weight.mean().item():.6f}")
print(f"  Std: {offset_weight.std().item():.6f}")
print(f"  Max abs: {offset_weight.abs().max().item():.6f}")

print()
print("Phase init statistics after 10 steps:")
phase_init_weight = layer0.to_phase_init.weight
print(f"  Mean: {phase_init_weight.mean().item():.6f}")
print(f"  Std: {phase_init_weight.std().item():.6f}")
print(f"  Max abs: {phase_init_weight.abs().max().item():.6f}")

print()
print("Integration scale statistics after 10 steps:")
integration_scale = layer0.integration_scale
print(f"  Initial: 0.001000")
print(f"  Mean: {integration_scale.abs().mean().item():.6f}")
print(f"  Min: {integration_scale.abs().min().item():.6f}")
print(f"  Max: {integration_scale.abs().max().item():.6f}")
if integration_scale.abs().mean().item() > 0.0015:
    print(f"  [OK] Scale has grown from initial value")
else:
    print(f"  [STABLE] Scale remains near initial value")
