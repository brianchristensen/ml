"""Debug decoupled key/query phases."""

import torch
from novel_attention import NovelAttentionLM

device = 'cpu'

print("Debugging decoupled key/query phases...")
print()

# Create small model
model = NovelAttentionLM(
    vocab_size=256,
    dim=32,
    num_layers=2,
    device=device
)

# Test input
batch_size = 2
seq_len = 10
x = torch.randint(0, 256, (batch_size, seq_len))

# Hook to capture phases
phi_keys = []
phi_queries = []

def capture_phases(module, input, output):
    # Access the module's forward pass internals
    pass

# Forward pass
print("Running forward pass...")
with torch.no_grad():
    logits = model(x)

print(f"Output shape: {logits.shape}")
print()

# Check if omega_key and omega_query are learning different things
print("Checking omega_key vs omega_query parameters:")
for name, param in model.named_parameters():
    if 'omega_key' in name or 'omega_query' in name:
        print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

print()
print("Analyzing parameter initialization similarity...")

# Get first layer's omega projections
layer0 = model.blocks[0].integration
omega_key_w = layer0.to_omega_key.weight
omega_query_w = layer0.to_omega_query.weight

# Check if they're identical (bad initialization)
diff = (omega_key_w - omega_query_w).abs().mean()
print(f"Mean absolute difference between omega_key and omega_query weights: {diff.item():.6f}")

if diff < 0.001:
    print("WARNING: omega_key and omega_query are nearly identical!")
    print("This defeats the purpose of decoupling.")
else:
    print("OK: omega_key and omega_query are different.")

print()
print("Testing forward pass with gradients...")

# Test with gradients
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.randint(0, 256, (batch_size, seq_len))

for step in range(5):
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256),
        target.reshape(-1)
    )
    loss.backward()

    # Check gradients
    key_grad = layer0.to_omega_key.weight.grad.abs().mean().item()
    query_grad = layer0.to_omega_query.weight.grad.abs().mean().item()

    print(f"Step {step}: loss={loss.item():.4f}, key_grad={key_grad:.6f}, query_grad={query_grad:.6f}")

    optimizer.step()

print()
print("Analysis:")
print("- If query_grad >> key_grad: retrieval is the bottleneck")
print("- If key_grad >> query_grad: storage is the bottleneck")
print("- If both small: model not learning properly")
