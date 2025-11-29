"""
Transformer Baseline for Comparison with PSI

This implements a standard Transformer architecture for the same tasks:
1. Sensor Fusion (state reconstruction from noisy observations)
2. Lorenz Fusion (state reconstruction from noisy observations)

The goal is to determine whether the "phase space" properties we observed
in PSI are unique to PSI or are simply properties of any trained neural network.

Key questions:
1. Do Transformer hidden states also correlate highly with true state?
2. Does the Transformer show the same reconstruction vs prediction difference?
3. Is PSI actually special, or just "a neural network that works"?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

        # Causal mask will be created in forward
        self.register_buffer('causal_mask', None)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Create causal mask if needed
        if self.causal_mask is None or self.causal_mask.shape[0] != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.causal_mask = mask

        # Self-attention with causal mask
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=self.causal_mask)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerSensorFusionModel(nn.Module):
    """
    Transformer-based sensor fusion model.

    Matches PSI model architecture but uses Transformer blocks instead of PSI blocks.
    """

    def __init__(self, input_dim, state_dim=6, hidden_dim=256, num_layers=6, num_heads=8):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            state: [batch, seq_len, state_dim]
        """
        batch_size, seq_len, _ = x.shape

        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :seq_len, :]

        for block in self.transformer_blocks:
            h = block(h)

        state = self.output_head(h)
        return state


class TransformerLorenzFusionModel(nn.Module):
    """
    Transformer-based Lorenz fusion model.

    For reconstructing Lorenz state from noisy observations.
    """

    def __init__(self, input_dim, state_dim=3, hidden_dim=128, num_layers=6, num_heads=8):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :seq_len, :]

        for block in self.transformer_blocks:
            h = block(h)

        # Output from last timestep only
        h_final = h[:, -1, :]
        state = self.output_head(h_final)
        return state


class TransformerWithHooks(nn.Module):
    """
    Wrapper to extract internal representations from Transformer.

    Captures the hidden states after each Transformer block,
    analogous to extracting phi from PSI.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activations = {}
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on Transformer blocks."""

        def make_hook(layer_idx):
            def hook(module, input, output):
                self.activations[f'layer_{layer_idx}_hidden'] = output.detach()
            return hook

        for i, block in enumerate(self.model.transformer_blocks):
            block.register_forward_hook(make_hook(i))

    def forward(self, x):
        self.activations = {}
        return self.model(x)

    def get_activations(self):
        return self.activations


def train_transformer_sensor_fusion(train_loader, val_loader, input_dim, epochs=50,
                                    hidden_dim=256, num_layers=6, lr=1e-4, device='cuda'):
    """Train Transformer on sensor fusion task."""

    model = TransformerSensorFusionModel(
        input_dim=input_dim,
        state_dim=6,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = nn.functional.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred = model(inputs)
                val_loss += nn.functional.mse_loss(pred, targets).item()
        val_loss /= len(val_loader)

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers
                }
            }, 'transformer_sensor_fusion_model.pt')
            print(f"  → Saved best model")

    return model, best_val_loss


def train_transformer_lorenz_fusion(train_loader, val_loader, input_dim, epochs=50,
                                    hidden_dim=128, num_layers=6, lr=1e-4, device='cuda'):
    """Train Transformer on Lorenz fusion task."""

    model = TransformerLorenzFusionModel(
        input_dim=input_dim,
        state_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = nn.functional.mse_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred = model(inputs)
                val_loss += nn.functional.mse_loss(pred, targets).item()
        val_loss /= len(val_loader)

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, 'transformer_lorenz_fusion_model.pt')
            print(f"  → Saved best model")

    return model, best_val_loss


def extract_transformer_representations(model, dataset, device, n_samples=500):
    """
    Extract hidden state representations from Transformer.

    Analogous to extracting phi from PSI.
    """
    hooked_model = TransformerWithHooks(model)
    hooked_model.eval()

    all_data = {
        'targets': [],
    }

    num_layers = len(model.transformer_blocks)
    for layer_idx in range(num_layers):
        all_data[f'hidden_layer_{layer_idx}'] = []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting representations"):
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)

            _ = hooked_model(inputs)
            activations = hooked_model.get_activations()

            # Handle different target shapes
            if len(targets.shape) == 1:
                all_data['targets'].append(targets.numpy())
            else:
                all_data['targets'].append(targets.numpy())

            for layer_idx in range(num_layers):
                hidden = activations[f'layer_{layer_idx}_hidden'].cpu().numpy()[0]
                # Take last timestep if sequence
                if len(hidden.shape) == 2:
                    hidden = hidden[-1, :]
                all_data[f'hidden_layer_{layer_idx}'].append(hidden)

    for key in all_data:
        all_data[key] = np.array(all_data[key])

    return all_data, num_layers


def compute_state_correlations(hidden_states, targets, state_names):
    """
    Compute correlation between hidden dimensions and state variables.

    Returns dict with max correlation per state variable.
    """
    n_samples, dim = hidden_states.shape

    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)

    n_states = targets.shape[-1] if len(targets.shape) > 1 else 1

    # Flatten if needed
    if len(targets.shape) > 2:
        targets = targets.reshape(-1, targets.shape[-1])
        hidden_states = hidden_states.reshape(-1, dim)

    correlations = {}
    best_dims = {}

    for s_idx in range(min(n_states, len(state_names))):
        s_name = state_names[s_idx]
        corrs = []
        for d in range(dim):
            c = np.corrcoef(hidden_states[:, d], targets[:, s_idx])[0, 1]
            corrs.append(c if not np.isnan(c) else 0)

        best_dim = np.argmax(np.abs(corrs))
        correlations[s_name] = corrs[best_dim]
        best_dims[s_name] = best_dim

    return correlations, best_dims


if __name__ == "__main__":
    print("=" * 80)
    print("TRANSFORMER BASELINE")
    print("=" * 80)
    print()
    print("This module provides Transformer baselines for comparison with PSI.")
    print("Use compare_psi_transformer.py to run the full comparison.")
