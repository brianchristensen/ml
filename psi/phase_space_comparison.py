"""
Compare: Do PSI and Transformer both learn phase space diffeomorphisms?

If both learn it, it's not special to PSI.
If only PSI learns it, that's PSI's unique strength.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Models
# =============================================================================

class PSIBlock(nn.Module):
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        h = self.norm1(x)
        g = self.gate(h)
        v = self.value(h)
        gv = g * v

        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            gv_padded = F.pad(gv, (0, 0, 0, pad_len))
            g_padded = F.pad(g, (0, 0, 0, pad_len))
        else:
            gv_padded = gv
            g_padded = g

        padded_len = gv_padded.shape[1]
        num_chunks = padded_len // self.chunk_size

        gv_chunked = gv_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        g_chunked = g_padded.view(batch_size, num_chunks, self.chunk_size, dim)
        cumsum_v = torch.cumsum(gv_chunked, dim=2)
        cumsum_g = torch.cumsum(g_chunked, dim=2) + 1e-6
        mem = cumsum_v / cumsum_g
        mem = mem.view(batch_size, padded_len, dim)
        if pad_len > 0:
            mem = mem[:, :seq_len, :]

        x = x + mem
        x = x + self.ffn(self.norm2(x))
        return x


class PSIModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=64, num_layers=4):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([PSIBlock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))

    def get_hidden(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=64, num_layers=4, num_heads=4):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(input_dim, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        seq_len = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    def get_hidden(self, x):
        h = self.input_proj(x)
        seq_len = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.norm(h)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim=64, num_layers=4):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(input_dim, dim)
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.head(self.norm(h))

    def get_hidden(self, x):
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.norm(h)


# =============================================================================
# Lorenz System
# =============================================================================

def generate_lorenz(n_steps, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, transient=1000):
    x, y, z = np.random.randn(3) * 0.1

    for _ in range(transient):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    trajectory = []
    for _ in range(n_steps):
        trajectory.append([x, y, z])
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    return np.array(trajectory)


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_correlation_dimension(points, n_samples=800):
    if len(points) > n_samples:
        idx = np.random.choice(len(points), n_samples, replace=False)
        points = points[idx]

    distances = pdist(points)

    r_min = np.percentile(distances, 1)
    r_max = np.percentile(distances, 50)
    r_values = np.logspace(np.log10(r_min + 1e-10), np.log10(r_max), 15)

    n = len(points)
    C_r = []

    for r in r_values:
        count = np.sum(distances < r)
        C_r.append(2.0 * count / (n * (n - 1) + 1e-10))

    C_r = np.array(C_r)
    valid = (C_r > 1e-10) & (C_r < 1)

    if np.sum(valid) < 3:
        return np.nan

    log_r = np.log(r_values[valid])
    log_C = np.log(C_r[valid])
    slope, _ = np.polyfit(log_r, log_C, 1)

    return slope


def test_distance_preservation(true_points, hidden_points, n_samples=400):
    n = min(len(true_points), len(hidden_points), n_samples)
    idx = np.random.choice(min(len(true_points), len(hidden_points)), n, replace=False)

    true_dist = pdist(true_points[idx])
    hidden_dist = pdist(hidden_points[idx])

    r, _ = spearmanr(true_dist, hidden_dist)
    return r


def test_neighborhood_preservation(true_points, hidden_points, k=10, n_samples=400):
    n = min(len(true_points), len(hidden_points), n_samples)
    idx = np.random.choice(min(len(true_points), len(hidden_points)), n, replace=False)

    true_sample = true_points[idx]
    hidden_sample = hidden_points[idx]

    true_dist = squareform(pdist(true_sample))
    hidden_dist = squareform(pdist(hidden_sample))

    preserved = 0
    for i in range(n):
        true_neighbors = set(np.argsort(true_dist[i])[1:k+1])
        hidden_neighbors = set(np.argsort(hidden_dist[i])[1:k+1])
        preserved += len(true_neighbors & hidden_neighbors)

    return preserved / (n * k)


# =============================================================================
# Main Comparison
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE SPACE DIFFEOMORPHISM: PSI vs Transformer vs LSTM")
    print("=" * 70)
    print()

    # Generate data
    print("Generating Lorenz attractor...")
    trajectory = generate_lorenz(4000)

    traj_mean = trajectory.mean(axis=0)
    traj_std = trajectory.std(axis=0)
    traj_norm = (trajectory - traj_mean) / traj_std

    X = torch.tensor(traj_norm[:-1], dtype=torch.float32).unsqueeze(0).to(device)
    Y = torch.tensor(traj_norm[1:], dtype=torch.float32).unsqueeze(0).to(device)

    true_states = trajectory[:-1]

    # True Lorenz properties
    print("Computing true Lorenz properties...")
    true_dim = compute_correlation_dimension(true_states)
    print(f"True Lorenz correlation dimension: {true_dim:.2f}")
    print()

    # Train and analyze each model
    models = [
        ("PSI", PSIModel),
        ("Transformer", TransformerModel),
        ("LSTM", LSTMModel),
    ]

    results = {}

    for name, ModelClass in models:
        print(f"\n{'='*70}")
        print(f"Training {name}...")
        print("=" * 70)

        model = ModelClass(input_dim=3, output_dim=3, dim=64, num_layers=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()

        print(f"Final loss: {loss.item():.6f}")

        # Extract hidden states
        model.eval()
        with torch.no_grad():
            hidden = model.get_hidden(X)[0].cpu().numpy()

        # PCA to 3D
        pca = PCA(n_components=3)
        hidden_3d = pca.fit_transform(hidden)
        pca_var = pca.explained_variance_ratio_.sum()

        # Compute metrics
        hidden_dim = compute_correlation_dimension(hidden_3d)
        dist_corr = test_distance_preservation(true_states, hidden_3d)
        neighbor_pres = test_neighborhood_preservation(true_states, hidden_3d, k=10)

        results[name] = {
            'dim': hidden_dim,
            'dist_corr': dist_corr,
            'neighbor': neighbor_pres,
            'pca_var': pca_var,
            'loss': loss.item()
        }

        print(f"\n{name} Results:")
        print(f"  Correlation dimension: {hidden_dim:.2f} (true: {true_dim:.2f})")
        print(f"  Distance preservation: r = {dist_corr:.3f}")
        print(f"  Neighborhood preserved: {neighbor_pres*100:.1f}%")
        print(f"  PCA variance explained: {pca_var:.3f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nTrue Lorenz dimension: {true_dim:.2f}")
    print()
    print(f"{'Model':<15} {'Dimension':<12} {'Dist Corr':<12} {'Neighbors':<12} {'Loss':<12}")
    print("-" * 63)

    for name in ["PSI", "Transformer", "LSTM"]:
        r = results[name]
        print(f"{name:<15} {r['dim']:<12.2f} {r['dist_corr']:<12.3f} {r['neighbor']*100:<11.1f}% {r['loss']:<12.6f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_dist = max(results.values(), key=lambda x: x['dist_corr'])
    best_neighbor = max(results.values(), key=lambda x: x['neighbor'])

    psi_dist = results['PSI']['dist_corr']
    trans_dist = results['Transformer']['dist_corr']
    lstm_dist = results['LSTM']['dist_corr']

    print(f"""
Distance Preservation (higher = better diffeomorphism):
  PSI:         {psi_dist:.3f}
  Transformer: {trans_dist:.3f}
  LSTM:        {lstm_dist:.3f}

Interpretation:
""")

    if psi_dist > trans_dist + 0.05 and psi_dist > lstm_dist + 0.05:
        print("  -> PSI learns phase space geometry BETTER than alternatives!")
        print("  -> This is PSI's unique strength: true dynamical system learning.")
    elif abs(psi_dist - trans_dist) < 0.05 and abs(psi_dist - lstm_dist) < 0.05:
        print("  -> All models learn similar geometry.")
        print("  -> Phase space learning is NOT unique to PSI.")
    else:
        winner = max(results.items(), key=lambda x: x[1]['dist_corr'])[0]
        print(f"  -> {winner} learns geometry best.")


if __name__ == "__main__":
    main()
