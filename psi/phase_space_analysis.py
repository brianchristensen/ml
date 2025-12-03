"""
Is PSI Learning a Diffeomorphism to True Phase Space?

If PSI truly learns phase space geometry, then:
1. Its hidden states should form an attractor topologically equivalent to the true one
2. Nearby points in hidden space should map to nearby points in true phase space
3. The Lyapunov exponents should match (chaos preserved)
4. Embedding dimension should be correct

Test on Lorenz attractor where we know ground truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# PSI Model
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

    def forward(self, x, return_memory=False):
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

        out = x + mem
        out = out + self.ffn(self.norm2(out))

        if return_memory:
            return out, mem
        return out


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

    def get_hidden_states(self, x):
        """Extract hidden states from all layers."""
        h = self.input_proj(x)
        hidden_states = [h.detach()]
        for block in self.blocks:
            h = block(h)
            hidden_states.append(h.detach())
        return hidden_states


# =============================================================================
# Lorenz System
# =============================================================================

def generate_lorenz(n_steps, dt=0.01, sigma=10.0, rho=28.0, beta=8/3,
                    x0=None, transient=1000):
    """Generate Lorenz attractor trajectory."""
    if x0 is None:
        x0 = np.random.randn(3) * 0.1

    x, y, z = x0
    trajectory = []

    # Skip transient
    for _ in range(transient):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    # Record trajectory
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
# Phase Space Analysis
# =============================================================================

def compute_correlation_dimension(points, r_values=None):
    """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
    distances = pdist(points)

    if r_values is None:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)

    n = len(points)
    C_r = []

    for r in r_values:
        count = np.sum(distances < r)
        C_r.append(2.0 * count / (n * (n - 1)))

    C_r = np.array(C_r)

    # Estimate dimension from log-log slope
    valid = (C_r > 0) & (C_r < 1)
    if np.sum(valid) < 5:
        return np.nan

    log_r = np.log(r_values[valid])
    log_C = np.log(C_r[valid])

    # Linear fit in log-log space
    slope, _ = np.polyfit(log_r, log_C, 1)

    return slope


def compute_lyapunov_exponent(trajectory, dt=0.01, n_neighbors=5):
    """Estimate largest Lyapunov exponent from trajectory."""
    n = len(trajectory)

    # Build distance matrix
    distances = squareform(pdist(trajectory))

    # For each point, find nearest neighbors and track divergence
    divergences = []

    for i in range(n // 2):
        # Find nearest neighbors (excluding temporal neighbors)
        d = distances[i].copy()
        d[max(0, i-10):min(n, i+10)] = np.inf  # Exclude temporal neighbors

        neighbor_idx = np.argpartition(d, n_neighbors)[:n_neighbors]

        # Track divergence
        for j in neighbor_idx:
            if j + 50 < n and i + 50 < n:
                initial_dist = distances[i, j]
                final_dist = distances[i + 50, j + 50]

                if initial_dist > 0 and final_dist > 0:
                    divergences.append(np.log(final_dist / initial_dist) / (50 * dt))

    if len(divergences) > 0:
        return np.mean(divergences)
    return np.nan


def test_distance_preservation(true_points, hidden_points, n_samples=500):
    """Test if PSI preserves pairwise distances (diffeomorphism property)."""
    n = min(len(true_points), len(hidden_points), n_samples)
    idx = np.random.choice(len(true_points), n, replace=False)

    true_sample = true_points[idx]
    hidden_sample = hidden_points[idx]

    # Compute pairwise distances
    true_dist = pdist(true_sample)
    hidden_dist = pdist(hidden_sample)

    # Correlation between distance matrices
    pearson_r, _ = pearsonr(true_dist, hidden_dist)
    spearman_r, _ = spearmanr(true_dist, hidden_dist)

    return pearson_r, spearman_r


def test_neighborhood_preservation(true_points, hidden_points, k=10):
    """Test if k-nearest neighbors are preserved."""
    n = min(len(true_points), len(hidden_points))

    true_dist = squareform(pdist(true_points[:n]))
    hidden_dist = squareform(pdist(hidden_points[:n]))

    preserved = 0
    total = 0

    for i in range(min(500, n)):
        true_neighbors = set(np.argsort(true_dist[i])[1:k+1])
        hidden_neighbors = set(np.argsort(hidden_dist[i])[1:k+1])

        overlap = len(true_neighbors & hidden_neighbors)
        preserved += overlap
        total += k

    return preserved / total


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE SPACE DIFFEOMORPHISM ANALYSIS")
    print("=" * 70)
    print("Question: Does PSI learn the true phase space geometry of Lorenz?")
    print()

    # Generate Lorenz trajectory
    print("Generating Lorenz attractor...")
    trajectory = generate_lorenz(5000, dt=0.01)

    # Normalize for training
    traj_mean = trajectory.mean(axis=0)
    traj_std = trajectory.std(axis=0)
    traj_normalized = (trajectory - traj_mean) / traj_std

    # Create training data (next-step prediction)
    X = torch.tensor(traj_normalized[:-1], dtype=torch.float32).unsqueeze(0)
    Y = torch.tensor(traj_normalized[1:], dtype=torch.float32).unsqueeze(0)

    # Train PSI model
    print("Training PSI on Lorenz prediction...")
    model = PSIModel(input_dim=3, output_dim=3, dim=64, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    X_train = X.to(device)
    Y_train = Y.to(device)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.6f}")

    # Extract hidden states
    print("\nExtracting hidden representations...")
    model.eval()
    with torch.no_grad():
        hidden_states = model.get_hidden_states(X_train)
        # Use last layer hidden state
        hidden = hidden_states[-1][0].cpu().numpy()  # (seq_len, dim)

    true_states = trajectory[:-1]  # Align with hidden states

    # Reduce hidden dim for analysis
    print("Reducing hidden dimensions with PCA...")
    pca = PCA(n_components=3)
    hidden_3d = pca.fit_transform(hidden)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # =================================================================
    # Analysis 1: Correlation Dimension
    # =================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Correlation Dimension")
    print("=" * 70)
    print("Lorenz attractor has fractal dimension ~2.06")
    print("If PSI learns true geometry, its hidden space should match.")
    print()

    # Subsample for efficiency
    idx = np.random.choice(len(true_states), min(1000, len(true_states)), replace=False)

    true_dim = compute_correlation_dimension(true_states[idx])
    hidden_dim = compute_correlation_dimension(hidden_3d[idx])

    print(f"True Lorenz dimension: {true_dim:.2f}")
    print(f"PSI hidden dimension:  {hidden_dim:.2f}")
    print(f"Match: {'YES' if abs(true_dim - hidden_dim) < 0.5 else 'NO'}")

    # =================================================================
    # Analysis 2: Distance Preservation
    # =================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Distance Preservation (Diffeomorphism Test)")
    print("=" * 70)
    print("A diffeomorphism should preserve relative distances.")
    print("High correlation = geometry preserved.")
    print()

    pearson_r, spearman_r = test_distance_preservation(true_states, hidden_3d)

    print(f"Pairwise distance correlation:")
    print(f"  Pearson r:  {pearson_r:.3f}")
    print(f"  Spearman r: {spearman_r:.3f}")
    print(f"Geometry preserved: {'YES' if spearman_r > 0.5 else 'PARTIAL' if spearman_r > 0.2 else 'NO'}")

    # =================================================================
    # Analysis 3: Neighborhood Preservation
    # =================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Neighborhood Preservation")
    print("=" * 70)
    print("Do nearby points in true space remain nearby in hidden space?")
    print()

    for k in [5, 10, 20]:
        overlap = test_neighborhood_preservation(true_states, hidden_3d, k=k)
        random_baseline = k / len(true_states)
        print(f"k={k}: {overlap*100:.1f}% neighbors preserved (random: {random_baseline*100:.2f}%)")

    # =================================================================
    # Analysis 4: Lyapunov Exponent
    # =================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Lyapunov Exponent (Chaos Preservation)")
    print("=" * 70)
    print("Lorenz has positive Lyapunov exponent (~0.9).")
    print("If PSI learns dynamics, it should have similar divergence.")
    print()

    true_lyap = compute_lyapunov_exponent(true_states)
    hidden_lyap = compute_lyapunov_exponent(hidden_3d)

    print(f"True Lorenz Lyapunov:  {true_lyap:.3f}")
    print(f"PSI hidden Lyapunov:   {hidden_lyap:.3f}")

    # =================================================================
    # Visualization
    # =================================================================
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    fig = plt.figure(figsize=(15, 5))

    # True Lorenz attractor
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(true_states[::5, 0], true_states[::5, 1], true_states[::5, 2],
             'b-', alpha=0.5, linewidth=0.5)
    ax1.set_title('True Lorenz Attractor')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # PSI hidden space (PCA)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(hidden_3d[::5, 0], hidden_3d[::5, 1], hidden_3d[::5, 2],
             'r-', alpha=0.5, linewidth=0.5)
    ax2.set_title('PSI Hidden Space (PCA)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')

    # Distance correlation scatter
    ax3 = fig.add_subplot(133)
    idx_sample = np.random.choice(len(true_states), 300, replace=False)
    true_d = pdist(true_states[idx_sample])
    hidden_d = pdist(hidden_3d[idx_sample])
    ax3.scatter(true_d, hidden_d, alpha=0.1, s=1)
    ax3.set_xlabel('True pairwise distance')
    ax3.set_ylabel('Hidden pairwise distance')
    ax3.set_title(f'Distance Preservation (r={spearman_r:.2f})')

    plt.tight_layout()
    plt.savefig('phase_space_analysis.png', dpi=150)
    print("Saved phase_space_analysis.png")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Is PSI Learning a Diffeomorphism?")
    print("=" * 70)

    dim_match = abs(true_dim - hidden_dim) < 0.5
    geom_preserved = spearman_r > 0.3
    neighbors_preserved = test_neighborhood_preservation(true_states, hidden_3d, k=10) > 0.1

    print(f"""
Criteria for diffeomorphism:
  1. Dimension preserved:    {'✓' if dim_match else '✗'} (true={true_dim:.2f}, learned={hidden_dim:.2f})
  2. Distances correlated:   {'✓' if geom_preserved else '✗'} (r={spearman_r:.2f})
  3. Neighborhoods preserved: {'✓' if neighbors_preserved else '✗'}

Conclusion: PSI {'IS' if (dim_match and geom_preserved) else 'is NOT fully'} learning true phase space geometry.
""")

    if geom_preserved:
        print("""
The partial geometry preservation suggests PSI learns a SMOOTHED VERSION
of the phase space - the running average operation acts as a low-pass filter
on the dynamics, preserving large-scale structure but blurring fine details.

This is consistent with PSI being good at:
- Trend following (large scale dynamics)
- Smooth trajectory prediction
- Statistics of the attractor

But NOT good at:
- Fine-grained state reconstruction
- Exact position recall
- Sharp transitions
""")


if __name__ == "__main__":
    main()
