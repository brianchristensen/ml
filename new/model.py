# inverted_cognition_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandRouter(nn.Module):
    def __init__(self, d_model, k_query=32, k_topk=4):
        super().__init__()
        self.query_proj = nn.Linear(d_model, k_query)
        self.key_proj = nn.Linear(d_model, k_query)
        self.k_topk = k_topk

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.size()
        queries = self.query_proj(x)  # (B, T, K)
        keys = self.key_proj(x)      # (B, T, K)
        sim = torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, T)
        sim = sim / queries.size(-1)**0.5

        _, topk_indices = torch.topk(sim, self.k_topk, dim=-1)  # (B, T, k)

        # Gather neighbors using advanced indexing
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, T, k, D)
        x_expanded = x.unsqueeze(1).expand(B, T, T, D)                       # (B, T, T, D)
        gathered = torch.gather(x_expanded, 2, topk_indices_exp)            # (B, T, k, D)

        return gathered.mean(dim=2)  # (B, T, D)

class System1Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ffn(x))

class System2Graph(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        memory = torch.zeros(B, D, device=x.device)
        memories = []

        for t in range(T):  # Still per-time but vectorized over batch
            token = x[:, t]                          # (B, D)
            edge_input = torch.cat([token, memory], dim=-1)  # (B, 2D)
            proposed = self.transform(edge_input)            # (B, D)
            g = self.gate(proposed)                          # (B, D)
            memory = memory * (1 - g) + proposed * g         # (B, D)
            memories.append(memory.unsqueeze(1))             # (B, 1, D)

        return torch.cat(memories, dim=1)  # (B, T, D)

class InvertedCognitionModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = DemandRouter(d_model=hidden_dim)
        self.system1 = System1Block(d_model=hidden_dim)
        self.system2 = System2Graph(d_model=hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, return_routing_trace=False):
        """
        x: (B, T, D) — embedded input tokens
        return_routing_trace: if True, enables evaluation diagnostics (unused for now)
        returns: (B, D) — pooled output
        """
        x = self.router(x)         # Inverted attention routing
        x = self.system1(x)        # Fast reflexive processing
        x = self.system2(x)        # Graph-based symbolic planner
        pooled = x[:, -1]          # Take last token (sequence-aware output)
        return self.output_proj(pooled)  # Projected to hidden_dim
