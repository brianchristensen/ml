# inverted_cognition_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandRouter(nn.Module): # inverted attention
    def __init__(self, d_model, k_query=32, k_topk=4):
        super().__init__()
        self.query_proj = nn.Linear(d_model, k_query)
        self.key_proj = nn.Linear(d_model, k_query)
        self.k_topk = k_topk

    def forward(self, x):
        # x: (B, T, D)
        queries = self.query_proj(x)  # (B, T, K)
        keys = self.key_proj(x)      # (B, T, K)
        sim = torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, T)
        sim = sim / queries.size(-1)**0.5

        topk_values, topk_indices = torch.topk(sim, self.k_topk, dim=-1)

        B, T, D = x.size()
        gathered = torch.zeros_like(x)
        for b in range(B):
            for t in range(T):
                idx = topk_indices[b, t]
                neighbors = x[b, idx]  # (k, D)
                gathered[b, t] = neighbors.mean(dim=0)
                
        return gathered

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

    def forward(self, x):
        B, T, D = x.shape
        graph_output = []

        for b in range(B):
            memory = torch.zeros(D, device=x.device)
            token_nodes = []
            for t in range(T):
                token = x[b, t]
                edge_input = torch.cat([token, memory], dim=-1)
                proposed = self.transform(edge_input)
                g = self.gate(proposed)
                memory = memory * (1 - g) + proposed * g
                token_nodes.append(memory.clone())

            graph_output.append(torch.stack(token_nodes))

        return torch.stack(graph_output)

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
