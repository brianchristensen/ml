import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandRouter(nn.Module):
    def __init__(self, d_model, k_query=32, k_topk=4):
        super().__init__()
        self.query_proj = nn.Linear(d_model, k_query)
        self.key_proj = nn.Linear(d_model, k_query)
        self.k_topk = k_topk

    def forward(self, x):
        B, T, D = x.size()
        queries = self.query_proj(x)  # (B, T, K)
        keys = self.key_proj(x)       # (B, T, K)
        sim = torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, T)
        sim = sim / queries.size(-1)**0.5

        _, topk_indices = torch.topk(sim, self.k_topk, dim=-1)  # (B, T, k)
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        x_expanded = x.unsqueeze(1).expand(B, T, T, D)
        gathered = torch.gather(x_expanded, 2, topk_indices_exp)  # (B, T, k, D)
        sim_gathered = torch.gather(sim, 2, topk_indices)  # (B, T, k)

        return gathered, topk_indices, sim_gathered

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

class ReasoningFunction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.logic = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, token, routed_tokens, sim_row):
        # token: (B*T, D)
        # routed_tokens: (B*T, K, D)
        # sim_row: (B*T, K)

        B_T, K, D = routed_tokens.shape

        # Expand token to match K arguments
        token_expanded = token.unsqueeze(1).expand(-1, K, -1)  # (B*T, K, D)
        combined = torch.cat([token_expanded, routed_tokens], dim=-1)  # (B*T, K, 2D)

        fused = self.logic(combined)  # (B*T, K, D)
        sim_weights = torch.softmax(sim_row, dim=-1).unsqueeze(-1)  # (B*T, K, 1)

        arg_summary = torch.sum(sim_weights * fused, dim=1)  # (B*T, D)

        gate = self.gate(torch.cat([token, arg_summary], dim=-1))  # (B*T, D)
        return arg_summary, gate
    
class FunctionLibrary(nn.Module):
    def __init__(self, d_model, n_functions=4):
        super().__init__()
        self.fns = nn.ModuleList([ReasoningFunction(d_model) for _ in range(n_functions)])
        self.fn_selector = nn.Linear(d_model, n_functions)

    def forward(self, token, routed_tokens, sim_row):
        B_T, K, D = routed_tokens.size()
        fn_weights = torch.softmax(self.fn_selector(token), dim=-1)  # (B*T, n)               # (B*T, k)

        proposed_list = []
        gate_list = []

        for fn in self.fns:
            out, gate = fn(token, routed_tokens, sim_row)  # (B*T, D), (B*T, D)
            proposed_list.append(out)
            gate_list.append(gate)

        proposed_stack = torch.stack(proposed_list, dim=1)  # (B*T, n, D)
        gate_stack = torch.stack(gate_list, dim=1)          # (B*T, n, D)

        proposed = torch.sum(fn_weights.unsqueeze(-1) * proposed_stack, dim=1)
        gate = torch.sum(fn_weights.unsqueeze(-1) * gate_stack, dim=1)

        return proposed, gate

class System2Graph(nn.Module):
    def __init__(self, d_model, k_topk=4):
        super().__init__()
        self.k_topk = k_topk
        self.logic_fn = FunctionLibrary(d_model)

    def forward(self, x, routed_tokens, sim):
        B, T, D = x.shape
        K = self.k_topk

        x_flat = x.view(B * T, D)
        routed_flat = routed_tokens.view(B * T, K, D)
        sim_flat = sim.view(B * T, K)

        proposed, gate = self.logic_fn(x_flat, routed_flat, sim_flat)
        updated = proposed * gate
        return updated.view(B, T, D)

class CognitionModel(nn.Module):
    def __init__(self, hidden_dim, k_topk=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = DemandRouter(d_model=hidden_dim, k_topk=k_topk)
        self.system1 = System1Block(d_model=hidden_dim)
        self.system2 = System2Graph(d_model=hidden_dim, k_topk=k_topk)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, return_routing_trace=False):
        routed_tokens, topk_indices, sim = self.router(x)  # (B, T, k, D), (B, T, k), (B, T, k)
        routed_mean = routed_tokens.mean(dim=2)
        x = self.system1(routed_mean)                      # (B, T, D)
        x = self.system2(x, routed_tokens, sim)            # (B, T, D)
        pooled = x[:, -1]                                  # (B, D)
        return self.output_proj(pooled)
