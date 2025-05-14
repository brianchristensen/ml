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

        # Also gather similarities corresponding to topk
        sim_gathered = torch.gather(sim, 2, topk_indices)  # (B, T, k)

        return gathered, topk_indices, sim_gathered  # (B, T, k, D), (B, T, k), (B, T, k)

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
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, token, argument):
        inp = torch.cat([token, argument], dim=-1)
        proposed = self.net(inp)      # (B*T, D)
        g = self.gate(inp)            # (B*T, D)
        return proposed, g

class FunctionLibrary(nn.Module):
    def __init__(self, d_model, n_functions=4):
        super().__init__()
        self.fns = nn.ModuleList([ReasoningFunction(d_model) for _ in range(n_functions)])
        self.fn_selector = nn.Linear(d_model, n_functions)

    def forward(self, token, routed_tokens, sim_row):
        fn_weights = torch.softmax(self.fn_selector(token), dim=-1)  # (B*T, n)
        arg_weights = torch.softmax(sim_row, dim=-1)                 # (B*T, k)
        argument = torch.bmm(arg_weights.unsqueeze(1), routed_tokens).squeeze(1)  # (B*T, D)

        proposed_list = []
        gate_list = []

        for fn in self.fns:
            p, g = fn(token, argument)  # each: (B*T, D)
            proposed_list.append(p)
            gate_list.append(g)

        proposed_stack = torch.stack(proposed_list, dim=1)  # (B*T, n, D)
        gate_stack = torch.stack(gate_list, dim=1)          # (B*T, n, D)

        # Weighted sum over functions
        weighted_proposed = torch.sum(fn_weights.unsqueeze(-1) * proposed_stack, dim=1)  # (B*T, D)
        weighted_gate = torch.sum(fn_weights.unsqueeze(-1) * gate_stack, dim=1)          # (B*T, D)

        return weighted_proposed, weighted_gate

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

        proposed, g = self.logic_fn(x_flat, routed_flat, sim_flat)  # each: (B*T, D)
        updated = proposed * g                                      # gated memory update
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
