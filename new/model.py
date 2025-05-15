import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Routing ------------------

class DemandRouter(nn.Module):
    def __init__(self, d_model, k_query=32, top_k=4):
        super().__init__()
        self.query_proj = nn.Linear(d_model, k_query)
        self.key_proj = nn.Linear(d_model, k_query)
        self.top_k = top_k

    def forward(self, x):
        B, T, D = x.size()
        queries = self.query_proj(x)  # (B, T, K)
        keys = self.key_proj(x)       # (B, T, K)
        sim = torch.matmul(queries, keys.transpose(-1, -2)) / queries.size(-1)**0.5
        _, topk_indices = torch.topk(sim, self.top_k, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        x_expanded = x.unsqueeze(1).expand(B, T, T, D)
        gathered = torch.gather(x_expanded, 2, topk_indices_exp)
        sim_gathered = torch.gather(sim, 2, topk_indices)
        return gathered, topk_indices, sim_gathered

# ------------------ Fast Feedforward ------------------

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

# ------------------ Temporal Reasoning ------------------

class TemporalEncoding(nn.Module):
    def __init__(self, max_delta=256, d_model=128):
        super().__init__()
        self.max_delta = max_delta
        self.embedding = nn.Embedding(2 * max_delta + 1, d_model)

    def forward(self, delta):
        delta_clamped = torch.clamp(delta + self.max_delta, 0, 2 * self.max_delta)
        return self.embedding(delta_clamped)

# ------------------ Reasoning Functions ------------------

class ReasoningFunction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.logic = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

    def forward(self, token, routed_tokens, sim_row, delta_emb):
        B_T, K, D = routed_tokens.shape
        token_expanded = token.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([token_expanded, routed_tokens, delta_emb], dim=-1)
        fused = self.logic(combined)
        sim_weights = torch.softmax(sim_row, dim=-1).unsqueeze(-1)
        arg_summary = torch.sum(sim_weights * fused, dim=1)
        gate = self.gate(torch.cat([token, arg_summary, arg_summary], dim=-1))
        return arg_summary, gate

class FunctionLibrary(nn.Module):
    def __init__(self, d_model, n_functions=4):
        super().__init__()
        self.fns = nn.ModuleList([ReasoningFunction(d_model) for _ in range(n_functions)])
        self.fn_selector = nn.Linear(d_model, n_functions)

    def forward(self, token, routed_tokens, sim_row, delta_emb):
        fn_weights = torch.softmax(self.fn_selector(token), dim=-1)
        proposed_list, gate_list = [], []
        for fn in self.fns:
            out, gate = fn(token, routed_tokens, sim_row, delta_emb)
            proposed_list.append(out)
            gate_list.append(gate)
        proposed_stack = torch.stack(proposed_list, dim=1)
        gate_stack = torch.stack(gate_list, dim=1)
        proposed = torch.sum(fn_weights.unsqueeze(-1) * proposed_stack, dim=1)
        gate = torch.sum(fn_weights.unsqueeze(-1) * gate_stack, dim=1)
        return proposed, gate

# ------------------ Concept Graph (EMA-VQ) ------------------

class ConceptGraph(nn.Module):
    def __init__(self, d_model, n_concepts=32, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_concepts = n_concepts
        self.d_model = d_model
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_concepts, d_model)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(n_concepts))
        self.register_buffer("embedding_avg", embed.clone())
        self.commitment_cost = 0.25
        self.vq_loss = 0.0
        self.concept_adj = torch.zeros(n_concepts, n_concepts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        distances = (
            x_flat.pow(2).sum(1, keepdim=True)
            - 2 * x_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*T, C)

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_concepts).type(x.dtype)
        quantized = torch.matmul(encodings, self.embedding)

        # EMA update
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            embed_sum = encodings.t() @ x_flat
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.n_concepts * self.eps)
                * n
            )
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        # VQ loss
        self.vq_loss = F.mse_loss(quantized.detach(), x_flat) + \
                       self.commitment_cost * F.mse_loss(quantized, x_flat.detach())

        quantized = x_flat + (quantized - x_flat).detach()
        return quantized.view(B, T, D)

# ------------------ System 2: Logic + Concepts ------------------

class System2Graph(nn.Module):
    def __init__(self, d_model, top_k=4, max_delta=256):
        super().__init__()
        self.top_k = top_k
        self.logic_fn = FunctionLibrary(d_model)
        self.delta_embed = TemporalEncoding(max_delta=max_delta, d_model=d_model)
        self.concept_graph = ConceptGraph(d_model)

    def forward(self, x, routed_tokens, sim, topk_indices):
        B, T, D = x.shape
        K = self.top_k

        # Compute Δt for temporal embedding
        t = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(2)  # (1, T, 1)
        delta = t - topk_indices  # (B, T, k)
        delta_emb = self.delta_embed(delta).view(B * T, K, D)

        x_flat = x.view(B * T, D)
        routed_flat = routed_tokens.view(B * T, K, D)
        sim_flat = sim.view(B * T, K)

        proposed, gate = self.logic_fn(x_flat, routed_flat, sim_flat, delta_emb)
        updated = proposed * gate
        updated = updated.view(B, T, D)

        concepts = self.concept_graph(updated)
        return concepts

# ------------------ Full Model ------------------

class CognitionModel(nn.Module):
    def __init__(self, hidden_dim, top_k=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = DemandRouter(d_model=hidden_dim, top_k=top_k)
        self.system1 = System1Block(d_model=hidden_dim)
        self.system2 = System2Graph(d_model=hidden_dim, top_k=top_k)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, return_routing_trace=False):
        routed_tokens, topk_indices, sim = self.router(x)
        routed_mean = routed_tokens.mean(dim=2)
        x = self.system1(routed_mean)
        x = self.system2(x, routed_tokens, sim, topk_indices)
        pooled = x[:, -1]
        return self.output_proj(pooled)
