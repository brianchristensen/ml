# -- [model.py] --
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
        self.routing_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_mask=None):
        B, T, D = x.size()
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        sim = torch.matmul(queries, keys.transpose(-1, -2)) / queries.size(-1)**0.5
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            sim = sim.masked_fill(mask == 0, -1e9)
        gates = self.routing_gate(x).squeeze(-1)
        sim = sim * gates.unsqueeze(2)
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

# ------------------ Concept Classifier ------------------

class ConceptClassifier(nn.Module):
    def __init__(self, d_model, concept_dims):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, dim) for name, dim in concept_dims.items()
        })

    def forward(self, x):  # (B, T, D)
        return {name: head(x) for name, head in self.heads.items()}

# ------------------ Program Generator ------------------

class ProgramGenerator(nn.Module):
    def __init__(self, d_model, op_vocab_size, max_len=4):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, op_vocab_size)
        self.max_len = max_len

    def forward(self, context):  # (B, D)
        B, D = context.size()
        h0 = context.unsqueeze(0)
        rnn_input = torch.zeros(B, self.max_len, D, device=context.device)
        rnn_out, _ = self.rnn(rnn_input, h0)
        return self.proj(rnn_out)  # (B, L, V)

# ------------------ Neural Operator Library ------------------

class NeuralOpLibrary(nn.Module):
    def __init__(self, d_model, num_ops):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_ops)
        ])

    def forward(self, op_logits, token_feats):  # (B, L, V), (B, T, D)
        B, T, D = token_feats.size()
        L, V = op_logits.shape[1:3]
        token_feats_exp = token_feats.unsqueeze(1).expand(B, L, T, D)  # (B, L, T, D)
        outputs = []
        for op in self.ops:
            outputs.append(op(token_feats_exp))  # (B, L, T, D)
        outputs = torch.stack(outputs, dim=3)  # (B, L, T, V, D)
        weights = torch.softmax(op_logits, dim=-1).unsqueeze(2)  # (B, L, 1, V)
        fused = (outputs * weights.unsqueeze(-1)).sum(dim=3)  # (B, L, T, D)
        result = fused.mean(dim=1)  # (B, T, D)
        return result

# ------------------ Temporal Encoding ------------------

class TemporalEncoding(nn.Module):
    def __init__(self, max_delta=256, d_model=128):
        super().__init__()
        self.max_delta = max_delta
        self.embedding = nn.Embedding(2 * max_delta + 1, d_model)

    def forward(self, delta):
        delta_clamped = torch.clamp(delta + self.max_delta, 0, 2 * self.max_delta)
        return self.embedding(delta_clamped)

# ------------------ System 2: Neural-Symbolic Execution ------------------

class System2Graph(nn.Module):
    def __init__(self, d_model, top_k=4, max_delta=256, concept_dims=None, op_vocab_size=8, prog_len=4):
        super().__init__()
        self.top_k = top_k
        self.concept_classifier = ConceptClassifier(d_model, concept_dims or {"token_type": 6, "polarity": 2})
        self.program_generator = ProgramGenerator(d_model, op_vocab_size, prog_len)
        self.op_library = NeuralOpLibrary(d_model, op_vocab_size)

    def forward(self, x, routed_tokens, sim, topk_indices):  # x: (B, T, D)
        B, T, D = x.shape
        # Step 1: Classify token concepts
        concepts = self.concept_classifier(x)  # dict of (B, T, C)

        # Step 2: Generate soft program
        pooled = x.mean(dim=1)  # use mean pooled context for program generator
        program_logits = self.program_generator(pooled)  # (B, L, V)

        # Step 3: Execute soft program using learned ops
        out = self.op_library(program_logits, x)  # (B, T, D)
        return out

# ------------------ Full Model ------------------

class CognitionModel(nn.Module):
    def __init__(self, hidden_dim, top_k=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = DemandRouter(d_model=hidden_dim, top_k=top_k)
        self.system1 = System1Block(d_model=hidden_dim)
        self.system2 = System2Graph(
            d_model=hidden_dim,
            top_k=top_k,
            concept_dims={"token_type": 6, "polarity": 2},
            op_vocab_size=8,
            prog_len=4
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None, return_routing_trace=False):
        routed_tokens, topk_indices, sim = self.router(x, attention_mask)
        routed_mean = routed_tokens.mean(dim=2)
        x = self.system1(routed_mean)
        x = self.system2(x, routed_tokens, sim, topk_indices)
        pooled = torch.max(x, dim=1)[0]
        return self.output_proj(pooled)
