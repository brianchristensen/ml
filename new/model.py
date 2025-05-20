import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Memory(nn.Module):
    def __init__(self, hidden_dim, concept_dim_total, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim + concept_dim_total, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim + concept_dim_total, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=128, padding=127)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, concept_tensor):
        xt = torch.cat([x, concept_tensor], dim=-1)
        gate = torch.sigmoid(self.gate_proj(xt))
        v = self.dropout(gate * self.input_proj(xt))
        v = v.transpose(1, 2)
        y = self.conv(v)[:, :, :x.shape[1]]
        y = y.transpose(1, 2)
        return self.out_proj(x + y)

class ConceptClassifier(nn.Module):
    def __init__(self, hidden_dim, concept_dims, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            ) for name, dim in concept_dims.items()
        })

    def forward(self, x):
        x = self.dropout(x)
        return {name: head(x) for name, head in self.heads.items()}


class CognitionModel(nn.Module):
    def __init__(self, hidden_dim, concept_dims={
        "question_focus": 5,
        "answer_role": 4,
        "negation_scope": 2
    }):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concept_dims = concept_dims
        self.concept_dim_total = sum(concept_dims.values())

        self.concept_classifier = ConceptClassifier(hidden_dim, concept_dims)
        self.mem1 = Memory(hidden_dim, self.concept_dim_total)
        self.mem2 = Memory(hidden_dim, self.concept_dim_total)
        self.mem3 = Memory(hidden_dim, self.concept_dim_total)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        
        # Step 1: Predict token-level symbolic concepts
        concepts = self.concept_classifier(x)  # dict of (B, T, C)
        concept_tensor = torch.cat([concepts[k] for k in self.concept_dims.keys()], dim=-1)  # (B, T, C_total)

        # Concept dropout
        if self.training:
            if torch.rand(1).item() < 0.3:
                drop_key = random.choice(list(concepts.keys()))
                concepts[drop_key] = torch.zeros_like(concepts[drop_key])

        # Step 2: Symbol-guided state-space encoding
        x = self.mem1(x, concept_tensor)  # (B, T, D)
        x = self.mem2(x, concept_tensor)
        x = self.mem3(x, concept_tensor)

        # Step 3: Hybrid Pooling
        max_pooled = torch.max(x, dim=1)[0]
        mean_pooled = x.mean(dim=1)
        pooled = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 2D)
        z = self.output_proj(pooled)  # (B, D)

        symbolic_entropy = sum(
            (-logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(-1).mean()
            for logits in concepts.values()
        )
        
        return z, x, symbolic_entropy, concepts
