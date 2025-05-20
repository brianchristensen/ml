import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class SSM(nn.Module):
    def __init__(self, hidden_dim, kernel_size=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Parameterized kernel decay dynamics
        self.raw_gamma = nn.Parameter(torch.randn(hidden_dim))  # learnable γ
        self.raw_beta = nn.Parameter(torch.randn(hidden_dim))   # input scale
        self.C = nn.Parameter(torch.randn(hidden_dim))          # output scale

        # Dynamic input-conditioned modulation
        self.dynamic_proj = nn.Linear(hidden_dim, hidden_dim)   # input -> kernel mod
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def generate_kernel(self):
        gamma = F.softplus(self.raw_gamma) + 1e-4  # stabilize: γ ∈ (0, ∞)
        t = torch.arange(self.kernel_size, device=gamma.device).float().view(1, -1)  # (1, K)
        decay = torch.exp(-t * gamma.view(-1, 1))  # (D, K)
        kernel = self.C.view(-1, 1) * decay        # (D, K)
        kernel = kernel / (kernel.norm(dim=-1, keepdim=True) + 1e-6)
        return kernel  # (D, K)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        assert D == self.hidden_dim

        # Project input for dynamic kernel modulation
        dynamic_scale = torch.sigmoid(self.dynamic_proj(x))  # (B, T, D)

        # Project input
        beta = torch.tanh(self.raw_beta)  # values in [-1, 1]
        u = x * beta.view(1, 1, -1)

        # Generate base kernel
        base_kernel = self.generate_kernel()  # (D, K)

        # Dynamic modulation: scale base kernel per timestep
        # Reshape for grouped conv: (D, 1, K)
        k = base_kernel.unsqueeze(1)  # (D, 1, K)

        # Input preparation
        u = u.transpose(1, 2)  # (B, D, T)
        u_padded = F.pad(u, (self.kernel_size - 1, 0))  # (B, D, T + K - 1)

        # Perform grouped convolution
        y = F.conv1d(u_padded, k, groups=D)  # (B, D, T)

        # Apply dynamic scale
        y = y.transpose(1, 2)  # (B, T, D)
        y = y * dynamic_scale  # dynamic modulation

        # Output projection
        return self.output_proj(y + x)

class LambdaSSM:
    def __init__(self, module):
        self.module = module

    def __call__(self, x):
        return self.module(x)

    def compose(self, other):
        return LambdaSSM(lambda x: self(other(x)))

    def __matmul__(self, other):
        return self.compose(other)

class ConceptClassifier(nn.Module):
    def __init__(self, hidden_dim, concept_dims: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) for name, dim in concept_dims.items()
        })

    def forward(self, x):  # (B, T, D)
        return {k: head(x) for k, head in self.heads.items()}

class ConceptComposer(nn.Module):
    def __init__(self, lambda_bank: List[LambdaSSM], concept_dim_total: int):
        super().__init__()
        self.bank = nn.ModuleList([l.module for l in lambda_bank])
        self.router = nn.Linear(concept_dim_total, len(lambda_bank))
        self.norm = nn.LayerNorm(lambda_bank[0].module.hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, concept_tensor):  # x: (B, T, D), concept_tensor: (B, T, C)
        weights = torch.softmax(self.router(concept_tensor), dim=-1)  # (B, T, K)
        outputs = torch.stack([fn(x) for fn in self.bank], dim=-1)  # (B, T, D, K)
        routed = (outputs * weights.unsqueeze(2)).sum(dim=-1)  # (B, T, D)
        x_routed = self.dropout(self.norm(routed))
        return x_routed, weights

class CognitionModel(nn.Module):
    def __init__(self, hidden_dim, concept_dims={
        "question_focus": 5,
        "answer_role": 4,
        "negation_scope": 2
        }, num_primitives=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concept_dims = concept_dims
        self.concept_dim_total = sum(concept_dims.values())

        # Concept bottleneck
        self.concept_classifier = ConceptClassifier(hidden_dim, concept_dims)

        # Primitive rational SSMs
        self.primitives = [LambdaSSM(SSM(hidden_dim)) for _ in range(num_primitives)]

        # Concept-guided composition
        self.composer = ConceptComposer(self.primitives, self.concept_dim_total)

        # Output projection after pooling
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, attention_mask=None):
        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Step 1: Predict symbolic concepts
        concepts = self.concept_classifier(x)
        concept_tensor = torch.cat([concepts[k] for k in self.concept_dims], dim=-1)

        # Step 2: Route through symbolic concept interpreter
        x_routed, routing_weights = self.composer(x, concept_tensor)

        # Step 3: Pooling
        max_pooled = torch.max(x_routed, dim=1)[0]
        mean_pooled = x_routed.mean(dim=1)
        pooled = torch.cat([max_pooled, mean_pooled], dim=-1)

        # Final representation
        z = self.output_proj(pooled)

        # Optional symbolic entropy for interpretability regularization
        symbolic_entropy = sum(
            (-logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(-1).mean()
            for logits in concepts.values()
        )

        return z, x_routed, symbolic_entropy, concepts#
