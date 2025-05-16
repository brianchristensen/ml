import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Encode Temporal Structure of Sequence ------------------

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out + x

# ------------------ Concept Classifier ------------------

class ConceptClassifier(nn.Module):
    def __init__(self, hidden_dim, concept_dims):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) for name, dim in concept_dims.items()
        })

    def forward(self, x):  # (B, T, D)
        return {name: head(x) for name, head in self.heads.items()}

# ------------------ Program Generator ------------------

class ProgramGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, op_vocab_size, max_len=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.context_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, op_vocab_size)
        self.max_len = max_len

    def forward(self, context):  # (B, input_dim)
        B, D = context.size()
        h0 = self.context_to_hidden(context).unsqueeze(0)  # (1, B, hidden_dim)
        rnn_input = torch.zeros(B, self.max_len, D, device=context.device)
        rnn_out, _ = self.rnn(rnn_input, h0)
        return self.proj(rnn_out)  # (B, L, V)

# ------------------ Neural Operator Library ------------------

class NeuralOpLibrary(nn.Module):
    def __init__(self, hidden_dim, num_ops, concept_dim_total=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = hidden_dim + concept_dim_total
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_ops)
        ])

    def forward(self, op_logits, token_feats, concepts=None):  # (B, L, V), (B, T, D)
        if concepts:
            concept_features = torch.cat([v for v in concepts.values()], dim=-1)  # (B, T, C)
            token_feats = torch.cat([token_feats, concept_features], dim=-1)      # (B, T, D + C)
        
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

# ------------------ System 2: Neural-Symbolic Program Compiler ------------------

class Compiler(nn.Module):
    def __init__(self, hidden_dim, concept_dims=None, op_vocab_size=8, prog_len=4):
        super().__init__()
        concept_dim_total = sum(concept_dims.values())
        self.program_generator = ProgramGenerator(
            input_dim=hidden_dim + concept_dim_total,
            hidden_dim=hidden_dim,
            op_vocab_size=op_vocab_size,
            max_len=prog_len
        )
        self.concept_classifier = ConceptClassifier(hidden_dim, concept_dims or {"token_type": 6, "polarity": 2})
        self.op_library = NeuralOpLibrary(hidden_dim, op_vocab_size, concept_dim_total)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        # Step 1: Classify token concepts
        concepts = self.concept_classifier(x)  # dict of (B, T, C)
        concept_summary = torch.cat([
            c.mean(dim=1) for c in concepts.values()
        ], dim=-1)

        # Step 2: Generate soft program
        pooled = x.mean(dim=1)
        context = torch.cat([pooled, concept_summary], dim=-1)
        program_logits = self.program_generator(context)  # (B, L, V)

        # Step 3: Execute soft program using learned ops
        out = self.op_library(program_logits, x, concepts)  # (B, T, D)
        return out, program_logits, concepts, pooled

# ------------------ Full Model ------------------

class CognitionModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc = TemporalEncoder(hidden_dim=hidden_dim)
        self.synth = Compiler(
            hidden_dim=hidden_dim,
            concept_dims={"token_type": 6, "polarity": 2},
            op_vocab_size=8,
            prog_len=4
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            x = x * mask  # zero out padding
   
        x = self.enc(x)
        out, program_logits, concepts, pooled = self.synth(x)

        max_pooled = torch.max(out, dim=1)[0]   # (B, D)
        mean_pooled = out.mean(dim=1)           # (B, D)
        pooled = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 2D)
        z = self.output_proj(pooled)

        # --- Program Reuse Loss (invariance via shared programs) ---
        B, L, V = program_logits.shape
        program_logits_flat = program_logits.view(B, -1)  # (B, L*V)

        # Similarity between inputs (latent space) — cosine sim
        pooled_norm = F.normalize(pooled, dim=-1)  # (B, D)
        sim_matrix = torch.matmul(pooled_norm, pooled_norm.T)  # (B, B)

        # Target: similar inputs should have similar programs
        program_dists = torch.cdist(program_logits_flat, program_logits_flat, p=2)  # (B, B)

        # Encourage small distance for similar inputs
        reuse_loss = (sim_matrix * program_dists).mean()

        # --- Program Contrastive Loss (turn program space into a metric space) ---
        # 1. Normalize pooled embeddings and program logits
        pooled_norm = F.normalize(pooled, dim=-1)                 # (B, D)
        program_norm = F.normalize(program_logits_flat, dim=-1)   # (B, P)

        # 2. Cosine similarity between pooled vectors → similarity proxy
        sim_matrix = torch.matmul(pooled_norm, pooled_norm.T)     # (B, B)

        # 3. Cosine similarity between programs
        prog_sim = torch.matmul(program_norm, program_norm.T)     # (B, B)

        # 4. Build contrastive masks
        sim_threshold = 0.7
        margin = 0.5
        pos_mask = (sim_matrix > sim_threshold).float()           # Similar pairs
        neg_mask = (sim_matrix < sim_threshold).float()           # Dissimilar pairs

        # 5. Contrastive components
        # Pull similar programs together
        pos_loss = (1 - prog_sim) * pos_mask

        # Push dissimilar programs apart (only if too similar)
        neg_loss = F.relu(prog_sim - margin) * neg_mask

        # 6. Combine and normalize
        contrastive_loss = (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + neg_mask.sum() + 1e-6)

        return z, reuse_loss, contrastive_loss
