import torch
import torch.nn as nn
import torch.nn.functional as F

class Temper(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.joint_policy = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.gate_slope = nn.Parameter(torch.tensor(1.0))
        self.gate_offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, message):
        input_cat = torch.cat([x, message], dim=-1)
        proposed = self.joint_policy(input_cat)
        gate = torch.tanh(self.gate_slope * message + self.gate_offset)
        delta = proposed - x
        tempered = x + gate * delta

        stability = -F.mse_loss(tempered, x, reduction='none').mean(dim=-1)
        instability = (tempered - x).pow(2).sum(dim=-1).clamp(max=1e4)

        return tempered, stability, instability


class TemperGraph(nn.Module):
    def __init__(self, input_dim, latent_dim, num_tempers, top_k=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.1)
        )

        self.tempers = nn.ModuleList([
            Temper(latent_dim, latent_dim) for _ in range(num_tempers)
        ])

        self.aux_heads = nn.ModuleList([
            nn.Linear(latent_dim, 10) for _ in range(num_tempers)
        ])

        self.msg_linear = nn.Linear(latent_dim, latent_dim)
        self.msg_update = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.adjacency = nn.Parameter(torch.rand(num_tempers, num_tempers))

        self.final_predictor = nn.Linear(latent_dim * num_tempers, 10)
        self.top_k = top_k

        self.log_history = {
            'mean_gate': [],
            'instability': [],
            'stability': [],
            'proposal_norm': [],
            'gate_var': [],
            'rewrite_magnitude': [],
            'grad_norm': []
        }

    def forward(self, x):
        B = x.size(0)
        z = self.encoder(x)

        proposals = [tmpr.joint_policy(torch.cat([z, z], dim=-1)) for tmpr in self.tempers]
        norm_keys = [F.normalize(self.key_proj(p), dim=-1) for p in proposals]
        norm_queries = [F.normalize(self.query_proj(p), dim=-1) for p in proposals]

        messages = []
        stacked_keys = torch.stack(norm_keys, dim=1)
        stacked_vals = torch.stack(proposals, dim=1)

        for i, q in enumerate(norm_queries):
            q_exp = q.unsqueeze(1)
            attn_scores = (q_exp * stacked_keys).sum(dim=-1)
            modulated_scores = attn_scores * torch.sigmoid(self.adjacency[i]).unsqueeze(0)

            topk_indices = torch.topk(modulated_scores, self.top_k, dim=-1).indices
            mask = torch.zeros_like(modulated_scores).scatter(1, topk_indices, 1.0)
            attn_weights = F.softmax(modulated_scores.masked_fill(mask == 0, -1e9), dim=-1)

            msg = (attn_weights.unsqueeze(-1) * stacked_vals).sum(dim=1)
            updated_msg = self.msg_update(torch.cat([msg, proposals[i]], dim=-1))
            messages.append(updated_msg)

        tempers, instabilities = [], []
        gate_vars, rewrite_mags, proposal_norms = [], [], []

        for i, tmpr in enumerate(self.tempers):
            tr, stab, instab = tmpr(z, messages[i])
            tempers.append(tr)
            instabilities.append(instab)

            if self.training:
                with torch.no_grad():
                    proposal = tmpr.joint_policy(torch.cat([z, messages[i]], dim=-1))
                    proposal_norm = proposal.norm(dim=-1).mean().item()
                    gate_values = torch.tanh(tmpr.gate_slope * messages[i] + tmpr.gate_offset)
                    gate_var = gate_values.var().item()
                    rewrite_magnitude = (tr - z).norm(dim=-1).mean().item()
                    grad_norm = 0.0
                    for p in tmpr.joint_policy.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item()

                    self.log_history['mean_gate'].append(gate_values.mean().item())
                    self.log_history['instability'].append(instab.mean().item())
                    self.log_history['stability'].append(stab.mean().item())
                    self.log_history['proposal_norm'].append(proposal_norm)
                    self.log_history['gate_var'].append(gate_var)
                    self.log_history['rewrite_magnitude'].append(rewrite_magnitude)
                    self.log_history['grad_norm'].append(grad_norm)

                gate_vars.append(gate_var)
                rewrite_mags.append(rewrite_magnitude)
                proposal_norms.append(proposal_norm)

        fused = torch.cat(tempers, dim=-1)
        logits = self.final_predictor(fused)

        instability_tensor = torch.stack(instabilities, dim=1)
        scale_tensor = torch.tensor([
            (rm / (pn + 1e-6)) for rm, pn in zip(rewrite_mags, proposal_norms)
        ], device=instability_tensor.device).view(1, -1)
        weighted_instability = torch.log1p((instability_tensor * scale_tensor).mean())

        return logits, weighted_instability, tempers

    def report_log_summary(self):
        print("📊 Temper Metrics Summary")
        for i in range(len(self.tempers)):
            print(f"Temper {i:02d} | Gate: {self.log_history['mean_gate'][i]:.4f}, Instab: {self.log_history['instability'][i]:.4f}, Stab: {self.log_history['stability'][i]:.4f}, ProposalNorm: {self.log_history['proposal_norm'][i]:.4f}, GateVar: {self.log_history['gate_var'][i]:.4f}, RewriteMag: {self.log_history['rewrite_magnitude'][i]:.4f}, GradNorm: {self.log_history['grad_norm'][i]:.4f}")
        self.log_history = {k: [] for k in self.log_history}
        print("")
