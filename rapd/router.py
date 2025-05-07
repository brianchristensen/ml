import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, latent_dim, symbolic_dim, num_nodes, transformer_hidden, nhead, num_layers):
        super(Router, self).__init__()
        self.num_nodes = num_nodes
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(transformer_hidden, nhead),
            num_layers
        )
        self.query_proj = nn.Linear(latent_dim, transformer_hidden)
        self.symbolic_proj = nn.Linear(symbolic_dim, transformer_hidden)
        self.decoder_out = nn.Linear(transformer_hidden, num_nodes)

    def forward(self, latent, symbolic_embeds, gem_embeds=None):
        batch_size = latent.size(0)

        # Combine symbolic + GEM if available
        if gem_embeds is not None:
            symbolic = torch.cat([symbolic_embeds, gem_embeds], dim=1)
        else:
            symbolic = symbolic_embeds

        # Shape symbolic as (seq_len, batch, embed)
        symbolic_proj = self.symbolic_proj(symbolic)  # (batch, num_nodes, hidden)
        symbolic_proj = symbolic_proj.permute(1, 0, 2)  # (num_nodes, batch, hidden)

        # Query as (1, batch, embed)
        query = self.query_proj(latent).unsqueeze(0)  # (1, batch, hidden)

        out = self.transformer(query, symbolic_proj).squeeze(0)  # (batch, hidden)
        logits = self.decoder_out(out)  # (batch, num_nodes)
        probs = torch.softmax(logits, dim=-1)

        return probs
