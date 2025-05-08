import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, symbolic_dim, num_nodes, transformer_hidden=128, nhead=4, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.symbolic_proj = nn.Linear(symbolic_dim, transformer_hidden)
        self.gem_proj = nn.Linear(symbolic_dim, transformer_hidden)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(transformer_hidden, nhead),
            num_layers=num_layers
        )
        self.out_proj = nn.Linear(transformer_hidden, num_nodes + 1)  # +1 for HALT

    def forward(self, z, symbolic_embeds, gem_embeds=None, max_ops=4):
        batch_size, num_nodes, sym_dim = symbolic_embeds.size()
        hidden = z.unsqueeze(0)

        # Prepare keys: symbolic + GEM
        proj_sym = self.symbolic_proj(symbolic_embeds).permute(1, 0, 2)  # (num_nodes, batch, hidden)
        key = proj_sym
        if gem_embeds is not None and gem_embeds.size(0) > 0:
            gem_proj = self.gem_proj(gem_embeds)  # [k, hidden]
            gem_proj = gem_proj.unsqueeze(1).expand(-1, batch_size, -1)  # [k, batch, hidden]
            key = torch.cat([key, gem_proj], dim=0)

        program_indices = []

        for step in range(max_ops):
            out = self.transformer(hidden, key).squeeze(0)
            logits = self.out_proj(out)

            if step == 0:
                logits[:, self.num_nodes] = float('-inf')  # block HALT at first step

            probs = torch.softmax(logits, dim=-1)
            selected = torch.multinomial(probs, 1).squeeze(-1)
            program_indices.append(selected)

            if (selected == self.num_nodes).all():
                break

            hidden = out.unsqueeze(0)

        # Pad if fewer than max_ops
        while len(program_indices) < max_ops:
            pad = torch.full((batch_size,), self.num_nodes, device=z.device, dtype=torch.long)  # HALT
            program_indices.append(pad)

        program_indices = torch.stack(program_indices, dim=1)  # (batch, max_ops)
        return program_indices
