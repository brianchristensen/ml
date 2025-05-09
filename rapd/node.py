import torch.nn as nn

class Node (nn.Module):
    def __init__(self, latent_dim, symbolic_dim, num_heads=4, num_layers=2):
        super(Node, self).__init__()
        self.latent_dim = latent_dim
        self.symbolic_dim = symbolic_dim
        self.num_layers = num_layers

        # Symbolic head → single symbolic token
        self.symbolic_head = nn.Linear(latent_dim, symbolic_dim)

        # Project symbolic to latent query
        self.q_proj = nn.Linear(symbolic_dim, latent_dim)

        # Project latent to key/value
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)

        # Stacked multihead attention
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Final latent update
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def emit_operator(self, z):
        batch_size = z.size(0)

        # Symbolic embedding: [batch, symbolic_dim]
        symbolic = self.symbolic_head(z)

        def op_fn(z_in, symbolic_in):
            # Prepare shapes
            q = self.q_proj(symbolic_in).unsqueeze(1)  # [batch, 1, latent_dim]
            k = self.k_proj(z_in).unsqueeze(1)         # [batch, 1, latent_dim]
            v = self.v_proj(z_in).unsqueeze(1)         # [batch, 1, latent_dim]

            attn_output = q
            for layer in self.attention_layers:
                attn_output, _ = layer(attn_output, k, v)  # [batch, 1, latent_dim]

            # Squeeze sequence dimension
            out = self.out_proj(attn_output.squeeze(1))    # [batch, latent_dim]
            return out

        return op_fn, symbolic
