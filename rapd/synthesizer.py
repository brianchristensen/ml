import torch
from collections import Counter

class Synthesizer:
    def __init__(self, nodes, router, gem, device='cuda'):
        self.nodes = nodes
        self.router = router
        self.gem = gem
        self.selection_counter = Counter()
        self.device = device
        
    def forward(self, x, use_gem=True, k=5):
        batch_size = x.size(0)

        # Run all nodes on input batch
        transform_outs = []
        symbolic_embeds = []
        for node in self.nodes:
            transform, symbolic = node(x)  # each: (batch, dim)
            transform_outs.append(transform.unsqueeze(1))  # (batch, 1, dim)
            symbolic_embeds.append(symbolic.unsqueeze(1))  # (batch, 1, embed_dim)

        # Stack across nodes: (batch, num_nodes, dim)
        transform_outs = torch.cat(transform_outs, dim=1)
        symbolic_embeds = torch.cat(symbolic_embeds, dim=1)

        # GEM retrieval (optional)
        gem_embeds = None
        if use_gem:
            retrieved = self.gem.retrieve(x)  # make sure gem.retrieve returns (batch, k, embed_dim)
            if retrieved:
                gem_embeds = retrieved.to(self.device)

        # Router picks per-batch probabilities
        probs = self.router(x, symbolic_embeds, gem_embeds)  # (batch, num_nodes)

        # Sample indices per batch
        selected_idx = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)
        for idx in selected_idx.tolist():
            self.selection_counter[idx] += 1

        # Gather outputs per batch
        batch_indices = torch.arange(batch_size, device=self.device)
        out = transform_outs[batch_indices, selected_idx, :]  # (batch, dim)

        return out, selected_idx, probs
