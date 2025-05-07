import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import faiss
import numpy as np
import torch

class GEM:
    def __init__(self, symbolic_dim, max_gem=1000, device='cuda'):
        self.symbolic_dim = symbolic_dim
        self.device = device
        self.max_gem = max_gem

        self.index = faiss.IndexFlatL2(symbolic_dim)
        self.symbolic_embeds = []  # [torch.Tensor]
        self.programs = []         # [List[int]]
        self.rewards = []          # [float]

    def insert(self, embedding, program, reward):
        if len(self.symbolic_embeds) >= self.max_gem:
            # Prune lowest reward
            min_idx = np.argmin(self.rewards)
            self.symbolic_embeds.pop(min_idx)
            self.programs.pop(min_idx)
            self.rewards.pop(min_idx)
            self.rebuild_index()

        embed_np = embedding.detach().cpu().numpy()
        self.index.add(embed_np[np.newaxis, :])
        self.symbolic_embeds.append(embedding)
        self.programs.append(program)
        self.rewards.append(reward)

    def insert_batch(self, embeddings_np, programs, rewards):
        # Convert existing to numpy
        existing_embeds_np = np.stack([e.detach().cpu().numpy() for e in self.symbolic_embeds], axis=0) if self.symbolic_embeds else np.empty((0, self.symbolic_dim))
        existing_programs = self.programs
        existing_rewards = self.rewards

        # Combine all
        combined_embeds = np.concatenate([existing_embeds_np, embeddings_np], axis=0)
        combined_programs = existing_programs + programs
        combined_rewards = existing_rewards + rewards

        # Keep top max_gem
        top_indices = np.argsort(combined_rewards)[-self.max_gem:][::-1]
        kept_embeds = combined_embeds[top_indices]
        kept_programs = [combined_programs[i] for i in top_indices]
        kept_rewards = [combined_rewards[i] for i in top_indices]

        # Update internal state
        self.symbolic_embeds = [torch.tensor(e, dtype=torch.float32, device=self.device) for e in kept_embeds]
        self.programs = kept_programs
        self.rewards = kept_rewards

        # Rebuild index once
        self.index = faiss.IndexFlatL2(self.symbolic_dim)
        self.index.add(kept_embeds)

    def rebuild_index(self):
        self.index = faiss.IndexFlatL2(self.symbolic_dim)
        for e in self.symbolic_embeds:
            self.index.add(e.detach().cpu().numpy()[np.newaxis, :])

    def retrieve(self, query, k=5):
        if len(self.symbolic_embeds) == 0:
            return []
        
        _, idxs = self.index.search(query[np.newaxis, :], k)
        idxs = idxs[0]
        results = []
        for i in idxs:
            results.append({
                'symbolic': self.symbolic_embeds[i],
                'program': self.programs[i],
                'reward': self.rewards[i]
            })
        return results

    def get_top_k(self, k=10):
        if len(self.rewards) == 0:
            return []
        top_indices = np.argsort(self.rewards)[-k:][::-1]
        results = []
        for i in top_indices:
            results.append({
                'symbolic': self.symbolic_embeds[i],
                'program': self.programs[i],
                'reward': self.rewards[i]
            })
        return results
