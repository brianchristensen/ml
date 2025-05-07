import faiss
import numpy as np
import torch

class GEM:
    def __init__(self, embed_dim, device='cuda'):
        self.embed_dim = embed_dim
        self.device = device

        self.index = faiss.IndexFlatL2(embed_dim)
        self.embeds = []   # List[torch.Tensor]
        self.programs = []  # List[List[int]]
        self.rewards = []   # List[float]

    def insert(self, embedding: torch.Tensor, program: list, reward: float):
        embedding_np = embedding.detach().cpu().numpy()
        self.index.add(embedding_np[np.newaxis, :])
        self.embeds.append(embedding)
        self.programs.append(program)
        self.rewards.append(reward)

    def retrieve(self, query: torch.Tensor, k=5):
        if len(self.embeds) == 0:
            return []

        query_np = query.detach().cpu().numpy()
        _, idxs = self.index.search(query_np[np.newaxis, :], k)
        idxs = idxs[0]

        results = []
        for idx in idxs:
            results.append({
                'embedding': self.embeds[idx],
                'program': self.programs[idx],
                'reward': self.rewards[idx]
            })
        return results

    def prune(self, max_size):
        if len(self.embeds) <= max_size:
            return
        keep_idxs = np.argsort(self.rewards)[-max_size:]
        self.index = faiss.IndexFlatL2(self.embed_dim)
        new_embeds = []
        new_programs = []
        new_rewards = []
        for i in keep_idxs:
            embedding_np = self.embeds[i].detach().cpu().numpy()
            self.index.add(embedding_np[np.newaxis, :])
            new_embeds.append(self.embeds[i])
            new_programs.append(self.programs[i])
            new_rewards.append(self.rewards[i])
        self.embeds = new_embeds
        self.programs = new_programs
        self.rewards = new_rewards

    def get_top_k(self, k=10):
        if len(self.rewards) == 0:
            return []
        top_indices = np.argsort(self.rewards)[-k:][::-1]  # top-k by reward
        results = []
        for i in top_indices:
            results.append({
                'embedding': self.embeds[i],
                'program': self.programs[i],
                'reward': self.rewards[i]
            })
        return results
