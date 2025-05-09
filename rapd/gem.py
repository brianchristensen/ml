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
        # Convert all programs to tuple form for hashability
        program_tuples = [tuple(prog) for prog in programs]

        # Build a lookup: program tuple → (embedding, reward)
        combined_dict = {}

        # First, add existing entries
        for i, prog in enumerate(self.programs):
            prog_tuple = tuple(prog)
            if prog_tuple not in combined_dict:
                combined_dict[prog_tuple] = (self.symbolic_embeds[i].detach().cpu().numpy(), self.rewards[i])
            else:
                # Keep max reward
                combined_dict[prog_tuple] = (
                    combined_dict[prog_tuple][0],
                    max(combined_dict[prog_tuple][1], self.rewards[i])
                )

        # Then, merge in the new batch
        for i, prog_tuple in enumerate(program_tuples):
            if prog_tuple not in combined_dict:
                combined_dict[prog_tuple] = (embeddings_np[i], rewards[i])
            else:
                combined_dict[prog_tuple] = (
                    combined_dict[prog_tuple][0],
                    max(combined_dict[prog_tuple][1], rewards[i])
                )

        # Convert back to lists
        all_embeds = []
        all_programs = []
        all_rewards = []
        for prog_tuple, (embed, reward) in combined_dict.items():
            all_embeds.append(embed)
            all_programs.append(list(prog_tuple))
            all_rewards.append(reward)

        # Limit to top max_gem by reward
        sorted_indices = np.argsort(all_rewards)[-self.max_gem:][::-1]
        kept_embeds = np.stack([all_embeds[i] for i in sorted_indices], axis=0)
        kept_programs = [all_programs[i] for i in sorted_indices]
        kept_rewards = [all_rewards[i] for i in sorted_indices]

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

        # Filter by reward threshold
        results = []
        for i in idxs:
            results.append({
                'symbolic': self.symbolic_embeds[i],
                'program': self.programs[i],
                'reward': self.rewards[i]
            })

        # Optional: sort final k by reward descending
        results.sort(key=lambda r: r['reward'], reverse=True)

        return results

    def mutate_top_programs(self, k=10, num_mutations=5, num_nodes=20, max_ops=10):
        top_programs = self.get_top_k(k)
        mutated = []

        for entry in top_programs:
            prog = entry['program']
            for _ in range(num_mutations):
                # Simple random mutation: swap, replace, or extend
                mutated_prog = prog.copy()
                if np.random.rand() < 0.33 and len(mutated_prog) > 1:
                    # Swap two ops
                    i, j = np.random.choice(len(mutated_prog), 2, replace=False)
                    mutated_prog[i], mutated_prog[j] = mutated_prog[j], mutated_prog[i]
                elif np.random.rand() < 0.66 and len(mutated_prog) < max_ops:
                    # Extend
                    mutated_prog.append(np.random.randint(0, num_nodes))
                else:
                    # Replace one op
                    idx = np.random.randint(0, len(mutated_prog))
                    mutated_prog[idx] = np.random.randint(0, num_nodes)
                mutated.append(mutated_prog)

        return mutated

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
    
    def print_top_n(self, n=5):
        if len(self.rewards) == 0:
            print("GEM is empty.")
            return

        top_indices = np.argsort(self.rewards)[-n:][::-1]  # top-n highest rewards
        print(f"\nTop {n} GEM - ({len(self.rewards)} total):")
        for rank, i in enumerate(top_indices, start=1):
            reward = self.rewards[i]
            program = self.programs[i]
            print(f"Rank {rank}: Reward = {reward:.4f}, Program = {program}")
        print("")