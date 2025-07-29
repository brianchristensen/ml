import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dm_control import suite
from collections import deque
import random

# --- Config ---
HIDDEN = 200
LATENT_DIM = 30
BATCH_SIZE = 32
SEQ_LEN = 50
IMAG_HORIZON = 15
LR = 1e-3
GAMMA = 0.99
LAMBDA = 0.95
MAX_EPISODES = 300
MAX_STEPS = 500
REPLAY_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))

    def sample_seq(self, batch_size, seq_len):
        episodes = []
        while len(episodes) < batch_size:
            idx = random.randint(0, len(self.buffer) - seq_len - 1)
            seq = self.buffer[idx:idx + seq_len + 1]
            if any(x[3] for x in seq[:-1]):  # skip sequences that terminate early
                continue
            episodes.append(seq)
        obs, act, rew, done = zip(*[[list(e[i]) for e in episodes] for i in range(4)])
        return (
            torch.tensor(obs, dtype=torch.float32).to(device),
            torch.tensor(act, dtype=torch.float32).to(device),
            torch.tensor(rew, dtype=torch.float32).unsqueeze(-1).to(device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1).to(device),
        )

# --- Models ---
class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.rnn = nn.GRU(latent_dim + act_dim, latent_dim, batch_first=True)

    def forward(self, obs_seq, act_seq):
        z_seq = self.obs_enc(obs_seq)
        x = torch.cat([z_seq[:, :-1], act_seq[:, :-1]], dim=-1)
        h_seq, _ = self.rnn(x)
        return h_seq  # [B, T, latent_dim]

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)

class Actor(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)

# --- Env setup ---
env = suite.load("reacher", "hard")
action_spec = env.action_spec()
obs_spec = env.observation_spec()
obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
act_dim = action_spec.shape[0]

def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()])

# --- Model Init ---
rssm = RSSM(obs_dim, act_dim, LATENT_DIM, HIDDEN).to(device)
reward_predictor = RewardPredictor(LATENT_DIM, HIDDEN).to(device)
actor = Actor(LATENT_DIM, act_dim, HIDDEN).to(device)
critic = Critic(LATENT_DIM, HIDDEN).to(device)

opt_world = optim.Adam(list(rssm.parameters()) + list(reward_predictor.parameters()), lr=LR)
opt_actor = optim.Adam(actor.parameters(), lr=LR)
opt_critic = optim.Adam(critic.parameters(), lr=LR)

replay = ReplayBuffer(REPLAY_SIZE)

# --- Training ---
def imagine_rollout(z0):
    zs, actions, rewards = [], [], []
    z = z0
    for _ in range(IMAG_HORIZON):
        a = actor(z)
        z, _ = rssm.rnn(torch.cat([z, a], dim=-1).unsqueeze(1), z.unsqueeze(0))
        z = z.squeeze(1)
        r = reward_predictor(z)
        zs.append(z)
        actions.append(a)
        rewards.append(r)
    return torch.stack(zs, dim=1), torch.stack(rewards, dim=1)

for ep in range(MAX_EPISODES):
    ts = env.reset()
    obs = flatten_obs(ts.observation)
    ep_reward = 0

    for t in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            z = rssm.obs_enc(obs_tensor)
            action = actor(z).cpu().numpy().flatten()
        action = np.clip(action, action_spec.minimum, action_spec.maximum)

        ts = env.step(action)
        next_obs = flatten_obs(ts.observation)
        reward = -np.linalg.norm(env.physics.named.data.geom_xpos['finger'] -
                                 env.physics.named.data.geom_xpos['target'])
        done = ts.last()

        replay.add(obs, action, reward, done)
        obs = next_obs
        ep_reward += reward
        if done:
            break

    print(f"Episode {ep} Reward: {ep_reward:.2f}")

    # --- Train world model ---
    if len(replay.buffer) > 1000:
        obs_seq, act_seq, rew_seq, done_seq = replay.sample_seq(BATCH_SIZE, SEQ_LEN)
        z_seq = rssm(obs_seq, act_seq)
        pred_rew = reward_predictor(z_seq)
        loss_world = nn.functional.mse_loss(pred_rew, rew_seq[:, 1:])
        opt_world.zero_grad()
        loss_world.backward()
        opt_world.step()

        # --- Imagination ---
        with torch.no_grad():
            z0 = z_seq[:, -1]
            zs, rewards = imagine_rollout(z0)
            returns = []
            G = torch.zeros_like(rewards[:, 0])
            for r in reversed(rewards.transpose(0, 1)):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns = torch.stack(returns, dim=1)

        # --- Train actor/critic ---
        values = critic(zs.detach())
        critic_loss = nn.functional.mse_loss(values, returns.detach())
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        actor_loss = -critic(zs).mean()
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
