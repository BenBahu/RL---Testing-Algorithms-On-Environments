import os
import csv
import random
import collections
import math

import numpy as np
# Workaround for Gym’s numpy.bool8 check
if not hasattr(np, "bool8"):
    np.bool8 = bool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.wrappers import RecordVideo

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Q-Network definition (128–128 hidden)
class DQNNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.net(x)

# 2) Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.vstack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(dones,   dtype=np.uint8),
        )
    def __len__(self):
        return len(self.buffer)

# 3) Training loop: no fixed cap, only stop when solved
def train_dqn(seed,
              solve_thresh: float = 475.0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    obs = env.reset()
    state = obs[0] if isinstance(obs, tuple) else obs

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = DQNNet(obs_dim, act_dim).to(device)
    target_net = DQNNet(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer    = ReplayBuffer(capacity=100_000)

    batch_size      = 64
    gamma           = 0.99
    learning_starts = 500
    target_update   = 500

    # ε-decay down to ~0
    eps_start = 1.0
    eps_end   = 0.001
    eps_decay = 5_000
    steps_done = 0

    all_returns   = []
    recent_returns = collections.deque(maxlen=100)
    episode = 0

    while True:
        episode += 1
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        ep_ret = 0.0
        done   = False

        while not done:
            # exponential ε decay
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st_v = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(st_v).argmax(dim=1).item()

            out = env.step(action)
            if len(out) == 5:
                nxt, r, term, trunc, _ = out
                done = term or trunc
            else:
                nxt, r, done, _ = out

            buffer.push(state, action, r, nxt, done)
            state = nxt
            ep_ret += r
            steps_done += 1

            if steps_done > learning_starts and len(buffer) >= batch_size:
                s, a, rew, s2, d = buffer.sample(batch_size)
                sv  = torch.FloatTensor(s).to(device)
                av  = torch.LongTensor(a).unsqueeze(1).to(device)
                rv  = torch.FloatTensor(rew).unsqueeze(1).to(device)
                s2v = torch.FloatTensor(s2).to(device)
                dv  = torch.FloatTensor(d).unsqueeze(1).to(device)

                q     = policy_net(sv).gather(1, av)
                with torch.no_grad():
                    nq   = target_net(s2v).max(1)[0].unsqueeze(1)
                    tgt  = rv + gamma * nq * (1 - dv)
                loss = F.mse_loss(q, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_returns.append(ep_ret)
        recent_returns.append(ep_ret)

        # Logging every 50 episodes
        if episode % 50 == 0:
            last50 = np.mean(all_returns[-50:])
            print(f"Seed {seed} | Ep {episode:4d} | last50 avg = {last50:.1f} | ε = {eps:.3f}")

        # Check solve condition
        if len(recent_returns) == 100:
            avg100 = np.mean(recent_returns)
            if avg100 >= solve_thresh:
                print(f"\n🎉 Seed {seed} SOLVED at episode {episode}! avg100={avg100:.1f}\n")
                break

    env.close()
    return policy_net, all_returns

# 4) Save CSV
def save_rewards_to_csv(rewards, seed):
    csv_dir = r"C:\Users\Kilidjian\OneDrive\Documents\RL_git\RL_Project\src\CSV"
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f"DQN_cartpole_rewards_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Episode","Return"])
        for i, r in enumerate(rewards, start=1):
            w.writerow([i, r])
    print(f"Rewards saved to {path}")

# 5) Record video
def record_video(seed, policy_net):
    vid_dir = rf"C:\Users\Kilidjian\OneDrive\Documents\RL_git\RL_Project\src\Videos\seed{seed}"
    os.makedirs(vid_dir, exist_ok=True)
    env = RecordVideo(
        gym.make("CartPole-v1"),
        video_folder=vid_dir,
        episode_trigger=lambda _: True,
        name_prefix=f"seed{seed}"
    )
    obs = env.reset()
    state = obs[0] if isinstance(obs, tuple) else obs
    done = False
    while not done:
        with torch.no_grad():
            st_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(st_v).argmax(1).item()
        out = env.step(action)
        if len(out) == 5:
            state, _, term, trunc, _ = out
            done = term or trunc
        else:
            state, _, done, _ = out
    env.close()
    print(f"Video saved in {vid_dir}")

# 6) Main
def main():
    seeds = [0, 42, 123]
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        policy_net, rets = train_dqn(seed)
        save_rewards_to_csv(rets, seed)
        record_video(seed, policy_net)

if __name__ == "__main__":
    main()