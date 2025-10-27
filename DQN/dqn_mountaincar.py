import os
import csv
import random
import collections

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import RecordVideo

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# ----------------------------
# 0) Paths for CSV and Video
# ----------------------------
CSV_DIR   = r"C:\Users\Kilidjian\OneDrive\Documents\RL_git\RL_Project\src\CSV"
VIDEO_DIR = r"C:\Users\Kilidjian\OneDrive\Documents\RL_git\RL_Project\src\Videos"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ----------------------------
# 1) Q-Network definition
# ----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# 2) Replay buffer
# ----------------------------
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
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

# ------------------------------------------
# 3) Save per-episode returns to CSV
# ------------------------------------------
def save_rewards_to_csv(rewards, seed):
    path = os.path.join(CSV_DIR, f"DQN_mountaincar_rewards_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Episode", "Return"])
        for i, r in enumerate(rewards, start=1):
            w.writerow([i, r])
    print(f"Rewards saved to {path}")

# ----------------------------------------------------
# 4) Training loop: run up to max_episodes episodes
#    stop early if avg 100-episode return ≥ -110
# ----------------------------------------------------
def train_dqn(seed,
              env_name='MountainCar-v0',
              solve_thresh=-110,
              max_episodes=1100):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    memory = ReplayBuffer(capacity=50_000)

    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.01, 0.995
    epsilon = eps_start
    batch_size = 128
    target_update = 5  # sync every 5 episodes

    all_returns     = []
    recent_rewards  = collections.deque(maxlen=100)

    for ep in range(1, max_episodes + 1):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_r = 0.0
        done    = False

        while not done:
            # linear ε decay
            epsilon = max(eps_end, epsilon * eps_decay)

            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                st_v = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy_net(st_v).max(1)[1].item()

            out = env.step(action)
            if len(out) == 5:
                next_state, reward, term, trunc, _ = out
                done = term or trunc
            else:
                next_state, reward, done, _ = out

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward

            if len(memory) >= batch_size:
                s, a, r, s2, d = memory.sample(batch_size)
                s_v  = torch.FloatTensor(s).to(device)
                a_v  = torch.LongTensor(a).unsqueeze(1).to(device)
                r_v  = torch.FloatTensor(r).unsqueeze(1).to(device)
                s2_v = torch.FloatTensor(s2).to(device)
                d_v  = torch.FloatTensor(d).unsqueeze(1).to(device)

                q_vals = policy_net(s_v).gather(1, a_v)
                with torch.no_grad():
                    next_q = target_net(s2_v).max(1)[0].unsqueeze(1)
                    q_tgt  = r_v + gamma * next_q * (1 - d_v)
                loss = F.mse_loss(q_vals, q_tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_returns.append(total_r)
        recent_rewards.append(total_r)

        # sync target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # logging
        if len(recent_rewards) == 100:
            avg100 = np.mean(recent_rewards)
            print(f"Seed {seed} | Ep {ep:4d} | Avg100 = {avg100:6.2f} | ε = {epsilon:.3f}")
            if avg100 >= solve_thresh:
                print(f"\n🎉 Seed {seed} solved at episode {ep}! Avg100={avg100:.2f}\n")
                break
        else:
            print(f"Seed {seed} | Ep {ep:4d} | Return = {total_r:6.2f} | ε = {epsilon:.3f}")

    env.close()
    return policy_net, all_returns

# ----------------------------------------------------
# 5) Evaluation + video recording (one episode per seed)
# ----------------------------------------------------
def evaluate_and_record(policy_net, seed,
                        env_name='MountainCar-v0'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_folder = os.path.join(VIDEO_DIR, f"seed{seed}")
    os.makedirs(video_folder, exist_ok=True)

    # Wrap env with RecordVideo
    env = gym.make(env_name)
    env = RecordVideo(env,
                      video_folder=video_folder,
                      episode_trigger=lambda e: True,
                      name_prefix=f"seed{seed}")
    obs = env.reset()
    state = obs[0] if isinstance(obs, tuple) else obs

    total = 0.0
    done  = False
    while not done:
        st_v = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(st_v).max(1)[1].item()

        out = env.step(action)
        if len(out) == 5:
            state, reward, term, trunc, _ = out
            done = term or trunc
        else:
            state, reward, done, _ = out

        total += reward

    env.close()
    print(f"Recorded video for seed {seed} → total reward = {total:.2f}")

# ----------------------------
# 6) Main: multi-seed runner
# ----------------------------
def main():
    seeds = [0, 42, 123]
    for seed in seeds:
        print(f"\n=== Run seed {seed} ===")
        policy_net, returns = train_dqn(seed)
        save_rewards_to_csv(returns, seed)
        evaluate_and_record(policy_net, seed)

if __name__ == "__main__":
    main()