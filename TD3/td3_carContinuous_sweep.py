# TD3 Hyperparameter Sweep for MountainCarContinuous-v0
# Full script with sweep over actor_lr, critic_lr, tau, policy_noise
# Organized logging and plotting per seed and averaged

import os
import gym
import torch
import torch.nn as nn
import random
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base directory for run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.join("results", f"Run_SIGMA")
CSV_DIR = os.path.join(BASE_DIR, "csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Sweep toggles and values
SWEEP_ACTOR_LR = False
SWEEP_CRITIC_LR = False
SWEEP_TAU = False
SWEEP_POLICY_NOISE = True 

ACTOR_LR_VALUES = [1e-4, 5e-4, 1e-3]
CRITIC_LR_VALUES = [1e-4, 5e-4, 1e-3]
TAU_VALUES = [0.001, 0.005, 0.01]
POLICY_NOISE_VALUES = [0.1, 0.15, 0.2]

# Training constants
SEEDS = [0, 42, 123]
ENV_NAME = "MountainCarContinuous-v0"
MAX_EPISODES = 200
MAX_TIMESTEPS = 999
BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)
GAMMA = 0.99
NOISE_CLIP = 0.3
POLICY_FREQ = 2
SOLVE_SCORE = 90
SOLVE_WINDOW = 20
WARMUP_STEPS = 5000
EARLY_STOP = True

# Environment wrapper
class MomentumRewardWrapper:
    def __init__(self, env_name, momentum_factor=2.0, height_factor=1.0):
        self.env = gym.make(env_name)
        self.momentum_factor = momentum_factor
        self.height_factor = height_factor
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            next_state, env_reward, done, info = out
            terminated, truncated = done, False
        else:
            next_state, env_reward, terminated, truncated, info = out

        position, velocity = next_state
        shaped_reward = env_reward + self.momentum_factor * abs(velocity) + self.height_factor * (position + 1.2) / 1.7
        info['original_reward'] = env_reward
        return next_state, shaped_reward, terminated, truncated, info

    def close(self):
        self.env.close()

# Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim):
        self.state = np.zeros((BUFFER_SIZE, state_dim))
        self.action = np.zeros((BUFFER_SIZE, action_dim))
        self.next_state = np.zeros((BUFFER_SIZE, state_dim))
        self.reward = np.zeros((BUFFER_SIZE, 1))
        self.done = np.zeros((BUFFER_SIZE, 1))
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0
        self.sum, self.sum_sq, self.count = 0.0, 0.0, 0

    def add(self, s, a, s2, r, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s2
        self.reward[self.ptr] = r
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % BUFFER_SIZE
        self.size = min(self.size + 1, BUFFER_SIZE)

        self.count += 1
        self.sum += r
        self.sum_sq += r ** 2
        self.mean = self.sum / self.count
        self.std = np.sqrt(max(self.sum_sq / self.count - self.mean ** 2, 1e-8))

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        r = torch.FloatTensor(self.reward[idx]).to(device)
        r = (r - self.mean) / (self.std + 1e-8)
        return (torch.FloatTensor(self.state[idx]).to(device),
                torch.FloatTensor(self.action[idx]).to(device),
                torch.FloatTensor(self.next_state[idx]).to(device),
                r, torch.FloatTensor(self.done[idx]).to(device))

# Actor & Critic
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        def build():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 1)
            )
        self.q1 = build()
        self.q2 = build()

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa)

# TD3 Agent
class TD3:
    def __init__(self, s_dim, a_dim, max_a, actor_lr, critic_lr, tau, policy_noise):
        self.actor = Actor(s_dim, a_dim, max_a).to(device)
        self.actor_target = Actor(s_dim, a_dim, max_a).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_a = max_a
        self.total_it = 0
        self.tau = tau
        self.policy_noise = policy_noise

    def select_action(self, s, noise=0.1):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        a = self.actor(s).cpu().data.numpy().flatten()
        return np.clip(a + np.random.normal(0, noise), -self.max_a, self.max_a)

    def train(self, buffer):
        self.total_it += 1
        s, a, s2, r, d = buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-NOISE_CLIP, NOISE_CLIP)
            a2 = (self.actor_target(s2) + noise).clamp(-self.max_a, self.max_a)
            q1, q2 = self.critic_target(s2, a2)
            q = r + (1 - d) * GAMMA * torch.min(q1, q2)

        c_q1, c_q2 = self.critic(s, a)
        critic_loss = torch.nn.MSELoss()(c_q1, q) + torch.nn.MSELoss()(c_q2, q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_it % POLICY_FREQ == 0:
            actor_loss = -self.critic.q1_value(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# Train function

def train_td3(seed, actor_lr, critic_lr, tau, policy_noise):
    env = MomentumRewardWrapper(ENV_NAME)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.env.reset(seed=seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_a = float(env.action_space.high[0])
    agent = TD3(s_dim, a_dim, max_a, actor_lr, critic_lr, tau, policy_noise)
    buffer = ReplayBuffer(s_dim, a_dim)

    rewards, avg_rewards = [], []
    episode_rewards = deque(maxlen=SOLVE_WINDOW)
    csv_file = os.path.join(CSV_DIR, f"seed{seed}_actor{actor_lr}_critic{critic_lr}_tau{tau}_pn{policy_noise}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Avg_Reward"])

    total_steps = 0
    for ep in range(1, MAX_EPISODES + 1):
        s = env.reset()
        if isinstance(s, tuple): s = s[0]
        ep_r = 0
        noise_lvl = max(0.5 * (1 - ep / 1200), 0.1)

        for t in range(MAX_TIMESTEPS):
            a = env.action_space.sample() if total_steps < WARMUP_STEPS else agent.select_action(s, noise_lvl)
            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            buffer.add(s, a, s2, r, done)
            s = s2
            ep_r += info['original_reward']
            total_steps += 1
            if buffer.size > BATCH_SIZE:
                agent.train(buffer)
            if done:
                break

        rewards.append(ep_r)
        episode_rewards.append(ep_r)
        avg = np.mean(episode_rewards)
        avg_rewards.append(avg)
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([ep, ep_r, avg])
        print(f"Seed {seed}, Ep {ep}, R {ep_r:.1f}, Avg {avg:.1f}")

    env.close()
    return rewards, avg_rewards

# Plot

def plot_sweep(sweep_rewards, sweep_avg, param_name, param_values):
    for seed in SEEDS:
        plt.figure(figsize=(12, 6))
        for val in param_values:
            plt.plot(sweep_avg[val][seed], label=f"{param_name}={val}")
        plt.title(f"Seed {seed} - {param_name} Sweep")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f"{param_name}_seed{seed}.png"))
        plt.close()

    plt.figure(figsize=(12, 6))
    for val in param_values:
        max_len = max(len(sweep_avg[val][s]) for s in SEEDS)
        mean_curve = np.mean([sweep_avg[val][s][:max_len] for s in SEEDS], axis=0)
        plt.plot(mean_curve, label=f"{param_name}={val}")
    plt.title(f"Mean Avg Reward - {param_name} Sweep")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{param_name}_mean.png"))
    plt.close()

# Main
if __name__ == "__main__":
    if SWEEP_ACTOR_LR:
        values = ACTOR_LR_VALUES
        name = "actor_lr"
    elif SWEEP_CRITIC_LR:
        values = CRITIC_LR_VALUES
        name = "critic_lr"
    elif SWEEP_TAU:
        values = TAU_VALUES
        name = "tau"
    elif SWEEP_POLICY_NOISE:
        values = POLICY_NOISE_VALUES
        name = "policy_noise"
    else:
        raise ValueError("Enable at least one sweep.")

    sweep_rewards = {val: {} for val in values}
    sweep_avg = {val: {} for val in values}

    for val in values:
        for seed in SEEDS:
            actor_lr = val if name == "actor_lr" else 5e-4
            critic_lr = val if name == "critic_lr" else 5e-4
            tau = val if name == "tau" else 0.005
            policy_noise = val if name == "policy_noise" else 0.15
            r, ar = train_td3(seed, actor_lr, critic_lr, tau, policy_noise)
            sweep_rewards[val][seed] = r
            sweep_avg[val][seed] = ar

    plot_sweep(sweep_rewards, sweep_avg, name, values)
