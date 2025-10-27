import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os
import csv
from collections import deque
import uuid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.15
NOISE_CLIP = 0.3
POLICY_FREQ = 2
ACTOR_LR = 5e-4
CRITIC_LR = 5e-4
BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)
MAX_EPISODES = 1500
MAX_TIMESTEPS = 999
SOLVE_SCORE = 90
SOLVE_WINDOW = 20
SEEDS = [0, 42, 123]
WARMUP_STEPS = 5000
ENV_NAME = "MountainCarContinuous-v0"
EARLY_STOP = True

# ----- Custom Reward Wrapper -----
class MomentumRewardWrapper:
    def __init__(self, env_name="MountainCarContinuous-v0", momentum_factor=2.0, height_factor=1.0):
        self.env = gym.make(env_name)
        self.momentum_factor = momentum_factor
        self.height_factor = height_factor
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            next_state, env_reward, done, info = out
            terminated, truncated = done, False
        else:
            next_state, env_reward, terminated, truncated, info = out

        position, velocity = next_state
        momentum_reward = self.momentum_factor * abs(velocity)
        height_reward = self.height_factor * (position + 1.2) / 1.7
        shaped_reward = env_reward + momentum_reward + height_reward

        info['original_reward'] = env_reward
        info['shaped_reward'] = shaped_reward

        return next_state, shaped_reward, terminated, truncated, info

    def close(self):
        self.env.close()

# ----- TD3 Networks -----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim):
        self.state = np.zeros((BUFFER_SIZE, state_dim))
        self.action = np.zeros((BUFFER_SIZE, action_dim))
        self.next_state = np.zeros((BUFFER_SIZE, state_dim))
        self.reward = np.zeros((BUFFER_SIZE, 1))
        self.done = np.zeros((BUFFER_SIZE, 1))
        self.ptr = 0
        self.size = 0
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_sum = 0.0
        self.reward_sum_sq = 0.0
        self.reward_count = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % BUFFER_SIZE
        self.size = min(self.size + 1, BUFFER_SIZE)

        # Update running reward statistics
        self.reward_count += 1
        self.reward_sum += reward
        self.reward_sum_sq += reward ** 2
        self.reward_mean = self.reward_sum / self.reward_count
        self.reward_std = np.sqrt(max(self.reward_sum_sq / self.reward_count - self.reward_mean ** 2, 1e-8))

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        rewards = torch.FloatTensor(self.reward[ind]).to(device)
        # Normalize rewards
        rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        return (torch.FloatTensor(self.state[ind]).to(device),
                torch.FloatTensor(self.action[ind]).to(device),
                torch.FloatTensor(self.next_state[ind]).to(device),
                rewards,
                torch.FloatTensor(self.done[ind]).to(device))

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train(self, replay_buffer):
        self.total_it += 1
        state, action, next_state, reward, done = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * GAMMA * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if self.total_it % POLICY_FREQ == 0:
            actor_loss = -self.critic.q1_value(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# ---- Training ----
def train_td3(seed):
    env = MomentumRewardWrapper(ENV_NAME)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(state_dim, action_dim)

    # Initialize replay buffer with random actions
    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    for _ in range(2000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.add(state, action, next_state, reward, terminated or truncated)
        state = next_state
        if terminated or truncated:
            state = env.reset()
            if isinstance(state, tuple): state = state[0]

    rewards, avg_rewards = [], []
    best_avg = -float('inf')
    best_model_path = None
    episode_rewards = deque(maxlen=SOLVE_WINDOW)
    total_timesteps = 0

    csv_path = f"td3_mountaincar_seed{seed}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Avg_Reward_Last_20'])

    for ep in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        ep_reward = 0

        noise_level = max(0.5 * (1 - ep / 1200), 0.1)

        for t in range(MAX_TIMESTEPS):
            if total_timesteps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=noise_level)

            next_state, shaped_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, next_state, shaped_reward, done)

            state = next_state
            ep_reward += info['original_reward']
            total_timesteps += 1

            if buffer.size > BATCH_SIZE:
                agent.train(buffer)

            if done:
                break

        episode_rewards.append(ep_reward)
        rewards.append(ep_reward)
        avg_r = np.mean(episode_rewards)
        avg_rewards.append(avg_r)

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, ep_reward, avg_r])

        print(f"Seed {seed}, Episode {ep}, Reward: {ep_reward:.2f}, AvgReward: {avg_r:.2f}")

        if avg_r > best_avg:
            best_avg = avg_r
            best_model_path = f'td3_mountaincar_best_seed{seed}.pth'
            torch.save(agent.actor.state_dict(), best_model_path)

        if len(episode_rewards) == SOLVE_WINDOW and avg_r >= SOLVE_SCORE and EARLY_STOP:
            print(f"🎉 Environment solved in {ep} episodes! 🎉")
            break

    env.close()
    return rewards, avg_rewards, best_model_path

# ---- Plotting ----
def plot_results(all_rewards, all_avg_rewards):
    plt.figure(figsize=(12, 6))
    min_len = min(len(r) for r in all_rewards)
    truncated = np.array([r[:min_len] for r in all_rewards])
    plt.plot(np.mean(truncated, axis=0), color='black', linewidth=2, label='Mean Reward')
    for i, seed in enumerate(SEEDS):
        plt.plot(all_rewards[i][:min_len], alpha=0.3, label=f'Seed {seed}')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TD3 MountainCarContinuous - Rewards")
    plt.legend()
    plt.savefig('td3_mountaincar_rewards.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    truncated_avg = np.array([r[:min_len] for r in all_avg_rewards])
    plt.plot(np.mean(truncated_avg, axis=0), color='black', linewidth=2, label='Mean Avg Reward')
    for i, seed in enumerate(SEEDS):
        plt.plot(all_avg_rewards[i][:min_len], alpha=0.5, label=f'Seed {seed}')
    plt.axhline(SOLVE_SCORE, color='r', linestyle='--', label='Solve Threshold')
    plt.xlabel("Episode")
    plt.ylabel(f"{SOLVE_WINDOW}-Episode Avg Reward")
    plt.title("TD3 MountainCarContinuous - Running Averages")
    plt.legend()
    plt.savefig('td3_mountaincar_avg_rewards.png')
    plt.close()

# ---- Evaluation ----
def evaluate_td3(model_path, video_folder="videos_mountaincar"):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    video_path = os.path.join(video_folder, os.path.basename(model_path).split('.')[0])
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x == 0)

    state = env.reset()
    if isinstance(state, tuple): state = state[0]
    ep_reward = 0
    done = False

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0])).to(device)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    while not done:
        action = actor(torch.FloatTensor(state.reshape(1, -1)).to(device)).cpu().data.numpy().flatten()
        next_state, reward, done, _, _ = env.step(action)
        ep_reward += reward
        state = next_state

    print(f"Evaluation Reward: {ep_reward:.2f}")
    env.close()

# ---- Main ----
if __name__ == "__main__":
    all_rewards, all_avg_rewards, best_models, best_avgs = [], [], [], []

    for seed in SEEDS:
        rewards, avg_rewards, best_model = train_td3(seed)
        all_rewards.append(rewards)
        all_avg_rewards.append(avg_rewards)
        best_models.append(best_model)
        best_avgs.append(np.max(avg_rewards))

    plot_results(all_rewards, all_avg_rewards)

    best_idx = np.argmax(best_avgs)
    print(f"Evaluating best model of Seed {SEEDS[best_idx]}")
    evaluate_td3(best_models[best_idx])