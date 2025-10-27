# --- IMPORTS ---
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

# --- DEVICE SETUP ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- GLOBAL SETTINGS ---
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 150
MAX_TIMESTEPS = 200
SOLVE_SCORE = -200
SEEDS = [0, 42, 123]
WARMUP_STEPS = 5000
EARLY_STOP = False
BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)

# --- OUTPUT DIRECTORY ---
BASE_DIR = "Run_CRITIC_LR"  # Configurable base directory
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- SWEEP TOGGLES & VALUES ---
sweep_actor_lr = False
sweep_critic_lr = True
sweep_policy_noise = False
sweep_tau = False

# --- 1ST TEST 1 --- 
#actor_lr_values = [1e-4, 1e-3, 3e-3]
#critic_lr_values = [1e-4, 1e-3, 3e-3]
#policy_noise_values = [0.05, 0.1, 0.2]
#tau_values = [0.001, 0.005, 0.01]
# --- 2ND TEST ---
actor_lr_values = [0.0005, 0.001, 0.002]
critic_lr_values = [0.0005, 0.001, 0.002]
tau_values = [0.002, 0.005, 0.008]
policy_noise_values = [0.075, 0.1, 0.15]

# --- NETWORK DEFINITIONS ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim):
        self.state = np.zeros((BUFFER_SIZE, state_dim))
        self.action = np.zeros((BUFFER_SIZE, action_dim))
        self.next_state = np.zeros((BUFFER_SIZE, state_dim))
        self.reward = np.zeros((BUFFER_SIZE, 1))
        self.done = np.zeros((BUFFER_SIZE, 1))
        self.ptr, self.size = 0, 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % BUFFER_SIZE
        self.size = min(self.size + 1, BUFFER_SIZE)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.state[ind]).to(device),
                torch.FloatTensor(self.action[ind]).to(device),
                torch.FloatTensor(self.next_state[ind]).to(device),
                torch.FloatTensor(self.reward[ind]).to(device),
                torch.FloatTensor(self.done[ind]).to(device))

# --- TD3 AGENT ---
class TD3:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, tau, policy_noise):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.total_it = 0
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = 0.2
        self.policy_freq = 2
        self.gamma = 0.99

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
        return action

    def train(self, replay_buffer):
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_value(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- TRAINING FUNCTION ---
def train_td3(seed, actor_lr, critic_lr, policy_noise, tau):
    env = gym.make(ENV_NAME)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action, actor_lr, critic_lr, tau, policy_noise)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    rewards = []
    avg_rewards = []
    episode_rewards = deque(maxlen=10)

    total_timesteps = 0
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        episode_reward = 0

        for t in range(MAX_TIMESTEPS):
            action = env.action_space.sample() if total_timesteps < WARMUP_STEPS else agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            total_timesteps += 1

            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer)

            if done:
                break

        rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(np.mean(episode_rewards))

        print(f"Seed {seed}, Episode {episode}, Reward: {episode_reward:.2f}, AvgReward: {avg_reward:.2f}")

    env.close()
    return rewards, avg_rewards

# --- PLOTTING FUNCTION ---
def plot_sweep_results(sweep_avg_rewards_dict, hyperparam_name, hyperparam_values, seeds):
    for seed in seeds:
        plt.figure(figsize=(12, 6))
        for val in hyperparam_values:
            plt.plot(sweep_avg_rewards_dict[val][seed], label=f"{hyperparam_name}={val}")
        plt.xlabel("Episode")
        plt.ylabel("10-Episode Average Reward")
        plt.title(f"TD3 - Seed {seed} - Varying {hyperparam_name}")
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f"td3_{hyperparam_name}_seed{seed}.png"))
        plt.close()

    plt.figure(figsize=(12, 6))
    for val in hyperparam_values:
        min_len = min(len(sweep_avg_rewards_dict[val][s]) for s in seeds)
        avg_curves = np.array([sweep_avg_rewards_dict[val][s][:min_len] for s in seeds])
        mean_curve = np.mean(avg_curves, axis=0)
        plt.plot(mean_curve, label=f"{hyperparam_name}={val}")
    plt.xlabel("Episode")
    plt.ylabel("10-Episode Average Reward")
    plt.title(f"TD3 - Average Over Seeds - Varying {hyperparam_name}")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"td3_{hyperparam_name}_avg_over_seeds.png"))
    plt.close()

# --- MAIN LOOP ---
if __name__ == "__main__":
    sweep_params = []
    if sweep_actor_lr:
        sweep_params.append(("actor_lr", actor_lr_values))
    if sweep_critic_lr:
        sweep_params.append(("critic_lr", critic_lr_values))
    if sweep_policy_noise:
        sweep_params.append(("policy_noise", policy_noise_values))
    if sweep_tau:
        sweep_params.append(("tau", tau_values))

    if not sweep_params:
        raise ValueError("Enable at least one hyperparameter sweep.")

    for sweep_name, sweep_values in sweep_params:
        avg_rewards_dict = {val: {} for val in sweep_values}
        for val in sweep_values:
            for seed in SEEDS:
                actor_lr = val if sweep_name == "actor_lr" else 1e-3
                critic_lr = val if sweep_name == "critic_lr" else 1e-3
                policy_noise = val if sweep_name == "policy_noise" else 0.1
                tau = val if sweep_name == "tau" else 0.005

                _, avg_rewards = train_td3(seed, actor_lr, critic_lr, policy_noise, tau)
                avg_rewards_dict[val][seed] = avg_rewards

        plot_sweep_results(avg_rewards_dict, sweep_name, sweep_values, SEEDS)