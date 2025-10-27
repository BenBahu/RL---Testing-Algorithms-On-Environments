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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.1
NOISE_CLIP = 0.2
POLICY_FREQ = 2
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)
MAX_EPISODES = 100
MAX_TIMESTEPS = 200
SOLVE_SCORE = -200
SEEDS = [0, 42, 123]
ENV_NAME = "Pendulum-v1"
EARLY_STOP = False
WARMUP_STEPS = 5000

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
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
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

def train_td3(seed):
    env = gym.make(ENV_NAME)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    rewards = []
    avg_rewards = []
    best_avg = -float('inf')
    best_model_path = None

    episode_rewards = deque(maxlen=10)
    solved = False
    total_timesteps = 0

    # CSV Logging Setup
    csv_filename = f"CSV/TD3_{ENV_NAME}_rewards_seed{seed}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Reward', 'Avg_Reward_Last_10']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        episode_reward = 0

        for t in range(MAX_TIMESTEPS):
            if total_timesteps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=0.1)

            next_state, reward, done, _ = env.step(action)

            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            total_timesteps += 1

            if replay_buffer.size > BATCH_SIZE:
                agent.train(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        print(f"Seed {seed}, Episode {episode}, Reward: {episode_reward:.2f}, AvgReward: {avg_reward:.2f}")

        # CSV Logging
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Episode', 'Reward', 'Avg_Reward_Last_10'])
            writer.writerow({
                'Episode': episode,
                'Reward': episode_reward,
                'Avg_Reward_Last_10': avg_reward
            })

        if avg_reward > best_avg:
            best_avg = avg_reward
            best_model_path = f'td3_pendulum_best_seed{seed}.pth'
            torch.save(agent.actor.state_dict(), best_model_path)
            print(f"New best model saved with avg reward: {best_avg:.2f}")

        if avg_reward >= SOLVE_SCORE and not solved:
            print(f"\n🎉 Environment solved in {episode} episodes! 🎉\n")
            solved = True
            if EARLY_STOP:
                break

    env.close()
    return rewards, avg_rewards, best_model_path

def evaluate_td3(model_path, eval_episodes=5, video_folder="videos_pendulum"):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    video_path = os.path.join(video_folder, os.path.basename(model_path).split('.')[0])
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_path,
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=f"td3_pendulum"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    avg_reward = 0

    for ep in range(eval_episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = actor(state_tensor).cpu().data.numpy().flatten()
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state

        avg_reward += episode_reward
        print(f"Evaluation Episode {ep+1}: Reward = {episode_reward:.2f}")

    avg_reward /= eval_episodes
    print(f"Average Reward over {eval_episodes} evaluation episodes: {avg_reward:.2f}")
    env.close()

def plot_results(all_rewards, all_avg_rewards, seeds):
    plt.figure(figsize=(12, 6))
    for i, seed in enumerate(seeds):
        plt.plot(all_rewards[i], alpha=0.3, label=f'Seed {seed} Rewards')

    min_len = min(len(rew) for rew in all_rewards)
    truncated_rewards = np.array([rew[:min_len] for rew in all_rewards])
    plt.plot(np.mean(truncated_rewards, axis=0), color='black', linewidth=2, label='Mean Reward')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Pendulum - Per Episode Reward')
    plt.legend()
    plt.savefig('td3_pendulum_rewards.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, seed in enumerate(seeds):
        plt.plot(all_avg_rewards[i], alpha=0.5, label=f'Seed {seed} Avg Reward')

    min_len = min(len(avg_rew) for avg_rew in all_avg_rewards)
    truncated_avg_rewards = np.array([avg_rew[:min_len] for avg_rew in all_avg_rewards])
    plt.plot(np.mean(truncated_avg_rewards, axis=0), color='black', linewidth=2, label='Mean Avg Reward')

    plt.axhline(y=SOLVE_SCORE, color='r', linestyle='--', label='Solve Threshold')
    plt.xlabel('Episode')
    plt.ylabel('10-Episode Average Reward')
    plt.title('TD3 Pendulum - Running Average Rewards')
    plt.legend()
    plt.savefig('td3_pendulum_avg_rewards.png')
    plt.show()

if __name__ == "__main__":
    all_rewards, all_avg_rewards = [], []
    best_models = []
    best_avgrewards = []

    for seed in SEEDS:
        rewards, avg_rewards, best_model = train_td3(seed)
        all_rewards.append(rewards)
        all_avg_rewards.append(avg_rewards)
        best_models.append(best_model)
        best_avgrewards.append(np.max(avg_rewards))

    plot_results(all_rewards, all_avg_rewards, SEEDS)

    best_model_idx = np.argmax(best_avgrewards)
    print(f"\n--- Evaluating Best Model of Seed {SEEDS[best_model_idx]} ---")
    print(f"Best Avg Reward: {best_avgrewards[best_model_idx]:.2f}")
    evaluate_td3(best_models[best_model_idx])
