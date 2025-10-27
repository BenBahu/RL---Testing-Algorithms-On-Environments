import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import csv
from gymnasium.wrappers import RecordVideo

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Apply correction for Tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

# Critic Network (Q-Value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

# SAC Agent
class SAC:
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.4,
        automatic_entropy_tuning=True,
        hidden_dim=256,
        lr=0.0003,
        buffer_size=1000000,
        batch_size=256
    ):
        self.gamma = discount
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize actor
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Initialize critics
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize target critics
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        if evaluate:
            # Use mean action for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.actor.max_action
        else:
            # Sample action for training
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        
        return action.cpu().data.numpy().flatten()
    
    def update_parameters(self):
        # Check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
        
        with torch.no_grad():
            # Sample next action and compute Q target
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        # Compute current Q values
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        action, log_prob = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter alpha (if automatic)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update of target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

# Modified Training function to incorporate CSV logging and early stopping
def train_sac_with_seed(seed, env_name="Pendulum-v1", max_episodes=1000, max_steps=500, 
                        consecutive_episodes=100, target_avg_reward=-200):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = SAC(state_dim, action_dim, max_action)
    
    rewards = []
    running_avg_rewards = []
    best_avg_reward = -float('inf')
    best_model_path = None
    
    # Create CSV file to store rewards
    csv_filename = f"../src/CSV/SAC_{env_name}_rewards_seed{seed}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Reward', 'Avg_Reward_Last_100']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Track if the task has been solved
    solved = False
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=episode+seed*1000)  # Different seed per episode
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            # Update agent
            agent.update_parameters()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        # Calculate running average over the last 'consecutive_episodes' episodes
        window_size = min(consecutive_episodes, episode + 1)
        avg_reward = np.mean(rewards[-window_size:])
        running_avg_rewards.append(avg_reward)
        
        # Write to CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Episode', 'Reward', 'Avg_Reward_Last_100'])
            writer.writerow({
                'Episode': episode + 1,
                'Reward': episode_reward,
                'Avg_Reward_Last_100': avg_reward
            })
        
        print(f"Seed {seed}, Episode {episode+1}, Reward: {episode_reward:.2f}, "
              f"Avg Reward over last {window_size}: {avg_reward:.2f}")
        
        # Save the best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            
            # Remove previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = f'models/sac_{env_name}_best_seed{seed}.pth'
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'reward': best_avg_reward,
                'episode': episode + 1
            }, best_model_path)
            
            print(f"New best model saved with avg reward: {best_avg_reward:.2f}")
        
        # Check if environment is solved (avg reward exceeds target over consecutive_episodes)
        if window_size == consecutive_episodes and avg_reward >= target_avg_reward and not solved:
            print(f"\n🎉 Environment solved in {episode+1} episodes! 🎉")
            print(f"Average reward of {avg_reward:.2f} achieved over {consecutive_episodes} consecutive episodes.\n")
            solved = True
            
            # Save the model that achieved this milestone
            solved_model_path = f'models/sac_{env_name}_solved_seed{seed}.pth'
            os.makedirs(os.path.dirname(solved_model_path), exist_ok=True)
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'reward': avg_reward,
                'episode': episode + 1
            }, solved_model_path)
            print(f"Solved model saved to {solved_model_path}")
            
            # Early stopping - break out of the training loop
            break
    
    # Save final model if not solved
    if not solved:
        final_model_path = f'models/sac_{env_name}_final_seed{seed}.pth'
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'reward': running_avg_rewards[-1] if running_avg_rewards else -float('inf'),
            'episode': max_episodes
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    env.close()
    return rewards, running_avg_rewards, best_model_path, solved

# Function to evaluate trained agents and record video
def evaluate_sac_with_video(model_path, env_name="Pendulum-v1", eval_episodes=5, video_folder="../src/videos"):
    # Create folders if they don't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Load model - explicitly set weights_only=False to handle the PyTorch 2.6 change
    try:
        checkpoint = torch.load(model_path, weights_only=False)
    except RuntimeError:
        # Fallback for compatibility with older PyTorch versions
        checkpoint = torch.load(model_path)
    
    # Create environment with video recording
    video_path = os.path.join(video_folder, f"{os.path.basename(model_path).split('.')[0]}")
    os.makedirs(video_path, exist_ok=True)
    
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_path,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"SAC_{env_name.split('-')[0]}"
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )
    
    # Load state dictionary
    agent.actor.load_state_dict(checkpoint['actor'])
    
    avg_reward = 0
    
    for ep in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        avg_reward += episode_reward
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    avg_reward /= eval_episodes
    
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.2f}")
    
    env.close()
    print(f"Videos saved to {video_path}")
    return avg_reward, video_path

# New function to run multiple seeds with early stopping
def run_multiple_seeds(seeds=[0, 42, 123], env_name="Pendulum-v1", 
                      max_episodes=1000, max_steps=500, consecutive_episodes=100, target_avg_reward=-200):
    all_rewards = []
    all_avg_rewards = []
    best_models = []
    episodes_to_solve = []
    
    # Train with different seeds
    for seed in seeds:
        print(f"\n--- Starting training with seed {seed} ---\n")
        rewards, avg_rewards, best_model_path, solved = train_sac_with_seed(
            seed, env_name, max_episodes, max_steps, consecutive_episodes, target_avg_reward
        )
        all_rewards.append(rewards)
        all_avg_rewards.append(avg_rewards)
        best_models.append((best_model_path, np.max(avg_rewards)))
        
        if solved:
            episodes_to_solve.append(len(rewards))
        else:
            episodes_to_solve.append(max_episodes)
    
    # Combine all seed data into a single CSV
    combined_csv_filename = f"../src/CSV/SAC_{env_name}_rewards_combined.csv"
    os.makedirs(os.path.dirname(combined_csv_filename), exist_ok=True)
    with open(combined_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Seed', 'Episode', 'Reward', 'Avg_Reward_Last_100']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, seed in enumerate(seeds):
            for episode in range(len(all_rewards[i])):
                window_size = min(consecutive_episodes, episode + 1)
                avg_reward = np.mean(all_rewards[i][max(0, episode - consecutive_episodes + 1):episode + 1])
                
                writer.writerow({
                    'Seed': seed,
                    'Episode': episode + 1,
                    'Reward': all_rewards[i][episode],
                    'Avg_Reward_Last_100': avg_reward
                })
    
    # Plot individual rewards and mean
    plt.figure(figsize=(12, 6))
    
    # Plot individual seed results
    colors = ['blue', 'green', 'red']
    for i, seed in enumerate(seeds):
        plt.plot(all_rewards[i], alpha=0.3, color=colors[i], label=f'Seed {seed}')
    
    # Calculate and plot mean rewards across seeds up to the minimum length
    min_length = min(len(rewards) for rewards in all_rewards)
    truncated_rewards = [rewards[:min_length] for rewards in all_rewards]
    mean_rewards = np.mean(truncated_rewards, axis=0)
    plt.plot(mean_rewards, color='black', linewidth=2, label='Mean Reward')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'SAC Training on {env_name} with Multiple Seeds')
    plt.legend()
    plt.savefig(f'sac_{env_name}_rewards.png')
    
    # Plot average rewards (running averages)
    plt.figure(figsize=(12, 6))
    
    # Plot individual seed averages
    for i, seed in enumerate(seeds):
        plt.plot(all_avg_rewards[i], alpha=0.5, color=colors[i], label=f'Avg Reward Seed {seed} (Last {consecutive_episodes})')
    
    # Calculate and plot mean of average rewards up to the minimum length
    min_avg_length = min(len(rewards) for rewards in all_avg_rewards)
    truncated_avg_rewards = [rewards[:min_avg_length] for rewards in all_avg_rewards]
    mean_avg_rewards = np.mean(truncated_avg_rewards, axis=0)
    plt.plot(mean_avg_rewards, color='black', linewidth=2, label=f'Mean Avg Reward (Last {consecutive_episodes})')
    
    # Add a horizontal line at the target average reward
    plt.axhline(y=target_avg_reward, color='r', linestyle='--', label=f'Target Avg Reward ({target_avg_reward})')
    
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (Last {consecutive_episodes} Episodes)')
    plt.title(f'SAC Training on {env_name} with Multiple Seeds - Running Averages')
    plt.legend()
    plt.savefig(f'sac_{env_name}_avg_rewards.png')
    plt.show()
    
    # Print final statistics
    print("\n--- Final Results ---")
    for i, seed in enumerate(seeds):
        final_avg = all_avg_rewards[i][-1]
        episodes_count = len(all_rewards[i])
        print(f"Seed {seed} - Episodes: {episodes_count}/{max_episodes}, Final Average Reward: {final_avg:.2f}")
    
    # Calculate overall statistics
    solved_count = sum(1 for eps in episodes_to_solve if eps < max_episodes)
    print(f"\nSolved in {solved_count}/{len(seeds)} seeds")
    
    if solved_count > 0:
        solved_episodes = [eps for eps in episodes_to_solve if eps < max_episodes]
        avg_episodes_to_solve = np.mean(solved_episodes)
        print(f"Average episodes to solve: {avg_episodes_to_solve:.1f}")
    
    # Find the best model across all seeds
    best_model = max(best_models, key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} with avg reward: {best_model[1]:.2f}")
    
    # Record video of the best model
    print("\n--- Recording video of best model ---")
    avg_reward, video_path = evaluate_sac_with_video(best_model[0], env_name=env_name)
    print(f"Best model evaluation average reward: {avg_reward:.2f}")
    print(f"Video saved to: {video_path}")
    
    return all_rewards, all_avg_rewards, best_model[0]

# Run training with multiple seeds
if __name__ == "__main__":
    seeds = [0, 42, 123]  # Common seeds for reproducibility
    all_rewards, all_avg_rewards, best_model_path = run_multiple_seeds(
        seeds, 
        env_name="Pendulum-v1", 
        max_episodes=1000,
        max_steps=500,
        consecutive_episodes=100,
        target_avg_reward=-200  # Target reward for Pendulum-v1 (adjust based on environment)
    )