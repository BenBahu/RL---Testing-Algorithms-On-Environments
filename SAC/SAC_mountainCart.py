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

# Reward Shaping Wrapper
class MomentumRewardWrapper:
    def __init__(self, env_name="MountainCarContinuous-v0", momentum_factor=10.0, height_factor=5.0):
        self.env = gym.make(env_name)
        self.momentum_factor = momentum_factor
        self.height_factor = height_factor
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        next_state, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Original reward from environment (preserved for evaluation)
        original_reward = env_reward
        
        # Apply reward shaping - focus on momentum and height
        position, velocity = next_state
        
        # Reward for momentum (absolute velocity)
        momentum_reward = self.momentum_factor * abs(velocity)
        
        # Reward for height (position closer to goal at 0.5)
        height_reward = self.height_factor * (position + 1.2) / 1.7  # Normalize position from [-1.2, 0.5] to [0, 1]
        
        # Combine rewards for training
        shaped_reward = original_reward + momentum_reward + height_reward
        
        # Store original reward in info for logging
        info['original_reward'] = original_reward
        info['shaped_reward'] = shaped_reward
        
        return next_state, shaped_reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

# SAC Agent
class SAC:
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.01,
        alpha=0.5,
        automatic_entropy_tuning=True,
        hidden_dim=256,
        lr=0.0006,
        buffer_size=1000000,
        batch_size=256,
        reward_scale=1.0  # Reduced since we're using shaped rewards
    ):
        self.gamma = discount
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.reward_scale = reward_scale
        
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
        # Apply reward scaling
        reward_batch = torch.FloatTensor(reward_batch * self.reward_scale).to(device).unsqueeze(1)
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

# Create the video directory if it doesn't exist
os.makedirs("videos", exist_ok=True)

# Create a data directory for CSV files
os.makedirs("data", exist_ok=True)

# Modified Training function with shaped rewards and early stopping based on 100-episode average
def train_sac_with_seed(seed, env_name="MountainCarContinuous-v0", max_episodes=1000, max_steps=999,
                        momentum_factor=10.0, height_factor=5.0, consecutive_episodes=100, target_avg_reward=90):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment with reward shaping
    env = MomentumRewardWrapper(env_name, momentum_factor, height_factor)
    state, _ = env.reset(seed=seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create agent with environment-specific parameters
    if env_name == "MountainCarContinuous-v0":
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=0.99,
            tau=0.01,
            alpha=0.5,
            reward_scale=1.0,  # Reduced since we're using shaped rewards
            hidden_dim=256,
            lr=0.0006
        )
    else:
        agent = SAC(state_dim, action_dim, max_action)
    
    # Track rewards
    original_rewards = []  # Track original environment rewards
    shaped_rewards = []    # Track shaped rewards
    running_avg_rewards = []  # Track average over the last 'consecutive_episodes' episodes
    best_avg_reward = -float('inf')
    best_model_path = None
    
    # Create CSV file to store rewards
    csv_filename = f"../src/CSV/SAC_MoutainCartRewards_seed{seed}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'Original_Reward', 'Shaped_Reward', 'Avg_Original_Reward_Last_100']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Track if the task has been solved
    solved = False
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=episode+seed*1000)  # Different seed per episode
        episode_original_reward = 0
        episode_shaped_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store the original and shaped rewards
            original_reward = info['original_reward']
            shaped_reward = reward  # This is the shaped reward
            
            # Store transition in replay buffer (using shaped reward for training)
            agent.replay_buffer.push(state, action, shaped_reward, next_state, float(done))
            
            # Update agent
            agent.update_parameters()
            
            state = next_state
            episode_original_reward += original_reward
            episode_shaped_reward += shaped_reward
            
            if done:
                break
        
        # Track both original and shaped rewards
        original_rewards.append(episode_original_reward)
        shaped_rewards.append(episode_shaped_reward)
        
        # Calculate running average over the last 'consecutive_episodes' episodes
        # For episodes < consecutive_episodes, we calculate over all available episodes
        window_size = min(consecutive_episodes, episode + 1)
        avg_reward = np.mean(original_rewards[-window_size:])
        running_avg_rewards.append(avg_reward)
        
        # Write to CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Episode', 'Original_Reward', 'Shaped_Reward', 'Avg_Original_Reward_Last_100'])
            writer.writerow({
                'Episode': episode + 1,
                'Original_Reward': episode_original_reward,
                'Shaped_Reward': episode_shaped_reward,
                'Avg_Original_Reward_Last_100': avg_reward
            })
        
        print(f"Seed {seed}, Episode {episode+1}, Original Reward: {episode_original_reward:.2f}, "
              f"Shaped Reward: {episode_shaped_reward:.2f}, Avg Original Reward over last {window_size}: {avg_reward:.2f}")
        
        # Save the best model (based on original reward)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            
            # Remove previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            #best_model_path = f'sac_{env_name.split("-")[0]}_momentum_best_seed{seed}.pth'
            # Save only state dictionaries instead of objects
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'reward': best_avg_reward,
                'episode': episode + 1
            }, best_model_path)
            
            print(f"New best model saved with avg reward: {best_avg_reward:.2f}")
        
        # Check if MountainCar is solved (avg reward of target_avg_reward+ over consecutive_episodes episodes)
        if window_size == consecutive_episodes and avg_reward >= target_avg_reward and not solved:
            print(f"\n🎉 Environment solved in {episode+1} episodes! 🎉")
            print(f"Average reward of {avg_reward:.2f} achieved over {consecutive_episodes} consecutive episodes.\n")
            solved = True
            
            # Save the model that achieved this milestone
            solved_model_path = f'sac_{env_name.split("-")[0]}_momentum_solved_seed{seed}.pth'
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
        final_model_path = f'sac_{env_name.split("-")[0]}_momentum_final_seed{seed}.pth'
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'reward': running_avg_rewards[-1] if running_avg_rewards else -float('inf'),
            'episode': max_episodes
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    env.close()
    return original_rewards, shaped_rewards, running_avg_rewards, best_model_path, solved

# Function to evaluate trained agents and record video (using original environment without reward shaping)
def evaluate_sac_with_video(model_path, env_name="MountainCarContinuous-v0", eval_episodes=5, video_folder="../src/Videos"):
    # Create folders if they don't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Load model - explicitly set weights_only=False to handle the PyTorch 2.6 change
    try:
        checkpoint = torch.load(model_path, weights_only=False)
    except RuntimeError:
        # Fallback for compatibility with older PyTorch versions
        checkpoint = torch.load(model_path)
    
    # Create environment with video recording (original environment, no reward shaping)
    video_path = os.path.join(video_folder, f"{os.path.basename(model_path).split('.')[0]}")
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_path,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"SAC_{env_name.split('-')[0]}_MountainCartContinuous"
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
    
    # Load state dictionaries
    agent.actor.load_state_dict(checkpoint['actor'])
    
    avg_reward = 0
    successes = 0
    
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
            
            # For MountainCar, check if goal is reached
            if env_name == "MountainCarContinuous-v0" and terminated and reward >= 100:
                successes += 1
        
        avg_reward += episode_reward
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    avg_reward /= eval_episodes
    success_rate = successes / eval_episodes
    
    print(f"Average Reward over {eval_episodes} episodes: {avg_reward:.2f}")
    if env_name == "MountainCarContinuous-v0":
        print(f"Success Rate: {success_rate:.2%}")
    
    env.close()
    print(f"Videos saved to {video_path}")
    return avg_reward, success_rate, video_path

# New function to run multiple seeds with momentum-based reward shaping and early stopping
def run_multiple_seeds_with_momentum(seeds=[0, 42, 123], env_name="MountainCarContinuous-v0", 
                                    max_episodes=1000, max_steps=999, consecutive_episodes=100, 
                                    target_avg_reward=90, momentum_factor=10.0, height_factor=5.0):
    all_original_rewards = []
    all_shaped_rewards = []
    all_avg_rewards = []
    best_models = []
    episodes_to_solve = []
    
    # Train with different seeds
    for seed in seeds:
        print(f"\n--- Starting training with seed {seed} (Momentum Reward Shaping) ---\n")
        original_rewards, shaped_rewards, avg_rewards, best_model_path, solved = train_sac_with_seed(
            seed, env_name, max_episodes, max_steps, momentum_factor, height_factor, 
            consecutive_episodes, target_avg_reward
        )
        all_original_rewards.append(original_rewards)
        all_shaped_rewards.append(shaped_rewards)
        all_avg_rewards.append(avg_rewards)
        best_models.append((best_model_path, np.max(avg_rewards)))
        
        if solved:
            episodes_to_solve.append(len(original_rewards))
        else:
            episodes_to_solve.append(max_episodes)
    
    # Combine all seed data into a single CSV
    combined_csv_filename = f"data/combined_rewards_{env_name.split('-')[0]}.csv"
    with open(combined_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Seed', 'Episode', 'Original_Reward', 'Shaped_Reward', 'Avg_Original_Reward_Last_100']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, seed in enumerate(seeds):
            for episode in range(len(all_original_rewards[i])):
                window_size = min(consecutive_episodes, episode + 1)
                avg_reward = np.mean(all_original_rewards[i][max(0, episode - consecutive_episodes + 1):episode + 1])
                
                writer.writerow({
                    'Seed': seed,
                    'Episode': episode + 1,
                    'Original_Reward': all_original_rewards[i][episode],
                    'Shaped_Reward': all_shaped_rewards[i][episode],
                    'Avg_Original_Reward_Last_100': avg_reward
                })
    
    # Plot original rewards
    plt.figure(figsize=(12, 6))
    
    # Plot individual seed results
    colors = ['blue', 'green', 'red']
    for i, seed in enumerate(seeds):
        plt.plot(all_original_rewards[i], alpha=0.3, color=colors[i], label=f'Seed {seed} (Original)')
    
    # Calculate and plot mean original rewards across seeds up to the minimum length
    min_length = min(len(rewards) for rewards in all_original_rewards)
    truncated_rewards = [rewards[:min_length] for rewards in all_original_rewards]
    mean_original_rewards = np.mean(truncated_rewards, axis=0)
    plt.plot(mean_original_rewards, color='black', linewidth=2, label='Mean Original Reward')
    
    plt.xlabel('Episode')
    plt.ylabel('Original Reward')
    plt.title(f'SAC Training with Momentum Reward Shaping on {env_name}')
    plt.legend()
    plt.savefig(f'sac_{env_name.split("-")[0]}_momentum_original_rewards.png')
    
    # Plot shaped rewards
    plt.figure(figsize=(12, 6))
    
    # Plot individual seed shaped rewards
    for i, seed in enumerate(seeds):
        plt.plot(all_shaped_rewards[i], alpha=0.3, color=colors[i], label=f'Seed {seed} (Shaped)')
    
    # Calculate and plot mean shaped rewards across seeds up to the minimum length
    truncated_shaped_rewards = [rewards[:min_length] for rewards in all_shaped_rewards]
    mean_shaped_rewards = np.mean(truncated_shaped_rewards, axis=0)
    plt.plot(mean_shaped_rewards, color='purple', linewidth=2, label='Mean Shaped Reward')
    
    plt.xlabel('Episode')
    plt.ylabel('Shaped Reward')
    plt.title(f'Shaped Rewards During Training on {env_name}')
    plt.legend()
    plt.savefig(f'sac_{env_name.split("-")[0]}_momentum_shaped_rewards.png')
    
    # Plot average original rewards (running averages)
    plt.figure(figsize=(12, 6))
    
    # Plot individual seed averages
    for i, seed in enumerate(seeds):
        plt.plot(all_avg_rewards[i], alpha=0.5, color=colors[i], label=f'Avg Reward Seed {seed} (Last {consecutive_episodes})')
    
    # Calculate and plot mean of average original rewards up to the minimum length
    min_avg_length = min(len(rewards) for rewards in all_avg_rewards)
    truncated_avg_rewards = [rewards[:min_avg_length] for rewards in all_avg_rewards]
    mean_avg_rewards = np.mean(truncated_avg_rewards, axis=0)
    plt.plot(mean_avg_rewards, color='black', linewidth=2, label=f'Mean Avg Reward (Last {consecutive_episodes})')
    
    # Add a horizontal line at the target average reward
    plt.axhline(y=target_avg_reward, color='r', linestyle='--', label=f'Target Avg Reward ({target_avg_reward})')
    
    plt.xlabel('Episode')
    plt.ylabel(f'Average Original Reward (Last {consecutive_episodes} Episodes)')
    plt.title(f'SAC Training with Momentum Reward Shaping - Running Averages')
    plt.legend()
    plt.savefig(f'sac_{env_name.split("-")[0]}_momentum_avg_rewards.png')
    plt.show()
    
    # Print final statistics
    print("\n--- Final Results with Momentum Reward Shaping ---")
    for i, seed in enumerate(seeds):
        final_avg = all_avg_rewards[i][-1]
        episodes_count = len(all_original_rewards[i])
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
    evaluate_sac_with_video(best_model[0], env_name=env_name)
    
    return all_original_rewards, all_shaped_rewards, all_avg_rewards, best_model[0]

# Run training with multiple seeds
if __name__ == "__main__":
    seeds = [0, 42, 123]  # Use multiple seeds for better robustness
    all_original_rewards, all_shaped_rewards, all_avg_rewards, best_model_path = run_multiple_seeds_with_momentum(
        seeds, 
        env_name="MountainCarContinuous-v0", 
        max_episodes=1000,  # Increased maximum episodes to allow for early stopping
        max_steps=999,
        consecutive_episodes=100,  # Use 100 consecutive episodes for the average
        target_avg_reward=90,      # Stop when average reward is over 90
        momentum_factor=5.0,       # Weight for momentum reward component
        height_factor=0            # Weight for height reward component
    )