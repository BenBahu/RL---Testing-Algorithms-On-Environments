import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import os
import cv2
import csv

# PPO Hyperparameters
GAMMA = 0.99
EPS_CLIP = 0.2
LR = 0.002
K_EPOCH = 3
MAX_EPISODES = 2000
SOLVE_SCORE = 475
SEEDS = [0, 42, 123]  # Changed seeds to 0, 42, 123

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, 64)
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.policy(x), self.value(x)
    
    def get_action(self, state):
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

# Memory for storing episode data
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
    def add(self, state, action, log_prob, reward, is_terminal, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)

# PPO Agent Class
class PPO:
    def __init__(self, state_dim, action_dim):
        # Create model and optimizer
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = Memory()
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state_tensor)
        return action, log_prob, value
    
    def update(self):
        # Calculate advantages and returns
        returns = []
        advantages = []
        R = 0
        
        for r, d, v in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals), reversed(self.memory.values)):
            R = 0 if d else R
            R = r + GAMMA * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        values = torch.tensor(self.memory.values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(self.memory.states))
        old_actions = torch.LongTensor(self.memory.actions)
        old_log_probs = torch.FloatTensor(self.memory.log_probs)
        
        # PPO update for K epochs
        for _ in range(K_EPOCH):
            # Get current policy
            logits, state_values = self.model(old_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(old_actions)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            # Total loss (policy loss + value loss)
            loss = -torch.min(surr1, surr2).mean() + 0.5 * ((returns.detach() - state_values.squeeze()).pow(2)).mean()
            
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()

# Function to save rewards to CSV
def save_rewards_to_csv(rewards, seed):
    # Create directory for CSV files
    
    # Create CSV file
    file_path = f'../src/CSV/PPO_cartpole_rewards_seed{seed}.csv'
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward'])  # Header
        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])
    
    print(f"Rewards saved to {file_path}")

# Train function - runs training for a specific seed
def train_for_seed(seed, record_video=False):
    # Create environment
    env = gym.make('CartPole-v1')
    env.seed(seed)
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create PPO agent
    ppo_agent = PPO(state_dim, action_dim)
    
    # To store rewards for saving to CSV
    all_rewards = []
    
    # To track if environment is solved
    solved = False
    
    print(f"\nStarting training for seed {seed}")
    
    # Training loop
    for episode in range(MAX_EPISODES):
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # For different gym versions
            state = state[0]
        
        # Run episode
        episode_reward = 0
        
        while True:
            # Get action
            action, log_prob, value = ppo_agent.select_action(state)
            
            # Take action
            next_state, reward, done, *_ = env.step(action.item())
            if isinstance(next_state, tuple):  # For different gym versions
                next_state = next_state[0]
            
            # Store in memory
            ppo_agent.memory.add(state, action, log_prob, reward, done, value)
            
            # Update state and episode reward
            state = next_state
            episode_reward += reward
            
            # End episode if done
            if done:
                break
        
        all_rewards.append(episode_reward)
        
        # Update policy
        ppo_agent.update()
        
        # Print progress periodically
        if episode % 50 == 0:
            avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Episode {episode} | Reward: {episode_reward:.1f} | Avg Reward: {avg_reward:.1f}")
        
        # Check if solved
        if len(all_rewards) >= 100:
            avg_reward = np.mean(all_rewards[-100:])
            if avg_reward >= SOLVE_SCORE and not solved:
                solved = True
                print(f"✅ Solved at episode {episode} with avg reward {avg_reward:.2f}")
                
                # Save the model
                # torch.save(ppo_agent.model.state_dict(), f'../src/Results/ppo_cartpole_seed{seed}.pt')
                
                # Record video if needed
                if record_video:
                    save_video(ppo_agent.model, seed)
                
                break
    
    env.close()
    
    # Save rewards to CSV
    save_rewards_to_csv(all_rewards, seed)
    
    return all_rewards

# Save video
def save_video(model, seed=1):
    print("Recording video...")
    
    # Create environment
    env = gym.make('CartPole-v1')
    env.seed(seed)
    
    # Reset environment
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'../src/Videos/PPO_cartpole_seed{seed}.mov', fourcc, 30, (600, 400))
    
    # Run episode
    done = False
    episode_reward = 0
    
    while not done:
        # Render frame
        frame = env.render(mode='rgb_array')
        
        # Resize frame
        frame = cv2.resize(frame, (600, 400))
        
        # Convert RGB to BGR (for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write to video
        video.write(frame)
        
        # Get action
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action, _, _ = model.get_action(state_tensor)
        
        # Take action
        next_state, reward, done, *_ = env.step(action.item())
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        state = next_state
        episode_reward += reward
    
    # Close video writer and environment
    video.release()
    env.close()
    
    print(f"Video saved to videos/cartpole_seed{seed}.mov with reward {episode_reward}")

# Main function
def main():
    # Train for each seed
    for i, seed in enumerate(SEEDS):
        # Only record video for first seed
        record_video = (i == 0)
        train_for_seed(seed, record_video)
    
    print("\nTraining completed for all seeds. Results saved to CSV files.")

if __name__ == "__main__":
    main()