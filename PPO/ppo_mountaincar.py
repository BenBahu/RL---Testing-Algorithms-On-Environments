import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import os
import csv

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
EPOCHS = 10
BATCH_SIZE = 64
MAX_EPISODES = 1000
MAX_TIMESTEPS = 1000
SOLVE_SCORE = 90
SEEDS = [0, 42, 123]  # Changed seeds to match CartPole code

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Action standard deviation
        self.action_std = action_std_init
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = torch.full((self.actor[4].out_features,), new_action_std * new_action_std).to(device)
    
    def act(self, state):
        action_mean = self.actor(state)
        dist = Normal(action_mean, self.action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = Normal(action_mean, self.action_std)
        
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, lr_actor, lr_critic):
        self.action_std = action_std_init
        
        # Initialize actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        
        # Optimizers
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        
        # Old policy for updates
        self.old_policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Loss function
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.old_policy.act(state)
        
        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()
    
    def update(self, memory):
        # Get the rollout buffer data
        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_rewards = torch.FloatTensor(np.array(memory.rewards)).to(device)
        old_dones = torch.FloatTensor(np.array(memory.dones)).to(device)
        
        # Calculate advantages using GAE
        advantages = []
        gae = 0
        with torch.no_grad():
            values = self.old_policy.critic(old_states).squeeze()
            
            next_value = 0
            
            for t in reversed(range(len(old_rewards))):
                if old_dones[t]:
                    delta = old_rewards[t] - values[t]
                    gae = delta
                else:
                    if t == len(old_rewards) - 1:
                        delta = old_rewards[t] + GAMMA * next_value - values[t]
                    else:
                        delta = old_rewards[t] + GAMMA * values[t + 1] - values[t]
                    gae = delta + GAMMA * GAE_LAMBDA * gae
                
                advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for EPOCHS epochs
        for _ in range(EPOCHS):
            # Create random mini-batches
            batch_indices = np.random.permutation(len(old_states))
            batches = [batch_indices[i:i + BATCH_SIZE] for i in range(0, len(batch_indices), BATCH_SIZE)]
            
            for batch_idx in batches:
                # Convert batch indices to tensors
                batch_states = old_states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Evaluate actions and get new log probs and state values
                new_logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Calculate ratios
                ratios = torch.exp(new_logprobs - batch_logprobs)
                
                # Surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, batch_returns.unsqueeze(1))
                
                # Add entropy regularization
                entropy_loss = -0.01 * dist_entropy.mean()
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # Update actor
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()
        
        # Copy new weights to old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


# Function to save rewards to CSV
def save_rewards_to_csv(rewards, seed):
    # Use the same CSV path as in the updated CartPole code
    file_path = f'../src/CSV/PPO_mountaincar_rewards_seed{seed}.csv'
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward'])  # Header
        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])
    
    print(f"Rewards saved to {file_path}")


def train_for_seed(seed, record_video=False):
    # Create environment
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    # Get environment details
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Print environment information
    print(f"\nStarting training for seed {seed}")
    
    # Setup PPO agent
    action_std = 0.6  # Starting standard deviation for exploration
    ppo_agent = PPO(state_dim, action_dim, action_std, LR_ACTOR, LR_CRITIC)
    
    # Setup buffer
    memory = RolloutBuffer()
    
    # Lists to track progress
    episode_rewards = []
    
    # Action standard deviation decay
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = 200  # Every N episodes
    
    # Flag to track if environment is solved
    solved = False
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]  # Handle gym versions
        cumulative_reward = 0
        
        # Decay action std if needed
        if episode % action_std_decay_freq == 0:
            new_action_std = max(min_action_std, ppo_agent.action_std - action_std_decay_rate)
            ppo_agent.set_action_std(new_action_std)
            print(f"Episode {episode}: Action std changed to {new_action_std}")
        
        for t in range(MAX_TIMESTEPS):
            # Select action
            action, logprob = ppo_agent.select_action(state)
            
            # Take action in env
            step_result = env.step(action)
            
            # Handle different gym versions
            if len(step_result) == 4:  # Old gym API
                next_state, reward, done, _ = step_result
            else:  # New gym API
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Store in buffer
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.dones.append(done)
            
            # Update state
            state = next_state
            cumulative_reward += reward
            
            if done:
                break
        
        # Record episode stats
        episode_rewards.append(cumulative_reward)
        
        # Update PPO agent
        ppo_agent.update(memory)
        memory.clear()
        
        # Print episode statistics
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Seed {seed} | Episode {episode}\tAvg Reward: {avg_reward:.2f}")
        
        # Check if environment is solved
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward >= SOLVE_SCORE and not solved:  # MountainCarContinuous is solved when avg reward is 90+
                solved = True
                print(f"✅ Seed {seed} solved at episode {episode} with avg reward {avg_reward:.2f}")
                
                # Record video if needed
                if record_video:
                    save_video(ppo_agent, seed)
                
                # Break early if solved
                break
    
    # Close environment
    env.close()
    
    # Save rewards to CSV
    save_rewards_to_csv(episode_rewards, seed)
    
    return episode_rewards


def save_video(agent, seed=1):
    print("🎥 Recording video using Gym's RecordVideo wrapper...")

    env_name = "MountainCarContinuous-v0"
    save_dir = "../src/Videos"
    os.makedirs(save_dir, exist_ok=True)

    # Use RecordVideo wrapper
    env = gym.make(env_name)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=save_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=f"PPO_mountaincar_seed{seed}"
    )

    state = env.reset(seed=seed)
    if isinstance(state, tuple):
        state = state[0]

    done = False
    total_reward = 0

    while not done:
        action, _ = agent.select_action(state)
        result = env.step(action)

        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated

        if isinstance(next_state, tuple):
            next_state = next_state[0]

        state = next_state
        total_reward += reward

    env.close()
    print(f"✅ Video saved in {save_dir}/ | Total reward: {total_reward:.2f}")


def main():
    # Train for each seed
    for i, seed in enumerate(SEEDS):
        # Only record video for first seed
        record_video = (i == 0)
        train_for_seed(seed, record_video)
    
    print("\nTraining completed for all seeds. Results saved to CSV files.")


if __name__ == "__main__":
    main()