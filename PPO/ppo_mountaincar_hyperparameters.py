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

# Base PPO settings
BASE_CONFIG = {
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'CLIP_EPSILON': 0.2,
    'LR_ACTOR': 0.0003,
    'LR_CRITIC': 0.001,
    'EPOCHS': 10,
    'ACTION_STD': 0.6,
    'MIN_ACTION_STD': 0.1
}

# Hyperparameters to test
HYPERPARAMS = {
    'GAMMA': [0.9, 0.999],
    'GAE_LAMBDA': [0.8, 0.99],
    'CLIP_EPSILON': [0.1, 0.3],
    'LR_ACTOR': [0.0001, 0.001],
    'EPOCHS': [5, 20],
    'ACTION_STD': [0.4, 0.8]
}

# Fixed parameters
MAX_EPISODES = 1000
MAX_TIMESTEPS = 1000
BATCH_SIZE = 64
SOLVE_SCORE = 90
SEED = 0  # Just one seed for hyperparameter testing

# Results path
RESULTS_DIR = '../src/CSV/hyperparameters/PPO_MountainCar'
os.makedirs(RESULTS_DIR, exist_ok=True)
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, 'summary_results.csv')

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
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.action_std = config['ACTION_STD']
        
        # Initialize actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, self.action_std).to(device)
        
        # Optimizers
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=config['LR_ACTOR'])
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=config['LR_CRITIC'])
        
        # Old policy for updates
        self.old_policy = ActorCritic(state_dim, action_dim, self.action_std).to(device)
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
                        delta = old_rewards[t] + self.config['GAMMA'] * next_value - values[t]
                    else:
                        delta = old_rewards[t] + self.config['GAMMA'] * values[t + 1] - values[t]
                    gae = delta + self.config['GAMMA'] * self.config['GAE_LAMBDA'] * gae
                
                advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.config['EPOCHS']):
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
                surr2 = torch.clamp(ratios, 1 - self.config['CLIP_EPSILON'], 1 + self.config['CLIP_EPSILON']) * batch_advantages
                
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


def save_rewards_to_csv(rewards, param_name, param_value):
    """Save rewards for each episode to a separate CSV file"""
    filename = f"{param_name}_{param_value}.csv"
    file_path = os.path.join(RESULTS_DIR, filename)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward'])  # Header
        for i, reward in enumerate(rewards):
            writer.writerow([i+1, reward])
    
    print(f"Detailed rewards saved to {file_path}")


def run_experiment(param_name, param_value, config):
    """Run an experiment with a specific parameter value"""
    # Make a deep copy of the config to avoid contamination between runs
    config = config.copy()
    config[param_name] = param_value
    
    # Set up the environment
    env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Get environment details
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, config)
    
    # Initialize rollout buffer
    memory = RolloutBuffer()
    
    # Lists to track progress
    rewards = []
    solved = False
    solve_ep = None
    
    # Action standard deviation decay settings
    action_std_decay_rate = 0.05
    action_std_decay_freq = 100  # Every N episodes
    
    print(f"\nRunning experiment: {param_name}={param_value}")
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]  # Handle gym versions
        cumulative_reward = 0
        
        # Decay action std if needed
        if episode % action_std_decay_freq == 0:
            new_action_std = max(config['MIN_ACTION_STD'], ppo_agent.action_std - action_std_decay_rate)
            ppo_agent.set_action_std(new_action_std)
        
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
        rewards.append(cumulative_reward)
        
        # Update PPO agent
        ppo_agent.update(memory)
        memory.clear()
        
        # Print progress periodically
        if episode % 50 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode} | Param: {param_name}={param_value} | Avg Reward: {avg_reward:.1f}")
        
        # Check if environment is solved
        if len(rewards) >= 100:
            avg_reward = np.mean(rewards[-100:])
            if avg_reward >= SOLVE_SCORE and not solved:
                solved = True
                solve_ep = episode
                print(f"✅ Solved at episode {episode} with {param_name}={param_value}, avg reward {avg_reward:.2f}")
                break  # Stop once solved
    
    # Close environment
    env.close()
    
    # Save all rewards to a dedicated CSV file
    save_rewards_to_csv(rewards, param_name, param_value)
    
    # Calculate final average reward
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    
    # Return results summary
    return [param_name, param_value, int(solved), solve_ep if solved else "-", round(final_avg, 2)]


def main():
    """Run hyperparameter experiments for PPO on MountainCarContinuous-v0"""
    print("Starting PPO hyperparameter tuning for MountainCarContinuous-v0...")
    print(f"Device: {device}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Testing parameters: {HYPERPARAMS}")
    
    # Create summary CSV file
    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value", "solved", "episode_solved", "final_avg_reward"])
        
        # Run experiments for each hyperparameter
        for param_name, values in HYPERPARAMS.items():
            for val in values:
                result = run_experiment(param_name, val, BASE_CONFIG)
                print(f"RESULT: {result[0]}={result[1]} | Solved: {result[2]} | Ep: {result[3]} | Avg: {result[4]}")
                writer.writerow(result)
    
    print(f"\nExperiment complete! Summary results saved to {SUMMARY_CSV_PATH}")
    print(f"Detailed results for each experiment saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()