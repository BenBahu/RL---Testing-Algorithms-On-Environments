import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import csv
import os

# Base PPO settings
BASE_CONFIG = {
    'GAMMA': 0.99,
    'EPS_CLIP': 0.2,
    'LR': 0.002,
    'K_EPOCH': 3
}
MAX_EPISODES = 2000
SOLVE_SCORE = 475
SEED = 0  # Just one seed for quick experiments

# Variants to test
HYPERPARAMS = {
    'GAMMA': [0.90, 0.999],
    'EPS_CLIP': [0.1, 0.3],
    'LR': [0.0005, 0.01],
    'K_EPOCH': [1, 10]
}

# Results path - create directory structure
RESULTS_DIR = '../src/CSV/hyperparameters/PPO/'
os.makedirs(RESULTS_DIR, exist_ok=True)
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, 'summary_results.csv')

# Simple PPO setup
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        return self.policy(x), self.value(x)

    def get_action(self, state):
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['LR'])
        self.config = config
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state_tensor)
        return action.item(), log_prob, value.item()

    def store(self, s, a, lp, r, d, v):
        self.memory['states'].append(s)
        self.memory['actions'].append(a)
        self.memory['log_probs'].append(lp)
        self.memory['rewards'].append(r)
        self.memory['dones'].append(d)
        self.memory['values'].append(v)

    def update(self):
        # Calculate returns and advantages
        R = 0
        returns, adv = [], []
        for r, done, v in zip(reversed(self.memory['rewards']), reversed(self.memory['dones']), reversed(self.memory['values'])):
            R = 0 if done else R
            R = r + self.config['GAMMA'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.tensor(self.memory['values'])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(self.memory['states'])
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])

        for _ in range(self.config['K_EPOCH']):
            logits, state_values = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config['EPS_CLIP'], 1 + self.config['EPS_CLIP']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns - state_values.squeeze()) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = {k: [] for k in self.memory}

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

    env = gym.make('CartPole-v1')
    env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, config)

    rewards = []
    solved = False
    solve_ep = None

    print(f"Running experiment: {param_name}={param_value}")

    for ep in range(1, MAX_EPISODES + 1):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        ep_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, *_ = env.step(action)
            if isinstance(next_state, tuple): next_state = next_state[0]
            agent.store(state, action, log_prob, reward, done, value)
            state = next_state
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        agent.update()

        # Print progress periodically
        if ep % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {ep} | Parameter: {param_name}={param_value} | Reward: {ep_reward:.1f} | Avg Reward: {avg_reward:.1f}")

        if len(rewards) >= 100:
            avg = np.mean(rewards[-100:])
            if avg >= SOLVE_SCORE and not solved:
                solved = True
                solve_ep = ep
                print(f"✅ Solved at episode {ep} with {param_name}={param_value}, avg reward {avg:.2f}")
                break  # Stop once solved

    env.close()
    final_avg = np.mean(rewards[-100:])
    
    # Save all rewards to a dedicated CSV file
    save_rewards_to_csv(rewards, param_name, param_value)
    
    return [param_name, param_value, int(solved), solve_ep if solved else "-", round(final_avg, 2)]

# Run all tests
with open(SUMMARY_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["param", "value", "solved", "episode_solved", "final_avg_reward"])

    for param, values in HYPERPARAMS.items():
        for val in values:
            result = run_experiment(param, val, BASE_CONFIG)
            print(f"RESULT: {result[0]}={result[1]} | Solved: {result[2]} | Ep: {result[3]} | Avg: {result[4]}")
            writer.writerow(result)

print(f"\nExperiment complete! Summary results saved to {SUMMARY_CSV_PATH}")
print(f"Detailed results for each experiment saved in {RESULTS_DIR}")