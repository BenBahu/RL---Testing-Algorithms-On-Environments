import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set up a nice plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def read_rewards_csv(filename):
    """Read the rewards from a CSV file."""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def smooth_rewards(rewards, window_size=10):
    """Apply moving average smoothing to rewards."""
    if len(rewards) < 2:
        return rewards  # Not enough data to smooth
        
    # Create a proper moving average with correct handling of beginning and end
    smoothed = np.zeros_like(rewards)
    
    for i in range(len(rewards)):
        # Calculate window bounds
        start_idx = max(0, i - window_size + 1)
        # Use only available data points
        window = rewards[start_idx:i+1]
        smoothed[i] = np.mean(window)
    
    return smoothed

def get_reward_column_name(df):
    """Determine the reward column name based on dataframe columns."""
    # Print all columns for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    # Try different possible column names in order of preference
    if 'Return' in df.columns:
        return 'Return'
    elif 'Reward' in df.columns:
        return 'Reward'
    elif 'Original_Reward' in df.columns:
        return 'Original_Reward'
    elif 'reward' in df.columns:
        return 'reward'
    # Try to find any column with "reward" or "return" in the name (case insensitive)
    else:
        for col in df.columns:
            if 'reward' in col.lower() or 'return' in col.lower():
                return col
        
        # If no known reward column found
        print(f"Warning: Could not identify reward column.")
        return None

def get_episode_column_name(df):
    """Determine the episode column name based on dataframe columns."""
    # Try different possible column names
    if 'Episode' in df.columns:
        return 'Episode'
    elif 'episode' in df.columns:
        return 'episode'
    else:
        # Try to find any column with "episode" in the name (case insensitive)
        for col in df.columns:
            if 'episode' in col.lower():
                return col
        
        # If no known episode column found
        print(f"Warning: Could not identify episode column.")
        return None

def plot_cartpole_comparison():
    """Plot rewards for DQN and PPO algorithms on CartPole environment."""
    # Files to analyze
    dqn_file = "CSV/DQN_cartpole_rewards_seed0.csv"
    ppo_file = "CSV/PPO_cartpole_rewards_seed0.csv"
    
    # Check if files exist
    for file_path in [dqn_file, ppo_file]:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Please check the path.")
    
    # Read data
    dqn_df = read_rewards_csv(dqn_file)
    ppo_df = read_rewards_csv(ppo_file)
    
    # Initialize data dictionaries
    data = {}
    
    # Process DQN data
    if dqn_df is not None:
        print(f"DQN data columns: {dqn_df.columns.tolist()}")
        episode_col = get_episode_column_name(dqn_df)
        reward_col = get_reward_column_name(dqn_df)
        
        if episode_col is not None and reward_col is not None:
            data["DQN"] = {
                "episodes": dqn_df[episode_col].values,
                "rewards": dqn_df[reward_col].values,
            }
            data["DQN"]["smoothed"] = smooth_rewards(data["DQN"]["rewards"], window_size=10)
        else:
            print(f"Skipping DQN data due to missing columns.")
    
    # Process PPO data
    if ppo_df is not None:
        print(f"PPO data columns: {ppo_df.columns.tolist()}")
        episode_col = get_episode_column_name(ppo_df)
        reward_col = get_reward_column_name(ppo_df)
        
        if episode_col is not None and reward_col is not None:
            data["PPO"] = {
                "episodes": ppo_df[episode_col].values,
                "rewards": ppo_df[reward_col].values,
            }
            data["PPO"]["smoothed"] = smooth_rewards(data["PPO"]["rewards"], window_size=10)
        else:
            print(f"Skipping PPO data due to missing columns.")
    
    # If no valid data was found, return
    if not data:
        print(f"No valid data found for any algorithm. Exiting.")
        return False
    
    # Collect all rewards for y-axis limits
    all_rewards = []
    for algo, algo_data in data.items():
        all_rewards.extend(algo_data["rewards"])
    
    # Set consistent y-axis limits
    if all_rewards:
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
        margin = (max_reward - min_reward) * 0.1
        y_min = min_reward - margin
        y_max = max_reward + margin
    else:
        y_min, y_max = None, None
    
    # Define colors
    colors = {
        "DQN": '#45B5AA',  # Turquoise
        "PPO": '#EE7674'   # Soft red
    }
    
    # Create two plots: one for raw data, one for smoothed data
    for plot_type in ["raw", "smoothed"]:
        # Create first plot: regular x-axis
        plt.figure(figsize=(12, 8))
        
        for algo, algo_data in data.items():
            if plot_type == "raw":
                plt.plot(algo_data["episodes"], algo_data["rewards"], 
                         label=algo, 
                         color=colors[algo], 
                         linewidth=2.0)
            else:  # smoothed
                plt.plot(algo_data["episodes"], algo_data["smoothed"], 
                         label=algo, 
                         color=colors[algo], 
                         linewidth=2.5)
        
        # Add labels and title
        title_type = "Raw" if plot_type == "raw" else "Smoothed (10-episode window)"
        plt.title(f'CartPole-v1 Training Rewards (Algorithm Comparison) - {title_type}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=14, labelpad=10)
        plt.ylabel('Reward', fontsize=14, labelpad=10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(fontsize=12, frameon=True, facecolor='white', 
                   edgecolor='lightgray', loc='best')
        
        # Set consistent y-axis limits across plots
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        # Create Results directory if it doesn't exist
        os.makedirs('Results', exist_ok=True)
        
        # Save plot
        save_path = f'Results/cartpole_comparison_{plot_type}_rewards.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_type} plot saved to {save_path}")
        
        # Apply tight layout
        plt.tight_layout()
        plt.show()
        
        # Create second plot: log-scaled x-axis
        plt.figure(figsize=(12, 8))
        
        for algo, algo_data in data.items():
            # Add 1 to all episode values to avoid log(0)
            episodes = algo_data["episodes"] + 1
            
            if plot_type == "raw":
                plt.semilogx(episodes, algo_data["rewards"], 
                            label=algo, 
                            color=colors[algo], 
                            linewidth=2.0)
            else:  # smoothed
                plt.semilogx(episodes, algo_data["smoothed"], 
                            label=algo, 
                            color=colors[algo], 
                            linewidth=2.5)
        
        # Add labels and title
        title_type = "Raw" if plot_type == "raw" else "Smoothed (10-episode window)"
        plt.title(f'CartPole-v1 Training Rewards (Algorithm Comparison) - {title_type} (Log Scale)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode (Log Scale)', fontsize=14, labelpad=10)
        plt.ylabel('Reward', fontsize=14, labelpad=10)
        
        # Add grid with log-scaling support
        plt.grid(True, alpha=0.3, which='both')
        
        # Add legend
        plt.legend(fontsize=12, frameon=True, facecolor='white', 
                   edgecolor='lightgray', loc='best')
        
        # Set consistent y-axis limits across plots
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        # Create Results directory if it doesn't exist
        os.makedirs('Results', exist_ok=True)
        
        # Save plot
        save_path = f'Results/cartpole_comparison_{plot_type}_rewards_log.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_type} log-scale plot saved to {save_path}")
        
        # Apply tight layout
        plt.tight_layout()
        plt.show()
    
    return True

def main():
    """Plot rewards for DQN and PPO on CartPole-v1."""
    print("Plotting algorithm comparison for CartPole-v1...")
    plot_cartpole_comparison()
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()