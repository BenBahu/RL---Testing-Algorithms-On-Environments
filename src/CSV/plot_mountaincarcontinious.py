import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob

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
    # Try different possible column names in order of preference
    if 'Original_Reward' in df.columns:
        return 'Original_Reward'
    elif 'Reward' in df.columns:
        return 'Reward'
    elif 'reward' in df.columns:
        return 'reward'
    elif 'Return' in df.columns:
        return 'Return'
    # Try to find any column with "reward" or "return" in the name (case insensitive)
    else:
        for col in df.columns:
            if 'reward' in col.lower() or 'return' in col.lower():
                return col
        
        # If no known reward column found
        print(f"Warning: Could not identify reward column. Columns: {df.columns.tolist()}")
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
        print(f"Warning: Could not identify episode column. Columns: {df.columns.tolist()}")
        return None

def find_ppo_mountaincar_files():
    """Find all PPO MountainCar reward files for all seeds."""
    # Look in the CSV directory
    csv_path = os.path.join("CSV", "*.csv")
    
    # Get all CSV files in the directory
    all_csv_files = glob.glob(csv_path)
    
    # Filter for PPO MountainCar files
    ppo_mountaincar_files = []
    for file in all_csv_files:
        file_lower = file.lower()
        if "ppo" in file_lower and "mountain" in file_lower:
            ppo_mountaincar_files.append(file)
    
    return ppo_mountaincar_files

def calculate_ppo_average():
    """Calculate the average rewards across all PPO MountainCar seeds."""
    # Find all PPO MountainCar files
    ppo_files = find_ppo_mountaincar_files()
    
    if not ppo_files:
        print("No PPO MountainCar files found. Exiting.")
        return None
    
    print(f"Found {len(ppo_files)} PPO MountainCar files: {ppo_files}")
    
    # Extract and process data from all seed files
    seed_data = {}
    
    for file in ppo_files:
        # Extract seed from filename
        seed = None
        for s in ["0", "42", "123"]:
            if f"seed{s}" in file.lower():
                seed = s
                break
        
        if seed is None:
            print(f"Could not determine seed for file {file}, skipping.")
            continue
        
        # Read data
        df = read_rewards_csv(file)
        if df is None:
            continue
        
        # Determine column names
        episode_col = get_episode_column_name(df)
        reward_col = get_reward_column_name(df)
        
        if episode_col is None or reward_col is None:
            print(f"Skipping file {file} due to missing required columns")
            continue
        
        # Extract data
        seed_data[seed] = {
            'episodes': df[episode_col].values,
            'rewards': df[reward_col].values
        }
    
    if not seed_data:
        print("No valid data found in any file. Exiting.")
        return None
    
    print(f"Processed data for {len(seed_data)} seeds: {list(seed_data.keys())}")
    
    # Determine the maximum number of episodes across all seeds
    max_episodes = max(len(data['episodes']) for data in seed_data.values())
    
    # Create a common episode range
    common_episodes = np.arange(1, max_episodes + 1)
    
    # Interpolate rewards for each seed to a common episode range
    interpolated_data = {}
    
    for seed, data in seed_data.items():
        # Create interpolation function
        if len(data['episodes']) > 1:  # Need at least 2 points for interpolation
            # If episodes don't start from 1, adjust
            episodes = data['episodes']
            if episodes[0] != 1:
                episodes = episodes - episodes[0] + 1
            
            # Limit to max_episodes
            valid_indices = episodes <= max_episodes
            if not any(valid_indices):
                print(f"No valid episodes for seed {seed} within range, skipping.")
                continue
                
            episodes = episodes[valid_indices]
            rewards = data['rewards'][valid_indices]
            
            # Linear interpolation
            interpolated_rewards = np.interp(
                common_episodes,
                episodes,
                rewards,
                left=rewards[0],   # Extrapolate using first value
                right=rewards[-1]  # Extrapolate using last value
            )
            
            interpolated_data[seed] = {
                'episodes': common_episodes,
                'rewards': interpolated_rewards
            }
    
    # Initialize array for average rewards
    avg_rewards = np.zeros_like(common_episodes, dtype=float)
    count = np.zeros_like(common_episodes, dtype=int)
    
    # Sum rewards for each episode
    for data in interpolated_data.values():
        avg_rewards += data['rewards']
        count += 1
    
    # Calculate average, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_rewards = np.divide(avg_rewards, count, 
                              out=np.zeros_like(avg_rewards), 
                              where=count!=0)
    
    # Replace NaN values with 0
    avg_rewards = np.nan_to_num(avg_rewards)
    
    return {
        'episodes': common_episodes,
        'rewards': avg_rewards,
        'smoothed': smooth_rewards(avg_rewards, window_size=10)
    }

def plot_mountaincar_comparison_with_ppo_average():
    """
    Plot rewards for MountainCarContinuous environment with:
    - TD3 (seed 0)
    - SAC (seed 0)
    - PPO (average across all seeds but labeled simply as "PPO")
    """
    # Files to analyze for TD3 and SAC
    sac_file = "CSV/SAC_MoutainCartRewards_seed0.csv"
    td3_file = "CSV/TD3_mountaincar_seed0.csv"
    
    # Check if files exist
    for file_path in [sac_file, td3_file]:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Please check the path.")
    
    # Read data for TD3 and SAC
    sac_df = read_rewards_csv(sac_file)
    td3_df = read_rewards_csv(td3_file)
    
    # Get PPO average
    ppo_avg_data = calculate_ppo_average()
    
    # Initialize data dictionaries
    data = {}
    
    # Process PPO average data
    if ppo_avg_data is not None:
        data["PPO"] = ppo_avg_data
    else:
        print("No PPO average data available.")
    
    # Process SAC data
    if sac_df is not None:
        print(f"SAC data columns: {sac_df.columns.tolist()}")
        episode_col = get_episode_column_name(sac_df)
        reward_col = get_reward_column_name(sac_df)
        
        if episode_col is not None and reward_col is not None:
            data["SAC"] = {
                "episodes": sac_df[episode_col].values,
                "rewards": sac_df[reward_col].values,
            }
            data["SAC"]["smoothed"] = smooth_rewards(data["SAC"]["rewards"], window_size=10)
        else:
            print(f"Skipping SAC data due to missing columns.")
    
    # Process TD3 data
    if td3_df is not None:
        print(f"TD3 data columns: {td3_df.columns.tolist()}")
        episode_col = get_episode_column_name(td3_df)
        reward_col = get_reward_column_name(td3_df)
        
        if episode_col is not None and reward_col is not None:
            data["TD3"] = {
                "episodes": td3_df[episode_col].values,
                "rewards": td3_df[reward_col].values,
            }
            data["TD3"]["smoothed"] = smooth_rewards(data["TD3"]["rewards"], window_size=10)
        else:
            print(f"Skipping TD3 data due to missing columns.")
    
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
        "PPO": '#45B5AA',  # Turquoise
        "SAC": '#E6CCB2',  # Beige
        "TD3": '#EE7674'   # Soft red
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
        plt.title(f'MountainCarContinuous-v0 Training Rewards (Algorithm Comparison) - {title_type}', 
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
        save_path = f'Results/mountaincar_comparison_{plot_type}_rewards.png'
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
        plt.title(f'MountainCarContinuous-v0 Training Rewards (Algorithm Comparison) - {title_type} (Log Scale)', 
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
        save_path = f'Results/mountaincar_comparison_{plot_type}_rewards_log.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_type} log-scale plot saved to {save_path}")
        
        # Apply tight layout
        plt.tight_layout()
        plt.show()
    
    return True

def main():
    """
    Plot MountainCarContinuous comparison with:
    - TD3 (seed 0)
    - SAC (seed 0) 
    - PPO (average of all seeds, labeled as PPO)
    """
    print("Plotting algorithm comparison for MountainCarContinuous-v0...")
    print("Using seed 0 for TD3 and SAC, and average of all seeds for PPO (labeled as PPO)")
    plot_mountaincar_comparison_with_ppo_average()
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()