import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

# Directory paths
CSV_DIR = "CSV"  # CSV files are in src/CSV
RESULTS_DIR = "Results"  # Results should go in src/Results

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

def find_reward_files(algo, env_name):
    """Find all reward files for a given algorithm and environment."""
    # Look in the CSV directory
    csv_path = os.path.join(CSV_DIR, "*.csv")
    
    # Get all CSV files in the directory
    all_csv_files = glob.glob(csv_path)
    print(f"Searching for {algo} {env_name} files among: {all_csv_files}")
    
    # Filter by algorithm, environment name 
    files = []
    
    # Special case for DQN MountainCar files
    if algo.lower() == "dqn" and "mountain" in env_name.lower():
        for file in all_csv_files:
            file_lower = file.lower()
            if "dqn" in file_lower and "mountain" in file_lower:
                files.append(file)
                print(f"Found DQN MountainCar file: {file}")
    # Special case for TD3 Pendulum files
    elif algo.lower() == "td3" and "pendulum" in env_name.lower():
        for file in all_csv_files:
            file_lower = file.lower()
            if "td3" in file_lower and "pendulum" in file_lower:
                files.append(file)
                print(f"Found TD3 Pendulum file: {file}")
    # Special case for SAC MountainCar with "MoutainCartRewards" naming 
    elif algo.lower() == "sac" and "mountain" in env_name.lower():
        for file in all_csv_files:
            file_lower = file.lower()
            # Check for both official and typo versions of "mountaincar" in filenames
            if ("sac" in file_lower and 
                ("mountaincar" in file_lower or "moutaincart" in file_lower)):
                files.append(file)
                print(f"Found SAC MountainCar file: {file}")
    # Special case for SAC Pendulum-v1 combined file
    elif algo.lower() == "sac" and "pendulum" in env_name.lower():
        for file in all_csv_files:
            file_lower = file.lower()
            if "sac" in file_lower and "pendulum" in file_lower:
                files.append(file)
                print(f"Found SAC Pendulum file: {file}")
    # For regular files
    else:
        for file in all_csv_files:
            file_lower = file.lower()
            if algo.lower() in file_lower and env_name.lower() in file_lower:
                files.append(file)
                print(f"Found standard file: {file}")
    
    # If we found any files
    if files:
        return files
    else:
        print(f"No reward files found for {algo} {env_name}.")
        return []

def get_reward_column_name(df):
    """Determine the reward column name based on dataframe columns."""
    # Print all columns for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    # Try different possible column names in order of preference
    if 'Return' in df.columns:
        return 'Return'
    elif 'Original_Reward' in df.columns:
        return 'Original_Reward'
    elif 'Reward' in df.columns:
        return 'Reward'
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

def extract_seed_from_filename(filename):
    """Extract seed number from filename."""
    # Try to find seed pattern in filename
    for s in ["0", "42", "123"]:
        if f"seed{s}" in filename.lower():
            return s
    
    # If specific seeds not found, look for any number after "seed"
    import re
    match = re.search(r'seed(\d+)', filename.lower())
    if match:
        return match.group(1)
    
    # If no seed pattern found
    return None

def plot_rewards(algo, env_name):
    """Plot rewards for a specific algorithm and environment with separate raw and smoothed plots."""
    # Set up a nice plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Find reward files
    reward_files = find_reward_files(algo, env_name)
    
    if not reward_files:
        print(f"No files found for {algo} {env_name}. Skipping plot.")
        return False
    
    print(f"Found {len(reward_files)} files for {algo} {env_name}: {reward_files}")
    
    # Get proper environment name for display
    if env_name.lower() == 'cartpole':
        display_name = 'CartPole-v1'
    elif env_name.lower() == 'mountaincar' and algo.lower() == 'dqn':
        display_name = 'MountainCar-v0'  # Discrete version for DQN
    elif env_name.lower() == 'mountaincar':
        display_name = 'MountainCarContinuous-v0'  # Continuous version for other algos
    elif env_name.lower() == 'pendulum':
        display_name = 'Pendulum-v1'
    else:
        display_name = env_name
    
    # Store all rewards to set consistent y-axis limits across plots
    all_rewards = []
    
    # Collect data from all files
    data_by_seed = {}
    
    # Process each file
    for file in reward_files:
        # Read data
        df = read_rewards_csv(file)
        if df is None:
            print(f"Failed to read file: {file}")
            continue
        
        # Print raw data for debugging
        print(f"First few rows of {file}:")
        print(df.head())
        
        # For files with a Seed column (where multiple seeds are in one file)
        if 'Seed' in df.columns:
            # Group by seed
            for seed, group in df.groupby('Seed'):
                # Determine column names
                episode_col = get_episode_column_name(group)
                reward_col = get_reward_column_name(group)
                
                if episode_col is None or reward_col is None:
                    print(f"Skipping seed {seed} in file {file} due to missing required columns")
                    continue
                
                # Extract data
                episodes = group[episode_col].values
                rewards = group[reward_col].values
                
                # Handle empty data
                if len(rewards) == 0:
                    print(f"Empty data for seed {seed}")
                    continue
                
                # Store data
                seed_key = str(seed)
                data_by_seed[seed_key] = {
                    'episodes': episodes,
                    'rewards': rewards,
                    'smoothed': smooth_rewards(rewards, window_size=10)
                }
                
                # Collect all rewards for y-axis limits
                all_rewards.extend(rewards)
        else:
            # Determine column names
            episode_col = get_episode_column_name(df)
            reward_col = get_reward_column_name(df)
            
            if episode_col is None:
                print(f"No episode column found in {file}. Looking for first numeric column...")
                # Try to find the first numeric column
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if col != reward_col:  # Make sure it's not the reward column
                            episode_col = col
                            print(f"Using {col} as episode column")
                            break
            
            if reward_col is None:
                print(f"No reward column found in {file}. Looking for numeric columns...")
                # Try to find numeric columns that might be rewards
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]) and col != episode_col:
                        reward_col = col
                        print(f"Using {col} as reward column")
                        break
            
            if episode_col is None or reward_col is None:
                print(f"Skipping file {file} due to missing required columns")
                continue
            
            # Extract data
            episodes = df[episode_col].values
            rewards = df[reward_col].values
            
            # Handle empty data
            if len(rewards) == 0:
                print(f"Empty data for file {file}")
                continue
            
            # Extract seed from filename or use fallback
            seed = extract_seed_from_filename(file)
            if seed is None:
                # For files like combined pendulum data
                if "pendulum" in file.lower() and "combined" in file.lower():
                    seed = "combined"
                else:
                    seed = f"file{len(data_by_seed)}"
            
            # Store data
            data_by_seed[seed] = {
                'episodes': episodes,
                'rewards': rewards,
                'smoothed': smooth_rewards(rewards, window_size=10)
            }
            
            # Collect all rewards for y-axis limits
            all_rewards.extend(rewards)
    
    # If no valid data was found, return
    if not data_by_seed:
        print(f"No valid data found for {algo} {env_name}. Skipping plot.")
        return False
    
    # Create Results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
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
    colors = ['#45B5AA', '#E6CCB2', '#EE7674']  # Turquoise, Beige, Soft red
    
    # Create two plots: one for raw data, one for smoothed data
    for plot_type in ["raw", "smoothed"]:
        plt.figure(figsize=(12, 8))
        
        # Plot each seed
        for i, (seed, data) in enumerate(data_by_seed.items()):
            if seed == "combined":
                seed_label = "Data"
            else:
                seed_label = f"Seed {seed}"
            
            if plot_type == "raw":
                plt.plot(data['episodes'], data['rewards'], 
                         label=seed_label, 
                         color=colors[i % len(colors)], 
                         linewidth=2.0)
            else:  # smoothed
                plt.plot(data['episodes'], data['smoothed'], 
                         label=seed_label, 
                         color=colors[i % len(colors)], 
                         linewidth=2.5)
        
        # Add labels and title
        title_type = "Raw" if plot_type == "raw" else "Smoothed (10-episode window)"
        plt.title(f'{display_name} Training Rewards ({algo.upper()}) - {title_type}', 
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
        
        # Save plot
        save_path = os.path.join(RESULTS_DIR, 
                                f'{algo.upper()}_{env_name.lower()}_{plot_type}_rewards.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_type} plot saved to {save_path}")
        
        # Apply tight layout
        plt.tight_layout()
        plt.show()
    
    return True

def plot_ppo_rewards(env_name):
    """Plot PPO rewards for a specific environment with separate raw and smoothed plots."""
    return plot_rewards("PPO", env_name)

def plot_sac_rewards(env_name):
    """Plot SAC rewards for a specific environment with separate raw and smoothed plots."""
    return plot_rewards("SAC", env_name)

def plot_td3_rewards(env_name):
    """Plot TD3 rewards for a specific environment with separate raw and smoothed plots."""
    return plot_rewards("TD3", env_name)

def plot_dqn_rewards(env_name):
    """Plot DQN rewards for a specific environment with separate raw and smoothed plots."""
    return plot_rewards("DQN", env_name)

def main():
    """Plot rewards for multiple algorithms and environments, with separate raw and smoothed plots."""
    # Check if CSV directory exists
    if not os.path.exists(CSV_DIR):
        print(f"Error: CSV directory '{CSV_DIR}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print("Please make sure you're running this script from the 'src' directory.")
        return
    
    # List all CSV files in the CSV directory to help debug
    print(f"Available CSV files in {CSV_DIR} directory:")
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    for file in csv_files:
        print(f"  - {file}")
    
    print("\n")
    
    # Plot PPO CartPole rewards
    print("Plotting PPO CartPole rewards...")
    plot_ppo_rewards('cartpole')
    
    print("\n")
    
    # Plot PPO MountainCar rewards
    print("Plotting PPO MountainCar rewards...")
    plot_ppo_rewards('mountaincar')
    
    print("\n")
    
    # Plot SAC MountainCar rewards
    print("Plotting SAC MountainCar rewards...")
    plot_sac_rewards('mountaincar')
    
    print("\n")
    
    # Plot SAC Pendulum rewards
    print("Plotting SAC Pendulum rewards...")
    plot_sac_rewards('pendulum')
    
    print("\n")
    
    # Plot TD3 Pendulum rewards
    print("Plotting TD3 Pendulum rewards...")
    plot_td3_rewards('pendulum')
    
    print("\n")
    
    # Plot TD3 MountainCar rewards
    print("Plotting TD3 MountainCar rewards...")
    plot_td3_rewards('mountaincar')
    
    print("\n")
    
    # Plot DQN MountainCar rewards
    print("Plotting DQN MountainCar rewards...")
    plot_dqn_rewards('mountaincar')
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()