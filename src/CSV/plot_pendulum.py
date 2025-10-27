import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def plot_pendulum_comparison():
    """Plot rewards for SAC and TD3 algorithms on Pendulum-v1 environment."""
    # Files to analyze
    sac_file = "CSV/SAC_Pendulum-v1_rewards_seed0.csv"
    td3_file = "CSV/TD3_Pendulum-v1_rewards_seed0.csv"
    
    # Read data
    sac_df = read_rewards_csv(sac_file)
    td3_df = read_rewards_csv(td3_file)
    
    if sac_df is None or td3_df is None:
        print("Failed to read one or more files. Exiting.")
        return False
    
    # Print first few rows for debugging
    print(f"SAC data (first 5 rows):")
    print(sac_df.head())
    
    print(f"TD3 data (first 5 rows):")
    print(td3_df.head())
    
    # Extract data
    sac_episodes = sac_df['Episode'].values
    sac_rewards = sac_df['Reward'].values
    sac_smoothed = smooth_rewards(sac_rewards, window_size=10)
    
    td3_episodes = td3_df['Episode'].values
    td3_rewards = td3_df['Reward'].values
    td3_smoothed = smooth_rewards(td3_rewards, window_size=10)
    
    # Collect all rewards for y-axis limits
    all_rewards = list(sac_rewards) + list(td3_rewards)
    
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
    sac_color = '#45B5AA'  # Turquoise
    td3_color = '#EE7674'  # Soft red
    
    # Create two plots: one for raw data, one for smoothed data
    for plot_type in ["raw", "smoothed"]:
        plt.figure(figsize=(12, 8))
        
        if plot_type == "raw":
            plt.plot(sac_episodes, sac_rewards, 
                     label="SAC", 
                     color=sac_color, 
                     linewidth=2.0)
            plt.plot(td3_episodes, td3_rewards, 
                     label="TD3", 
                     color=td3_color, 
                     linewidth=2.0)
        else:  # smoothed
            plt.plot(sac_episodes, sac_smoothed, 
                     label="SAC", 
                     color=sac_color, 
                     linewidth=2.5)
            plt.plot(td3_episodes, td3_smoothed, 
                     label="TD3", 
                     color=td3_color, 
                     linewidth=2.5)
        
        # Add labels and title
        title_type = "Raw" if plot_type == "raw" else "Smoothed (10-episode window)"
        plt.title(f'Pendulum-v1 Training Rewards (SAC vs TD3) - {title_type}', 
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
        save_path = f'Results/pendulum_comparison_{plot_type}_rewards.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_type} plot saved to {save_path}")
        
        # Apply tight layout
        plt.tight_layout()
        plt.show()
    
    return True

def main():
    """Plot rewards for SAC and TD3 on Pendulum-v1."""
    print("Plotting SAC vs TD3 Pendulum rewards comparison...")
    plot_pendulum_comparison()
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()