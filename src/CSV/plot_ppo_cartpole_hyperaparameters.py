import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

# Directory paths - adjust if needed
CSV_DIR = "CSV"
RESULTS_DIR = "Results"
HYPERPARAMS_DIR = "CSV/hyperparameters/PPO"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define baseline config for reference
BASE_CONFIG = {
    'GAMMA': 0.99,
    'EPS_CLIP': 0.2,
    'LR': 0.002,
    'K_EPOCH': 3
}

# Define how files should be matched to parameters and their actual values
PARAM_MAPPING = {
    'GAMMA': ['GAMMA'],
    'EPS': ['EPS', 'EPS_CLIP'],
    'LR': ['LR'],
    'K': ['K_EPOCH', 'K']
}

# The actual hyperparameter values being tested
HYPERPARAM_VALUES = {
    'GAMMA': ['0.90', '0.999'],
    'EPS': ['0.1', '0.3'],
    'LR': ['0.0005', '0.01'],
    'K': ['1', '10']
}

# Define distinct colors - explicitly different
COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']  # red, blue, green, purple, orange, brown

def smooth_rewards(rewards, window_size=10):
    """Apply moving average smoothing to rewards."""
    if len(rewards) < 2:
        return rewards
        
    smoothed = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window = rewards[start_idx:i+1]
        smoothed[i] = np.mean(window)
    
    return smoothed

def read_reward_file(filepath):
    """Read a reward file and return episodes and rewards."""
    try:
        df = pd.read_csv(filepath)
        
        # Determine column names
        if 'Episode' in df.columns and 'Reward' in df.columns:
            episode_col = 'Episode'
            reward_col = 'Reward'
        else:
            # Try to guess columns
            for col in df.columns:
                if 'episode' in col.lower():
                    episode_col = col
                    break
            else:
                episode_col = df.columns[0]
                    
            for col in df.columns:
                if 'reward' in col.lower() or 'return' in col.lower():
                    reward_col = col
                    break
            else:
                reward_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        episodes = df[episode_col].values
        rewards = df[reward_col].values
        
        return episodes, rewards
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def load_baseline_data():
    """Load the baseline PPO CartPole data."""
    baseline_files = glob.glob(os.path.join(CSV_DIR, "PPO_cartpole*.csv"))
    print(f"Found {len(baseline_files)} baseline files")
    
    baseline_data = []
    
    for file in baseline_files:
        if "seed" in file.lower():  # Only process files with seed info
            episodes, rewards = read_reward_file(file)
            if episodes is not None and rewards is not None:
                smoothed_rewards = smooth_rewards(rewards)
                baseline_data.append((episodes, smoothed_rewards))
    
    # Calculate average baseline if multiple files exist
    if len(baseline_data) > 1:
        # Find the minimum episode length across all seeds
        min_len = min(len(data[0]) for data in baseline_data)
        
        # Truncate all data to the minimum length
        truncated_rewards = [data[1][:min_len] for data in baseline_data]
        
        # Calculate the average reward at each episode
        avg_rewards = np.mean(truncated_rewards, axis=0)
        
        return range(1, min_len+1), avg_rewards
    elif len(baseline_data) == 1:
        # Just one baseline
        return baseline_data[0]
    else:
        print("No baseline data found!")
        return None, None

def extract_param_info(filename):
    """Extract parameter name and value from filename."""
    # Try different formats
    for param_type, param_names in PARAM_MAPPING.items():
        for param_name in param_names:
            if param_name in filename:
                # First check if this is one of our known hyperparameter values
                for value in HYPERPARAM_VALUES.get(param_type, []):
                    if value in filename:
                        return param_type, value
                
                # If not found in our predefined values, try to extract from filename
                parts = filename.split(param_name)
                if len(parts) > 1:
                    # Try to extract numeric value
                    value_part = parts[1].lstrip('_').split('_')[0].split('.')[0]
                    # Return standardized parameter type and the actual value
                    return param_type, value_part
    
    return None, None

def create_plots():
    """Create separate comparison plots for each parameter type."""
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Load baseline data
    baseline_episodes, baseline_rewards = load_baseline_data()
    if baseline_episodes is None:
        print("Cannot plot: No baseline data found")
        return
    
    # Get hyperparameter CSV files (excluding summary files)
    all_files = glob.glob(os.path.join(HYPERPARAMS_DIR, "*.csv"))
    files = [f for f in all_files if "summary" not in os.path.basename(f).lower()]
    print(f"Found {len(files)} hyperparameter files")
    
    # Group files by parameter type
    param_groups = {}
    
    for file in files:
        filename = os.path.basename(file)
        param_type, param_value = extract_param_info(filename)
        
        if param_type and param_value:
            if param_type not in param_groups:
                param_groups[param_type] = []
            param_groups[param_type].append((file, param_value))
    
    # Process each parameter type
    for param_type, file_pairs in param_groups.items():
        if not file_pairs:
            continue
            
        print(f"\nCreating plot for {param_type}...")
        plt.figure(figsize=(12, 8))
        
        # Get baseline value for this parameter
        if param_type == 'GAMMA':
            baseline_value = BASE_CONFIG['GAMMA']
        elif param_type == 'EPS':
            baseline_value = BASE_CONFIG['EPS_CLIP']
        elif param_type == 'LR':
            baseline_value = BASE_CONFIG['LR']
        elif param_type == 'K':
            baseline_value = BASE_CONFIG['K_EPOCH']
        else:
            baseline_value = 'N/A'
        
        # Plot baseline
        plt.plot(baseline_episodes, baseline_rewards, color='black', 
                linewidth=2.5, label=f'Baseline ({param_type}={baseline_value})')
        
        # Plot each parameter value with a different color
        for i, (file, param_value) in enumerate(file_pairs):
            color = COLORS[i % len(COLORS)]
            
            episodes, rewards = read_reward_file(file)
            if episodes is not None and rewards is not None:
                smoothed_rewards = smooth_rewards(rewards)
                
                # Check if param_value is one of our known values
                if param_type in HYPERPARAM_VALUES and param_value in HYPERPARAM_VALUES[param_type]:
                    label = f'{param_type}={param_value}'
                else:
                    # If it's not one of our predefined values, try to match it
                    matched = False
                    if param_type in HYPERPARAM_VALUES:
                        for known_value in HYPERPARAM_VALUES[param_type]:
                            if (param_value.startswith(known_value) or 
                                param_value.replace('.', '').startswith(known_value.replace('.', ''))):
                                label = f'{param_type}={known_value}'
                                matched = True
                                break
                    
                    if not matched:
                        label = f'{param_type}={param_value}'
                
                # Plot with clear label
                plt.plot(episodes, smoothed_rewards, color=color, 
                        linewidth=2.0, label=label)
                
                print(f"  Plotted {label} with color {color}")
        
        # Add threshold line
        plt.axhline(y=475, color='r', linestyle='--', alpha=0.7, 
                   label='Solve Threshold (475)')
        
        # Configure plot appearance
        plt.title(f'Effect of {param_type} on CartPole-v1 Performance', 
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=14, labelpad=10)
        plt.ylabel('Reward (10-episode moving average)', fontsize=14, labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2000)  # Fixed range for consistency
        plt.legend(fontsize=12, loc='lower right', frameon=True, 
                  facecolor='white', edgecolor='lightgray')
        
        # Save the plot
        save_path = os.path.join(RESULTS_DIR, f'PPO_{param_type}_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")

def main():
    """Create comparison plots for hyperparameter tuning results."""
    print("Starting PPO hyperparameter visualization...")
    
    # Check if directories exist
    if not os.path.exists(CSV_DIR):
        print(f"Error: CSV directory '{CSV_DIR}' not found!")
        return
    
    if not os.path.exists(HYPERPARAMS_DIR):
        print(f"Error: Hyperparameters directory '{HYPERPARAMS_DIR}' not found!")
        return
    
    # Create plots
    create_plots()
    
    print("Plotting complete! Check the Results directory for the saved plots.")

if __name__ == "__main__":
    main()