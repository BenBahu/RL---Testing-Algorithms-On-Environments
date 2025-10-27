import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

# Directory paths
CSV_DIR = "CSV"
RESULTS_DIR = "CSV/Results"
HYPERPARAMS_DIR = "CSV/hyperparameters/PPO_MountainCar"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define baseline config
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

# Hyperparameter values being tested
HYPERPARAMS = {
    'GAMMA': [0.9, 0.999],
    'GAE_LAMBDA': [0.8, 0.99],
    'CLIP_EPSILON': [0.1, 0.3],
    'LR_ACTOR': [0.0001, 0.001],
    'EPOCHS': [5, 20],
    'ACTION_STD': [0.4, 0.8]
}

# Colors for parameter values (first value=red, second value=blue)
COLORS = ['#d62728', '#1f77b4']

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
    """Load the baseline PPO MountainCar data - ONLY seed 0."""
    baseline_file = os.path.join("CSV", "PPO_mountaincar_rewards_seed0.csv")
    print(f"Loading baseline file: {baseline_file}")
    
    if os.path.exists(baseline_file):
        episodes, rewards = read_reward_file(baseline_file)
        if episodes is not None and rewards is not None:
            smoothed_rewards = smooth_rewards(rewards)
            return episodes, smoothed_rewards
    
    print("Baseline file not found!")
    return None, None

def plot_hyperparameter_comparisons():
    """Create separate plots for each hyperparameter type."""
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # Load baseline data once
    baseline_episodes, baseline_rewards = load_baseline_data()
    
    if baseline_episodes is None:
        print("Cannot plot: No baseline data found")
        return
    
    # Get all hyperparameter files
    all_files = glob.glob(os.path.join(HYPERPARAMS_DIR, "*.csv"))
    files = [f for f in all_files if "summary" not in os.path.basename(f).lower()]
    print(f"Found {len(files)} hyperparameter files")
    
    # Parameter display names
    param_display_names = {
        'GAMMA': 'Discount Factor (γ)',
        'GAE_LAMBDA': 'GAE Lambda (λ)',
        'CLIP_EPSILON': 'Clipping Parameter (ε)',
        'LR_ACTOR': 'Actor Learning Rate',
        'EPOCHS': 'Training Epochs',
        'ACTION_STD': 'Action Standard Deviation'
    }
    
    # Process each parameter type
    for param_name, param_values in HYPERPARAMS.items():
        print(f"\nCreating plot for {param_name}...")
        plt.figure(figsize=(12, 8))
        
        # Plot baseline
        baseline_value = BASE_CONFIG.get(param_name, "N/A")
        display_name = param_display_names.get(param_name, param_name)
        
        # Format baseline value for display
        if param_name in ['GAMMA', 'GAE_LAMBDA']:
            formatted_baseline = f"{float(baseline_value):.3f}"
        elif param_name in ['LR_ACTOR', 'LR_CRITIC']:
            formatted_baseline = f"{float(baseline_value):.4f}" if float(baseline_value) >= 0.001 else f"{float(baseline_value):.1e}"
        elif param_name in ['CLIP_EPSILON', 'ACTION_STD']:
            formatted_baseline = f"{float(baseline_value):.2f}"
        else:
            formatted_baseline = str(baseline_value)
        
        plt.plot(baseline_episodes, baseline_rewards, color='black', 
                linewidth=2.5, label=f'Baseline ({param_name}={formatted_baseline})')
        
        # Find and plot files for each parameter value
        for i, param_value in enumerate(param_values):
            # For GAMMA specifically
            if param_name == 'GAMMA':
                if param_value == 0.9:
                    filename = f"GAMMA_0.9.csv"
                elif param_value == 0.999:
                    filename = f"GAMMA_0.999.csv"
                else:
                    filename = f"{param_name}_{param_value}.csv"
            else:
                # For other parameters
                filename = f"{param_name}_{param_value}.csv"
            
            # Get the full path
            file_path = os.path.join(HYPERPARAMS_DIR, filename)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"  ⚠️ File not found: {file_path}")
                continue
            
            # Read the data
            episodes, rewards = read_reward_file(file_path)
            if episodes is not None and rewards is not None:
                smoothed_rewards = smooth_rewards(rewards)
                
                # Format parameter value for display
                if param_name in ['GAMMA', 'GAE_LAMBDA']:
                    formatted_value = f"{float(param_value):.3f}"
                elif param_name in ['LR_ACTOR', 'LR_CRITIC']:
                    formatted_value = f"{float(param_value):.4f}" if float(param_value) >= 0.001 else f"{float(param_value):.1e}"
                elif param_name in ['CLIP_EPSILON', 'ACTION_STD']:
                    formatted_value = f"{float(param_value):.2f}"
                else:
                    formatted_value = str(param_value)
                
                # Plot with the parameter value in the label
                color = COLORS[i % len(COLORS)]
                plt.plot(episodes, smoothed_rewards, color=color, 
                        linewidth=2.0, label=f'{param_name}={formatted_value}')
                
                print(f"  ✅ Plotted {param_name}={formatted_value}")
            else:
                print(f"  ❌ Could not read data from {file_path}")
        
        # Add threshold line
        plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, 
                   label='Solve Threshold (90)')
        
        # Configure plot appearance
        plt.title(f'Effect of {display_name} on MountainCarContinuous-v0 Performance', 
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=14, labelpad=10)
        plt.ylabel('Reward (10-episode moving average)', fontsize=14, labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1000)
        
        # Add legend with better positioning
        plt.legend(fontsize=12, loc='lower right', frameon=True, 
                  facecolor='white', edgecolor='lightgray')
        
        # Save the plot
        save_path = os.path.join(RESULTS_DIR, f'PPO_MountainCar_{param_name}_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")

def main():
    """Create comparison plots for MountainCar hyperparameter tuning."""
    print("Starting PPO MountainCar hyperparameter visualization...")
    
    # Check if directories exist
    if not os.path.exists(HYPERPARAMS_DIR):
        print(f"Error: Hyperparameters directory '{HYPERPARAMS_DIR}' not found!")
        return
    
    # Create plots
    plot_hyperparameter_comparisons()
    
    print("Plotting complete! Check the Results directory for the saved plots.")

if __name__ == "__main__":
    main()