import gradio as gr
import os

# Paper links and descriptions for each algorithm
# Implementation notes for algorithms on various environments
implementation_info = {
    "CartPole-v1_PPO": """
### 🛠️ Implementation Challenges for PPO on CartPole

1. **Discrete Actions Handling**: CartPole uses discrete actions (left/right), so we had to use a `Categorical` distribution instead of `Normal`. This changes how actions are sampled and log-probabilities are computed.
2. **Shared Network**: We used a single neural network with shared layers for both actor and critic, which helps with parameter efficiency but can lead to interference if not tuned well.
3. **Advantage Estimation**: We calculated advantages using the simple difference `returns - values`, and normalized them to stabilize training.
4. **Multiple Epoch Updates**: PPO requires updating the same batch several times. We had to carefully manage log probabilities and ratios to ensure stable learning.
5. **Gym Compatibility**: Recent changes in the Gym API required handling tuples when resetting or stepping the environment.
6. **Video Recording**: Gym's rendering had to be accessed using `render(mode='rgb_array')`, and OpenCV needed proper BGR conversion and resizing.

Despite being simpler than continuous control, PPO on CartPole still demanded precision in batching, advantage computation, and log-prob tracking.

### 📊 Hyperparameter Impact Analysis

*Note: Detailed hyperparameter experiments were conducted on this environment, with insights applicable to other discrete control tasks.*

Our hyperparameter tuning experiments revealed several key insights:

1. **Learning Rate (LR)**: 
   - Higher learning rate (0.01) led to significantly faster convergence
   - Lower learning rate (0.0005) struggled to reach the solving threshold

2. **Discount Factor (GAMMA)**:
   - Higher discount (0.999) had more variance but eventually solved
   - Lower discount (0.90) learned quickly initially but had stability issues

3. **Clipping Range (EPS_CLIP)**:
   - Both values (0.1 and 0.3) solved successfully
   - Higher clipping (0.3) had slightly better early performance

4. **Update Epochs (K_EPOCH)**:
   - Lower value (1) struggled with learning speed
   - Higher value (10) solved very quickly, showing more updates help
""",
    "MountainCarContinuous-v0_PPO": """
### 🛠️ Implementation Challenges for PPO on MountainCarContinuous

1. **Continuous Action Sampling**: We had to use a `Normal` distribution instead of `Categorical`, introducing the need to manage `action_std` and diagonal covariance matrices.
2. **Action Standard Deviation Decay**: To reduce exploration over time, we decayed `action_std` every 200 episodes to help the agent converge.
3. **Generalized Advantage Estimation (GAE)**: We implemented GAE to reduce variance in advantage estimates using a lambda-weighted future reward structure.
4. **Separate Actor/Critic Networks**: Continuous actions benefited from separate actor and critic networks for better learning stability.
5. **Entropy Regularization**: We added an entropy bonus to encourage exploration, which was essential in early episodes where rewards are sparse.
6. **Gym Compatibility + Video Capture**: Gym's new step API required checking `terminated` and `truncated`, and video recording had to handle raw RGB frames with OpenCV.

MountainCarContinuous was trickier than CartPole due to continuous actions and sparse rewards — we had to introduce action variance decay and GAE to learn successfully.

### 📊 Hyperparameter Impact Analysis

*Note: Detailed hyperparameter experiments were conducted on this environment, with insights applicable to other continuous control tasks.*

Our hyperparameter tuning experiments revealed several key insights:

1. **Action Standard Deviation**: 
   - Higher value (0.80) led to faster convergence by enabling greater exploration
   - Lower value (0.40) resulted in much slower learning due to limited exploration

2. **Clipping Parameter (EPS_CLIP)**:
   - Lower value (0.10) enabled faster learning and quicker convergence
   - Higher value (0.30) still solved the environment but took longer

3. **Training Epochs**:
   - Higher value (20) dramatically improved learning speed, solving in ~300 episodes
   - Lower value (5) struggled to make progress, highlighting the importance of sufficient updates

4. **GAE Lambda**:
   - Lower value (0.80) significantly improved learning speed, solving in ~400 episodes
   - Higher value (0.99) resulted in slower, more stable but less efficient learning
   
5. **Discount Factor (GAMMA)**:
   - Higher value (0.999) led to faster convergence by focusing on long-term returns
   - Lower value (0.90) resulted in slower learning due to shortsighted optimization

6. **Actor Learning Rate**:
   - Higher value (0.001) enabled faster policy updates and quicker convergence
   - Lower value (0.0001) resulted in slower but more stable learning
""",
    "MountainCarContinuous-v0_SAC": """
### 🛠️ Implementation Challenges for SAC on MountainCarContinuous

1. **Entropy Maximization**: Implementing the entropy term required careful balancing to ensure enough exploration without sacrificing performance.
2. **Twin Critics**: We needed two separate Q-networks to mitigate overestimation bias, requiring careful management of target networks.
3. **Automatic Temperature Tuning**: To automatically adjust the entropy coefficient, we had to implement a separate optimization process.
4. **Replay Buffer Management**: Efficient experience replay was crucial for off-policy learning in this sparse reward environment.
5. **Reward Scaling**: The large +100 reward for reaching the goal needed proper scaling to stabilize training.
6. **Action Squashing**: Ensuring actions fell within the environment limits using tanh and proper log probability calculations.
7. **Reward Shaping**: Unlike the standard SAC implementation, reward shaping was necessary to guide exploration in this sparse reward environment.

SAC's entropy maximization helped solve the exploration challenges in MountainCarContinuous where traditional methods struggle.

### 📊 Hyperparameter Impact Analysis

*Note: Detailed hyperparameter experiments were conducted on this environment, with insights applicable to other continuous control tasks.*

Our comprehensive hyperparameter study revealed critical insights:

1. **Target Update Rate (τ)**:
   - Lower values (0.005) provided excellent stability and fastest convergence around episode 20
   - Medium values (0.01) showed good performance but slightly slower convergence
   - Higher values (0.02) led to more volatile learning and delayed convergence

2. **Learning Rate**:
   - Higher learning rate (0.001) achieved fastest convergence and most stable performance
   - Medium rate (0.0006) showed good but slower convergence
   - Lower rate (0.0003) struggled significantly, taking much longer to reach optimal performance

3. **Temperature Parameter (α)**:
   - Lower values (0.1) led to fastest and most stable convergence, reaching ~95 reward consistently
   - Medium values (0.5) showed competitive performance but with more variability
   - Higher values (0.9) resulted in significantly slower learning and lower asymptotic performance

4. **Discount Factor (γ)**:
   - Higher values (0.995) demonstrated fastest convergence and excellent stability
   - Medium values (0.99) showed good performance but slower initial learning
   - Lower values (0.95) struggled with long-term planning, achieving lower final performance

**Key Finding**: SAC showed remarkable sensitivity to hyperparameter choices, with τ=0.005, LR=0.001, α=0.1, and γ=0.995 providing optimal performance.
""",
    "Pendulum-v1_SAC": """
### 🛠️ Implementation Challenges for SAC on Pendulum

1. **Continuous Torque Control**: Managing the continuous action space (-2 to 2) required proper scaling and action bounds.
2. **Negative Rewards**: Pendulum's negative rewards required careful Q-value initialization to avoid pessimistic starts.
3. **Dense Reward Function**: Unlike sparse reward environments, we needed to tune hyperparameters to handle the frequent feedback.
4. **Temperature Parameter Tuning**: Finding the right entropy coefficient was critical for balancing exploration and exploitation.
5. **Neural Network Architecture**: The relatively simple state space allowed for smaller networks, but required tuning layer sizes.
6. **Target Network Updates**: We used soft updates with polyak averaging to ensure stable learning.

SAC's ability to balance exploration and exploitation made it well-suited for the Pendulum's continuous control problem with its dense reward feedback.

### 📊 Hyperparameter Impact Analysis

*Note: Hyperparameter analysis was conducted on MountainCarContinuous-v0 and the insights apply to both environments due to similar continuous control characteristics.*

The hyperparameter insights from MountainCarContinuous transfer well to Pendulum:

1. **Target Update Rate (τ)**: Lower values (0.005) provide better stability for continuous control
2. **Learning Rates**: Higher learning rates (0.001) enable faster convergence in both environments
3. **Temperature Parameter (α)**: Lower values (0.1) balance exploration-exploitation effectively
4. **Discount Factor (γ)**: Higher values (0.995) support better long-term planning in both tasks
""",
    "MountainCarContinuous-v0_TD3": """
### 🛠️ Implementation Challenges for TD3 on MountainCarContinuous

1. **Twin Delayed Critics**: Implementing two critic networks and delaying policy updates required careful synchronization.
2. **Target Policy Smoothing**: Adding noise to target actions helped prevent exploitation of Q-function errors.
3. **Delayed Policy Updates**: Updating the policy less frequently than the critics required tracking update steps.
4. **Sparse Rewards**: The sparse reward structure of MountainCar required extended exploration periods.
5. **Action Bounds**: Ensuring actions stayed within [-1, 1] while calculating proper gradients needed special handling.
6. **Initialization Strategies**: Proper weight initialization was critical for stable learning in this environment.

TD3's conservative policy updates and overestimation bias mitigation proved effective for the challenging MountainCarContinuous task.

### 📊 Hyperparameter Impact Analysis

*Note: Hyperparameter analysis was conducted on Pendulum-v1 and the insights apply to both environments due to similar continuous control characteristics.*

TD3 required careful tuning for continuous control environments:

1. **Policy Noise**: Higher noise values improved exploration in sparse reward environments
2. **Target Update Frequency**: Delayed policy updates (every 2 critic updates) provided stability
3. **Learning Rates**: Balanced actor and critic learning rates were crucial for convergence
4. **Exploration Strategy**: For MountainCarContinuous, reward shaping was necessary to guide initial exploration
""",
    "Pendulum-v1_TD3": """
### 🛠️ Implementation Challenges for TD3 on Pendulum

1. **Exploration Strategy**: Balancing exploration noise magnitude was crucial for the pendulum's sensitive control.
2. **Clipped Double Q-learning**: Implementing the minimum of two critics required careful tensor operations.
3. **Target Networks**: Managing four separate networks (two critics, two targets) required organized code structure.
4. **Delayed Policy Updates**: Synchronizing updates at the right frequency was important for stability.
5. **Reward Scaling**: Pendulum's large negative rewards needed normalization to prevent value function saturation.
6. **Network Sizes**: Finding the right network capacity for both actor and critics affected learning speed.

TD3's focus on stable learning made it effective for Pendulum, where small action differences can lead to very different outcomes.

### 📊 Hyperparameter Impact Analysis

*Note: Detailed hyperparameter experiments were conducted on this environment, with insights applicable to other continuous control tasks.*

Our hyperparameter experiments on Pendulum revealed:

1. **Policy Noise (σ)**: 
   - Lower noise accelerated convergence and increased stability
   - Higher noise  provided more exploration but slower convergence

2. **Target Update Rate (τ)**:
   - Lower τ values led to slower but more stable convergence
   - Higher τ values enabled faster learning with acceptable stability

3. **Actor Learning Rate**:
   - Standard rate provided balanced learning speed and stability
   - Higher rates led to instability, lower rates slowed convergence

4. **Critic Learning Rate**:
   - Similar patterns to actor learning rate
   - Twin critics benefited from synchronized learning rates
""",
    "MountainCar-v0_DQN": """
### 🛠️ Implementation Challenges for DQN on MountainCar (Discrete)

1. **Discretized Action Space**: Working with the limited discrete actions (left, neutral, right) required effective exploration.
2. **Sparse Rewards**: The sparse reward structure meant the agent received almost no feedback until reaching the goal.
3. **Experience Replay**: Implementing a replay buffer to break correlations in the observation sequence was crucial.
4. **Target Network Updates**: Hard updates to the target network required careful timing to balance stability and learning speed.
5. **Epsilon Decay**: Finding the right exploration schedule was essential for the agent to discover the momentum-building strategy.
6. **Double DQN**: We implemented Double DQN to reduce overestimation bias, which was important for stable learning.

DQN required careful tuning to overcome the exploration challenges in MountainCar's sparse reward setting.

### 📊 Hyperparameter Impact Analysis

*Note: Hyperparameter analysis was conducted on CartPole-v1 and the insights apply to both discrete environments due to similar DQN architecture requirements.*

DQN's performance was highly sensitive to hyperparameter choices:

1. **Learning Rate**: Balanced rates provided steady convergence without instability
2. **Epsilon Decay**: Gradual decay from 1.0 to 0.01 over episodes enabled sufficient exploration
3. **Replay Buffer Size**: Large buffer helped provide diverse experiences for breaking correlations
4. **Target Network Update**: Regular updates balanced stability with learning speed
""",
    "CartPole-v1_DQN": """
### 🛠️ Implementation Challenges for DQN on CartPole

1. **Binary Action Selection**: Implementing efficient Q-value calculation for the two discrete actions (left/right).
2. **Reward Discount Tuning**: Finding the right gamma value for this task with potentially long episodes.
3. **Network Architecture**: Balancing network capacity with training stability for this relatively simple task.
4. **Epsilon Annealing**: Creating an effective exploration schedule that transitions from exploration to exploitation.
5. **Replay Buffer Size**: Tuning the memory size to balance between recent and diverse experiences.
6. **Update Frequency**: Determining how often to update the target network to maintain stability.

DQN's ability to learn value functions directly made it effective for CartPole, though careful exploration strategy was still necessary.

### 📊 Hyperparameter Impact Analysis

*Note: Detailed hyperparameter experiments were conducted on this environment, with insights applicable to other discrete control tasks.*

DQN demonstrated robust performance on CartPole across different hyperparameter settings:

1. **Learning Rate**: Higher rates led to faster convergence, lower rates were more stable
2. **Batch Size**: Medium batch sizes (64) provided good balance of gradient quality and computational efficiency
3. **Network Architecture**: Two hidden layers with 128 units each proved sufficient for this task
4. **Replay Buffer**: 100,000 transitions provided adequate experience diversity
"""
}

algo_info = {
    "PPO": {
        "description": "Proximal Policy Optimization (PPO) is a policy gradient method that uses a clipped surrogate objective to ensure stable and efficient updates.",
        "paper": "https://arxiv.org/abs/1707.06347",
        "equation": "L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}_t [ \\min(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1 - \\epsilon, 1 + \\epsilon)\\hat{A}_t) ]"
    },
    "DQN": {
        "description": "Deep Q-Network (DQN) uses deep neural networks to approximate the Q-value function in reinforcement learning.",
        "paper": "https://arxiv.org/abs/1312.5602",
        "equation": "L_i(\\theta_i) = \\mathbb{E}_{s,a,r,s'}[(r + \\gamma \\max_{a'} Q(s',a'; \\theta_i^-) - Q(s,a;\\theta_i))^2]"
    },
    "SAC": {
        "description": "Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm that maximizes a trade-off between expected return and entropy.",
        "paper": "https://arxiv.org/abs/1812.05905",
        "equation": "J(\\pi) = \\sum_t \\mathbb{E}_{(s_t, a_t) \\sim \\rho_\\pi} [r(s_t, a_t) + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t))]"
    },
    "TD3": {
        "description": "Twin Delayed DDPG (TD3) addresses overestimation bias in actor-critic methods by using two critics and target policy smoothing.",
        "paper": "https://arxiv.org/abs/1802.09477",
        "equation": "L(\\theta) = \\mathbb{E}[(r + \\gamma \\min_{i=1,2} Q_i(s', \\pi(s')) - Q(s,a))^2]"
    }
}

# Environment descriptions
env_info = {
    "CartPole-v1": "**CartPole-v1**\n\n- Goal: Keep the pole balanced upright on a moving cart.\n- Observation: Cart position/velocity, pole angle/angular velocity (4D).\n- Action Space: Discrete (left or right).\n- Reward: +1 per time step the pole is upright.\n- Termination: Pole falls or cart moves out of bounds.\n- Challenge: Requires rapid corrections; sensitive to delayed actions.",

    "MountainCarContinuous-v0": "**MountainCarContinuous-v0**\n\n- Goal: Drive the car up the right hill to reach the flag.\n- Observation: Position and velocity (2D).\n- Action Space: Continuous (thrust left/right).\n- Reward: +100 for reaching goal, small negative each step.\n- Termination: 200 steps or reaching the goal.\n- Challenge: Sparse reward, needs exploration to gain momentum.",
    
    "MountainCar-v0": "**MountainCar-v0**\n\n- Goal: Drive the car up the right hill to reach the flag.\n- Observation: Position and velocity (2D).\n- Action Space: Discrete (left, neutral, right).\n- Reward: -1 per step, 0 upon reaching goal.\n- Termination: 200 steps or reaching the goal.\n- Challenge: Very sparse reward, requires building momentum through oscillations.",
    
    "Pendulum-v1": "**Pendulum-v1**\n\n- Goal: Swing a pendulum to upright position and balance it.\n- Observation: Sine/cosine of angle, angular velocity (3D).\n- Action Space: Continuous (torque).\n- Reward: Negative cost based on angle from vertical and energy use.\n- Termination: After 200 steps (no early termination).\n- Challenge: Requires energy-efficient control and dealing with momentum."
}

# Mapping of algorithms to supported environments
algo_to_env = {
    "PPO": ["CartPole-v1", "MountainCarContinuous-v0"],
    "SAC": ["MountainCarContinuous-v0", "Pendulum-v1"],
    "TD3": ["MountainCarContinuous-v0", "Pendulum-v1"],
    "DQN": ["MountainCar-v0", "CartPole-v1"]
}

# Interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # Reinforcement Learning Algorithm Explorer
    Select an algorithm to learn more, then run it on a supported environment.

    **Environment**: A simulation where an agent takes actions to maximize rewards. Each interaction loop consists of: observation → action → reward → new state. The agent learns to optimize future rewards.
    """)

    algo_dropdown = gr.Dropdown(["PPO", "DQN", "SAC", "TD3"], label="Algorithm")
    algo_description = gr.Markdown()
    algo_equation = gr.Markdown()
    algo_link = gr.Markdown()

    env_dropdown = gr.Dropdown(label="Environment")
    env_description = gr.Markdown()

    run_button = gr.Button("Run")
    plot_output = gr.Image(label="Reward Curve")
    video_output = gr.Video(label="Agent Behavior Video")
    
    # Hyperparameter plot outputs
    hyperparams_accordion = gr.Accordion("Hyperparameter Analysis", open=False, visible=False)
    with hyperparams_accordion:
        gr.Markdown("### Hyperparameter Sensitivity Analysis")
        with gr.Row():
            hyperparam_img1 = gr.Image(label="", show_label=False, visible=False, height=400)
            hyperparam_img2 = gr.Image(label="", show_label=False, visible=False, height=400)
        with gr.Row():
            hyperparam_img3 = gr.Image(label="", show_label=False, visible=False, height=400)
            hyperparam_img4 = gr.Image(label="", show_label=False, visible=False, height=400)
        with gr.Row():
            hyperparam_img5 = gr.Image(label="", show_label=False, visible=False, height=400)
            hyperparam_img6 = gr.Image(label="", show_label=False, visible=False, height=400)
    
    # Implementation details
    implementation_output = gr.Markdown(label="Implementation Details")

    def update_algo_info(algo):
        info = algo_info.get(algo, {})
        return (
            info.get("description", ""),
            f"**Equation**: $${info.get('equation', '')}$$",
            f"[Read the paper]({info.get('paper', '#')})"
        )

    def update_env_info(env):
        return env_info.get(env, "")

    def filter_envs(algo):
        return gr.update(choices=algo_to_env.get(algo, []), value=algo_to_env.get(algo, [])[0] if algo_to_env.get(algo, []) else None)

    def serve_model(env_name, algorithm):
        combo_key = f"{env_name}_{algorithm}"
        
        # Show/hide hyperparameter accordion based on selection
        # Only show hyperparams for combinations that were actually tested
        show_hyperparams = combo_key in ["CartPole-v1_PPO", "MountainCarContinuous-v0_PPO", "Pendulum-v1_TD3", "CartPole-v1_DQN", "MountainCarContinuous-v0_SAC"]
        
        # Map each algorithm-environment combination to its plot and video paths
        # Updated paths based on your repository structure
        paths = {
            "CartPole-v1_PPO": ("src/Results/PPO_cartpole_smoothed_rewards.png", "src/Videos/PPO_cartpole_seed0.mp4"),
            "MountainCarContinuous-v0_PPO": ("src/Results/PPO_mountaincar_smoothed_rewards.png", "src/Videos/PPO_mountaincar_seed0-episode-0.mp4"),
            "MountainCarContinuous-v0_SAC": ("src/Results/SAC_mountaincar_smoothed_rewards.png", "src/Videos/SAC_MountainCarContinuous.mp4"),
            "Pendulum-v1_SAC": ("src/Results/SAC_pendulum_smoothed_rewards.png", "src/Videos/SAC_Pendulum.mp4"),
            "MountainCarContinuous-v0_TD3": ("src/Results/TD3_pendulum_smoothed_rewards.png", "src/Videos/TD3_MountainCarContinuous.mp4"),
            "Pendulum-v1_TD3": ("src/Results/TD3_pendulum_smoothed_rewards.png", "src/Videos/TD3_Pendulum.mp4"),
            "MountainCar-v0_DQN": ("src/Results/DQN_mountaincar_smoothed_rewards.png", "src/Videos/DQN_mountaincar_best.mp4"),
            "CartPole-v1_DQN": ("src/Results/cartpole_comparison_smoothed_rewards.png", "src/Videos/DQN_cartpole_best.mp4")
        }
        
        # Hyperparameter paths for different environments
        # Only include combinations that were actually tested
        hyperparameter_paths = {
            "CartPole-v1_PPO": [
                "src/Results/Hyperparameters/PPO_GAMMA_comparison.png",
                "src/Results/Hyperparameters/PPO_EPS_comparison.png",
                "src/Results/Hyperparameters/PPO_LR_comparison.png",
                "src/Results/Hyperparameters/PPO_K_comparison.png"
            ],
            "MountainCarContinuous-v0_PPO": [
                "src/Results/Hyperparameters/PPO_MountainCar_GAMMA_comparison.png",
                "src/Results/Hyperparameters/PPO_MountainCar_CLIP_EPSILON_comparison.png",
                "src/Results/Hyperparameters/PPO_MountainCar_EPOCHS_comparison.png",
                "src/Results/Hyperparameters/PPO_MountainCar_GAE_LAMBDA_comparison.png",
                "src/Results/PPO_MountainCar_ACTION_STD_comparison.png",
                "src/Results/Hyperparameters/PPO_MountainCar_LR_ACTOR_comparison.png"
            ],
            "Pendulum-v1_TD3": [
                "src/Results/Hyperparameters/td3_hyperparam.png"
            ],
            "CartPole-v1_DQN": [
                "src/Results/Hyperparameters/DQN_Hyperparameters.jpg"
            ],
            "MountainCarContinuous-v0_SAC": [
                "src/Results/Hyperparameters/SAC_tau.jpg",
                "src/Results/Hyperparameters/SAC_lr.jpg",
                "src/Results/Hyperparameters/SAC_alpha.jpg",
                "src/Results/Hyperparameters/SAC_Gamma.jpg"
            ]
        }
        
        if combo_key in paths:
            plot_path, video_path = paths[combo_key]
            
            # Check if the files exist
            plot_exists = os.path.exists(plot_path)
            video_exists = os.path.exists(video_path)
            
            if not plot_exists:
                print(f"Warning: Plot file {plot_path} not found.")
            
            if not video_exists:
                print(f"Warning: Video file {video_path} not found.")
            
            # Get implementation details
            implementation_details = implementation_info.get(combo_key, "Implementation details not available.")
            
            # Initialize all hyperparameter images as None
            img1 = img2 = img3 = img4 = img5 = img6 = None
            vis1 = vis2 = vis3 = vis4 = vis5 = vis6 = False
            
            # Get hyperparameter plots if applicable
            if combo_key in hyperparameter_paths:
                hyperparam_files = []
                for h_path in hyperparameter_paths[combo_key]:
                    if os.path.exists(h_path):
                        hyperparam_files.append(h_path)
                    else:
                        print(f"Warning: Hyperparameter plot {h_path} not found.")
                
                # Assign images to slots
                if len(hyperparam_files) >= 1:
                    img1, vis1 = hyperparam_files[0], True
                if len(hyperparam_files) >= 2:
                    img2, vis2 = hyperparam_files[1], True
                if len(hyperparam_files) >= 3:
                    img3, vis3 = hyperparam_files[2], True
                if len(hyperparam_files) >= 4:
                    img4, vis4 = hyperparam_files[3], True
                if len(hyperparam_files) >= 5:
                    img5, vis5 = hyperparam_files[4], True
                if len(hyperparam_files) >= 6:
                    img6, vis6 = hyperparam_files[5], True
                
                # Return all data including visibility update for accordion and individual images
                return (plot_path, video_path, implementation_details, gr.update(visible=show_hyperparams),
                        gr.update(value=img1, visible=vis1), gr.update(value=img2, visible=vis2),
                        gr.update(value=img3, visible=vis3), gr.update(value=img4, visible=vis4),
                        gr.update(value=img5, visible=vis5), gr.update(value=img6, visible=vis6))
            else:
                # Return without hyperparameter plots for other combinations
                return (plot_path, video_path, implementation_details, gr.update(visible=show_hyperparams),
                        gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                        gr.update(value=None, visible=False), gr.update(value=None, visible=False))
        else:
            return ("This combination is not supported yet.", None, "Implementation details not available.", gr.update(visible=False),
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False))

    algo_dropdown.change(fn=update_algo_info, inputs=algo_dropdown, outputs=[algo_description, algo_equation, algo_link])
    algo_dropdown.change(fn=filter_envs, inputs=algo_dropdown, outputs=env_dropdown)
    env_dropdown.change(fn=update_env_info, inputs=env_dropdown, outputs=env_description)
    run_button.click(fn=serve_model, inputs=[env_dropdown, algo_dropdown], 
                     outputs=[plot_output, video_output, implementation_output, hyperparams_accordion,
                             hyperparam_img1, hyperparam_img2, hyperparam_img3, 
                             hyperparam_img4, hyperparam_img5, hyperparam_img6])

if __name__ == "__main__":
    demo.launch()