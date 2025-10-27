# Deep RL Algorithm Comparison

Implementation and comparison of **PPO**, **DQN**, **SAC**, and **TD3** on OpenAI Gym environments.

## 🎯 Overview

Four deep RL algorithms tested across multiple environments with hyperparameter analysis and performance comparison.

**Algorithms**: PPO, DQN, SAC, TD3  
**Environments**: CartPole, MountainCar, MountainCarContinuous, Pendulum

## 🌐 Online Demo

You can explore the full project online with all videos, algorithm comparisons, and interactive visualizations at:

👉 [https://huggingface.co/spaces/zazo2002/RL_Applied_1_Project](https://huggingface.co/spaces/zazo2002/RL_Applied_1_Project)

Or Run `python Site.py` and open the provided URL.

## 🚀 Quick Start

### Run Individual Algorithms (Examples)
- PPO: `cd PPO && python ppo_cartpole.py`
- SAC: `cd SAC && python SAC_pendulum.py`  
- DQN: `cd DQN && python dqn_mountaincar.py`
- TD3: `cd TD3 && python td3_pendulum.py`

## 📁 Structure
RL_Project/
├── DQN/           
├── PPO/           
├── SAC/           
├── TD3/ 
├── requirements.txt         
├── src/
│   ├── CSV/       
│   ├── Results/   
│   └── Videos/    
└── Site.py       

## 🌟 Key Features

- **Interactive web interface** with algorithm comparisons
- **Multi-seed experiments** for statistical reliability  
- **Hyperparameter sensitivity analysis**
- **Video recordings** of trained policies
- **Comprehensive plotting** and data logging

## 📊 Results

| Algorithm | CartPole | MountainCar | MountainCarContinuous | Pendulum |
|-----------|----------|-------------|------------------------|----------|
| PPO       | ✅        | -           | ✅                      | -        |
| DQN       | ✅        | ✅          | -                      | -        |
| SAC       | -        | -           | ✅                      | ✅       |
| TD3       | -        | -           | ✅                      | ✅       |


**Features**:
- Algorithm equations and papers
- Environment descriptions  
- Performance plots
- Training videos
- Hyperparameter comparisons
- Implementation notes
