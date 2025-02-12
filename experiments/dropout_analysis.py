import os
import sys
# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rl.agents.ddpg import DDPGAgent
try:
    from env.virtual_taobao import VirtualTaobao
except ImportError:
    print("Warning: Could not import VirtualTaobao environment")
    print("Please ensure the env directory exists and contains virtual_taobao.py")
    sys.exit(1)

def analyze_uncertainty(agent, test_states, num_samples=100):
    """Analyze action uncertainty for different states"""
    uncertainties = []
    mean_actions = []
    
    for state in test_states:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, uncertainty = agent.actor.get_action_with_uncertainty(
            state_tensor, num_samples=num_samples
        )
        uncertainties.append(uncertainty.detach().mean().item())
        mean_actions.append(action.detach().numpy())
    
    return np.array(uncertainties), np.array(mean_actions)

def visualize_uncertainty(uncertainties, rewards, title):
    """Visualize the relationship between uncertainty and rewards"""
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainties, rewards, alpha=0.5)
    plt.xlabel('Action Uncertainty')
    plt.ylabel('Reward')
    plt.title(title)
    plt.savefig(f'uncertainty_vs_reward_{title}.png')
    plt.close()

def evaluate_action(action, env, state):
    """Evaluate the effect of an action"""
    next_state, reward, done, _ = env.step(action)
    return reward

def train_episode(agent, env, episode):
    """Train for one episode with improved update frequency"""
    state = env.reset()
    episode_reward = 0
    max_steps = 200
    
    # Add experience replay warmup
    if episode == 0:
        for _ in range(1000):  # Collect initial experiences
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state
    
    for step in range(max_steps):
        action, uncertainty = agent.select_action(state, explore=True)
        next_state, reward, done, _ = env.step(action)
        
        # Enhanced reward scaling for better stability
        reward = reward * 0.05  # Further scale down rewards
        
        agent.memory.push(state, action, reward, next_state, done)
        
        if len(agent.memory) > agent.batch_size:
            # Increase update frequency
            for _ in range(8):  # Increase updates per step from 4 to 8
                agent.update()
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    return episode_reward

def evaluate(agent, env, num_episodes=5):
    """Evaluate agent performance"""
    eval_rewards = []
    eval_uncertainties = []
    max_steps = 200
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_uncertainties = []
        
        for step in range(max_steps):
            # Get action and uncertainty
            action, uncertainty = agent.select_action(state, explore=False)
            next_state, reward, done, _ = env.step(action)  # Only use action
            
            episode_reward += reward
            episode_uncertainties.append(uncertainty)
            
            if done:
                break
            
            if step >= max_steps - 1:
                break
                
            state = next_state
            
        eval_rewards.append(episode_reward)
        eval_uncertainties.extend(episode_uncertainties)
    
    return np.mean(eval_rewards), np.mean(eval_uncertainties)

def compare_dropout_effects(results_with_dropout, results_without_dropout):
    """Compare the performance between agents with and without dropout"""
    plt.figure(figsize=(12, 5))
    
    # 1. Training rewards comparison
    plt.subplot(1, 2, 1)
    plt.plot(results_with_dropout['rewards'], label='With Dropout')
    plt.plot(results_without_dropout['rewards'], label='Without Dropout')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 2. Evaluation rewards comparison
    plt.subplot(1, 2, 2)
    plt.plot(results_with_dropout['eval_rewards'], label='With Dropout')
    plt.plot(results_without_dropout['eval_rewards'], label='Without Dropout')
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dropout_comparison.png')
    plt.close()

def main():
    print("Starting parameter search...")
    
    # Environment parameters
    state_dim = 91
    action_dim = 27
    
    # Create environment
    env = VirtualTaobao()
    
    # Define configurations for parameter search
    configs = [
        {'dropout_rate': 0.05, 'hidden_dim': 128},
        {'dropout_rate': 0.05, 'hidden_dim': 256},
        {'dropout_rate': 0.05, 'hidden_dim': 512},
        {'dropout_rate': 0.1, 'hidden_dim': 128},
        {'dropout_rate': 0.1, 'hidden_dim': 256},
        {'dropout_rate': 0.1, 'hidden_dim': 512},
        {'dropout_rate': 0.15, 'hidden_dim': 128},
        {'dropout_rate': 0.15, 'hidden_dim': 256},
        {'dropout_rate': 0.15, 'hidden_dim': 512}
    ]
    
    # Initialize tracking variables for best configuration
    best_config = None
    best_reward = float('-inf')
    results = {}
    
    print("\nPhase 1: Parameter Search")
    
    # Phase 1: Parameter Search
    for config in configs:
        dropout_rate = config['dropout_rate']
        hidden_dim = config['hidden_dim']
        
        print(f"\nTesting config: dropout_rate={dropout_rate}, hidden_dim={hidden_dim}")
        
        # Create agent
        agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Training and evaluation parameters
        rewards = []
        uncertainties = []
        num_test_episodes = 200  # Increased to 200 episodes for more thorough testing
        
        for episode in range(num_test_episodes):
            reward = train_episode(agent, env, episode)
            rewards.append(reward)
            
            if episode % 5 == 0:  # Evaluate every 5 episodes (40 evaluations total)
                eval_reward, eval_uncertainty = evaluate(agent, env, num_episodes=2)
                print(f"Episode {episode}, Eval Reward: {eval_reward:.3f}, "
                      f"Uncertainty: {eval_uncertainty:.3f}")
    
    # Phase 2: Comparing Best Config vs No Dropout
    print("\nPhase 2: Comparing Best Config vs No Dropout")
    
    # Initialize agents with and without dropout
    agent_with_dropout = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=best_config['hidden_dim'],
        dropout_rate=best_config['dropout_rate']
    )
    
    agent_without_dropout = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=best_config['hidden_dim'],
        dropout_rate=0.0
    )
    
    # Training parameters for comparison phase
    num_episodes = 200  # Total training episodes
    eval_interval = 5   # Evaluate every 5 episodes (40 evaluations total)
    
    # Storage for training and evaluation results
    results_with_dropout = {
        'rewards': [],      # Training rewards for 200 episodes
        'eval_rewards': []  # Evaluation rewards (40 measurements)
    }
    
    results_without_dropout = {
        'rewards': [],      # Training rewards for 200 episodes
        'eval_rewards': []  # Evaluation rewards (40 measurements)
    }
    
    # Training loop for both agents
    for episode in range(num_episodes):
        # Train and evaluate agent with dropout
        reward_with = train_episode(agent_with_dropout, env, episode)
        results_with_dropout['rewards'].append(reward_with)
        
        # Train and evaluate agent without dropout
        reward_without = train_episode(agent_without_dropout, env, episode)
        results_without_dropout['rewards'].append(reward_without)
        
        # Evaluate both agents
        if episode % eval_interval == 0:
            eval_reward_with, _ = evaluate(agent_with_dropout, env, num_episodes=2)
            eval_reward_without, _ = evaluate(agent_without_dropout, env, num_episodes=2)
            
            results_with_dropout['eval_rewards'].append(eval_reward_with)
            results_without_dropout['eval_rewards'].append(eval_reward_without)
            
            print(f"\nEpisode {episode}")
            print(f"With Dropout - Eval Reward: {eval_reward_with:.3f}")
            print(f"Without Dropout - Eval Reward: {eval_reward_without:.3f}")
    
    # Generate final comparison plot
    compare_dropout_effects(results_with_dropout, results_without_dropout)

if __name__ == '__main__':
    main() 