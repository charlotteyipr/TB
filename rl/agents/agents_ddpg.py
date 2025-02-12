import torch
import numpy as np
import random
from ..models.ddpg import Actor, Critic  # Import Actor and Critic from models module

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout_rate=0.1):
        """Initialize DDPG agent with Monte Carlo dropout
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability for uncertainty estimation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Create Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, dropout_rate)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim, dropout_rate)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)
        
        # Copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(1000000)
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.001
        
        # Exploration noise
        self.noise = OUNoise(action_dim)
        
        # Adjust exploration parameters for better stability
        self.uncertainty_threshold = 0.03   # Lower threshold for more stable exploration
        self.min_noise_scale = 0.01        # Reduce noise range
        self.max_noise_scale = 0.05        # Reduce maximum noise
        
    def select_action(self, state, explore=True):
        """Select action with adaptive exploration based on uncertainty
        
        Args:
            state: Current state
            explore: Whether to explore or exploit
            
        Returns:
            action: Selected action
            uncertainty_level: Estimated uncertainty
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if explore:
            action, uncertainty = self.actor.get_action_with_uncertainty(state)
            uncertainty_level = torch.mean(uncertainty).item()
            
            # Improve adaptive exploration strategy
            if uncertainty_level > self.uncertainty_threshold:
                noise_scale = min(self.max_noise_scale, 
                                self.min_noise_scale + uncertainty_level)
                self.actor_optimizer.param_groups[0]['lr'] *= 0.98  # Gentle learning rate adjustment
            else:
                noise_scale = max(self.min_noise_scale,
                                self.max_noise_scale * (uncertainty_level / self.uncertainty_threshold))
                self.actor_optimizer.param_groups[0]['lr'] *= 1.02
            
            noise = noise_scale * self.noise.sample()
            action = action + torch.FloatTensor(noise)
            action = torch.clamp(action, -1, 1)
            
            return action.detach().cpu().numpy().flatten(), uncertainty_level
        else:
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state)
            self.actor.train()
            return action.detach().cpu().numpy().flatten(), 0.0
    
    def update(self):
        """Update networks"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample from experience replay
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(self.batch_size)
            
        # Convert to tensor and normalize
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)
        
        # Normalize rewards to avoid numerical instability
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
        
        # Calculate target Q values
        with torch.no_grad():
            next_action_batch = self.target_actor(next_state_batch)
            next_q_values = self.target_critic(next_state_batch, next_action_batch)
            target_q_batch = reward_batch + self.gamma * (1 - done_batch) * next_q_values
        
        # Update Critic
        current_q_batch = self.critic(state_batch, action_batch)
        critic_loss = torch.nn.functional.mse_loss(current_q_batch, target_q_batch)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        # Adjust update strategy based on average uncertainty
        with torch.no_grad():
            states = torch.FloatTensor(np.array([x[0] for x in self.memory.buffer[-100:]]))
            _, uncertainties = self.actor.get_action_with_uncertainty(states)
            mean_uncertainty = torch.mean(uncertainties).item()
            
        # Adjust target network update rate with numerical stability
        if mean_uncertainty > self.uncertainty_threshold:
            self.tau = max(self.tau * 0.95, 0.001)  # Set lower bound
        else:
            self.tau = min(self.tau * 1.05, 0.1)    # Set upper bound
    
    def _soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """Add batch of experiences to buffer"""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.push(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state 