import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout_rate=0.1):
        """Initialize Actor network with dropout layers
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.ln3 = nn.LayerNorm(hidden_dim//2)
        self.dropout3 = nn.Dropout(p=dropout_rate/2)
        
        self.residual = nn.Linear(state_dim, hidden_dim//2)
        
        self.fc4 = nn.Linear(hidden_dim//2, action_dim)
        
        self.training_mode = True
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        
        residual = F.relu(self.residual(state))
        x = x + residual
        
        x = F.relu(x)
        x = self.dropout3(x)
        
        action = torch.tanh(self.fc4(x))
        
        return action
    
    def set_training_mode(self, mode=True):
        """Set whether to enable dropout for uncertainty estimation"""
        self.training_mode = mode
        
    def get_action_with_uncertainty(self, state, num_samples=10):
        """Estimate action uncertainty through multiple sampling
        
        Args:
            state: Current state
            num_samples: Number of Monte Carlo samples
            
        Returns:
            mean_action: Average action across samples
            uncertainty: Standard deviation of actions
        """
        if not self.training_mode:
            self.train()  # Enable dropout for uncertainty estimation
            
        actions = []
        for _ in range(num_samples):
            action = self.forward(state)
            actions.append(action)
            
        actions = torch.stack(actions)
        mean_action = actions.mean(dim=0)
        uncertainty = actions.std(dim=0)
        
        return mean_action, uncertainty

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value 

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset() 