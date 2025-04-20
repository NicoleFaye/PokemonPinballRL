"""
PufferLib-compatible models for Pokemon Pinball.
"""
import torch
import torch.nn as nn

import pufferlib.models

# Import our custom network architectures
from models.networks import SmallCNNPolicy


class Recurrent(pufferlib.models.LSTMWrapper):
    """
    Recurrent wrapper for Pokemon Pinball policies.
    Adds LSTM layer for temporal memory.
    """
    
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        """
        Initialize the recurrent wrapper.
        
        Args:
            env: Environment
            policy: Policy to wrap
            input_size: Size of input to LSTM
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__(env, policy, input_size, hidden_size, num_layers)
        
    def forward_for_pufferlib_lstm(self, x, action=None):
        """
        Special forward method for non-recurrent inference paths.
        Used when LSTM state is not available/needed.
        
        Args:
            x: Input tensor of observations
            action: Optional actions tensor for training
            
        Returns:
            Tuple of (actions, logprobs, entropy, values) matching the format
            expected by clean_pufferl's evaluate function.
        """
        # Create initial zero state
        batch_size = x.shape[0]
        empty_state = (
            torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_size).to(x.device),
            torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_size).to(x.device)
        )
        
        # Use the standard forward from parent class
        hidden, value, _ = super().forward(x, empty_state)
        
        # Create categorical distribution from logits
        action_probs = torch.softmax(hidden, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            # Evaluation mode - sample actions
            actions = action_dist.sample()
            logprobs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            return actions, logprobs, entropy, value
        else:
            # Training mode - use provided actions
            # Reshape the action to match the batch size of our output
            if action.shape != (batch_size,):
                # Flatten the action tensor if needed
                action = action.reshape(-1)
            logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            return None, logprobs, entropy, value


class CNNPolicy(pufferlib.models.Convolutional):
    """
    CNN policy for Pokemon Pinball.
    Uses the same architecture as PufferLib's Atari policy.
    """
    
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
                 framestack=4, flat_size=64*5*6):
        """
        Initialize the CNN policy.
        
        Args:
            env: Environment
            input_size: Input size to the network
            hidden_size: Hidden layer size
            output_size: Output size from the network
            framestack: Number of frames stacked
            flat_size: Size of the flattened CNN output
        """
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            # Game Boy screen data is channels-last
            channels_last=True,
            # Add downsampling to reduce resolution further
            downsample=2  # Downsample by factor of 2
        )
        
    def forward(self, observations, action=None):
        """
        Forward pass through the network.
        
        Args:
            observations: Input observations tensor
            action: Optional actions tensor for training
        
        Returns:
            Different return values depending on mode:
            - During evaluation: (actions, logprobs, entropy, value)
            - During training with provided actions: (None, logprobs, entropy, value)
        """
        # Call the parent class's forward method which returns (logits, value)
        logits, value = super().forward(observations)
        
        # Create categorical distribution
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Handle different modes
        if action is None:
            # Evaluation mode - sample actions
            actions = action_dist.sample()
            logprobs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            return actions, logprobs, entropy, value
        else:
            # Training mode - use provided actions
            logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            return None, logprobs, entropy, value


class MLPPolicy(nn.Module):
    """
    MLP policy for Pokemon Pinball.
    Simple network for flattened observations.
    """
    
    def __init__(self, env, hidden_size=512):
        """
        Initialize the MLP policy.
        
        Args:
            env: Environment
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        # Custom encoder for Game Boy screen (flattens 3D tensor)
        input_dim = int(torch.prod(torch.tensor(env.single_observation_space.shape)))
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        
        # Actor and critic heads
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0)
            
    def forward(self, observations, action=None):
        """
        Forward pass through the network.
        
        Args:
            observations: Input observations tensor
            action: Optional actions tensor for training
        
        Returns:
            Different return values depending on mode:
            - During evaluation: (actions, logprobs, entropy, value)
            - During training with provided actions: (None, logprobs, entropy, value)
        """
        # Flatten observations if needed
        if len(observations.shape) > 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)
        
        # Normalize if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        # Extract features
        features = self.encoder(observations)
        
        # Get action logits and value
        logits = self.actor(features)
        value = self.critic(features)
        
        # Create categorical distribution
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Handle different modes
        if action is None:
            # Evaluation mode - sample actions
            actions = action_dist.sample()
            logprobs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            return actions, logprobs, entropy, value
        else:
            # Training mode - use provided actions
            logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            return None, logprobs, entropy, value
        
    def encode_observations(self, observations):
        """Encode observations for PufferLib compatibility."""
        # Flatten observations if needed
        if len(observations.shape) > 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)
        
        # Normalize if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        # Extract features
        features = self.encoder(observations)
        return features, None
        
    def decode_actions(self, features, lookup):
        """Decode actions for PufferLib compatibility."""
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class ResNetPolicy(pufferlib.models.ProcgenResnet):
    """
    ResNet policy for Pokemon Pinball.
    More advanced architecture for better feature extraction.
    """
    
    def __init__(self, env, cnn_width=16, mlp_width=512):
        """
        Initialize the ResNet policy.
        
        Args:
            env: Environment
            cnn_width: Width of CNN layers
            mlp_width: Width of MLP layers
        """
        super().__init__(
            env=env,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
        )
        
    def forward(self, observations, action=None):
        """
        Forward pass through the network.
        
        Args:
            observations: Input observations tensor
            action: Optional actions tensor for training
        
        Returns:
            Different return values depending on mode:
            - During evaluation: (actions, logprobs, entropy, value)
            - During training with provided actions: (None, logprobs, entropy, value)
        """
        # Call the parent class's forward method which returns (logits, value)
        logits, value = super().forward(observations)
        
        # Create categorical distribution
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Handle different modes
        if action is None:
            # Evaluation mode - sample actions
            actions = action_dist.sample()
            logprobs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            return actions, logprobs, entropy, value
        else:
            # Training mode - use provided actions
            logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            return None, logprobs, entropy, value