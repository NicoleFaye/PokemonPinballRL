"""Neural network models for Pokemon Pinball."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SmallCNNPolicy(nn.Module):
    """
    Small CNN policy optimized for the tiny game area (16x20) in Pokemon Pinball.
    Uses smaller kernel sizes and strides to handle the small input dimensions.
    """
    
    def __init__(self, env, hidden_size=512, cnn_channels=16):
        """
        Initialize the Small CNN policy.
        
        Args:
            env: The environment
            hidden_size: Size of the hidden layer
            cnn_channels: Number of channels in CNN layers
        """
        super().__init__()
        
        # Extract observation shape
        obs_shape = env.observation_space.shape
        
        # Determine if we have a frame-stacked observation (height, width, frames)
        if len(obs_shape) == 3:
            height, width, frame_stack = obs_shape
        else:
            height, width = obs_shape
            frame_stack = 1
            
        # Small CNN feature extractor with small kernels and strides
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack, cnn_channels, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the feature size after convolution
        # With the above architecture, a 16x20 input becomes:
        # Conv1 (k=3,s=1,p=1): 16x20 -> 16x20
        # Conv2 (k=3,s=1,p=1): 16x20 -> 16x20
        # Conv3 (k=3,s=1,p=0): 16x20 -> 14x18
        # Final output: (cnn_channels*2) x 14 x 18
        conv_output_size = (cnn_channels * 2) * 14 * 18
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(conv_output_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, env.action_space.n), std=0.01)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(conv_output_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
            
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (B, H, W, C) - channels last format from game_area
            
        Returns:
            Tuple of (logits, value)
        """
        # x is in shape (batch, height, width, channels) - convert to channels first
        x = x.permute(0, 3, 1, 2)
        
        # Normalize input to [0, 1]
        x = x.float() / 255.0
            
        # Extract features
        features = self.features(x)
        
        # Compute action logits and value
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value


class CNNPolicy(nn.Module):
    """
    CNN policy network for processing game images.
    
    This is a convolutional neural network designed for visual
    observation spaces, with a structure similar to the one used
    in the original DQN paper for Atari games.
    """
    
    def __init__(self, env, hidden_size=512, cnn_channels=32):
        """
        Initialize the CNN policy.
        
        Args:
            env: The environment
            hidden_size: Size of the hidden layer
            cnn_channels: Number of channels in CNN layers
        """
        super().__init__()
        
        # Extract observation shape
        obs_shape = env.observation_space.shape
        
        # Determine if we have a frame-stacked observation
        if len(obs_shape) == 3:  # (num_frames, height, width)
            frame_stack = obs_shape[0]
            height, width = obs_shape[1], obs_shape[2]
        else:  # (height, width)
            frame_stack = 1
            height, width = obs_shape[0], obs_shape[1]
            
        # CNN feature extractor
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack, cnn_channels, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the feature size after convolution
        conv_output_size = self._get_conv_output(frame_stack, height, width)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(conv_output_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, env.action_space.n), std=0.01)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(conv_output_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
        
    def _get_conv_output(self, frame_stack, height, width):
        """
        Calculate the output size of the CNN layers.
        
        Args:
            frame_stack: Number of stacked frames
            height: Input height
            width: Input width
            
        Returns:
            Size of the flattened CNN output
        """
        with torch.no_grad():
            return self.features(torch.zeros(1, frame_stack, height, width)).shape[1]
            
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, value)
        """
        # Permute if necessary (only if input is in (batch, frame_stack, height, width) format)
        if len(x.shape) == 4:
            # Ensure channels-first format for CNN
            pass
        elif len(x.shape) == 3:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            
        # Normalize input to [0, 1]
        x = x / 255.0
            
        # Extract features
        features = self.features(x)
        
        # Compute action logits and value
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value


class MLPPolicy(nn.Module):
    """
    MLP policy network for processing flattened observations.
    
    This is a simple multi-layer perceptron for handling
    flattened observation spaces.
    """
    
    def __init__(self, env, hidden_size=256, activation_fn=nn.ReLU):
        """
        Initialize the MLP policy.
        
        Args:
            env: The environment
            hidden_size: Size of hidden layers
            activation_fn: Activation function to use
        """
        super().__init__()
        
        # Get input dimension
        obs_dim = np.prod(env.observation_space.shape)
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, env.action_space.n), std=0.01)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (logits, value)
        """
        # Ensure input is properly shaped
        if len(x.shape) > 2:
            # Flatten if not already
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 1:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            
        # Normalize input to [0, 1] if it's not already
        if x.max() > 1.0:
            x = x / 255.0
            
        # Compute action logits and value
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value


class RecurrentPolicy(nn.Module):
    """
    Recurrent policy network for handling sequential data.
    
    This policy uses an LSTM layer to maintain state across time steps,
    which can be helpful for partially observable environments.
    """
    
    def __init__(self, env, hidden_size=256, recurrent_layers=1):
        """
        Initialize the recurrent policy.
        
        Args:
            env: The environment
            hidden_size: Size of hidden layers
            recurrent_layers: Number of recurrent layers
        """
        super().__init__()
        
        # Get input dimension
        self.obs_dim = np.prod(env.observation_space.shape)
        self.hidden_size = hidden_size
        
        # Feature extractor
        self.features = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_size)),
            nn.Tanh()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=recurrent_layers,
            batch_first=True
        )
        
        # Actor (policy) head
        self.actor = layer_init(nn.Linear(hidden_size, env.action_space.n), std=0.01)
        
        # Critic (value) head
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
                
    def forward(self, x, state=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            state: LSTM hidden state tuple (h, c)
            
        Returns:
            Tuple of (logits, value, new_state)
        """
        # Reshape input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Normalize input to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # Extract features
        x = self.features(x)
        
        # Reshape for LSTM input
        x = x.unsqueeze(1)  # Add sequence length dimension
        
        # LSTM forward pass
        if state is None:
            x, new_state = self.lstm(x)
        else:
            x, new_state = self.lstm(x, state)
            
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Compute action logits and value
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value, new_state