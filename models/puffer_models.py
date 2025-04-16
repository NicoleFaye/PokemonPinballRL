"""
PufferLib-compatible models for Pokemon Pinball.
"""
import torch
import torch.nn as nn

import pufferlib.models


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
        )


class MLPPolicy(pufferlib.models.Default):
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
        super().__init__(env, hidden_size=hidden_size)
        
        # Custom encoder for Game Boy screen
        input_dim = int(torch.prod(torch.tensor(env.single_observation_space.shape)))
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(input_dim, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )


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