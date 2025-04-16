"""
PufferLib-compatible environment for Pokemon Pinball.
"""
import functools
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pufferlib
import pufferlib.emulation
import pufferlib.spaces
import pufferlib.postprocess

# Check for PyBoy availability
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False

from environment.pokemon_pinball_env import PokemonPinballEnv, RewardShaping


def env_creator(name='pokemon_pinball'):
    """
    PufferLib environment creator function.
    
    Args:
        name: Environment name (unused but kept for compatibility)
        
    Returns:
        Function to create environment
    """
    return functools.partial(make, name)


def make(name, rom_path=None, headless=True, reward_shaping="comprehensive", 
         frame_skip=4, framestack=4, render_mode="rgb_array", buf=None):
    """
    Create a PufferLib-compatible Pokemon Pinball environment.
    
    Args:
        name: Environment name (unused but kept for compatibility)
        rom_path: Path to Pokemon Pinball ROM file
        headless: Whether to run headless
        reward_shaping: Reward shaping strategy
        frame_skip: Number of frames to skip
        framestack: Number of frames to stack
        render_mode: Rendering mode
        buf: Optional buffer for PufferLib
        
    Returns:
        PufferLib-compatible environment
    """
    if not PYBOY_AVAILABLE:
        raise ImportError("PyBoy is not installed. Please install it with 'pip install pyboy'.")
        
    if rom_path is None:
        raise ValueError("ROM path must be provided")
        
    # Determine window type based on render mode
    window_type = "null" if headless else "SDL2"
    
    # Initialize PyBoy
    pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
    
    # Determine reward shaping function
    if isinstance(reward_shaping, str):
        if reward_shaping == "basic":
            reward_fn = RewardShaping.basic
        elif reward_shaping == "catch_focused":
            reward_fn = RewardShaping.catch_focused
        elif reward_shaping == "comprehensive":
            reward_fn = RewardShaping.comprehensive
        else:
            reward_fn = None
    else:
        reward_fn = reward_shaping
    
    # Create base environment
    env = PokemonPinballEnv(
        pyboy=pyboy,
        debug=not headless,
        headless=headless,
        reward_shaping=reward_fn,
        info_level=2
    )
    
    # Apply frame skip wrapper if needed
    if frame_skip > 1:
        from environment.wrappers import SkipFrame
        env = SkipFrame(env, skip=frame_skip)
    
    # Add render wrapper for PufferLib compatibility
    env = PinballRenderWrapper(env)
    
    # Apply frame stacking if specified
    if framestack > 1:
        env = gym.wrappers.FrameStack(env, framestack)
    
    # Apply PufferLib postprocessing
    env = PinballPostprocessor(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    
    # Convert to PufferLib env
    return PinballPufferEnv(env=env, num_agents=1, buf=buf)


class PinballRenderWrapper(gym.Wrapper):
    """Wrapper to make Pokemon Pinball compatible with PufferLib rendering."""
    
    def __init__(self, env):
        """Initialize the render wrapper."""
        super().__init__(env)
        
    @property
    def render_mode(self):
        """Return the render mode."""
        return 'rgb_array'
    
    def render(self):
        """
        Render the current state of the game.
        
        Returns:
            NumPy array of the game screen
        """
        # Get the current screen from PyBoy as a PIL image
        if hasattr(self.env.unwrapped, 'pyboy'):
            return np.array(self.env.unwrapped.pyboy.screen_image())
        return np.zeros((144, 160, 3), dtype=np.uint8)  # Fallback


class PinballPostprocessor(gym.Wrapper):
    """
    Postprocessor for Pokemon Pinball observations.
    Handles observation transformation for PufferLib.
    """
    
    def __init__(self, env):
        """Initialize the postprocessor."""
        super().__init__(env)
        # Get observation shape
        shape = env.observation_space.shape
        
        # Handle frame-stacked observations
        if len(shape) >= 3:  # (stack, height, width) or (height, width, stack)
            if shape[-1] in [1, 3, 4]:  # Channels-last format
                # Convert to channels-first (PufferLib standard)
                self.channels_last = True
                shape = (shape[-1], *shape[:-1])
            else:
                # Already in channels-first format
                self.channels_last = False
        else:  # Single frame (height, width)
            # Add channel dimension
            self.channels_last = False
            shape = (1, *shape)
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
    
    def _process_obs(self, obs):
        """Process observation to channels-first format."""
        if isinstance(obs, tuple):  # Handle frame stacks from gym
            obs = np.array(obs)
            
        # Handle FrameStack wrapper output - already in (stack, h, w) format
        if len(obs.shape) == 3 and obs.shape[0] == 4:  # 4 is the framestack value
            # This is already in correct (stack, h, w) format for the CNN
            return obs
            
        # Convert to channels-first format if needed
        if self.channels_last and len(obs.shape) > 2:
            if len(obs.shape) == 3:  # HWC -> CHW
                return np.transpose(obs, (2, 0, 1))
            else:
                # Unknown format, just return as is
                return obs
        elif len(obs.shape) == 2:  # HW -> CHW
            return np.expand_dims(obs, 0)
        
        return obs
    
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminal, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminal, truncated, info


class PinballPufferEnv(pufferlib.environment.PufferEnv):
    """
    PufferLib-compatible environment for Pokemon Pinball.
    Interfaces directly with PufferLib's system.
    """
    
    def __init__(self, env, num_agents=1, buf=None):
        """
        Initialize the PufferLib environment.
        
        Args:
            env: The base environment
            num_agents: Number of agents
            buf: Optional buffer for PufferLib
        """
        self.env = env
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.num_agents = num_agents
        
        # Initialize PufferEnv
        super().__init__(buf=buf)
        
        # Track episode returns and lengths
        self.episode_returns = [0.0] * num_agents
        self.episode_lengths = [0] * num_agents
        
    def reset(self, seed=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            
        Returns:
            Tuple of (observations, info)
        """
        # Handle seed properly - if it's a list, extract a single value
        if isinstance(seed, list):
            if len(seed) > 0:
                seed = seed[0]
            else:
                seed = None
                
        # Reset the environment
        obs, info = self.env.reset(seed=seed)
        
        # Store the observation in the buffer
        self.observations[0] = obs
        
        # Reset episode tracking
        self.episode_returns = [0.0] * self.num_agents
        self.episode_lengths = [0] * self.num_agents
        
        return self.observations, [info]
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: Actions for all agents
            
        Returns:
            Tuple of (observations, rewards, terminals, truncated, infos)
        """
        # Take the step with the first agent's action
        obs, reward, done, truncated, info = self.env.step(actions[0])
        
        # Store results in buffers
        self.observations[0] = obs
        self.rewards[0] = reward
        self.terminals[0] = done
        self.truncations[0] = truncated
        
        # Update episode tracking
        self.episode_returns[0] += reward
        self.episode_lengths[0] += 1
        
        # If episode ended, add to info
        if done or truncated:
            info['episode'] = {
                'r': self.episode_returns[0],
                'l': self.episode_lengths[0]
            }
            # Reset tracking
            self.episode_returns[0] = 0.0
            self.episode_lengths[0] = 0
        
        return self.observations, self.rewards, self.terminals, self.truncations, [info]
    
    def close(self):
        """Close the environment."""
        self.env.close()