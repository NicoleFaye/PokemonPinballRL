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
         frame_skip=4, framestack=4, render_mode="rgb_array", visual_mode="game_area",
         reduce_screen_resolution=True, buf=None):
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
        visual_mode: Visual observation mode ("game_area" or "screen")
        reduce_screen_resolution: Whether to downsample screen images
        buf: Optional buffer for PufferLib
        
    Returns:
        PufferLib-compatible environment
    """
    if not PYBOY_AVAILABLE:
        raise ImportError("PyBoy is not installed. Please install it with 'pip install pyboy'.")
        
    if rom_path is None:
        raise ValueError("ROM path must be provided")
        
    
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
    
    config = {
        "debug" : False,
        "headless": headless,
        "reward_shaping": reward_fn,
        "frame_skip": frame_skip,
        "framestack": framestack,
        "render_mode": render_mode,
        "info_level": 1,
        "frame_stack": framestack,
        "visual_mode": visual_mode,
        "frame_stack_extra_observation": False,
        "reduce_screen_resolution": reduce_screen_resolution,
    }


    # Create base environment
    env = PokemonPinballEnv(rom_path,config)
    
    # Add render wrapper for PufferLib compatibility
    env = PinballRenderWrapper(env)
    
    # Apply PufferLib postprocessing
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
        
    def step(self, action):
        """Pass through step with dictionary observation intact."""
        return self.env.step(action)
        
    def reset(self, **kwargs):
        """Pass through reset with dictionary observation intact."""
        return self.env.reset(**kwargs)



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
        # Extract the game_area space from the Dict space
        if isinstance(env.observation_space, spaces.Dict) and 'game_area' in env.observation_space.spaces:
            self.single_observation_space = env.observation_space.spaces['game_area']
        else:
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
        
        # Store the observation in the buffer - extract game_area if using Dict observation space
        if isinstance(obs, dict) and 'game_area' in obs:
            # The game_area should now be correctly frame-stacked by the base environment
            self.observations[0] = obs['game_area']
        else:
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
        
        # Store results in buffers - extract game_area if using Dict observation space
        if isinstance(obs, dict) and 'game_area' in obs:
            # The game_area should now be correctly frame-stacked by the base environment
            self.observations[0] = obs['game_area']
        else:
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