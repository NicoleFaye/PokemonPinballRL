"""
Environment wrappers for Pokemon Pinball.
"""
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from environment.pokemon_pinball_env import RewardShaping

# Set default for import check
PUFFERLIB_AVAILABLE = False

try:
    import pufferlib
    from pufferlib.environment import PufferEnv
    PUFFERLIB_AVAILABLE = True
except ImportError:
    pass


class NormalizedObservation(gym.ObservationWrapper):
    """
    Normalize observations to be in [0, 1] range and flatten them.
    This can be helpful for neural networks.
    """
    
    def __init__(self, env):
        """
        Initialize the normalization wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        
        normalized_spaces = {}
        
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Box) and space.dtype == np.uint8:
                # For image-like observations, normalize and flatten
                flat_dim = int(np.prod(space.shape))
                normalized_spaces[key] = spaces.Box(
                    low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
                )
            else:
                # Keep other spaces as they are
                normalized_spaces[key] = space
                
        self.observation_space = spaces.Dict(normalized_spaces)
    
    def observation(self, observation):
        """
        Normalize and flatten the observation.
        
        Args:
            observation: The raw observation
            
        Returns:
            Normalized, flattened observation
        """
        normalized_obs = {}
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray) and value.dtype == np.uint8:
                # Normalize to [0, 1] and flatten
                normalized = value.astype(np.float32) / 255.0
                normalized_obs[key] = normalized.flatten()
            else:
                # Keep other values as they are
                normalized_obs[key] = value
                
        return normalized_obs


class PufferWrapper(gym.Wrapper):
    """
    Wrapper to make Pokemon Pinball environment compatible with PufferLib.
    Handles dictionary observation spaces correctly.
    """
    
    def __init__(self, env, normalize_reward=True, clip_reward=10.0):
        """
        Initialize the PufferLib wrapper.
        
        Args:
            env: The base environment to wrap
            normalize_reward: Whether to normalize rewards
            clip_reward: Maximum absolute value for reward clipping
        """
        if not PUFFERLIB_AVAILABLE:
            raise ImportError(
                "pufferlib is not installed. "
                "Please install it with 'pip install pufferlib'."
            )
            
        super().__init__(env)
        self.normalize_reward = normalize_reward
        self.clip_reward = clip_reward
        self.reward_history = []
        self.returns = 0
        self.episode_length = 0
        
        # Check if we have a Dict observation space and update the observation space
        # PufferLib should handle Dict spaces natively
        
        # Store the PyBoy instance for proper cleanup
        if hasattr(env, 'pyboy'):
            self.pyboy_instance = env.pyboy
        else:
            # Try to get from unwrapped
            try:
                self.pyboy_instance = env.unwrapped.pyboy
            except:
                self.pyboy_instance = None
        
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        
        self.returns = 0
        self.episode_length = 0
        return obs, info
        
    def step(self, action):
        """
        Take a step with reward processing.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Process reward
        if self.normalize_reward:
            # Store reward for normalization
            self.reward_history.append(reward)
            if len(self.reward_history) > 1000:
                self.reward_history.pop(0)
                
            # Normalize reward if we have enough history
            if len(self.reward_history) > 10:
                reward_mean = np.mean(self.reward_history)
                reward_std = np.std(self.reward_history) + 1e-8  # Avoid division by zero
                reward = (reward - reward_mean) / reward_std
        
        # Clip reward if specified
        if self.clip_reward > 0:
            reward = np.clip(reward, -self.clip_reward, self.clip_reward)
            
        # Track returns and episode length
        self.returns += reward
        self.episode_length += 1
        
        # Add additional info - include both done and truncated conditions
        is_terminal = done or truncated
        info.update({
            'episode_return': self.returns if is_terminal else None,
            'episode_length': self.episode_length if is_terminal else None
        })
        
        # Reset tracking if episode is terminal
        if is_terminal:
            self.returns = 0
            self.episode_length = 0
        
        return obs, reward, done, truncated, info
        
    def close(self):
        """
        Close the environment and clean up resources.
        Ensures that PyBoy instances are properly closed.
        """
        # First close the wrapped environment
        try:
            self.env.close()
        except Exception as e:
            print(f"Error closing wrapped environment: {e}")
            
        # Also explicitly stop PyBoy if we have stored the instance
        if hasattr(self, 'pyboy_instance') and self.pyboy_instance is not None:
            try:
                if hasattr(self.pyboy_instance, 'tick_passed') and self.pyboy_instance.tick_passed > 0:
                    print("Explicitly stopping PyBoy instance")
                    self.pyboy_instance.stop()
            except Exception as e:
                print(f"Error stopping PyBoy instance: {e}")


class PokemonPinballPufferEnv(PufferEnv):
    """
    PufferLib environment for Pokemon Pinball.
    This implementation makes Pokemon Pinball compatible with PufferLib's
    native APIs for more efficient training.
    """
    
    def __init__(self, env_maker, env_kwargs=None, num_agents=1, **kwargs):
        """
        Initialize the PufferLib environment.
        
        Args:
            env_maker: Function that creates a single environment instance
            env_kwargs: Keyword arguments to pass to env_maker
            num_agents: Number of parallel environments to run
        """
        # Create the base environment using the provided function
        env_kwargs = env_kwargs or {}
        self.env = env_maker(**env_kwargs)
        
        # Set up observation and action spaces for PufferLib
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space
        self.num_agents = num_agents
        
        # Initialize the buffers for PufferLib
        super().__init__()
        
        # Track episodes for logging
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
        # Reset the environment
        obs, info = self.env.reset(seed=seed)
        
        # Handle dictionary observation space from Pokemon Pinball
        # No need to flatten - pufferlib can handle dictionary spaces directly
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
            Tuple of (observations, rewards, dones, truncated, infos)
        """
        # Take the step with the first agent's action
        obs, reward, done, truncated, info = self.env.step(actions[0])
        
        self.observations[0] = obs
        
        # Store other values
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


def create_env_factory(rom_path, debug=False, reward_shaping=None, frame_stack=4, skip_frames=4):
    """
    Create a factory function for environment instances.
    
    Args:
        rom_path: Path to ROM file
        debug: Enable debug mode
        reward_shaping: Reward shaping function
        frame_stack: Number of frames to stack
        skip_frames: Number of frames to skip
        
    Returns:
        Function that creates environment instances
    """
    from pyboy import PyBoy
    
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
        
    def env_factory(**kwargs):
        """Factory function to create environment instances."""
        # Setup base environment configuration
        info_level = kwargs.get("info_level", 2)  # Default to level 2 for more state information
        
        config = {
            "debug": debug,
            "headless": not debug,
            "reward_shaping": reward_fn,
            "info_level": info_level,
            "frame_stack": frame_stack,
            "frame_skip": skip_frames,
            "visual_mode": "game_area",
            "frame_stack_extra_observation": False,
            "reduce_screen_resolution": True,
        }
        
        # Setup base environment
        from environment.pokemon_pinball_env import PokemonPinballEnv
        
        env = PokemonPinballEnv(rom_path, config)
        
        # Customize based on kwargs
        policy_type = kwargs.get("policy_type", "cnn")
        
        if policy_type == "cnn":
            # For CNN policies, we want to preserve the 2D structure in the dict space
            pass
        else:
            # For MLP policies, we want to normalize and flatten
            from environment.wrappers import NormalizedObservation
            env = NormalizedObservation(env)
            
        # Apply PufferLib wrapper if requested
        if kwargs.get("use_puffer_wrapper", False):
            from environment.wrappers import PufferWrapper
            env = PufferWrapper(env)
            
        return env
        
    return env_factory


def make_puffer_env(env_factory, num_envs=4, **kwargs):
    """
    Create a vectorized environment compatible with PufferLib.
    
    Args:
        env_factory: Function that creates a single environment instance
        num_envs: Number of environments to run in parallel
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        A list of environments for PufferLib
    """
    if not PUFFERLIB_AVAILABLE:
        raise ImportError(
            "pufferlib is not installed. "
            "Please install it with 'pip install pufferlib'."
        )
    
    # Create base environments
    print(f"Creating {num_envs} environments for vectorized execution")
    envs = [PufferWrapper(env_factory(**kwargs)) for _ in range(num_envs)]
    
    # Return list of environments - PufferLib will handle vectorization
    return envs


def create_puffer_vectorized_env(env_factory, num_envs=4, **kwargs):
    """
    Create a native PufferLib vectorized environment.
    
    Args:
        env_factory: Function that creates environment instances
        num_envs: Number of environments to create
        **kwargs: Additional arguments to pass to the environments
        
    Returns:
        PufferLib vectorized environment
    """
    if not PUFFERLIB_AVAILABLE:
        raise ImportError(
            "pufferlib is not installed. "
            "Please install it with 'pip install pufferlib'."
        )
        
    import pufferlib.vector
    
    try:
        # Use PufferLib native vectorization with multiprocessing backend
        return pufferlib.vector.make(
            env_factory,
            env_kwargs=kwargs,
            num_envs=num_envs,
            backend=pufferlib.vector.Multiprocessing
        )
    except Exception as e:
        print(f"Error creating PufferLib vectorized environment: {e}")
        print("Falling back to manual environment creation")
        return make_puffer_env(env_factory, num_envs, **kwargs)