import os
import numpy as np
from typing import Dict, List, Any, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging environment metrics to TensorBoard.
    
    This callback tracks:
    - Episode length per episode
    - Score per episode
    - Rewards per episode
    - Number of Pokemon caught per episode
    - Number of ball upgrades per episode
    
    It calculates both individual episode values and rolling averages for cleaner visualization.
    Compatible with vectorized environments.
    """

    def __init__(self, log_dir: str, verbose: int = 0, window_size: int = 10):
        """
        Initialize the callback.
        
        Args:
            log_dir: Path to the log directory
            verbose: Verbosity level
            window_size: Size of the rolling window for averaging metrics
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.window_size = window_size
        
        # Initialize buffers for rolling averages
        self.episode_length_buffer = []
        self.score_buffer = []
        self.rewards_buffer = []
        self.pokemon_caught_buffer = []
        self.ball_upgrades_buffer = []
        
        # Counter for global steps and episodes
        self.global_step = 0
        self.episode_count = 0
        self.total_timesteps = 0
        
        # Keep track of step-to-episode conversion
        self.steps_per_episode = []

    def _on_training_start(self):
        """Initialize the model reference and custom panels when training starts."""
        # Get a reference to the model
        self.model = self.locals.get('self')
        
        # Add custom panel to WandB if available
        try:
            import wandb
            if wandb.run is not None:
                # Add a custom panel to make the step vs episode relationship clear
                wandb.run.config.update({
                    "IMPORTANT_NOTE": "In WandB graphs, 'Step' on x-axis means environment timesteps, not episodes"
                })
        except (ImportError, AttributeError):
            pass

    def _on_step(self) -> bool:
        """
        Called at each environment step during training.
        Checks if any environment completed an episode and logs metrics accordingly.
        """
        # Use the model num_timesteps as global step 
        if hasattr(self.model, 'num_timesteps'):
            self.global_step = self.model.num_timesteps
        else:
            self.global_step += 1
        
        # Update total timesteps
        self.total_timesteps = self.global_step
            
        # Check each environment for completed episodes
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            
            # Track episode completions
            episodes_this_step = 0
            
            # Process info for each environment that finished an episode
            for idx, done in enumerate(dones):
                if done:
                    # Increment episode counter
                    self.episode_count += 1
                    episodes_this_step += 1
                    
                    # Extract metrics for completed episode
                    info = infos[idx]
                    
                    # Get metrics from the info dictionary
                    # Note: In the environment, values are stored as lists (for PufferLib compatibility)
                    episode_length = info.get("episode_length", [0])[0]
                    score = info.get("score", [0])[0]
                    episode_return = info.get("episode_return", [0])[0]
                    pokemon_caught = info.get("pokemon_caught", [0])[0]
                    total_ball_upgrades = info.get("total_ball_upgrades", [0])[0]
                    
                    # Track steps per episode for conversion
                    self.steps_per_episode.append(episode_length)
                    
                    # Log conversion metrics with clearer names
                    self.logger.record("episode_tracking/total_episodes_so_far", self.episode_count)
                    self.logger.record("episode_tracking/episodes_completed_this_step", episodes_this_step)
                    self.logger.record("episode_tracking/timesteps_in_current_episode", episode_length)
                    if len(self.steps_per_episode) > 0:
                        self.logger.record("episode_tracking/avg_timesteps_per_episode", np.mean(self.steps_per_episode[-30:]))
                    
                    # Log individual episode values with clearer names
                    self.logger.record("episode_metrics/length_per_episode", episode_length)
                    self.logger.record("episode_metrics/score_per_episode", score)
                    self.logger.record("episode_metrics/reward_per_episode", episode_return)
                    self.logger.record("episode_metrics/pokemon_caught_per_episode", pokemon_caught)
                    self.logger.record("episode_metrics/ball_upgrades_per_episode", total_ball_upgrades)
                    
                    # Add to buffers for rolling averages
                    self.episode_length_buffer.append(episode_length)
                    self.score_buffer.append(score)
                    self.rewards_buffer.append(episode_return)
                    self.pokemon_caught_buffer.append(pokemon_caught)
                    self.ball_upgrades_buffer.append(total_ball_upgrades)
                    
                    # Keep buffers at window size
                    if len(self.episode_length_buffer) > self.window_size:
                        self.episode_length_buffer.pop(0)
                        self.score_buffer.pop(0)
                        self.rewards_buffer.pop(0)
                        self.pokemon_caught_buffer.pop(0)
                        self.ball_upgrades_buffer.pop(0)
                    
                    # Calculate and log rolling averages if we have at least 2 episodes
                    if len(self.episode_length_buffer) >= 2:
                        avg_episode_length = np.mean(self.episode_length_buffer)
                        avg_score = np.mean(self.score_buffer)
                        avg_rewards = np.mean(self.rewards_buffer)
                        avg_pokemon_caught = np.mean(self.pokemon_caught_buffer)
                        avg_ball_upgrades = np.mean(self.ball_upgrades_buffer)
                        
                        # Log rolling averages with clearer names
                        self.logger.record("rolling_averages/avg_episode_length", avg_episode_length)
                        self.logger.record("rolling_averages/avg_score_per_episode", avg_score)
                        self.logger.record("rolling_averages/avg_reward_per_episode", avg_rewards)
                        self.logger.record("rolling_averages/avg_pokemon_caught_per_episode", avg_pokemon_caught)
                        self.logger.record("rolling_averages/avg_ball_upgrades_per_episode", avg_ball_upgrades)
                        
                        # Histograms aren't showing up in WandB, so we'll skip them
                        
            # Log the overall relationship between steps and episodes
            if episodes_this_step > 0:
                steps_per_episode_ratio = self.total_timesteps / max(1, self.episode_count)
                self.logger.record("episode_tracking/total_environment_timesteps", self.total_timesteps)
                self.logger.record("episode_tracking/total_episodes_completed", self.episode_count)
                self.logger.record("episode_tracking/ratio_timesteps_per_episode", steps_per_episode_ratio)
        
        return True
    
    def _on_training_end(self):
        """Clean up when training ends."""
        # No need to close writer as we're using the logger's writer
        pass