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

    def __init__(self, log_dir: str, verbose: int = 0, window_size: int = 100, log_interval: int = 5000):
        """
        Initialize the callback.
        
        Args:
            log_dir: Path to the log directory
            verbose: Verbosity level
            window_size: Size of the rolling window for averaging metrics
            log_interval: Interval (in timesteps) for logging aggregated metrics
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.window_size = window_size
        self.log_interval = log_interval
        
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
        
        # Track cumulative metrics
        self.total_pokemon_caught = 0
        self.total_score = 0  # Added back for avg_score_per_episode
        
        # Keep track of step-to-episode conversion
        self.steps_per_episode = []
        
        # Track the best episodes for highlighting
        self.top_episodes = []  # List of (episode_number, score) tuples

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
                    
                    # Update total Pokemon caught and score
                    self.total_pokemon_caught += pokemon_caught
                    self.total_score += score
                    
                    # Track steps per episode for conversion
                    self.steps_per_episode.append(episode_length)
                    
                    # Log basic tracking metrics
                    self.logger.record("episode_tracking/total_episodes_completed", self.episode_count)
                    self.logger.record("episode_tracking/avg_env_timesteps_per_episode", 
                                     self.total_timesteps / max(1, self.episode_count))
                    self.logger.record("episode_tracking/total_pokemon_caught", self.total_pokemon_caught)
                    self.logger.record("episode_tracking/avg_pokemon_per_episode", 
                                     self.total_pokemon_caught / max(1, self.episode_count))
                    self.logger.record("episode_tracking/avg_score_per_episode", 
                                     self.total_score / max(1, self.episode_count))
                    
                    # Only log individual episode data periodically to reduce noise
                    if self.episode_count % 10 == 0:
                        # Log with episode number as x-axis value
                        self.logger.record("episodes/score", score, self.episode_count)
                        self.logger.record("episodes/length", episode_length, self.episode_count)
                        self.logger.record("episodes/reward", episode_return, self.episode_count)
                        self.logger.record("episodes/pokemon_caught", pokemon_caught, self.episode_count)
                        self.logger.record("episodes/ball_upgrades", total_ball_upgrades, self.episode_count)
                    
                    # Track top episodes
                    self.top_episodes.append((self.episode_count, score))
                    self.top_episodes.sort(key=lambda x: x[1], reverse=True)
                    self.top_episodes = self.top_episodes[:10]
                    
                    # Track current high score
                    if len(self.top_episodes) > 0:
                        self.logger.record("performance/all_time_high_game_score", max(x[1] for x in self.top_episodes))
                    
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
        
        # Log rolling averages at fixed intervals only
        if self.global_step % self.log_interval == 0 and len(self.score_buffer) >= 2:
            avg_episode_length = np.mean(self.episode_length_buffer)
            avg_score = np.mean(self.score_buffer)
            avg_rewards = np.mean(self.rewards_buffer)
            avg_pokemon_caught = np.mean(self.pokemon_caught_buffer)
            avg_ball_upgrades = np.mean(self.ball_upgrades_buffer)
            
            # Log rolling averages with consistent window size in name
            self.logger.record(f"rolling_averages/avg_episode_length_per_{self.window_size}_episodes", avg_episode_length)
            self.logger.record(f"rolling_averages/avg_game_score_per_{self.window_size}_episodes", avg_score)
            self.logger.record(f"rolling_averages/avg_reward_per_{self.window_size}_episodes", avg_rewards)
            self.logger.record(f"rolling_averages/avg_pokemon_caught_per_{self.window_size}_episodes", avg_pokemon_caught)
            self.logger.record(f"rolling_averages/avg_ball_upgrades_per_{self.window_size}_episodes", avg_ball_upgrades)
            
            # Calculate percentiles if we have enough data
            if len(self.score_buffer) >= 10:
                score_p10 = np.percentile(self.score_buffer, 10)  # 10th percentile (weakest episodes)
                score_p50 = np.percentile(self.score_buffer, 50)  # Median
                score_p90 = np.percentile(self.score_buffer, 90)  # 90th percentile (best episodes)
                
                # Record percentiles under performance with clear names
                self.logger.record("performance/game_score_bottom_10pct", score_p10)
                self.logger.record("performance/game_score_median", score_p50)
                self.logger.record("performance/game_score_top_10pct", score_p90)
                
                # Log max score in the same window
                max_score_in_window = np.max(self.score_buffer)
                self.logger.record(f"rolling_averages/max_game_score_per_{self.window_size}_episodes", max_score_in_window)
        
        return True
    
    def _on_training_end(self):
        """Clean up when training ends."""
        # No need to close writer as we're using the logger's writer
        pass