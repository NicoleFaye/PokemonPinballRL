import os
import numpy as np
from typing import Dict, List, Any, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging environment metrics to TensorBoard.
    
    This callback tracks episode-based metrics including:
    - Episode length
    - Game score
    - Episode rewards
    - Number of Pokemon caught
    - Number of ball upgrades
    
    It calculates both individual episode values and rolling averages, and ensures
    that episode-based metrics use episode count as the x-axis in visualizations.
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
        self.total_score = 0
        
        # Keep track of step-to-episode conversion
        self.steps_per_episode = []
        
        # Track the best episodes
        self.top_episodes = []  # List of (episode_number, score) tuples

    def _on_training_start(self):
        """Initialize the model reference when training starts."""
        # Get a reference to the model
        self.model = self.locals.get('self')

    def _on_step(self) -> bool:
        """
        Called at each environment step during training.
        Checks if any environment completed an episode and logs metrics accordingly.
        """
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
                    # Note: In the environment, values are stored as lists 
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
                    
                    self.logger.record("episode_tracking/total_episodes_completed", self.episode_count)
                    self.logger.record("episode_tracking/avg_timesteps_per_episode", 
                                     self.total_timesteps / max(1, self.episode_count))
                    self.logger.record("episode_tracking/total_pokemon_caught", self.total_pokemon_caught)
                    self.logger.record("episode_tracking/avg_pokemon_per_episode", 
                                     self.total_pokemon_caught / max(1, self.episode_count))
                    self.logger.record("episode_tracking/avg_score_per_episode", 
                                     self.total_score / max(1, self.episode_count))
                    
                    self.logger.record("episode_metrics/score_per_episode", score, self.episode_count)
                    self.logger.record("episode_metrics/length_per_episode", episode_length, self.episode_count)
                    self.logger.record("episode_metrics/reward_per_episode", episode_return, self.episode_count)
                    self.logger.record("episode_metrics/pokemon_caught_per_episode", pokemon_caught, self.episode_count)
                    self.logger.record("episode_metrics/ball_upgrades_per_episode", total_ball_upgrades, self.episode_count)
                    
                    # Additionally, record the current episode data without x-axis for SB3's default logging
                    # These will use global step as x-axis in TensorBoard
                    self.logger.record("timestep_metrics/score", score)
                    self.logger.record("timestep_metrics/episode_length", episode_length)
                    self.logger.record("timestep_metrics/episode_reward", episode_return)
                    
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
        
        # Log rolling averages at fixed intervals
        if len(self.score_buffer) >= 2:
            # Calculate rolling averages
            avg_episode_length = np.mean(self.episode_length_buffer)
            avg_score = np.mean(self.score_buffer)
            avg_rewards = np.mean(self.rewards_buffer)
            avg_pokemon_caught = np.mean(self.pokemon_caught_buffer)
            avg_ball_upgrades = np.mean(self.ball_upgrades_buffer)
            
            # Log rolling averages with episode count as x-axis
            curr_episode = self.episode_count
            
            # Log rolling averages with consistent window size in name
            self.logger.record(f"rolling_averages/avg_episode_length_per_{self.window_size}_episodes", 
                             avg_episode_length, curr_episode)
            self.logger.record(f"rolling_averages/avg_game_score_per_{self.window_size}_episodes", 
                             avg_score, curr_episode)
            self.logger.record(f"rolling_averages/avg_reward_per_{self.window_size}_episodes", 
                             avg_rewards, curr_episode)
            self.logger.record(f"rolling_averages/avg_pokemon_caught_per_{self.window_size}_episodes", 
                             avg_pokemon_caught, curr_episode)
            self.logger.record(f"rolling_averages/avg_ball_upgrades_per_{self.window_size}_episodes", 
                             avg_ball_upgrades, curr_episode)
            
            # Calculate percentiles if we have enough data
            if len(self.score_buffer) >= 10:
                score_p10 = np.percentile(self.score_buffer, 10)  # 10th percentile (weakest episodes)
                score_p50 = np.percentile(self.score_buffer, 50)  # Median
                score_p90 = np.percentile(self.score_buffer, 90)  # 90th percentile (best episodes)
                
                # Record percentiles with episode count as x-axis
                self.logger.record("performance/game_score_bottom_10pct", score_p10, curr_episode)
                self.logger.record("performance/game_score_median", score_p50, curr_episode)
                self.logger.record("performance/game_score_top_10pct", score_p90, curr_episode)
                
                # Log max score in the window
                max_score_in_window = np.max(self.score_buffer)
                self.logger.record(f"rolling_averages/max_game_score_per_{self.window_size}_episodes", 
                                 max_score_in_window, curr_episode)
                
                # Also log the same metrics with global step as x-axis for compatibility
                self.logger.record("timestep_metrics/rolling_avg_score", avg_score)
                self.logger.record("timestep_metrics/rolling_avg_reward", avg_rewards)
        
        return True
    
    def _on_training_end(self):
        """Clean up when training ends."""
        # No additional cleanup needed when using SB3's logger
        pass