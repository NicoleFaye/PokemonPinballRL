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
        
        # Counter for global steps
        self.global_step = 0

    def _on_training_start(self):
        """Initialize the TensorBoard writer when training starts."""
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'metrics'))

    def _on_step(self) -> bool:
        """
        Called at each environment step during training.
        Checks if any environment completed an episode and logs metrics accordingly.
        """
        # Increment global step counter
        self.global_step += 1
        
        # Check each environment for completed episodes
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            
            # Process info for each environment that finished an episode
            for idx, done in enumerate(dones):
                if done:
                    # Extract metrics for completed episode
                    info = infos[idx]
                    
                    # Get metrics from the info dictionary
                    # Note: In the environment, values are stored as lists (for PufferLib compatibility)
                    episode_length = info.get("episode_length", [0])[0]
                    score = info.get("score", [0])[0]
                    episode_return = info.get("episode_return", [0])[0]
                    pokemon_caught = info.get("pokemon_caught", [0])[0]
                    total_ball_upgrades = info.get("total_ball_upgrades", [0])[0]
                    
                    # Log individual episode values
                    self.logger.record("metrics/episode_length", episode_length)
                    self.logger.record("metrics/score", score)
                    self.logger.record("metrics/episode_return", episode_return)
                    self.logger.record("metrics/pokemon_caught", pokemon_caught)
                    self.logger.record("metrics/ball_upgrades", total_ball_upgrades)
                    
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
                        
                        self.logger.record("metrics/avg_episode_length", avg_episode_length)
                        self.logger.record("metrics/avg_score", avg_score)
                        self.logger.record("metrics/avg_episode_return", avg_rewards)
                        self.logger.record("metrics/avg_pokemon_caught", avg_pokemon_caught)
                        self.logger.record("metrics/avg_ball_upgrades", avg_ball_upgrades)
                        
                        # Add histograms to visualize distributions
                        if self.writer:
                            self.writer.add_histogram("metrics_dist/episode_length", 
                                                    np.array(self.episode_length_buffer), 
                                                    self.global_step)
                            self.writer.add_histogram("metrics_dist/score", 
                                                    np.array(self.score_buffer), 
                                                    self.global_step)
                            self.writer.add_histogram("metrics_dist/episode_return", 
                                                    np.array(self.rewards_buffer), 
                                                    self.global_step)
        
        return True
    
    def _on_training_end(self):
        """Clean up when training ends."""
        if self.writer:
            self.writer.close()