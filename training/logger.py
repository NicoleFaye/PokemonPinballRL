"""
Logging utilities for Pokemon Pinball RL training.
"""
import datetime
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class MetricLogger:
    """Logs metrics during training and generates plots."""
    
    def __init__(self, log_dir, resume=False, metadata=None, max_history=None, json_save_freq=10000):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs and plots
            resume: Whether to resume logging from existing files
            metadata: Dictionary of metadata to include in log files
            max_history: Maximum history length to keep in memory
            json_save_freq: Frequency (in steps) to save metrics JSON file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frequency for metrics file (in steps)
        self.json_save_freq = json_save_freq
        
        # Current episode count
        self.episode_count = 0
        
        # Maximum history to keep in memory
        self.max_history = max_history
        
        # Setup metrics tracking
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        
        # Moving averages
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        
        # Keep track of step metrics
        self.step_losses = []
        self.step_q_values = []
        self.metrics_path = self.log_dir / "metrics.json"
        
        # Buffers for the current episode
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_q_length = 0
        
        # Metadata about this run
        self.metadata = {
            'start_time': datetime.datetime.now().isoformat(),
            'total_steps_completed': 0,
            'total_episodes_completed': 0,
            'training_completed': False
        }
        
        # Add additional metadata if provided
        if metadata:
            self.metadata.update(metadata)
            
        # Resume from existing log if specified
        if resume and self.metrics_path.exists():
            self.load_metrics()
        
        # Log last save time
        self.last_save_time = time.time()
        self.last_save_step = 0
        
        # Write initial metrics file
        self.save_metrics_json()
        
    def log_step(self, reward, loss=None, q=None, info=None):
        """
        Log data from a single step.
        
        Args:
            reward: The reward received
            loss: The loss value (if any)
            q: The Q-value (if any)
            info: Additional info dict
        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        
        if loss is not None:
            self.step_losses.append(loss)
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1
            
        if q is not None:
            if isinstance(q, (list, np.ndarray)) and len(q) > 0:
                q = float(np.mean(q))
            self.step_q_values.append(q)
            self.curr_ep_q += q
            self.curr_ep_q_length += 1
        
    def log_episode(self):
        """Log data from a completed episode."""
        # Calculate episode averages
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
            
        if self.curr_ep_q_length == 0:
            ep_avg_q = 0
        else:
            ep_avg_q = self.curr_ep_q / self.curr_ep_q_length
            
        # Record metrics
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        
        # Increment episode counter
        self.episode_count += 1
        
        # Update metadata
        self.metadata['total_episodes_completed'] = self.episode_count
        
        # Keep only max_history items if specified
        if self.max_history:
            self.ep_rewards = self.ep_rewards[-self.max_history:]
            self.ep_lengths = self.ep_lengths[-self.max_history:]
            self.ep_avg_losses = self.ep_avg_losses[-self.max_history:]
            self.ep_avg_qs = self.ep_avg_qs[-self.max_history:]
            
        # Calculate moving averages
        self.moving_avg_ep_rewards.append(np.mean(self.ep_rewards[-100:]))
        self.moving_avg_ep_lengths.append(np.mean(self.ep_lengths[-100:]))
        self.moving_avg_ep_avg_losses.append(np.mean(self.ep_avg_losses[-100:]))
        self.moving_avg_ep_avg_qs.append(np.mean(self.ep_avg_qs[-100:]))
        
        # Reset episode metrics
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_q_length = 0
        
    def log_puffer_update(self, data):
        """
        Log data from a PufferLib update.
        
        Args:
            data: Data dictionary from PufferLib update
        """
        # Extract and log relevant data
        step = data.get("global_step", 0)
        reward = data.get("reward", 0)
        done = data.get("done", [False])
        loss = data.get("loss", None)
        q_values = data.get("q_values", None)
        
        # Update step information
        self.metadata['total_steps_completed'] = step
        
        # Log reward (first element if it's a list/array)
        if isinstance(reward, (list, np.ndarray, torch.Tensor)) and len(reward) > 0:
            reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward[0].item()
        else:
            reward_val = reward
            
        self.log_step(reward_val, loss, q_values)
        
        # Check for episode completion
        if isinstance(done, (list, np.ndarray, torch.Tensor)) and len(done) > 0:
            # Episode is done for the first environment
            if done[0]:
                self.log_episode()
        elif done:
            # For scalar done value
            self.log_episode()
            
        # Save metrics periodically based on step count
        if step > 0 and (step - self.last_save_step >= self.json_save_freq):
            self.save_metrics_json()
            self.last_save_step = step
            
    def record(self, episode, epsilon=None, step=None):
        """
        Record the current state, creating plots and saving metrics.
        
        Args:
            episode: Current episode number
            epsilon: Current exploration rate (for DQN)
            step: Current global step count
        """
        if step:
            self.metadata['total_steps_completed'] = step
        
        # Update or save plots/metrics every 10 episodes
        if episode % 10 == 0:
            # Generate plots
            self.plot("Reward", self.ep_rewards, self.moving_avg_ep_rewards)
            self.plot("Episode Length", self.ep_lengths, self.moving_avg_ep_lengths)
            
            if len(self.ep_avg_losses) > 0:
                self.plot("Loss", self.ep_avg_losses, self.moving_avg_ep_avg_losses)
                
            if len(self.ep_avg_qs) > 0:
                self.plot("Q", self.ep_avg_qs, self.moving_avg_ep_avg_qs)
            
            # Save metrics
            self.save_metrics_json()
            
    def plot(self, name, values, moving_avgs):
        """
        Generate and save a plot.
        
        Args:
            name: Plot name/title
            values: Values to plot
            moving_avgs: Moving averages to plot
        """
        # Skip empty plots
        if not values:
            return
            
        plt.figure(figsize=(8, 4))
        plt.title(f"{name} Over Time")
        plt.xlabel("Episode")
        plt.ylabel(name)
        
        # Plot actual values
        plt.plot(values, label=f"{name}")
        
        # Plot moving average if available
        if moving_avgs:
            plt.plot(moving_avgs, label=f"Moving Avg ({name})")
            
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.log_dir / f"ep_{name.lower()}_plot.jpg", dpi=150)
        plt.close()
        
    def save_metrics_json(self):
        """Save current metrics and metadata to a JSON file."""
        current_time = time.time()
        self.last_save_time = current_time
        
        metrics = {
            # Metadata
            "metadata": self.metadata,
            
            # Episode metrics
            "episode_rewards": self.ep_rewards if len(self.ep_rewards) < 1000 else self.ep_rewards[-1000:],
            "episode_lengths": self.ep_lengths if len(self.ep_lengths) < 1000 else self.ep_lengths[-1000:],
            "episode_avg_losses": self.ep_avg_losses if len(self.ep_avg_losses) < 1000 else self.ep_avg_losses[-1000:],
            "episode_avg_qs": self.ep_avg_qs if len(self.ep_avg_qs) < 1000 else self.ep_avg_qs[-1000:],
            
            # Moving averages
            "moving_avg_ep_rewards": self.moving_avg_ep_rewards if len(self.moving_avg_ep_rewards) < 500 else self.moving_avg_ep_rewards[-500:],
            "moving_avg_ep_lengths": self.moving_avg_ep_lengths if len(self.moving_avg_ep_lengths) < 500 else self.moving_avg_ep_lengths[-500:],
            "moving_avg_ep_avg_losses": self.moving_avg_ep_avg_losses if len(self.moving_avg_ep_avg_losses) < 500 else self.moving_avg_ep_avg_losses[-500:],
            "moving_avg_ep_avg_qs": self.moving_avg_ep_avg_qs if len(self.moving_avg_ep_avg_qs) < 500 else self.moving_avg_ep_avg_qs[-500:],
            
            # Summary stats
            "best_episode_reward": max(self.ep_rewards) if self.ep_rewards else 0,
            "best_episode_length": max(self.ep_lengths) if self.ep_lengths else 0,
            "most_recent_episode_reward": self.ep_rewards[-1] if self.ep_rewards else 0,
            "most_recent_episode_length": self.ep_lengths[-1] if self.ep_lengths else 0,
            "most_recent_episode_loss": self.ep_avg_losses[-1] if self.ep_avg_losses else 0,
            "most_recent_episode_q": self.ep_avg_qs[-1] if self.ep_avg_qs else 0,
            
            # Current episode (in-progress)
            "current_episode_reward": self.curr_ep_reward,
            "current_episode_length": self.curr_ep_length,
            
            # Timestamps
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Convert numpy values to Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)):
                metrics_json[key] = [float(v) if isinstance(v, np.number) else v for v in value]
            elif isinstance(value, np.number):
                metrics_json[key] = float(value)
            else:
                metrics_json[key] = value
                
        # Write to file
        temp_path = self.metrics_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
            
        # Rename to final path (to avoid corruption if we crash during write)
        temp_path.rename(self.metrics_path)
        
    def load_metrics(self):
        """Load metrics from JSON file when resuming training."""
        try:
            with open(self.metrics_path, "r") as f:
                metrics = json.load(f)
                
            # Load metadata
            if "metadata" in metrics:
                self.metadata.update(metrics["metadata"])
                self.episode_count = self.metadata.get('total_episodes_completed', 0)
                
            # Load episode metrics
            self.ep_rewards = metrics.get("episode_rewards", [])
            self.ep_lengths = metrics.get("episode_lengths", [])
            self.ep_avg_losses = metrics.get("episode_avg_losses", [])
            self.ep_avg_qs = metrics.get("episode_avg_qs", [])
            
            # Load moving averages
            self.moving_avg_ep_rewards = metrics.get("moving_avg_ep_rewards", [])
            self.moving_avg_ep_lengths = metrics.get("moving_avg_ep_lengths", [])
            self.moving_avg_ep_avg_losses = metrics.get("moving_avg_ep_avg_losses", [])
            self.moving_avg_ep_avg_qs = metrics.get("moving_avg_ep_avg_qs", [])
            
            print(f"Resumed from existing metrics file with {len(self.ep_rewards)} episodes")
            
        except Exception as e:
            print(f"Error loading metrics: {e}")
            print("Starting with fresh metrics")
            
    def save(self):
        """Save final metrics and plots."""
        # Update metadata
        self.metadata['training_completed'] = True
        self.metadata['end_time'] = datetime.datetime.now().isoformat()
        
        # Generate final plots
        self.plot("Reward", self.ep_rewards, self.moving_avg_ep_rewards)
        self.plot("Episode Length", self.ep_lengths, self.moving_avg_ep_lengths)
        
        if len(self.ep_avg_losses) > 0:
            self.plot("Loss", self.ep_avg_losses, self.moving_avg_ep_avg_losses)
            
        if len(self.ep_avg_qs) > 0:
            self.plot("Q", self.ep_avg_qs, self.moving_avg_ep_avg_qs)
        
        # Save final metrics
        self.save_metrics_json()
        
        # Also write a simple stats file for quick reference
        stats_path = self.log_dir / "stats_summary.txt"
        with open(stats_path, "w") as f:
            f.write(f"Training completed: {self.metadata['training_completed']}\n")
            f.write(f"Start time: {self.metadata['start_time']}\n")
            f.write(f"End time: {self.metadata['end_time']}\n")
            f.write(f"Total episodes: {self.episode_count}\n")
            f.write(f"Total steps: {self.metadata['total_steps_completed']}\n")
            f.write(f"Best episode reward: {max(self.ep_rewards) if self.ep_rewards else 0}\n")
            f.write(f"Final 100-episode average reward: {np.mean(self.ep_rewards[-100:]) if len(self.ep_rewards) >= 100 else np.mean(self.ep_rewards)}\n")
            

# Import torch if available at module level
try:
    import torch
except ImportError:
    pass


class PufferMetricLogger(MetricLogger):
    """
    Extended version of MetricLogger for PufferLib training.
    Adds support for PufferLib-specific metrics and logging.
    """
    
    def __init__(self, log_dir, resume=False, metadata=None, max_history=None, json_save_freq=10000):
        """
        Initialize the PufferLib-compatible logger.
        
        Args:
            log_dir: Directory to save logs and plots
            resume: Whether to resume logging from existing files
            metadata: Dictionary of metadata to include in log files
            max_history: Maximum history length to keep in memory
            json_save_freq: Frequency (in steps) to save metrics JSON file
        """
        super().__init__(log_dir, resume, metadata, max_history, json_save_freq)
        
        # PufferLib-specific metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        self.learning_rates = []
        self.sps_values = []  # Steps per second
        
        # Moving window metrics (last 100 values)
        self.window_size = 100
        self.reward_window = deque(maxlen=self.window_size)
        self.policy_loss_window = deque(maxlen=self.window_size)
        self.value_loss_window = deque(maxlen=self.window_size)
        self.entropy_window = deque(maxlen=self.window_size)
        self.kl_window = deque(maxlen=self.window_size)
        
    def log_puffer_metrics(self, stats, losses):
        """
        Log metrics from PufferLib training.
        
        Args:
            stats: Dictionary of environment statistics
            losses: Dictionary of loss values
        """
        # Log basic environment stats
        if 'episode_return' in stats:
            self.reward_window.append(stats['episode_return'])
            self.ep_rewards.append(stats['episode_return'])
            
        if 'episode_length' in stats:
            self.ep_lengths.append(stats['episode_length'])
            
        # Log losses
        if hasattr(losses, 'policy_loss'):
            self.policy_loss_window.append(losses.policy_loss)
            self.policy_losses.append(losses.policy_loss)
            
        if hasattr(losses, 'value_loss'):
            self.value_loss_window.append(losses.value_loss)
            self.value_losses.append(losses.value_loss)
            
        if hasattr(losses, 'entropy'):
            self.entropy_window.append(losses.entropy)
            self.entropy_losses.append(losses.entropy)
            
        if hasattr(losses, 'approx_kl'):
            self.kl_window.append(losses.approx_kl)
            self.kl_divergences.append(losses.approx_kl)
            
        # Calculate moving averages
        if len(self.ep_rewards) > 0:
            self.moving_avg_ep_rewards.append(np.mean(self.reward_window) if self.reward_window else 0)
            
        if len(self.policy_losses) > 0:
            self.moving_avg_ep_avg_losses.append(np.mean(self.policy_loss_window) if self.policy_loss_window else 0)
            
    def save_metrics_json(self):
        """Save current metrics and metadata to a JSON file."""
        # Call parent method first
        super().save_metrics_json()
        
        # Add PufferLib specific plots
        if len(self.policy_losses) > 10:
            self.plot("Policy Loss", self.policy_losses, self.moving_avg_ep_avg_losses)
            
        if len(self.value_losses) > 10:
            self.plot("Value Loss", self.value_losses, None)
            
        if len(self.entropy_losses) > 10:
            self.plot("Entropy", self.entropy_losses, None)
            
        if len(self.kl_divergences) > 10:
            self.plot("KL Divergence", self.kl_divergences, None)
            
    def log_training_config(self, config):
        """
        Log training configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Save config as JSON
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            # Convert all values to JSON-serializable
            json_config = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
                    json_config[k] = v
                else:
                    json_config[k] = str(v)
                    
            json.dump(json_config, f, indent=2)
            
        # Also update metadata
        self.metadata.update({f"config_{k}": v for k, v in json_config.items()})