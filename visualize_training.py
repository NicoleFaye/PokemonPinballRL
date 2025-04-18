#!/usr/bin/env python3
"""
Script to visualize training results from Pokemon Pinball RL.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Pokemon Pinball RL training results")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to checkpoint directory")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory for plots (defaults to checkpoint dir)")
    parser.add_argument("--smooth", type=int, default=10, 
                       help="Window size for smoothing")
    parser.add_argument("--save-format", type=str, default="png", 
                       choices=["png", "jpg", "pdf", "svg"],
                       help="Format for saved plots")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display plots (save only)")
    
    return parser.parse_args()


def smooth_data(data, window=10):
    """Apply smoothing to data series."""
    if not data or len(data) < window:
        return data
        
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        window_vals = data[start:end]
        smoothed.append(sum(window_vals) / len(window_vals))
        
    return smoothed


def find_checkpoints(base_dir):
    """Find all checkpoint directories in the base directory."""
    base_path = Path(base_dir)
    
    # If path is a direct checkpoint directory
    if (base_path / "metrics.json").exists():
        return [base_path]
        
    # Otherwise look for subdirectories with metrics.json
    checkpoints = []
    for path in base_path.glob("**/metrics.json"):
        checkpoints.append(path.parent)
        
    return checkpoints


def load_metrics(checkpoint_dir):
    """Load metrics from a checkpoint directory."""
    metrics_path = Path(checkpoint_dir) / "metrics.json"
    
    if not metrics_path.exists():
        print(f"No metrics.json found in {checkpoint_dir}")
        return None
        
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            return metrics
    except Exception as e:
        print(f"Error loading metrics from {metrics_path}: {e}")
        return None


def plot_episode_rewards(metrics, output_dir, window=10, fmt="png", display=True):
    """Plot episode rewards."""
    plt.figure(figsize=(10, 6))
    
    # Get reward data
    rewards = metrics.get("episode_rewards", [])
    if not rewards:
        print("No reward data found")
        return
        
    # Calculate moving average
    moving_avg = smooth_data(rewards, window)
    
    # Plot
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label="Episode Reward")
    plt.plot(episodes, moving_avg, 'r-', label=f"Moving Average (window={window})")
    
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Get best and latest results
    if rewards:
        best_reward = max(rewards)
        best_episode = rewards.index(best_reward) + 1
        latest_reward = rewards[-1]
        
        plt.axhline(y=best_reward, color='g', linestyle='--', alpha=0.7, 
                   label=f"Best: {best_reward:.1f} (Ep {best_episode})")
        
        # Add text annotation
        plt.text(len(rewards) * 0.7, best_reward * 1.05, 
                f"Best: {best_reward:.1f} (Ep {best_episode})", 
                color='g')
        plt.text(len(rewards) * 0.7, latest_reward * 1.05, 
                f"Latest: {latest_reward:.1f}", 
                color='b')
    
    # Save
    plt.tight_layout()
    output_path = Path(output_dir) / f"episode_rewards.{fmt}"
    plt.savefig(output_path, dpi=150)
    print(f"Saved episode rewards plot to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close()


def plot_episode_lengths(metrics, output_dir, window=10, fmt="png", display=True):
    """Plot episode lengths."""
    plt.figure(figsize=(10, 6))
    
    # Get length data
    lengths = metrics.get("episode_lengths", [])
    if not lengths:
        print("No episode length data found")
        return
        
    # Calculate moving average
    moving_avg = smooth_data(lengths, window)
    
    # Plot
    episodes = range(1, len(lengths) + 1)
    plt.plot(episodes, lengths, 'b-', alpha=0.3, label="Episode Length")
    plt.plot(episodes, moving_avg, 'r-', label=f"Moving Average (window={window})")
    
    plt.title("Episode Lengths Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Length (steps)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Get best and latest results
    if lengths:
        best_length = max(lengths)
        best_episode = lengths.index(best_length) + 1
        latest_length = lengths[-1]
        
        plt.axhline(y=best_length, color='g', linestyle='--', alpha=0.7)
        
        # Add text annotation
        plt.text(len(lengths) * 0.7, best_length * 1.05, 
                f"Best: {best_length} (Ep {best_episode})", 
                color='g')
        plt.text(len(lengths) * 0.7, latest_length * 1.05, 
                f"Latest: {latest_length}", 
                color='b')
    
    # Save
    plt.tight_layout()
    output_path = Path(output_dir) / f"episode_lengths.{fmt}"
    plt.savefig(output_path, dpi=150)
    print(f"Saved episode lengths plot to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close()


def plot_training_progress(metrics, output_dir, window=10, fmt="png", display=True):
    """Plot combined training progress metrics."""
    # Setup multiple plots in one figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Get data
    rewards = metrics.get("episode_rewards", [])
    lengths = metrics.get("episode_lengths", [])
    
    if not rewards and not lengths:
        print("No training data found")
        return
        
    episodes = range(1, max(len(rewards), len(lengths)) + 1)
    
    # Plot rewards
    if rewards:
        smooth_rewards = smooth_data(rewards, window)
        axes[0].plot(episodes[:len(rewards)], rewards, 'b-', alpha=0.3, label="Episode Reward")
        axes[0].plot(episodes[:len(rewards)], smooth_rewards, 'r-', label=f"Moving Avg (w={window})")
        axes[0].set_title("Episode Rewards")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Add best reward line
        best_reward = max(rewards)
        axes[0].axhline(y=best_reward, color='g', linestyle='--', alpha=0.7)
        
    # Plot lengths
    if lengths:
        smooth_lengths = smooth_data(lengths, window)
        axes[1].plot(episodes[:len(lengths)], lengths, 'b-', alpha=0.3, label="Episode Length")
        axes[1].plot(episodes[:len(lengths)], smooth_lengths, 'r-', label=f"Moving Avg (w={window})")
        axes[1].set_title("Episode Lengths")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Length (steps)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Add best length line
        best_length = max(lengths)
        axes[1].axhline(y=best_length, color='g', linestyle='--', alpha=0.7)
    
    # Add metadata from checkpoint
    metadata = metrics.get("metadata", {})
    if metadata:
        episodes_completed = metadata.get("total_episodes_completed", 0)
        steps_completed = metadata.get("total_steps_completed", 0)
        fig.suptitle(f"Training Progress - {episodes_completed} Episodes, {steps_completed} Steps")
    
    # Save
    plt.tight_layout()
    output_path = Path(output_dir) / f"training_progress.{fmt}"
    plt.savefig(output_path, dpi=150)
    print(f"Saved training progress plot to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close()


def plot_losses(metrics, output_dir, window=10, fmt="png", display=True):
    """Plot losses."""
    # Setup figure
    plt.figure(figsize=(10, 6))
    
    # Get loss data
    losses = metrics.get("episode_avg_losses", [])
    if not losses or all(l == 0 for l in losses):
        print("No loss data found")
        return
        
    # Calculate moving average
    moving_avg = smooth_data(losses, window)
    
    # Plot
    episodes = range(1, len(losses) + 1)
    plt.plot(episodes, losses, 'b-', alpha=0.3, label="Loss")
    plt.plot(episodes, moving_avg, 'r-', label=f"Moving Average (window={window})")
    
    plt.title("Loss Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save
    plt.tight_layout()
    output_path = Path(output_dir) / f"losses.{fmt}"
    plt.savefig(output_path, dpi=150)
    print(f"Saved losses plot to {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Find checkpoint directories
    checkpoints = find_checkpoints(args.checkpoint)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint}")
        return
        
    print(f"Found {len(checkpoints)} checkpoint(s)")
    
    # Process each checkpoint
    for checkpoint_dir in checkpoints:
        print(f"Processing checkpoint: {checkpoint_dir}")
        
        # Load metrics
        metrics = load_metrics(checkpoint_dir)
        if not metrics:
            continue
            
        # Determine output directory
        output_dir = args.output if args.output else checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        plot_episode_rewards(metrics, output_dir, args.smooth, args.save_format, not args.no_display)
        plot_episode_lengths(metrics, output_dir, args.smooth, args.save_format, not args.no_display)
        plot_training_progress(metrics, output_dir, args.smooth, args.save_format, not args.no_display)
        plot_losses(metrics, output_dir, args.smooth, args.save_format, not args.no_display)
        
        # Print summary stats
        print("\nTraining Summary:")
        metadata = metrics.get("metadata", {})
        print(f"Episodes completed: {metadata.get('total_episodes_completed', 0)}")
        print(f"Steps completed: {metadata.get('total_steps_completed', 0)}")
        
        rewards = metrics.get("episode_rewards", [])
        if rewards:
            print(f"Best episode reward: {max(rewards):.1f}")
            print(f"Latest episode reward: {rewards[-1]:.1f}")
            print(f"Average reward (last 10 episodes): {np.mean(rewards[-10:]):.1f}")
            print(f"Average reward (all episodes): {np.mean(rewards):.1f}")
        
        lengths = metrics.get("episode_lengths", [])
        if lengths:
            print(f"Best episode length: {max(lengths)}")
            print(f"Latest episode length: {lengths[-1]}")
            print(f"Average length (last 10 episodes): {np.mean(lengths[-10:]):.1f}")
        
        
if __name__ == "__main__":
    main()