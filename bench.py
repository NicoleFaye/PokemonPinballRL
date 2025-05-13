#!/usr/bin/env python3
"""
Environment Benchmark Script for Pokemon Pinball RL

This script benchmarks training performance with different numbers of parallel environments
(8, 16, and 32) to determine the optimal configuration for training efficiency.

Usage:
    python environment_benchmark.py [--iterations 3] [--timesteps 50000] [--rom-path PATH]
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from pokemon_pinball_env import PokemonPinballEnv

class TimingCallback(BaseCallback):
    """Custom callback for timing training steps"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = None
        self.iter_start = None
        self.times = []
        self.rewards = []
        self.update_count = 0
    
    def _on_training_start(self) -> None:
        """Called at the start of training"""
        self.training_start = time.time()
        self.iter_start = self.training_start
    
    def _on_step(self) -> bool:
        """Called at each step. Required implementation for BaseCallback."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout"""
        # Calculate time for this iteration
        now = time.time()
        elapsed = now - self.iter_start
        self.iter_start = now
        self.times.append(elapsed)
        
        # Track rewards
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.rewards.append(mean_reward)
        else:
            self.rewards.append(0)
            
        self.update_count += 1
        
        if self.verbose > 0:
            print(f"Update {self.update_count}: {elapsed:.2f} seconds")
    
    def get_metrics(self):
        """Return the collected timing metrics"""
        total_time = time.time() - self.training_start
        return {
            'times': self.times,
            'total_time': total_time,
            'updates': self.update_count,
            'rewards': self.rewards
        }

def make_env(rank, env_conf, seed=0):
    """Create a function to instantiate environments"""
    def _init():
        env = PokemonPinballEnv(env_conf.get("rom_path", "./roms/pokemon_pinball.gbc"), env_conf)
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def run_benchmark(num_envs, timesteps, env_config, seed=0):
    """Run a benchmark with the given number of environments"""
    print(f"\n{'='*50}")
    print(f"Benchmarking with {num_envs} environments")
    print(f"{'='*50}")
    
    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, env_config, seed=seed+i) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=5.0, gamma=0.99)
    
    # Create PPO model with minimal training parameters
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=0,
        n_steps=1024,          # Smaller steps for faster iterations
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        learning_rate=3e-4
    )
    
    # Create timing callback
    timing_cb = TimingCallback(verbose=1)
    
    # Time the training
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps, callback=timing_cb)
    
    train_time = time.time() - start_time
    
    # Get metrics
    metrics = timing_cb.get_metrics()
    
    # Close environments
    env.close()
    
    # Calculate performance metrics
    fps = timesteps / train_time
    steps_per_update = timesteps / metrics['updates'] if metrics['updates'] > 0 else 0
    avg_update_time = np.mean(metrics['times']) if metrics['times'] else 0
    
    result = {
        'num_envs': num_envs,
        'total_time': train_time,
        'frames_per_second': fps,
        'steps_per_update': steps_per_update,
        'avg_update_time': avg_update_time,
        'update_times': metrics['times'],
        'rewards': metrics['rewards'],
        'updates': metrics['updates']
    }
    
    print(f"Results for {num_envs} environments:")
    print(f"  Total time: {train_time:.2f} seconds")
    print(f"  Steps per second: {fps:.2f}")
    print(f"  Steps per update: {steps_per_update:.0f}")
    print(f"  Average update time: {avg_update_time:.2f} seconds")
    
    return result

def plot_results(results, output_dir):
    """Generate performance comparison plots"""
    # Extract data
    envs = [r['num_envs'] for r in results]
    fps = [r['frames_per_second'] for r in results]
    update_times = [r['avg_update_time'] for r in results]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot frames per second
    ax1.bar(envs, fps, color='royalblue')
    ax1.set_xlabel('Number of Environments')
    ax1.set_ylabel('Frames Per Second')
    ax1.set_title('Training Speed')
    ax1.set_xticks(envs)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(fps):
        ax1.text(envs[i], v + max(fps)*0.05, f"{v:.0f}", ha='center')
    
    # Plot update times
    ax2.bar(envs, update_times, color='forestgreen')
    ax2.set_xlabel('Number of Environments')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Average Update Time')
    ax2.set_xticks(envs)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(update_times):
        ax2.text(envs[i], v + max(update_times)*0.05, f"{v:.2f}s", ha='center')
    
    # Add speedup calculation as text
    baseline_fps = fps[0]  # 8 environments is our baseline
    speedup_text = "Speedup relative to 8 envs:\n"
    for i, env_count in enumerate(envs):
        if i == 0:
            continue  # Skip baseline
        speedup = fps[i] / baseline_fps
        speedup_text += f"{env_count} envs: {speedup:.2f}x\n"
    
    fig.text(0.02, 0.02, speedup_text, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=200)
    plt.close()
    
    # Create update time series plot
    plt.figure(figsize=(10, 6))
    for result in results:
        times = result['update_times']
        updates = range(1, len(times) + 1)
        plt.plot(updates, times, label=f"{result['num_envs']} envs")
    
    plt.xlabel('Update Number')
    plt.ylabel('Update Time (seconds)')
    plt.title('Update Times Throughout Training')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "update_times.png"), dpi=200)
    plt.close()
    
    # Create rewards plot if available
    plt.figure(figsize=(10, 6))
    for result in results:
        rewards = result['rewards']
        if rewards and any(r != 0 for r in rewards):
            updates = range(1, len(rewards) + 1)
            plt.plot(updates, rewards, label=f"{result['num_envs']} envs")
    
    plt.xlabel('Update Number')
    plt.ylabel('Mean Episode Reward')
    plt.title('Rewards Throughout Training')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rewards.png"), dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Benchmark Pokemon Pinball training with different numbers of environments')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations to run for each config')
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of timesteps to train for each run')
    parser.add_argument('--rom-path', type=str, default="./roms/pokemon_pinball.gbc", help='Path to Pokemon Pinball ROM')
    args = parser.parse_args()
    
    # Environment configurations to test
    env_counts = [2, 8, 16, 32, 64]
    
    # Base environment configuration
    env_config = {
        'headless': True,
        'debug': False,
        'reward_shaping': 'basic',
        'info_level': 1,
        'frame_stack': 4,
        'frame_skip': 4,
        'visual_mode': 'screen',
        'frame_stack_extra_observation': True,
        'reduce_screen_resolution': True,
        'max_episode_frames': 10000,  # Limit episode length for benchmark
        'rom_path': args.rom_path
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmarks/benchmark_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config = {
        'iterations': args.iterations,
        'timesteps': args.timesteps,
        'env_counts': env_counts,
        'env_config': env_config,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run benchmarks
    all_results = []
    
    for num_envs in env_counts:
        env_results = []
        
        for i in range(args.iterations):
            print(f"\nIteration {i+1}/{args.iterations} for {num_envs} environments")
            result = run_benchmark(num_envs, args.timesteps, env_config, seed=i)
            env_results.append(result)
        
        # Average results across iterations
        avg_result = {
            'num_envs': num_envs,
            'total_time': np.mean([r['total_time'] for r in env_results]),
            'frames_per_second': np.mean([r['frames_per_second'] for r in env_results]),
            'steps_per_update': np.mean([r['steps_per_update'] for r in env_results]),
            'avg_update_time': np.mean([r['avg_update_time'] for r in env_results]),
            'update_times': env_results[0]['update_times'],  # Just use first iteration for plots
            'rewards': env_results[0]['rewards'],
            'updates': env_results[0]['updates']
        }
        
        all_results.append(avg_result)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'num_envs': r['num_envs'],
            'frames_per_second': r['frames_per_second'],
            'avg_update_time': r['avg_update_time'],
            'total_time': r['total_time']
        } for r in all_results
    ])
    
    results_df.to_csv(output_dir / "benchmark_results.csv", index=False)
    
    # Create summary
    summary = "Environment Count Benchmark Results\n"
    summary += "=" * 40 + "\n\n"
    summary += f"Timesteps per run: {args.timesteps}\n"
    summary += f"Iterations per config: {args.iterations}\n\n"
    
    summary += "Performance Summary:\n"
    for result in all_results:
        summary += f"\n{result['num_envs']} Environments:\n"
        summary += f"  - Frames per second: {result['frames_per_second']:.2f}\n"
        summary += f"  - Avg update time: {result['avg_update_time']:.2f} seconds\n"
        summary += f"  - Total time: {result['total_time']:.2f} seconds\n"
    
    # Add speedup calculation to summary
    baseline_fps = all_results[0]['frames_per_second']  # 8 environments is our baseline
    summary += "\nSpeedup relative to 8 envs:\n"
    for i, result in enumerate(all_results):
        if i == 0:
            continue  # Skip baseline
        speedup = result['frames_per_second'] / baseline_fps
        summary += f"  - {result['num_envs']} envs: {speedup:.2f}x\n"
    
    # Add recommendation
    best_index = np.argmax([r['frames_per_second'] for r in all_results])
    best_env_count = all_results[best_index]['num_envs']
    
    summary += f"\nRecommendation: Based on raw performance, {best_env_count} environments "
    summary += "provides the best training throughput.\n\n"
    
    # Factor in diminishing returns
    summary += "Efficiency Analysis:\n"
    for i in range(1, len(all_results)):
        prev_fps = all_results[i-1]['frames_per_second']
        curr_fps = all_results[i]['frames_per_second']
        env_increase = all_results[i]['num_envs'] / all_results[i-1]['num_envs']
        efficiency = (curr_fps / prev_fps) / env_increase
        
        summary += f"  - Scaling from {all_results[i-1]['num_envs']} to {all_results[i]['num_envs']} envs:\n"
        summary += f"    * Speed increase: {curr_fps/prev_fps:.2f}x\n"
        summary += f"    * Resource efficiency: {efficiency:.2f}x\n"
        summary += f"    * Return per added env: {(curr_fps-prev_fps)/(all_results[i]['num_envs']-all_results[i-1]['num_envs']):.2f} FPS/env\n"
    
    # Final recommendation considering diminishing returns
    summary += "\nFinal Recommendation: "
    if len(all_results) > 1:
        efficiencies = []
        for i in range(1, len(all_results)):
            prev_fps = all_results[i-1]['frames_per_second']
            curr_fps = all_results[i]['frames_per_second']
            env_increase = all_results[i]['num_envs'] / all_results[i-1]['num_envs']
            efficiency = (curr_fps / prev_fps) / env_increase
            efficiencies.append((all_results[i]['num_envs'], efficiency))
        
        # Find point of diminishing returns (efficiency < 0.7)
        diminishing_point = next((env_count for env_count, eff in efficiencies if eff < 0.7), best_env_count)
        
        if diminishing_point < best_env_count:
            summary += f"While {best_env_count} envs provides the highest raw throughput, "
            summary += f"significant diminishing returns occur after {diminishing_point} envs. "
            summary += f"Consider using {diminishing_point} envs for better resource efficiency."
        else:
            summary += f"Use {best_env_count} environments for optimal training performance."
    else:
        summary += f"Use {best_env_count} environments for optimal training performance."
    
    # Write summary to file
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    # Generate plots
    plot_results(all_results, output_dir)
    
    print("\nBenchmark complete!")
    print(f"Results saved to: {output_dir}")
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()