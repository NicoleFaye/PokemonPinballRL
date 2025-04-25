# Pokemon Pinball RL

This project uses the Stable Baselines 3 reinforcement learning library to train an agent to play Pokemon Pinball.

## Overview

Pokemon Pinball RL uses deep reinforcement learning to play Pokemon Pinball. The system:

1. Uses PyBoy to emulate the Game Boy environment
2. Implements a Gymnasium-compatible environment for Pokemon Pinball
3. Uses Stable Baselines 3 for efficient agent training with vectorized environments
4. Supports various reward shaping strategies to guide learning

## Installation

### Prerequisites

- Python 3.8+
- ROM file for Pokemon Pinball (not included)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/pokemon-pinball-rl.git
cd pokemon-pinball-rl
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train an agent on Pokemon Pinball:

```bash
python train.py [--timesteps TIMESTEPS] [--window_size WINDOW_SIZE] [--reward_mode REWARD_MODE]
```

Key parameters:
- `--timesteps`: Number of timesteps to train for (default: 10,000,000)
- `--window_size`: Size of window for rolling metrics (default: 100)
- `--reward_mode`: Reward shaping mode - basic, catch_focused, or comprehensive (default: basic)
- `--headless`: Run in headless mode without visualization
- `--debug`: Enable debug mode
- `--no_wandb`: Disable WandB logging

To resume training from a checkpoint:
```bash
python train.py --resume runs/basic_20250424_145544/poke_1 --timesteps 5000000
```

Resume parameters:
- `--resume`: Path to checkpoint file (without .zip extension)

When you provide a checkpoint path in the format `runs/SESSION_ID/checkpoint_name`, the training will automatically continue in the same session directory. Otherwise, it will create a new session directory with "_resumed_" in the name.

You can combine the resume parameter with any other parameters like `--timesteps`, `--window_size`, or `--reward_mode`. The resumed training will use the new parameters provided.

For more options:
```bash
python train.py --help
```

## Metrics Guide

When training, the following metrics are tracked:

**X-axis in graphs**: "Step" in WandB refers to environment timesteps (individual actions), not episodes.

### Game Performance Metrics:
- **performance/all_time_high_game_score**: Highest game score achieved so far
- **performance/game_score_median**: Median game score (50th percentile)
- **performance/game_score_bottom_10pct**: Low-end game scores (10th percentile)
- **performance/game_score_top_10pct**: High-end game scores (90th percentile)

### Rolling Averages:
- **rolling_averages/avg_game_score_per_window**: Rolling average of game scores
- **rolling_averages/max_game_score_per_window**: Maximum score in each window
- **rolling_averages/avg_reward_per_window**: Rolling average of RL rewards
- **rolling_averages/avg_episode_length_per_window**: Rolling average of episode lengths
- **rolling_averages/avg_pokemon_caught_per_window**: Rolling average of Pokemon caught
- **rolling_averages/avg_ball_upgrades_per_window**: Rolling average of ball upgrades

### Raw Episode Data:
- **episode_metrics/score_per_episode**: Raw game scores (note: shows sampled points)
- **episode_metrics/reward_per_episode**: RL reward values received
- **episode_metrics/length_per_episode**: Episode lengths in environment timesteps
- **episode_metrics/pokemon_caught_per_episode**: Number of Pokemon caught
- **episode_metrics/ball_upgrades_per_episode**: Number of ball upgrades

### Episode/Timestep Tracking:
- **episode_tracking/total_episodes_completed**: Total game episodes completed
- **episode_tracking/avg_env_timesteps_per_episode**: Average env timesteps per episode

### Understanding the Data:
- All metrics use a configurable rolling window size for averaging
- All episode data is recorded, but WandB samples points when zoomed out
- The rolling averages give the clearest picture of learning progress
- **performance/all_time_high_game_score** tracks your best achievement

## Project Structure

- `environment/`: Contains the Gymnasium environment implementation
  - `pokemon_pinball_env.py`: Core environment implementation
  - `rewards.py`: Various reward shaping strategies
  - `tensorboard_callback.py`: Custom TensorBoard logging
- `checkpoints/`: Saved model checkpoints
- `train.py`: Main training script using Stable Baselines 3

## Training Strategies

The project implements different reward shaping strategies:

1. **Basic**: Simple reward based on score difference
2. **Catch-focused**: Emphasizes catching Pokemon with large rewards
3. **Comprehensive**: Balanced approach with rewards for multiple game objectives:
   - Scoring points
   - Catching Pokemon
   - Evolving Pokemon
   - Completing stages
   - Ball upgrades
   - Survival time

## Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3): Reinforcement learning library
- [Gymnasium](https://gymnasium.farama.org/): Reinforcement learning environment interface
- [WandB](https://wandb.ai/): Experiment tracking