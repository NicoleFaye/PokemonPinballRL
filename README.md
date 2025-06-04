# Pokemon Pinball RL

This project uses the Stable Baselines 3 reinforcement learning library to train an agent to play Pokemon Pinball.

## Example RL agent Gameplay
<img src="https://github.com/Baekalfen/PyBoy/blob/master/extras/README/pinball.gif" width="480">

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
- Pyboy with updated pokemon pinball game wrapper updated to have reset_tracking method (may require manual build of current master branch of pyboy)

## Usage

### Training

To train an agent on Pokemon Pinball:

```bash
python train.py [--timesteps TIMESTEPS] [--window_size WINDOW_SIZE] [--reward_mode REWARD_MODE]
```

Key parameters:
- `--timesteps`: Number of timesteps to train for (default: 10,000,000)
- `--reward_mode`: Reward shaping mode - basic, catch_focused, or comprehensive (default: basic)
- `--headless`: Run in headless mode without visualization
- `--debug`: Enable debug mode
- `--no_wandb`: Disable WandB logging

To resume training from a checkpoint:
```bash
python train.py --resume runs/basic_20250424_145544/poke_1 --timesteps 5000000
```

Resume parameters:
- `--resume`: Path to checkpoint file 

When you provide a checkpoint path in the format `runs/SESSION_ID/checkpoint_name`, the training will automatically continue in the same session directory. Otherwise, it will create a new session directory with "_resumed_" in the name.

You can combine the resume parameter with any other parameters like `--timesteps`, `--window_size`, or `--reward_mode`. The resumed training will use the new parameters provided.

For more options:
```bash
python train.py --help
```

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

Thank you to the following projects for either their library usage or for my reference for the creation of this repo

- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator
- [Pokemon Red Experiments](https://github.com/PWhiddy/PokemonRedExperiments): Pokemon Red RL project using pyboy
- [Pret Pokemon Pinball Dissassebly](https://github.com/pret/pokepinball): Dissassembly of Pokemon pinball for GBC
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3): Reinforcement learning library
- [Gymnasium](https://gymnasium.farama.org/): Reinforcement learning environment interface
- [WandB](https://wandb.ai/): Experiment tracking
