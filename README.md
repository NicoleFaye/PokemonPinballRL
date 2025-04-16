# Pokemon Pinball RL with PufferLib

This project combines PyBoy, a Game Boy emulator with Python interface, with PufferLib, a reinforcement learning library, to train an agent to play Pokemon Pinball.

## Overview

Pokemon Pinball RL uses deep reinforcement learning to play Pokemon Pinball. The system:

1. Uses PyBoy to emulate the Game Boy environment
2. Implements a Gymnasium-compatible environment for Pokemon Pinball
3. Leverages PufferLib for efficient agent training with vectorized environments
4. Uses PufferLib's optimized model architectures and PPO implementation
5. Supports various reward shaping strategies to guide learning

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

To train an agent on Pokemon Pinball using PufferLib:

```bash
python train_puffer.py --rom path/to/pokemon_pinball.gbc --reward-shaping comprehensive --policy-type cnn
```

Key parameters:
- `--rom`: Path to the Pokemon Pinball ROM file
- `--reward-shaping`: Reward function (`basic`, `catch_focused`, or `comprehensive`)
- `--policy-type`: Network architecture (`cnn`, `mlp`, or `resnet`)
- `--recurrent`: Add this flag to use an LSTM-based recurrent policy
- `--num-envs`: Number of parallel environments (default: 4)
- `--framestack`: Number of frames to stack (default: 4)
- `--frame-skip`: Number of frames to skip (default: 4)

For more options:
```bash
python train_puffer.py --help
```

### Evaluation

To evaluate a trained agent:

```bash
python evaluate_puffer.py --rom path/to/pokemon_pinball.gbc --model checkpoints/model_name/final.pt --render
```

Key parameters:
- `--rom`: Path to the Pokemon Pinball ROM file
- `--model`: Path to the trained model file
- `--render`: Enable visualization
- `--speed`: Game speed for visualization (default: 1.0)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--record`: Record videos of evaluation episodes

For more options:
```bash
python evaluate_puffer.py --help
```

## Project Structure

- `environment/`: Contains the Gymnasium environment implementation
  - `pokemon_pinball_env.py`: Core environment implementation
  - `wrappers.py`: Various environment wrappers
  - `puffer_env.py`: PufferLib-compatible environment
- `models/`: Neural network models
  - `puffer_models.py`: PufferLib-compatible policy models
- `train_puffer.py`: Main training script using PufferLib
- `evaluate_puffer.py`: Evaluation script for trained agents

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
- [PufferLib](https://github.com/pufferlib/pufferlib): Reinforcement learning library
- [Gymnasium](https://gymnasium.farama.org/): Reinforcement learning environment interface