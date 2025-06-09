#!/usr/bin/env python3
"""
Pokemon Pinball training script using PufferLib
"""
import os
import sys
import argparse
import uuid
import torch
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime

import pufferlib
import pufferlib.vector
import pufferlib.utils
import pufferlib.emulation
import pufferlib.models
import pufferlib.cleanrl

from pokemon_pinball_env import PokemonPinballEnv, EnvironmentConfig, RenderWrapper
import clean_pufferl


def make_pokemon_env(config_dict):
    """Create a Pokemon Pinball environment factory."""
    def _make(**kwargs):
        rom_path = config_dict.get('rom_path', './roms/pokemon_pinball.gbc')
        env_config = EnvironmentConfig.from_dict(config_dict)
        env = PokemonPinballEnv(rom_path, env_config)
        env = RenderWrapper(env)
        env = pufferlib.postprocess.EpisodeStats(env)
        return pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return _make


def make_policy(env, policy_cls, rnn_cls, args):
    """Create and wrap policy with PufferLib's cleanrl wrappers."""
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])


def init_wandb(args, name, id=None, resume=True):
    """Initialize Weights & Biases logging."""
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball with PufferLib')
    
    # Environment configuration
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--rom-path', type=str, default='./roms/pokemon_pinball.gbc',
                          help='Path to Pokemon Pinball ROM')
    env_group.add_argument('--episode-mode', type=str, default='life',
                          choices=['ball', 'life', 'game'], help='Episode termination mode')
    env_group.add_argument('--reset-condition', type=str, default='game',
                          choices=['ball', 'life', 'game'], help='Reset condition')
    env_group.add_argument('--reward-shaping', type=str, default='comprehensive',
                          choices=['basic', 'catch_focused', 'comprehensive', 'progressive'],
                          help='Reward shaping mode')
    env_group.add_argument('--info-level', type=int, default=0, choices=[0, 1, 2, 3],
                          help='Information level in observations')
    env_group.add_argument('--visual-mode', type=str, default='screen',
                          choices=['screen', 'game_area'], help='Visual observation mode')
    env_group.add_argument('--frame-skip', type=int, default=4,
                          help='Number of frames to skip')
    env_group.add_argument('--reduce-screen-resolution', action='store_true', default=True,
                          help='Reduce screen resolution for faster processing')
    env_group.add_argument('--headless', action='store_true',
                          help='Run without rendering')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num-envs', type=int, default=8,
                           help='Number of parallel environments')
    train_group.add_argument('--num-workers', type=int, default=8,
                           help='Number of worker processes')
    train_group.add_argument('--total-timesteps', type=int, default=10_000_000,
                           help='Total training timesteps')
    train_group.add_argument('--batch-size', type=int, default=32768,
                           help='Batch size for training')
    train_group.add_argument('--minibatch-size', type=int, default=8192,
                           help='Minibatch size')
    train_group.add_argument('--bptt-horizon', type=int, default=16,
                           help='Backprop through time horizon')
    train_group.add_argument('--learning-rate', type=float, default=3e-4,
                           help='Learning rate')
    train_group.add_argument('--gamma', type=float, default=0.99,
                           help='Discount factor')
    train_group.add_argument('--gae-lambda', type=float, default=0.95,
                           help='GAE lambda')
    train_group.add_argument('--update-epochs', type=int, default=4,
                           help='Number of update epochs')
    train_group.add_argument('--clip-coef', type=float, default=0.2,
                           help='PPO clipping coefficient')
    train_group.add_argument('--ent-coef', type=float, default=0.01,
                           help='Entropy coefficient')
    train_group.add_argument('--vf-coef', type=float, default=0.5,
                           help='Value function coefficient')
    train_group.add_argument('--max-grad-norm', type=float, default=0.5,
                           help='Max gradient norm')
    train_group.add_argument('--device', type=str, default='cuda',
                           choices=['cpu', 'cuda'], help='Device to use')
    
    # Policy configuration
    policy_group = parser.add_argument_group('Policy')
    policy_group.add_argument('--hidden-size', type=int, default=512,
                            help='Hidden layer size')
    
    # Vectorization settings
    vector_group = parser.add_argument_group('Vectorization')
    vector_group.add_argument('--vec', type=str, default='multiprocessing',
                            choices=['serial', 'multiprocessing', 'ray', 'native'],
                            help='Vectorization backend')
    vector_group.add_argument('--vec-overwork', action='store_true',
                            help='Allow vectorization to use >1 worker/core')
    
    # Logging and experiment tracking
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument('--track', action='store_true',
                             help='Track with WandB')
    logging_group.add_argument('--wandb-project', type=str, default='pokemon-pinball-pufferlib',
                             help='WandB project name')
    logging_group.add_argument('--wandb-group', type=str, default='debug',
                             help='WandB group name')
    logging_group.add_argument('--exp-id', type=str, default=None,
                             help='Experiment ID (for resuming)')
    
    # Miscellaneous
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    
    return parser


def create_config_from_args(args):
    """Create nested configuration structure from command line arguments."""
    
    # Environment configuration
    env_config = {
        'rom_path': args.rom_path,
        'episode_mode': args.episode_mode,
        'reset_condition': args.reset_condition,
        'reward_shaping': args.reward_shaping,
        'info_level': args.info_level,
        'visual_mode': args.visual_mode,
        'frame_skip': args.frame_skip,
        'reduce_screen_resolution': args.reduce_screen_resolution,
        'headless': args.headless,
        'debug': False,
        'num_agents': args.num_envs,
    }
    
    # Training configuration
    train_config = {
        'seed': args.seed,
        'torch_deterministic': True,
        'device': args.device,
        
        # Environment parallelization
        'num_envs': args.num_envs,
        'num_workers': args.num_workers,
        'env_batch_size': args.num_envs // args.num_workers,
        'zero_copy': True,
        
        # Technical settings
        'cpu_offload': False,
        
        # PPO hyperparameters
        'total_timesteps': args.total_timesteps,
        'batch_size': args.batch_size,
        'minibatch_size': args.minibatch_size,
        'bptt_horizon': args.bptt_horizon,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'update_epochs': args.update_epochs,
        'clip_coef': args.clip_coef,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'vf_clip_coef': 0.1,
        'max_grad_norm': args.max_grad_norm,
        'norm_adv': True,
        'clip_vloss': True,
        'target_kl': None,
        'anneal_lr': True,
    }
    
    # Policy configuration
    policy_config = {
        'hidden_size': args.hidden_size,
    }
    
    # Main configuration structure
    config = {
        'env': env_config,
        'train': train_config,
        'policy': policy_config,
        'rnn': {},  # Empty for non-recurrent policies
        'vec': args.vec,
        'vec_overwork': args.vec_overwork,
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'exp_id': args.exp_id,
        'track': args.track,
    }
    
    return config


def train(args, make_env, policy_cls, rnn_cls, wandb_run):
    """Execute the main training loop."""
    
    # Configure vectorization backend
    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.environment.PufferEnv
    else:
        raise ValueError(f'Invalid --vec ({args["vec"]}). Use serial/multiprocessing/ray/native.')

    # Create vectorized environment
    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args['env'],
        num_envs=args['train']['num_envs'],
        num_workers=args['train']['num_workers'],
        batch_size=args['train']['env_batch_size'],
        zero_copy=args['train']['zero_copy'],
        overwork=args['vec_overwork'],
        backend=vec,
    )

    # Create and initialize policy
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    
    # Create training configuration namespace
    env_name = 'pokemon_pinball'
    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    
    # Initialize training data structure
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb_run)
    
    try:
        # Main training loop
        while data.global_step < train_config.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        # Cleanup resources
        clean_pufferl.close(data)
        print(f"\nTraining completed! Final steps: {data.global_step:,}")


def main():
    """Main entry point for Pokemon Pinball training."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration structure
    config = create_config_from_args(args)
    
    # Print training configuration
    print(f"Starting Pokemon Pinball training with PufferLib")
    print(f"Experiment ID: {config['exp_id'] or 'auto-generated'}")
    print(f"Device: {config['train']['device']}")
    print(f"Environments: {config['train']['num_envs']}")
    print(f"Workers: {config['train']['num_workers']}")
    print(f"Total timesteps: {config['train']['total_timesteps']:,}")
    print(f"Vectorization: {config['vec']}")
    
    # Initialize experiment tracking
    wandb_run = None
    if config['track']:
        wandb_run = init_wandb(config, 'pokemon_pinball', id=config['exp_id'])
    
    # Create environment factory
    make_env = make_pokemon_env(config['env'])
    
    # Configure policy architecture
    policy_cls = pufferlib.models.Default
    rnn_cls = None  # No recurrent layers
    
    # Execute training
    train(config, make_env, policy_cls, rnn_cls, wandb_run)


if __name__ == '__main__':
    main()