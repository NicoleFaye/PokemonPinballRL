#!/usr/bin/env python3
"""
Pokemon Pinball training script using PufferLib with default policy
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

# Import your custom modules
from pokemon_pinball_env import PokemonPinballEnv, EnvironmentConfig
import clean_pufferl


def make_pokemon_env(config_dict):
    """Create a Pokemon Pinball environment"""
    def _make(buf=None, **kwargs):
        rom_path = config_dict.get('rom_path', './roms/pokemon_pinball.gbc')
        env_config = EnvironmentConfig.from_dict(config_dict)
        env = PokemonPinballEnv(rom_path, env_config)
        
        # Wrap with PufferLib emulation layer (required for vectorization)
        return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return _make


def create_argument_parser():
    """Create argument parser for Pokemon Pinball training"""
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball with PufferLib')
    
    # Environment arguments
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
    
    # Training arguments
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
    
    # Policy arguments
    policy_group = parser.add_argument_group('Policy')
    policy_group.add_argument('--policy-type', type=str, default='default',
                            choices=['default', 'convolutional'], 
                            help='Type of policy to use')
    policy_group.add_argument('--hidden-size', type=int, default=512,
                            help='Hidden layer size')
    
    # Logging and checkpoints
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument('--track', action='store_true',
                             help='Track with WandB')
    logging_group.add_argument('--wandb-project', type=str, default='pokemon-pinball-pufferlib',
                             help='WandB project name')
    logging_group.add_argument('--wandb-group', type=str, default='debug',
                             help='WandB group name')
    logging_group.add_argument('--exp-id', type=str, default=None,
                             help='Experiment ID (for resuming)')
    logging_group.add_argument('--checkpoint-interval', type=int, default=100,
                             help='Checkpoint save interval')
    logging_group.add_argument('--data-dir', type=str, default='./experiments',
                             help='Data directory for checkpoints')
    
    # Misc
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    misc_group.add_argument('--compile', action='store_true',
                          help='Use torch.compile')
    misc_group.add_argument('--compile-mode', type=str, default='default',
                          help='Torch compile mode')
    
    return parser


def create_config_from_args(args):
    """Create configuration dictionaries from parsed arguments"""
    
    # Environment config
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
    }
    
    # Training config
    train_config = pufferlib.namespace(
        env='pokemon_pinball',
        exp_id=args.exp_id or f'pokemon-pinball-{str(uuid.uuid4())[:8]}',
        seed=args.seed,
        torch_deterministic=True,
        device=args.device,
        
        # Training hyperparameters
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        bptt_horizon=args.bptt_horizon,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        norm_adv=True,
        clip_vloss=True,
        vf_clip_coef=0.1,
        target_kl=None,
        anneal_lr=True,
        
        # Vectorization
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        env_batch_size=args.num_envs // args.num_workers,
        zero_copy=True,
        
        # Technical
        compile=args.compile,
        compile_mode=args.compile_mode,
        cpu_offload=False,
        
        # Checkpointing
        checkpoint_interval=args.checkpoint_interval,
        data_dir=args.data_dir,
    )
    
    # Policy config
    policy_config = {
        'policy_type': args.policy_type,
        'hidden_size': args.hidden_size,
    }
    
    return env_config, train_config, policy_config


def create_policy(env, policy_config):
    """Create policy based on configuration"""
    policy_type = policy_config['policy_type']
    hidden_size = policy_config['hidden_size']
    
    if policy_type == 'default':
        # Use PufferLib's default policy - works with any observation space
        policy = pufferlib.models.Default(env, hidden_size=hidden_size)
    elif policy_type == 'convolutional':
        # Use PufferLib's convolutional policy for image observations
        policy = pufferlib.models.Convolutional(
            env, 
            input_size=hidden_size, 
            hidden_size=hidden_size,
            framestack=1,
            flat_size=hidden_size  # Will be calculated automatically
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policy


def init_wandb(args, train_config):
    """Initialize Weights & Biases logging"""
    if not args.track:
        return None
        
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=train_config.exp_id,
        config={
            'env': vars(args),  # Environment config
            'train': dict(train_config),  # Training config
        },
        save_code=True,
    )
    return wandb


def main():
    """Main training function"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configurations
    env_config, train_config, policy_config = create_config_from_args(args)
    
    # Set up paths
    os.makedirs(train_config.data_dir, exist_ok=True)
    
    print(f"Starting Pokemon Pinball training with PufferLib")
    print(f"Experiment ID: {train_config.exp_id}")
    print(f"Device: {train_config.device}")
    print(f"Environments: {train_config.num_envs}")
    print(f"Workers: {train_config.num_workers}")
    print(f"Total timesteps: {train_config.total_timesteps:,}")
    print(f"Policy type: {policy_config['policy_type']}")
    
    # Initialize WandB
    wandb_run = init_wandb(args, train_config)
    
    # Create vectorized environment
    vecenv = pufferlib.vector.make(
        make_pokemon_env(env_config),
        num_envs=train_config.num_envs,
        num_workers=train_config.num_workers,
        batch_size=train_config.env_batch_size,
        zero_copy=train_config.zero_copy,
        backend=pufferlib.vector.Multiprocessing,  # or Serial for debugging
    )
    
    # Create policy using PufferLib's built-in policies
    policy = create_policy(vecenv.driver_env, policy_config)
    policy = policy.to(train_config.device)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    
    # Create training data structure
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
        # Clean up
        clean_pufferl.close(data)
        print(f"\nTraining completed! Final steps: {data.global_step:,}")


if __name__ == '__main__':
    main()