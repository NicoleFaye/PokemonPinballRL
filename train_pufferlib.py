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

# Import your custom modules
from pokemon_pinball_env import PokemonPinballEnv, EnvironmentConfig
import clean_pufferl


class PokemonPinballPolicy(torch.nn.Module):
    """Simple CNN-based policy for Pokemon Pinball"""
    
    def __init__(self, env, hidden_size=512, cnn_channels=[32, 64, 64]):
        super().__init__()
        
        # Get observation space info
        obs_space = env.single_observation_space
        if hasattr(obs_space, 'spaces'):
            # Dict observation space - use visual representation
            visual_shape = obs_space['visual_representation'].shape
            self.use_dict_obs = True
        else:
            # Single observation space
            visual_shape = obs_space.shape
            self.use_dict_obs = False
            
        self.action_space = env.single_action_space
        
        # CNN for visual processing
        channels = visual_shape[0] if len(visual_shape) == 3 else 1
        height, width = visual_shape[-2:]
        
        layers = []
        in_channels = channels
        for out_channels in cnn_channels:
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ])
            in_channels = out_channels
            
        self.cnn = torch.nn.Sequential(*layers)
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            cnn_out = self.cnn(dummy_input)
            cnn_output_size = cnn_out.numel()
        
        # Additional feature processing for dict observations
        if self.use_dict_obs:
            # Ball position and velocity (4 features)
            # Game state features (varies by info_level)
            additional_features = 10  # Estimate based on typical info_level
            total_input_size = cnn_output_size + additional_features
        else:
            total_input_size = cnn_output_size
            
        # Shared network
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(total_input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        
        # Policy and value heads
        self.actor = torch.nn.Linear(hidden_size, self.action_space.n)
        self.critic = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, obs, state=None, action=None):
        if self.use_dict_obs:
            if isinstance(obs, dict):
                visual = obs['visual_representation']
                additional_features = []
                
                # Add ball features if available
                if 'ball_x' in obs:
                    additional_features.extend([
                        obs['ball_x'], obs['ball_y'],
                        obs['ball_x_velocity'], obs['ball_y_velocity']
                    ])
                
                # Add game state features if available
                if 'current_stage' in obs:
                    additional_features.extend([
                        obs['current_stage'], obs['ball_type'],
                        obs['special_mode'], obs['special_mode_active'],
                        obs['saver_active']
                    ])
                    
                if 'pikachu_saver_charge' in obs:
                    additional_features.append(obs['pikachu_saver_charge'])
                    
            else:
                # Flattened observation - assume visual comes first
                batch_size = obs.shape[0]
                visual_size = np.prod(self.cnn_input_shape)
                visual = obs[:, :visual_size].reshape(batch_size, *self.cnn_input_shape)
                additional_features = obs[:, visual_size:]
        else:
            visual = obs
            additional_features = []
        
        # Process visual input
        if len(visual.shape) == 3:
            visual = visual.unsqueeze(0)
        elif len(visual.shape) == 4 and visual.shape[1] != 1:
            # Convert from HWC to CHW if needed
            visual = visual.permute(0, 3, 1, 2)
        
        # Ensure single channel for grayscale
        if visual.shape[1] == 3:
            visual = torch.mean(visual, dim=1, keepdim=True)
        elif visual.shape[1] != 1:
            visual = visual.unsqueeze(1)
            
        cnn_features = self.cnn(visual.float() / 255.0)
        cnn_features = cnn_features.flatten(start_dim=1)
        
        # Combine features
        if self.use_dict_obs and len(additional_features) > 0:
            if isinstance(additional_features, list):
                additional_features = torch.cat(additional_features, dim=-1)
            features = torch.cat([cnn_features, additional_features.float()], dim=-1)
        else:
            features = cnn_features
            
        # Shared processing
        shared_features = self.shared(features)
        
        # Policy and value outputs
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        if action is None:
            # Sampling action
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprob, entropy, value
        else:
            # Computing logprob for given action
            dist = torch.distributions.Categorical(logits=logits)
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprob, entropy, value


def make_pokemon_env(config_dict):
    """Create a Pokemon Pinball environment"""
    def _make(buf=None, **kwargs):  # Accept buf parameter and any other kwargs
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
    policy_group.add_argument('--hidden-size', type=int, default=512,
                            help='Hidden layer size')
    policy_group.add_argument('--cnn-channels', type=int, nargs='+', default=[32, 64, 64],
                            help='CNN channel sizes')
    
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
        'hidden_size': args.hidden_size,
        'cnn_channels': args.cnn_channels,
    }
    
    return env_config, train_config, policy_config


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
    
    # Create policy
    policy = PokemonPinballPolicy(vecenv.driver_env, **policy_config)
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