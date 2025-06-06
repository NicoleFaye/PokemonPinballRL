import sys
import argparse
import functools
from os.path import exists
from os import makedirs
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import uuid

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.environments
import pufferlib.emulation
import pufferlib.postprocess
import pufferlib.cleanrl
import pufferlib.models
import pufferlib.pytorch


def create_argument_parser():
    """Create and configure the argument parser for training configuration."""
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball RL agent with PufferLib CleanRL')
    
    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--rom-path', type=str, default='./roms/pokemon_pinball.gbc',
                          help='Path to Pokemon Pinball ROM file')
    env_group.add_argument('--episode-mode', type=str, default='life', 
                          choices=['ball', 'life', 'game'], help='Episode mode')
    env_group.add_argument('--reset-condition', type=str, default='game', 
                          choices=['ball', 'life', 'game'], help='Reset condition')
    env_group.add_argument('--seed', type=int, default=1, 
                          help='Random seed')
    env_group.add_argument('--reward-mode', type=str, default='comprehensive', 
                          choices=['basic', 'catch_focused', 'comprehensive', 'progressive'], 
                          help='Reward shaping mode')
    env_group.add_argument('--frame-skip', type=int, default=4, 
                          help='Number of frames to skip')
    env_group.add_argument('--visual-mode', type=str, default='screen', 
                          choices=['screen', 'game_area'], help='Visual observation mode')
    env_group.add_argument('--max-episode-frames', type=int, default=0, 
                          help='Max frames per episode (0 = no limit)')
    env_group.add_argument('--no-reduce-screen-resolution', dest='reduce_screen_resolution', 
                          action='store_false', help='Disable screen downsample')
    env_group.add_argument('--clip-reward', action='store_true', 
                          help='Enable reward clipping')
    env_group.add_argument('--reward-clip-max', type=float, default=10.0, 
                          help='Maximum reward clipping value')
    env_group.add_argument('--reward-clip-min', type=float, default=-10.0, 
                          help='Minimum reward clipping value')
    
    # Training hyperparameters (PufferLib style)
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument('--total-timesteps', type=int, default=10_000_000, 
                               help='Total timesteps to train for')
    training_group.add_argument('--learning-rate', type=float, default=2.5e-4, 
                               help='Learning rate')
    training_group.add_argument('--num-envs', type=int, default=8, 
                               help='Number of parallel environments')
    training_group.add_argument('--num-workers', type=int, default=None,
                               help='Number of workers (default: same as num_envs)')
    training_group.add_argument('--env-batch-size', type=int, default=None,
                               help='Environment batch size')
    training_group.add_argument('--batch-size', type=int, default=32768, 
                               help='Batch size for training')
    training_group.add_argument('--bptt-horizon', type=int, default=16,
                               help='Backprop through time horizon')
    training_group.add_argument('--minibatch-size', type=int, default=8192, 
                               help='Minibatch size')
    training_group.add_argument('--update-epochs', type=int, default=1, 
                               help='Number of update epochs')
    training_group.add_argument('--gamma', type=float, default=0.99, 
                               help='Discount factor')
    training_group.add_argument('--gae-lambda', type=float, default=0.95, 
                               help='GAE lambda')
    training_group.add_argument('--clip-coef', type=float, default=0.2, 
                               help='PPO clipping coefficient')
    training_group.add_argument('--vf-coef', type=float, default=0.5, 
                               help='Value function coefficient')
    training_group.add_argument('--vf-clip-coef', type=float, default=0.2,
                               help='Value function clipping coefficient')
    training_group.add_argument('--ent-coef', type=float, default=0.01, 
                               help='Entropy coefficient')
    training_group.add_argument('--max-grad-norm', type=float, default=0.5, 
                               help='Maximum gradient norm')
    training_group.add_argument('--target-kl', type=float, default=None, 
                               help='Target KL divergence')
    training_group.add_argument('--anneal-lr', action='store_true',
                               help='Anneal learning rate')
    training_group.add_argument('--norm-adv', action='store_true',
                               help='Normalize advantages')
    training_group.add_argument('--clip-vloss', action='store_true',
                               help='Clip value loss')
    
    # CNN Architecture
    cnn_group = parser.add_argument_group('CNN Architecture')
    cnn_group.add_argument('--frame-stack', type=int, default=4, 
                          help='Number of frames to stack')
    cnn_group.add_argument('--cnn-hidden-size', type=int, default=512,
                          help='CNN hidden layer size')
    
    # System configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default='cuda', 
                             choices=['cpu', 'cuda'], help='Device to use')
    system_group.add_argument('--torch-deterministic', action='store_true', 
                             help='Make PyTorch operations deterministic')
    system_group.add_argument('--compile', action='store_true',
                             help='Compile the model with torch.compile')
    system_group.add_argument('--compile-mode', type=str, default='default',
                             help='PyTorch compile mode')
    system_group.add_argument('--cpu-offload', action='store_true',
                             help='Offload data to CPU')
    system_group.add_argument('--zero-copy', action='store_true',
                             help='Use zero-copy optimization')
    
    # Logging and checkpointing
    logging_group = parser.add_argument_group('Logging and Checkpointing')
    logging_group.add_argument('--headless', action='store_true', 
                              help='Run without rendering')
    logging_group.add_argument('--debug', action='store_true', 
                              help='Enable debug mode')
    logging_group.add_argument('--track', action='store_true', 
                              help='Track experiment with Weights & Biases')
    logging_group.add_argument('--wandb-project', type=str, default='pokemon-pinball-pufferlib', 
                              help='WandB project name')
    logging_group.add_argument('--wandb-group', type=str, default='cnn-training',
                              help='WandB group name')
    logging_group.add_argument('--exp-id', type=str, default=None,
                              help='Experiment ID')
    logging_group.add_argument('--data-dir', type=str, default='experiments',
                              help='Data directory for checkpoints')
    logging_group.add_argument('--checkpoint-interval', type=int, default=100, 
                              help='Checkpoint save interval (in updates)')
    
    # Vectorization
    vec_group = parser.add_argument_group('Vectorization')
    vec_group.add_argument('--vec', type=str, default='ray',
                          choices=['serial', 'multiprocessing', 'ray'],
                          help='Vectorization backend')
    vec_group.add_argument('--vec-overwork', action='store_true',
                          help='Allow vec overwork')
    
    return parser


def make_pokemon_env(env_config, rom_path):
    """Environment creation function compatible with PufferLib."""
    def _make():
        from pokemon_pinball_env import PokemonPinballEnv
        
        # Create the base Pokemon Pinball environment
        env = PokemonPinballEnv(rom_path, env_config)
        
        # Apply postprocessing wrappers similar to Atari preprocessing
        if env_config.get('clip_reward', False):
            env = pufferlib.postprocess.ClipReward(
                env, 
                env_config.get('reward_clip_min', -10.0),
                env_config.get('reward_clip_max', 10.0)
            )
        
        if env_config.get('reduce_screen_resolution', True):
            env = pufferlib.postprocess.ResizeObservation(env, downscale=2)
            
        if env_config.get('frame_stack', 1) > 1:
            env = pufferlib.postprocess.FrameStack(env, env_config['frame_stack'])
            
        if env_config.get('max_episode_frames', 0) > 0:
            env = pufferlib.postprocess.TimeLimit(env, env_config['max_episode_frames'])
            
        # Add episode statistics tracking
        env = pufferlib.postprocess.EpisodeStats(env)
        
        # Convert to PufferLib format
        env = pufferlib.emulation.GymnasiumPufferEnv(env)
        
        return env
    
    return _make


def create_env_config(args):
    """Create environment configuration dictionary from parsed arguments."""
    return {
        'headless': args.headless,
        'debug': args.debug,
        'reward_shaping': args.reward_mode,
        'info_level': 0,  # Use only visual observations for CNN
        'frame_stack': args.frame_stack,
        'frame_skip': args.frame_skip,
        'visual_mode': args.visual_mode,
        'reduce_screen_resolution': args.reduce_screen_resolution,
        'max_episode_frames': args.max_episode_frames,
        'episode_mode': args.episode_mode,
        'reset_condition': args.reset_condition,
        'clip_reward': args.clip_reward,
        'reward_clip_max': args.reward_clip_max,
        'reward_clip_min': args.reward_clip_min,
    }


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 0.0  # Assume terminal
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """PPO Trainer using PufferLib components."""
    
    def __init__(self, config, vecenv, policy, optimizer=None, wandb=None):
        self.config = config
        self.vecenv = vecenv
        self.policy = policy
        self.wandb = wandb
        
        # Set up optimizer
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate, eps=1e-5
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.start_time = None
        
        # Storage for rollouts
        self.batch_size = config.batch_size
        self.num_envs = vecenv.num_agents
        self.num_steps = config.batch_size // self.num_envs
        
        # Initialize storage tensors
        self.obs = torch.zeros((self.num_steps, self.num_envs) + vecenv.single_observation_space.shape, 
                              dtype=torch.uint8).to(config.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long).to(config.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(config.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(config.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(config.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(config.device)
        
        # Environment state
        self.next_obs = None
        self.next_done = torch.zeros(self.num_envs).to(config.device)
        
        print(f"PPO Trainer initialized:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Steps per env: {self.num_steps}")
        print(f"  Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    def collect_rollouts(self):
        """Collect a batch of rollouts from the environments."""
        # Initialize if needed
        if self.next_obs is None:
            self.next_obs = torch.tensor(self.vecenv.reset()[0]).to(self.config.device)
            self.start_time = time.time()
        
        # Collect rollouts
        for step in range(self.num_steps):
            self.global_step += self.num_envs
            
            # Store current observations
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            
            # Get action from policy
            with torch.no_grad():
                # Convert to float and normalize
                obs_input = self.next_obs.float() / 255.0
                action, logprob, _, value = self.policy.get_action_and_value(obs_input)
                self.values[step] = value.flatten()
                
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # Execute actions
            obs, reward, done, truncated, info = self.vecenv.step(action.cpu().numpy())
            
            # Store transitions
            self.rewards[step] = torch.tensor(reward).to(self.config.device).view(-1)
            self.next_obs = torch.tensor(obs).to(self.config.device)
            self.next_done = torch.tensor(np.logical_or(done, truncated)).to(self.config.device)
    
    def compute_advantages(self):
        """Compute advantages and returns using GAE."""
        with torch.no_grad():
            # Get value of next observation
            next_obs_input = self.next_obs.float() / 255.0
            next_value = self.policy.get_value(next_obs_input).reshape(1, -1)
            
            # Convert to numpy for GAE computation
            rewards_np = self.rewards.cpu().numpy()
            values_np = self.values.cpu().numpy()
            dones_np = self.dones.cpu().numpy()
            
            # Add next value to values for GAE computation
            values_with_next = np.concatenate([values_np, next_value.cpu().numpy()], axis=0)
            
            # Compute GAE
            advantages = np.zeros_like(rewards_np)
            gae = 0
            
            for t in reversed(range(self.num_steps)):
                next_non_terminal = 1.0 - dones_np[t]
                next_value = values_with_next[t + 1]
                delta = rewards_np[t] + self.config.gamma * next_value * next_non_terminal - values_np[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
            
            returns = advantages + values_np
            
            # Convert back to tensors
            self.advantages = torch.tensor(advantages).to(self.config.device)
            self.returns = torch.tensor(returns).to(self.config.device)
    
    def update_policy(self):
        """Update the policy using PPO."""
        # Flatten batch dimensions
        b_obs = self.obs.reshape(-1, *self.vecenv.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # Normalize advantages
        if self.config.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Training loop
        batch_indices = np.arange(self.batch_size)
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(batch_indices)
            
            for start in range(0, self.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = batch_indices[start:end]
                
                # Get minibatch
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_values = b_values[mb_indices]
                
                # Forward pass
                obs_input = mb_obs.float() / 255.0
                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    obs_input, mb_actions
                )
                
                # Compute losses
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -self.config.vf_clip_coef,
                        self.config.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        # Learning rate annealing
        if self.config.anneal_lr:
            frac = 1.0 - self.global_step / self.config.total_timesteps
            lrnow = frac * self.config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
        
        # Compute explained variance
        y_pred = self.values.cpu().numpy().flatten()
        y_true = self.returns.cpu().numpy().flatten()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'explained_variance': explained_var,
            'learning_rate': self.optimizer.param_groups[0]["lr"],
        }
    
    def train_step(self):
        """Perform one training step."""
        # Collect rollouts
        self.collect_rollouts()
        
        # Compute advantages
        self.compute_advantages()
        
        # Update policy
        metrics = self.update_policy()
        
        self.epoch += 1
        return metrics
    
    def log_metrics(self, metrics):
        """Log training metrics."""
        if self.start_time is None:
            return
            
        # Calculate SPS
        elapsed_time = time.time() - self.start_time
        sps = self.global_step / elapsed_time if elapsed_time > 0 else 0
        
        # Print progress
        if self.epoch % 10 == 0:
            print(f"Epoch {self.epoch} | Step {self.global_step:,} | SPS {sps:.0f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Explained Var: {metrics['explained_variance']:.4f}")
            print(f"  Learning Rate: {metrics['learning_rate']:.6f}")
            print()
        
        # WandB logging
        if self.wandb is not None:
            self.wandb.log({
                'charts/learning_rate': metrics['learning_rate'],
                'charts/SPS': sps,
                'charts/global_step': self.global_step,
                'charts/epoch': self.epoch,
                'losses/policy_loss': metrics['policy_loss'],
                'losses/value_loss': metrics['value_loss'],
                'losses/entropy': metrics['entropy'],
                'losses/explained_variance': metrics['explained_variance'],
            }, step=self.global_step)
    
    def save_checkpoint(self, path):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def close(self):
        """Clean up resources."""
        self.vecenv.close()
        if self.wandb is not None:
            self.wandb.finish()
    """Calculate the flattened output size after CNN layers."""
    h, w = input_shape[:2]
    
    # Conv layer calculations for standard Atari CNN:
    # Conv1: kernel=8, stride=4 -> output = (input - 8) // 4 + 1
    # Conv2: kernel=4, stride=2 -> output = (input - 4) // 2 + 1  
    # Conv3: kernel=3, stride=1 -> output = input - 3 + 1
    
    h1 = (h - 8) // 4 + 1
    w1 = (w - 8) // 4 + 1
    
    h2 = (h1 - 4) // 2 + 1
    w2 = (w1 - 4) // 2 + 1
    
    h3 = h2 - 3 + 1
    w3 = w2 - 3 + 1
    
    # Final output: h3 * w3 * 64 (64 channels from last conv layer)
    flat_size = h3 * w3 * 64
    
def calculate_cnn_output_size(input_shape, frame_stack):
    """Calculate the flattened output size after CNN layers."""
    h, w = input_shape[:2]
    
    # Conv layer calculations for standard Atari CNN:
    # Conv1: kernel=8, stride=4 -> output = (input - 8) // 4 + 1
    # Conv2: kernel=4, stride=2 -> output = (input - 4) // 2 + 1  
    # Conv3: kernel=3, stride=1 -> output = input - 3 + 1
    
    h1 = (h - 8) // 4 + 1
    w1 = (w - 8) // 4 + 1
    
    h2 = (h1 - 4) // 2 + 1
    w2 = (w1 - 4) // 2 + 1
    
    h3 = h2 - 3 + 1
    w3 = w2 - 3 + 1
    
    # Final output: h3 * w3 * 64 (64 channels from last conv layer)
    flat_size = h3 * w3 * 64
    
    return flat_size


def make_policy(env, args):
    """Create policy using PufferLib's built-in CNN model."""
    obs_shape = env.single_observation_space.shape
    print(f"Observation space shape: {obs_shape}")
    
    if len(obs_shape) != 2:
        raise ValueError(f"Expected 2D observation space for CNN, got shape {obs_shape}")
    
    # Calculate the flat size for the CNN
    flat_size = calculate_cnn_output_size(obs_shape, args.frame_stack)
    print(f"Calculated CNN flat size: {flat_size}")
    
    # Create CNN policy
    policy = pufferlib.models.Convolutional(
        env, 
        framestack=args.frame_stack,
        flat_size=flat_size,
        hidden_size=args.cnn_hidden_size,
        channels_last=False  # PufferLib expects channels first
    )
    
    # Wrap with CleanRL policy wrapper
    policy = pufferlib.cleanrl.Policy(policy)
    
    return policy.to(args.device)


def init_wandb(args):
    """Initialize Weights & Biases tracking."""
    if not args.track:
        return None
        
    import wandb
    
    exp_id = args.exp_id or f"pokemon-pinball-cnn-{str(uuid.uuid4())[:8]}"
    
    wandb.init(
        id=exp_id,
        project=args.wandb_project,
        group=args.wandb_group,
        config=vars(args),
        name=exp_id,
        save_code=True,
        resume='allow'
    )
    
    return wandb


def setup_session_paths(args):
    """Setup session directories and paths for logging."""
    if args.exp_id:
        exp_id = args.exp_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"pokemon-pinball-cnn-{timestamp}"

    data_dir = Path(args.data_dir)
    makedirs(data_dir, exist_ok=True)
    
    return exp_id


def main():
    """Main training function using PufferLib with custom PPO trainer."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set defaults for optional arguments
    if args.num_workers is None:
        args.num_workers = args.num_envs
    if args.env_batch_size is None:
        args.env_batch_size = args.num_envs

    # Create configurations
    env_config = create_env_config(args)
    exp_id = setup_session_paths(args)
    args.exp_id = exp_id
    
    # Set up experiment tracking
    wandb = init_wandb(args)
    
    print(f"Starting Pokemon Pinball CNN Training with PufferLib")
    print(f"Experiment ID: {exp_id}")
    print(f"Device: {args.device}")
    print(f"Environments: {args.num_envs}")
    print(f"Workers: {args.num_workers}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"CNN Hidden size: {args.cnn_hidden_size}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Visual mode: {args.visual_mode}")
    print(f"Reward mode: {args.reward_mode}")
    print()
    
    # Seed everything
    pufferlib.utils.seed_everything(args.seed, args.torch_deterministic)
    
    # Create vectorized environments using PufferLib
    env_creator = make_pokemon_env(env_config, args.rom_path)
    
    # Choose vectorization backend
    if args.vec == 'serial':
        vec_backend = pufferlib.vector.Serial
    elif args.vec == 'multiprocessing':
        vec_backend = pufferlib.vector.Multiprocessing
    elif args.vec == 'ray':
        vec_backend = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vec choice: {args.vec}')
    
    print(f"Creating {args.num_envs} environments with {args.vec} backend...")
    vecenv = pufferlib.vector.make(
        env_creator,
        env_kwargs={},  # env_config is baked into the creator
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        batch_size=args.env_batch_size,
        zero_copy=args.zero_copy,
        backend=vec_backend,
    )
    
    # Create policy
    print("Creating CNN policy...")
    policy = make_policy(vecenv.driver_env, args)
    
    # Create training configuration
    train_config = pufferlib.namespace(
        # Environment settings
        env='pokemon_pinball',
        exp_id=exp_id,
        
        # Training hyperparameters
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        bptt_horizon=args.bptt_horizon,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        vf_clip_coef=args.vf_clip_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        
        # System settings
        device=args.device,
        torch_deterministic=args.torch_deterministic,
        compile=args.compile,
        compile_mode=args.compile_mode,
        cpu_offload=args.cpu_offload,
        seed=args.seed,
        
        # Logging
        data_dir=args.data_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    # Create trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(train_config, vecenv, policy, wandb=wandb)
    
    # Create checkpoint directory
    checkpoint_dir = Path(train_config.data_dir) / exp_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training loop...")
    print()
    
    # Main training loop
    try:
        while trainer.global_step < train_config.total_timesteps:
            # Perform training step
            metrics = trainer.train_step()
            
            # Log metrics
            trainer.log_metrics(metrics)
            
            # Save checkpoint
            if trainer.epoch % train_config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{trainer.epoch:06d}.pt"
                trainer.save_checkpoint(checkpoint_path)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        # Save final checkpoint
        final_checkpoint = checkpoint_dir / "final_checkpoint.pt"
        trainer.save_checkpoint(final_checkpoint)
        
        # Clean up
        print("Closing environments...")
        trainer.close()
        print("Training completed!")


if __name__ == "__main__":
    import time
    main()