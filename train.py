import sys
import argparse
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
import suppress_warnings  # Import the warning suppression module first
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn, constant_fn, set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from pokemon_pinball_env import PokemonPinballEnv

#import signal  # Aggressively exit on ctrl+c
#signal.signal(signal.SIGINT, lambda sig, frame: _exit(0))


class VecNormCallback(BaseCallback):
    """Callback for saving VecNormalize statistics at regular intervals."""
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Save normalization stats every save_freq calls
        if self.n_calls % self.save_freq == 0:
            vecnorm = self.model.get_vec_normalize_env()
            # Use num_timesteps for naming consistency
            vecnorm.save(f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}_vecnormalize.pkl")
        return True


class RewardMonitorCallback(BaseCallback):
    """Monitor different reward components and normalized rewards"""
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.raw_rewards = []
        self.shaped_rewards = []
        self.normalized_rewards = []
        self.clipped_rewards = []
        
    def _on_step(self) -> bool:
        # Only log periodically to avoid slowdown
        if self.n_calls % self.log_freq == 0:
            # Get the VecNormalize wrapper
            vec_normalize = self.model.get_vec_normalize_env()
            
            # Get raw rewards (before normalization)
            if hasattr(vec_normalize, 'return_rms'):
                # Sample some recent rewards
                recent_rewards = vec_normalize.unwrapped.get_attr('_fitness')[:5]  # First 5 envs
                recent_raw_diffs = vec_normalize.unwrapped.get_attr('_fitness')[:5] - \
                                   vec_normalize.unwrapped.get_attr('_previous_fitness')[:5]
                
                # Calculate normalized values (mimicking VecNormalize internal logic)
                if hasattr(vec_normalize.return_rms, 'mean') and hasattr(vec_normalize.return_rms, 'var'):
                    norm_mean = vec_normalize.return_rms.mean
                    norm_var = vec_normalize.return_rms.var
                    norm_std = norm_var ** 0.5
                    
                    # Calculate what normalized values would be
                    normalized = [(r - norm_mean) / (norm_std + 1e-8) for r in recent_raw_diffs]
                    # Calculate what clipped values would be
                    clip_val = vec_normalize.clip_reward
                    clipped = [max(min(r, clip_val), -clip_val) for r in normalized]
                    
                    # Log these values
                    self.logger.record("reward/raw_rewards_mean", float(sum(recent_raw_diffs))/len(recent_raw_diffs))
                    self.logger.record("reward/raw_rewards_min", float(min(recent_raw_diffs)))
                    self.logger.record("reward/raw_rewards_max", float(max(recent_raw_diffs)))
                    self.logger.record("reward/norm_mean", float(norm_mean))
                    self.logger.record("reward/norm_std", float(norm_std))
                    self.logger.record("reward/normalized_rewards_mean", float(sum(normalized))/len(normalized))
                    self.logger.record("reward/normalized_rewards_min", float(min(normalized)))
                    self.logger.record("reward/normalized_rewards_max", float(max(normalized)))
                    self.logger.record("reward/clipped_percent", 
                                      100 * sum(1 for r in normalized if r > clip_val or r < -clip_val) / len(normalized))
        return True


class LearningRateMonitorCallback(BaseCallback):
    """Callback for monitoring learning rate during training."""
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get current learning rate from the optimizer
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    current_lr = param_group['lr']
                    
                    # Log to console
                    print(f"Step: {self.n_calls}, Current learning rate: {current_lr}")
                    
                    # Log to tensorboard
                    self.logger.record("train/learning_rate", current_lr)
                    
                    # Log to wandb if available
                    if 'wandb' in sys.modules and hasattr(sys.modules['wandb'], 'log'):
                        import wandb
                        wandb.log({"learning_rate": current_lr}, step=self.num_timesteps)
        return True


class ClipRangeMonitorCallback(BaseCallback):
    """Callback for monitoring clip range during training."""
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get current clip range value
            if hasattr(self.model, 'clip_range'):
                if callable(self.model.clip_range):
                    # If it's a function, we need to calculate the current value
                    # based on the remaining progress
                    total_timesteps = self.model._total_timesteps
                    progress_remaining = 1.0 - (self.num_timesteps / total_timesteps)
                    current_clip_range = self.model.clip_range(progress_remaining)
                else:
                    # If it's a constant
                    current_clip_range = self.model.clip_range
                
                # Log to console
                print(f"Step: {self.n_calls}, Current clip range: {current_clip_range}")
                
                # Log to tensorboard
                self.logger.record("train/clip_range", current_clip_range)
                
                # Log to wandb if available
                if 'wandb' in sys.modules and hasattr(sys.modules['wandb'], 'log'):
                    import wandb
                    wandb.log({"clip_range": current_clip_range}, step=self.num_timesteps)
        return True


def configure_decay_rate(initial_value, schedule_type, final_value_fraction=0.1):
    """
    Configure the learning rate scheduler based on the specified type.
    
    Args:
        initial_value: Initial value
        schedule_type: Type of schedule ('constant', 'linear', or 'exponential')
        final_value_fraction: Final value as a fraction of initial (for non-constant schedules)
        
    Returns:
        A callable schedule function
    """
    final_value = initial_value * final_value_fraction
    
    if schedule_type == 'constant':
        # No decay - constant value
        return constant_fn(initial_value)
    
    elif schedule_type == 'linear':
        # Linear decay from initial_value to final_value
        return get_linear_fn(initial_value, final_value, 1.0)
    
    elif schedule_type == 'exponential':
        # Custom exponential decay function
        def exponential_schedule(progress_remaining):
            # Exponential decay from initial_value to final_value
            # progress_remaining goes from 1.0 to 0.0
            return final_value + (initial_value - final_value) * progress_remaining ** 2
        
        return exponential_schedule
    
    else:
        # Default to constant if unrecognized schedule type
        print(f"Warning: Unrecognized schedule type '{schedule_type}'. Using constant value.")
        return constant_fn(initial_value)


def make_env(rank, env_conf, seed=0):
    def _init():
        env = PokemonPinballEnv("./roms/pokemon_pinball.gbc", env_conf)
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball RL agent')
    # Training hyperparameters
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total timesteps to train for')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps per update (n_steps)')
    parser.add_argument('--batch-size', type=int, default=512, help='Mini-batch size for PPO')
    parser.add_argument('--n-epochs', type=int, default=1, help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--policy', type=str, default='MultiInputPolicy', help='PPO policy')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm for clipping')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda for advantage estimation')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,help='Initial learning rate (default: 2.5e-4 as recommended for Atari games)')
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'linear', 'exponential'],help='Learning rate schedule type (default: constant)')
    parser.add_argument('--clip-range-schedule', type=str, default='constant', choices=['constant', 'linear', 'exponential'],help='Clip range schedule type (default: constant)')
    parser.add_argument('--final-lr-fraction', type=float, default=0.1,help='Final learning rate as a fraction of initial rate (default: 0.1, 10% of initial rate)')
    parser.add_argument('--final-clip-range-fraction', type=float, default=0.1,help='Final clip value as a fraction of initial rate (default: 0.1, 10% of initial value)')
    # Environment configuration
    parser.add_argument('--seed', type=int, default=0, help='Random seed for environment')
    parser.add_argument('--reward-mode', type=str, default='basic', choices=['basic', 'catch_focused', 'comprehensive','progressive'], help='Reward shaping mode')
    parser.add_argument('--frame-stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--frame-skip', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--visual-mode', type=str, default='screen', choices=['screen', 'game_area'], help='Visual observation mode')
    parser.add_argument('--info-level', type=int, default=1, choices=[0, 1, 2, 3], help='Info level for environment')
    parser.add_argument('--max-episode-frames', type=int, default=0, help='Max frames per episode')
    parser.add_argument('--no-frame-stack-extra-observation',dest='frame_stack_extra_observation', action='store_false', help='Include extra frame-stack observations (positions & velocities)')
    parser.add_argument('--no-reduce-screen-resolution', dest='reduce_screen_resolution', action='store_false', help='Disable downsampling screen resolution')
    parser.add_argument('--reward-clip', type=float, default=3.0, help='Reward clipping value')
    parser.set_defaults(reduce_screen_resolution=True)
    parser.set_defaults(frame_stack_extra_observation=True)
    # Logging and runtime options
    parser.add_argument('--headless', action='store_true', help='Run without rendering')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (normal speed)')
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='pokemon-train-test', help='WandB project name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (.zip)')
    parser.add_argument('--num-cpu', type=int, default=6, help='Number of parallel environments')
    parser.add_argument('--save-freq-divisor', type=int, default=200, help='Divisor for checkpoint save frequency')
    parser.add_argument('--log-freq', type=int, default=100, help='Frequency for logging reward statistics')
    args = parser.parse_args()

    # Assign parameters
    time_steps = args.timesteps
    train_steps_batch = args.n_steps
    gamma = args.gamma
    use_wandb = not args.no_wandb

    # Build environment configuration
    env_config = {
        'headless': args.headless,
        'debug': args.debug,
        'reward_shaping': args.reward_mode,
        'info_level': args.info_level,
        'frame_stack': args.frame_stack,
        'frame_skip': args.frame_skip,
        'visual_mode': args.visual_mode,
        'frame_stack_extra_observation': args.frame_stack_extra_observation,
        'reduce_screen_resolution': args.reduce_screen_resolution,
        'max_episode_frames': args.max_episode_frames,
    }

    from datetime import datetime
    if args.resume:
        resume_path = Path(args.resume)
        reset_flag = False
        sess_id = resume_path.parent.name
    else:
        reset_flag = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_id = f"{env_config['reward_shaping']}_{timestamp}"

    sess_path = Path("runs/" + sess_id)
    makedirs(sess_path, exist_ok=True)
    print("Environment configuration:", env_config)

    num_cpu = args.num_cpu
    save_freq = max(1, time_steps // args.save_freq_divisor)

    # Create vectorized environments
    env = SubprocVecEnv([make_env(i, env_config, seed=args.seed) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, 
                       clip_obs=5.0, clip_reward=args.reward_clip, 
                       gamma=gamma, epsilon=1e-8)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path, name_prefix="poke")
    normalize_callback = VecNormCallback(save_freq=save_freq, save_path=sess_path, name_prefix="poke")
    reward_monitor_callback = RewardMonitorCallback(log_freq=args.log_freq)
    lr_monitor_callback = LearningRateMonitorCallback(log_freq=max(1, time_steps//100))  # Log LR ~100 times during training
    clip_range_monitor_callback = ClipRangeMonitorCallback(log_freq=max(1, time_steps//100))  # Log clip range ~100 times during training
    callbacks = [checkpoint_callback, normalize_callback, reward_monitor_callback, lr_monitor_callback, clip_range_monitor_callback]

    # Optional WandB logging
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        makedirs(sess_path, exist_ok=True)
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        # Combine configs for WandB
        wandb_config = {**env_config,
                        'gamma': gamma,
                        'n_steps': train_steps_batch,
                        'batch_size': args.batch_size,
                        'n_epochs': args.n_epochs,
                        'ent_coef': args.ent_coef,
                        'gae_lambda': args.gae_lambda,
                        'num_cpu': num_cpu,
                        'reward_clip': args.reward_clip,
                        'clip_range': args.clip_range,
                        'clip_range_schedule': args.clip_range_schedule,
                        'final_clip_range_fraction': args.final_clip_range_fraction,  
                        'max_grad_norm': args.max_grad_norm,
                        'learning_rate': args.learning_rate,
                        'lr_schedule': args.lr_schedule,
                        'final_lr_fraction': args.final_lr_fraction,
                        'seed': args.seed,
                        }
        run = wandb.init(project=args.wandb_project, id=sess_id, resume="allow",
                         name=sess_id, config=wandb_config, sync_tensorboard=True,
                         monitor_gym=True, save_code=True, dir=str(sess_path))
        callbacks.append(WandbCallback(verbose=1))

    # Configure learning rate based on arguments
    lr_schedule = configure_decay_rate(
        args.learning_rate, 
        args.lr_schedule, 
        args.final_lr_fraction
    )
    
    # Configure clip range based on arguments
    clip_range_schedule = configure_decay_rate(
        args.clip_range, 
        args.clip_range_schedule, 
        args.final_clip_range_fraction
    )
    
    print(f"Learning rate configuration:")
    print(f"  Initial rate: {args.learning_rate}")
    print(f"  Schedule: {args.lr_schedule}")
    if args.lr_schedule != 'constant':
        print(f"  Final rate: {args.learning_rate * args.final_lr_fraction}")

    print(f"Clip range configuration:")
    print(f"  Initial clip range: {args.clip_range}")
    print(f"  Schedule: {args.clip_range_schedule}")
    if args.clip_range_schedule != 'constant':
        print(f"  Final clip range: {args.clip_range * args.final_clip_range_fraction}")

    # Initialize or resume model
    if args.resume and exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        norm_path = args.resume.replace(".zip", "") + "_vecnormalize.pkl"
        if exists(norm_path):
            print(f"Loading normalization stats from: {norm_path}")
            env = VecNormalize.load(norm_path, env)
            env.training = True
            env.norm_obs = False
            # Make sure to update the clip_reward value to the current setting
            env.clip_reward = args.reward_clip
        else:
            print("No normalization stats found, using default initialization")
        
        model = PPO.load(args.resume, env=env)
        
        # Note about learning rate and clip range when resuming
        print("NOTE: When resuming, the original learning rate and clip range schedules from the checkpoint are used.")
        print("      The learning rate and clip range arguments provided now won't affect the resumed training.")
        print("      This is a limitation of the current Stable Baselines 3 implementation.")
        
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))
        train_target = model.num_timesteps + time_steps
    else:
        model = PPO(args.policy, env, verbose=1, n_steps=train_steps_batch,
                    batch_size=args.batch_size, n_epochs=args.n_epochs, gae_lambda=args.gae_lambda,
                    gamma=gamma, ent_coef=args.ent_coef, tensorboard_log=None,
                    clip_range=clip_range_schedule,  
                    learning_rate=lr_schedule,  
                    max_grad_norm=args.max_grad_norm)
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))
        train_target = time_steps

    print(model.policy)
    model.learn(total_timesteps=train_target, callback=CallbackList(callbacks), reset_num_timesteps=reset_flag)

    # Save final model
    final_model_path = sess_path / "poke_final"
    model.save(str(final_model_path))
    env.save(str(final_model_path) + "_vecnormalize.pkl")

    if use_wandb:
        run.finish()