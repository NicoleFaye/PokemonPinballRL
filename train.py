import sys
import argparse
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn, constant_fn, set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from gymnasium import wrappers
from pokemon_pinball_env import PokemonPinballEnv


def create_argument_parser():
    """Create and configure the argument parser for training configuration."""
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball RL agent')
    
    # Training hyperparameters
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument('--timesteps', type=int, default=10_000_000, 
                               help='Total timesteps to train for')
    training_group.add_argument('--n-steps', '--n_steps', type=int, default=2048, 
                               help='Number of steps per update (n_steps)')
    training_group.add_argument('--batch-size', '--batch_size', type=int, default=512, 
                               help='Mini-batch size for PPO')
    training_group.add_argument('--n-epochs', '--n_epochs', type=int, default=10, 
                               help='Number of epochs per update')
    training_group.add_argument('--gamma', type=float, default=0.99, 
                               help='Discount factor')
    training_group.add_argument('--ent-coef', '--ent_coef', type=float, default=0.01, 
                               help='Entropy coefficient')
    training_group.add_argument('--gae-lambda', '--gae_lambda', type=float, default=0.95, 
                               help='GAE lambda for advantage estimation')
    training_group.add_argument('--learning-rate', '--learning_rate', type=float, default=2.5e-4, 
                               help='Initial learning rate')
    training_group.add_argument('--lr-schedule', '--learning_rate_schedule', type=str, default='constant', 
                               choices=['constant', 'linear', 'exponential'], 
                               help='Learning rate schedule')
    training_group.add_argument('--clip-range', '--clip_range', type=float, default=0.2, 
                               help='PPO clip range')
    training_group.add_argument('--clip-range-schedule', type=str, default='constant', 
                               choices=['constant', 'linear', 'exponential'], 
                               help='Clip range schedule')
    training_group.add_argument('--final-lr-fraction', type=float, default=0.1, 
                               help='Final learning rate fraction')
    training_group.add_argument('--final-clip-range-fraction', type=float, default=0.1, 
                               help='Final clip range fraction')
    training_group.add_argument('--policy', type=str, default='MultiInputPolicy', 
                               help='PPO policy')
    training_group.add_argument('--max-grad-norm', '--max_grad_norm', type=float, default=0.5, 
                               help='Max gradient norm for clipping')
    training_group.add_argument('--target-kl', '--target_kl', type=float, default=None, 
                               help='Target KL divergence for early stopping')
    training_group.add_argument('--vf-coef', '--vf_coef', type=float, default=0.5, 
                               help='Value function coefficient')
    training_group.add_argument('--normalize-advantage', '--normalize_advantage', type=bool, default=True, 
                               help='Normalize advantage estimates')

    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--episode-mode', type=str, default='life', 
                          choices=['ball', 'life', 'game'], help='Episode mode')
    env_group.add_argument('--reset-condition', type=str, default='game', 
                          choices=['ball', 'life', 'game'], help='Reset condition')
    env_group.add_argument('--seed', type=int, default=0, 
                          help='Random seed for environment')
    env_group.add_argument('--reward-mode', '--reward_mode', type=str, default='basic', 
                          choices=['basic', 'catch_focused', 'comprehensive', 'progressive'], 
                          help='Reward shaping mode')
    env_group.add_argument('--info-level', '--info_level', type=int, default=1, 
                          choices=[0, 1, 2, 3], help='Info level for environment')
    env_group.add_argument('--frame-stack', type=int, default=4, 
                          help='Number of frames to stack')
    env_group.add_argument('--frame-skip', type=int, default=4, 
                          help='Number of frames to skip')
    env_group.add_argument('--visual-mode', type=str, default='screen', 
                          choices=['screen', 'game_area'], help='Visual observation mode')
    env_group.add_argument('--max-episode-frames', type=int, default=0, 
                          help='Max frames per episode')
    env_group.add_argument('--no-reduce-screen-resolution', dest='reduce_screen_resolution', 
                          action='store_false', help='Disable screen downsample')
    env_group.add_argument('--clip-reward', '--clip_reward', dest='clip_reward', 
                          action='store_true', help='Reward clipping value')
    env_group.add_argument('--reward-clip-max', type=float, default=10.0, 
                          help='Maximum reward clipping value')
    env_group.add_argument('--reward-clip-min', type=float, default=-10.0, 
                          help='Minimum reward clipping value')
    env_group.add_argument('--normalize-reward', dest='normalize_reward', 
                          action='store_true', help='Normalize rewards using VecNormalize')
    
    # Parallel environments
    parallel_group = parser.add_argument_group('Parallel Environments')
    parallel_group.add_argument('--num-cpu', '--n-envs', '--n_envs', type=int, default=24, 
                               help='Number of parallel environments')
    
    # Logging and runtime
    logging_group = parser.add_argument_group('Logging and Runtime')
    logging_group.add_argument('--headless', action='store_true', 
                              help='Run without rendering')
    logging_group.add_argument('--debug', action='store_true', 
                              help='Enable debug mode')
    logging_group.add_argument('--no-wandb', dest='no_wandb', action='store_true', 
                              help='Disable WandB logging')
    logging_group.add_argument('--wandb-project', type=str, default='pokemon-pinball-ppo', 
                              help='WandB project name')
    logging_group.add_argument('--resume', type=str, 
                              help='Path to checkpoint to resume from (.zip)')
    logging_group.add_argument('--save-freq-divisor', type=int, default=200, 
                              help='Divisor for checkpoint save frequency')
    logging_group.add_argument('--log-freq', type=int, default=100, 
                              help='Frequency for logging reward stats')
    
    return parser


def create_env_config(args):
    """Create environment configuration dictionary from parsed arguments."""
    return {
        'headless': args.headless,
        'debug': args.debug,
        'reward_shaping': args.reward_mode,
        'info_level': args.info_level,
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
        'normalize_reward': args.normalize_reward,
    }


def create_wandb_config(args, env_config, num_cpu):
    """Create WandB configuration dictionary from arguments."""
    return {
        **env_config,
        'gamma': args.gamma,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'ent_coef': args.ent_coef,
        'gae_lambda': args.gae_lambda,
        'n_envs': num_cpu,
        'learning_rate': args.learning_rate,
        'lr_schedule': args.lr_schedule,
        'final_lr_fraction': args.final_lr_fraction,
        'seed': args.seed,
        'target_kl': args.target_kl,
        'clip_range': args.clip_range,
        'clip_range_schedule': args.clip_range_schedule,
        'final_clip_range_fraction': args.final_clip_range_fraction,
        'clip_reward': args.clip_reward,
        'reward_clip_max': args.reward_clip_max,
        'reward_clip_min': args.reward_clip_min,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'normalize_advantage': args.normalize_advantage,
        'rollout_buffer_size': args.n_steps * num_cpu,
    }


def setup_session_paths(args, env_config):
    """Setup session directories and paths for logging."""
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
    
    return sess_path, sess_id, reset_flag


def print_schedule_info(args):
    """Print learning rate and clip range schedule information."""
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


def configure_decay_rate(initial_value, schedule_type, final_value_fraction=0.1):
    """Configure learning rate or clip range decay schedule."""
    final_value = initial_value * final_value_fraction
    if schedule_type == 'constant':
        return constant_fn(initial_value)
    elif schedule_type == 'linear':
        return get_linear_fn(initial_value, final_value, 1.0)
    elif schedule_type == 'exponential':
        def exponential_schedule(progress_remaining):
            return final_value + (initial_value - final_value) * progress_remaining ** 2
        return exponential_schedule
    else:
        print(f"Warning: Unrecognized schedule type '{schedule_type}'. Using constant value.")
        return constant_fn(initial_value)


def make_env(rank, env_conf, seed=0):
    """Create a single environment instance."""
    def _init():
        env = PokemonPinballEnv("./roms/pokemon_pinball.gbc", env_conf)
        env = Monitor(env)
        env = wrappers.FlattenObservation(env)
        env = wrappers.FrameStack(env, env_conf['frame_stack'])
        #if env_conf['clip_reward'] is not None:
        #    env = wrappers.ClipReward(env, env_conf['reward_clip_min'], env_conf['reward_clip_max'])
        if env_conf['normalize_reward']:
            env = wrappers.NormalizeReward(env)
        if env_conf['max_episode_frames'] > 0:
            env = wrappers.TimeLimit(env, max_episode_steps=env_conf['max_episode_frames'])
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def setup_wandb_logging(args, sess_path, sess_id, wandb_config):
    """Setup WandB logging if enabled."""
    if args.no_wandb:
        return []
    
    import wandb
    from wandb.integration.sb3 import WandbCallback
    
    makedirs(sess_path, exist_ok=True)
    wandb.tensorboard.patch(root_logdir=str(sess_path))

    run = wandb.init(
        project=args.wandb_project, 
        id=sess_id, 
        resume="allow",
        name=sess_id, 
        config=wandb_config, 
        sync_tensorboard=True,
        monitor_gym=True, 
        save_code=True, 
        dir=str(sess_path)
    )
    
    return [WandbCallback(verbose=1)]


def create_or_resume_model(args, env, sess_path, lr_schedule, clip_range_schedule, 
                          train_steps_batch, num_cpu, time_steps):
    """Create a new model or resume from checkpoint."""
    if args.resume and exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        
        model = PPO.load(args.resume, env=env)
        
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
        model = PPO(
            args.policy, env, verbose=1, n_steps=train_steps_batch,
            batch_size=args.batch_size, n_epochs=args.n_epochs, 
            gae_lambda=args.gae_lambda, gamma=args.gamma, ent_coef=args.ent_coef, 
            tensorboard_log=None, clip_range=clip_range_schedule,  
            learning_rate=lr_schedule, max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl, vf_coef=args.vf_coef,
            normalize_advantage=args.normalize_advantage,
        )
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))
        train_target = time_steps
    
    return model, train_target


def main():
    """Main training function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configurations
    env_config = create_env_config(args)
    sess_path, sess_id, reset_flag = setup_session_paths(args, env_config)
    
    # Setup environment
    num_cpu = args.num_cpu
    save_freq = max(1, args.timesteps // args.save_freq_divisor)
    env = SubprocVecEnv([make_env(i, env_config, seed=args.seed) for i in range(num_cpu)])

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path, name_prefix="poke")
    callbacks = [checkpoint_callback]

    # Setup WandB if enabled
    if not args.no_wandb:
        wandb_config = create_wandb_config(args, env_config, num_cpu)
        wandb_callbacks = setup_wandb_logging(args, sess_path, sess_id, wandb_config)
        callbacks.extend(wandb_callbacks)

    # Configure schedules
    lr_schedule = configure_decay_rate(args.learning_rate, args.lr_schedule, args.final_lr_fraction)
    clip_range_schedule = configure_decay_rate(args.clip_range, args.clip_range_schedule, args.final_clip_range_fraction)
    
    # Print configuration
    print_schedule_info(args)

    # Create or resume model
    model, train_target = create_or_resume_model(
        args, env, sess_path, lr_schedule, clip_range_schedule, 
        args.n_steps, num_cpu, args.timesteps
    )

    # Train model
    print(model.policy)
    model.learn(total_timesteps=train_target, callback=CallbackList(callbacks), reset_num_timesteps=reset_flag)

    # Save final model
    final_model_path = sess_path / "poke_final"
    model.save(str(final_model_path))
    env.save(str(final_model_path) + "_vecnormalize.pkl")

    # Clean up WandB if used
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
