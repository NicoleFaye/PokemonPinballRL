import sys
import argparse
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
import suppress_warnings  # Import the warning suppression module first
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
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
    parser.add_argument('--gamma', type=float, default=0.997, help='Discount factor')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--policy', type=str, default='MultiInputPolicy', help='PPO policy')
    # Environment configuration
    parser.add_argument('--reward-mode', type=str, default='basic', choices=['basic', 'catch_focused', 'comprehensive'], help='Reward shaping mode')
    parser.add_argument('--frame-stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--frame-skip', type=int, default=2, help='Number of frames to skip')
    parser.add_argument('--visual-mode', type=str, default='screen', choices=['screen', 'game_area'], help='Visual observation mode')
    parser.add_argument('--info-level', type=int, default=2, choices=[0, 1, 2, 3], help='Info level for environment')
    parser.add_argument('--frame-stack-extra-observation', action='store_true', help='Include extra frame-stack observations (positions & velocities)')
    parser.add_argument('--no-reduce-screen-resolution', dest='reduce_screen_resolution', action='store_false', help='Disable downsampling screen resolution')
    parser.set_defaults(reduce_screen_resolution=True)
    # Logging and runtime options
    parser.add_argument('--headless', action='store_true', help='Run without rendering')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (normal speed)')
    parser.add_argument('--no-wandb', dest='no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='pokemon-train-test', help='WandB project name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (.zip)')
    parser.add_argument('--num-cpu', type=int, default=6, help='Number of parallel environments')
    parser.add_argument('--save-freq-divisor', type=int, default=200, help='Divisor for checkpoint save frequency')
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
        'reduce_screen_resolution': args.reduce_screen_resolution
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
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=gamma, epsilon=1e-8)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path, name_prefix="poke")
    normalize_callback = VecNormCallback(save_freq=save_freq, save_path=sess_path, name_prefix="poke")
    callbacks = [checkpoint_callback, normalize_callback]

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
                        'num_cpu': num_cpu}
        run = wandb.init(project=args.wandb_project, id=sess_id, resume="allow",
                         name=sess_id, config=wandb_config, sync_tensorboard=True,
                         monitor_gym=True, save_code=True, dir=str(sess_path))
        callbacks.append(WandbCallback(verbose=1))

    # Initialize or resume model
    if args.resume and exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        norm_path = args.resume.replace(".zip", "") + "_vecnormalize.pkl"
        if exists(norm_path):
            print(f"Loading normalization stats from: {norm_path}")
            env = VecNormalize.load(norm_path, env)
            env.training = True
            env.norm_obs = False
        else:
            print("No normalization stats found, using default initialization")
        model = PPO.load(args.resume, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))
        train_target = model.num_timesteps + time_steps
    else:
        model = PPO(args.policy, env, verbose=1, n_steps=train_steps_batch,
                    batch_size=args.batch_size, n_epochs=args.n_epochs,
                    gamma=gamma, ent_coef=args.ent_coef, tensorboard_log=None)
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
