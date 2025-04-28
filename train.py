import sys
import argparse
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
import suppress_warnings  # Import the warning suppression module first
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from pokemon_pinball_env import PokemonPinballEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: _exit(0))

class VecNormCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vecnormalize = self.model.get_vec_normalize_env()
            vecnormalize.save(f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}_vecnormalize.pkl")
        return True

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PokemonPinballEnv("./roms/pokemon_pinball.gbc",env_conf)
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Pokemon Pinball RL agent')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                        help='Number of timesteps to train for (default: 10,000,000)')
    parser.add_argument('--reward_mode', type=str, default='basic', choices=['basic', 'catch_focused', 'comprehensive'],
                        help='Reward shaping mode (default: basic)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from (with .zip extension)')
    args = parser.parse_args()

    use_wandb_logging = not args.no_wandb

    time_steps = args.timesteps

    env_config = {
        'headless': args.headless,
        'debug': args.debug,
        'reward_shaping': args.reward_mode,
        'info_level': 2,
        'frame_stack': 4,
        'frame_skip': 2,
        'visual_mode': 'screen',
        'frame_stack_extra_observation': False,
        'reduce_screen_resolution': True
    }

    gamma = 0.997

    from datetime import datetime
    
    # Create a unique timestamp for this run or extract session ID from resume path
    if args.resume:
        # Extract the session ID from the resume path
        resume_path = Path(args.resume)
        reset_flag = False
        sess_id = resume_path.parent.name
    else:
        # New run
        reset_flag = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_id = f"{env_config['reward_shaping']}_{timestamp}"
    
    sess_path = Path("runs/"+sess_id)
    makedirs(sess_path, exist_ok=True)
    
    print(env_config)
    
    num_cpu = 6
    save_freq_divisor = 200
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # Add VecNormalize wrapper
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        epsilon=1e-8
    )
    
    # Create the regular checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=time_steps//save_freq_divisor, 
        save_path=sess_path,
        name_prefix="poke"
    )
    normalize_callback = VecNormCallback(
        save_freq=time_steps//save_freq_divisor,
        save_path=sess_path,
        name_prefix="poke"
    )
    
    callbacks = [checkpoint_callback, normalize_callback]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        
        # Make sure runs directory exists
        makedirs(sess_path, exist_ok=True)
        
        # Patch TensorBoard at the root level to capture all grouped metrics
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        
        # Initialize WandB
        run = wandb.init(
            project="pokemon-train-test",
            id=sess_id,
            resume="allow",
            name=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=str(sess_path),  
        )
        
        wandb_callback = WandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    train_steps_batch = 2048  # Standard PPO default

    # Check for resume checkpoint
    checkpoint_path = args.resume if args.resume else ""
    
    # If not specified via args, check stdin (for backwards compatibility)
    if not checkpoint_path and not sys.stdin.isatty():
        checkpoint_path = sys.stdin.read().strip()
    
    if checkpoint_path and exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        
        # Check if normalization stats exist alongside the checkpoint
        norm_path = checkpoint_path.replace(".zip", "") + "_vecnormalize.pkl"
        if exists(norm_path):
            print(f"Loading normalization stats from: {norm_path}")
            env = VecNormalize.load(norm_path, env)
            # Keep collecting running statistics during training
            env.training = True
            # Ensure observation normalization is disabled
            env.norm_obs = False
        else:
            print("No normalization stats found, using default initialization")
        
        # Load the model with the normalized environment
        model = PPO.load(checkpoint_path, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()

        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))

        train_target = model.num_timesteps + time_steps
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, 
              gamma=gamma, ent_coef=0.01, tensorboard_log=None)
        
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))

        train_target = time_steps
    
    print(model.policy)

    model.learn(total_timesteps=train_target, callback=CallbackList(callbacks),reset_num_timesteps=reset_flag)

    # Save final model and normalization stats
    final_model_path = f"{sess_path}/poke_final"
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")

    if use_wandb_logging:
        run.finish()