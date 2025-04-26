import sys
import argparse
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
import suppress_warnings  # Import the warning suppression module first
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from pokemon_pinball_env import PokemonPinballEnv


import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: _exit(0))

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
    parser.add_argument('--window_size', type=int, default=100,
                        help='Size of window for rolling metrics (default: 100)')
    parser.add_argument('--reward_mode', type=str, default='basic', choices=['basic', 'catch_focused', 'comprehensive'],
                        help='Reward shaping mode (default: basic)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from (without .zip extension)')
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

    from datetime import datetime
    
    # Create a unique timestamp for this run or extract session ID from resume path
    if args.resume:
        # Extract the session ID from the resume path
        resume_path = Path(args.resume)
        # If the resume path includes a runs directory, use that session ID
        if "runs" in str(resume_path):
            # Get the parent directory which should be the session directory
            sess_id = resume_path.parent.name
        else:
            # If just a checkpoint name was provided, use a new session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sess_id = f"{env_config['reward_shaping']}_resumed_{timestamp}"
    else:
        # New run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_id = f"{env_config['reward_shaping']}_{timestamp}"
    
    sess_path = Path("runs/"+sess_id)
    makedirs(sess_path, exist_ok=True)
    
    print(env_config)
    
    num_cpu = 6
    save_freq_divisor = 200
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=time_steps//save_freq_divisor, save_path=sess_path,
                                     name_prefix="poke")
    
    
    callbacks = [checkpoint_callback]

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
            name=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=str(sess_path),  # Store wandb data in the session folder
        )
        
        
        # Configure WandB callback with minimal options
        wandb_callback = WandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    #env_checker.check_env(env)

    # Define a smaller n_steps for the buffer size to avoid memory issues
    train_steps_batch = 2048  # Standard PPO default

    # Check for resume checkpoint
    checkpoint_path = args.resume if args.resume else ""
    
    # If not specified via args, check stdin (for backwards compatibility)
    if not checkpoint_path and not sys.stdin.isatty():
        checkpoint_path = sys.stdin.read().strip()
    
    if checkpoint_path and exists(checkpoint_path + ".zip"):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # Set PPO to log directly to the metrics directory, without any prefix
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, 
              gamma=0.997, ent_coef=0.01, tensorboard_log=None)
              
        # Configure the logger to use the session path
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(str(sess_path), ["stdout", "tensorboard"]))
    
    print(model.policy)

    # Use the timesteps parameter for training
    # The buffer size (n_steps=2048) is already set properly in the model
    model.learn(total_timesteps=time_steps, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()