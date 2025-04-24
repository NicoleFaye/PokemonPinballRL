import sys
from os.path import exists
from os import _exit, makedirs
from pathlib import Path
import suppress_warnings  # Import the warning suppression module first
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from environment import PokemonPinballEnv, Actions, RewardShaping
from environment.tensorboard_callback import TensorboardCallback


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
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    use_wandb_logging = True 
    ep_length = 2048 * 80

    env_config = {
        'headless': False,
        'debug': False,
        'reward_shaping': 'basic',
        'info_level': 2,
        'frame_stack': 4,
        'frame_skip': 2,
        'visual_mode': 'screen',
        'frame_stack_extra_observation': False,
        'reduce_screen_resolution': True
    }

    from datetime import datetime
    
    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_id = f"{env_config['reward_shaping']}_{timestamp}"
    sess_path = Path("runs/"+sess_id)
    
    print(env_config)
    
    num_cpu = 6 # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    # Add our custom TensorBoard callback
    tensorboard_callback = TensorboardCallback(log_dir=sess_path, window_size=20)
    
    callbacks = [checkpoint_callback, tensorboard_callback]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        
        # Make sure runs directory exists
        makedirs("./runs", exist_ok=True)
        
        # Patch TensorBoard at the root level to capture all grouped metrics
        wandb.tensorboard.patch(root_logdir="./runs")
        
        # Initialize WandB
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir="./runs",  # Set the wandb directory to avoid nesting
        )
        
        # Create notes explaining the metrics and x-axis
        run.notes = """
## Pokemon Pinball RL Training Metrics Guide

**X-axis in graphs**: "Step" in WandB refers to environment timesteps (individual actions), not episodes.

### Metric Groups:
1. **episode_metrics/**: Raw values for each completed episode
2. **rolling_averages/**: Smoothed metrics over multiple episodes (window size = 20)
3. **episode_tracking/**: Relationship between timesteps and episodes

### Key Metrics:
- **episode_tracking/total_episodes_completed**: How many full episodes have been played
- **episode_tracking/ratio_timesteps_per_episode**: Average timesteps needed to complete an episode
- **episode_tracking/total_environment_timesteps**: Total action steps taken across all environments

### Understanding Steps vs Episodes:
- WandB's x-axis "Step" = Environment timesteps (individual actions)
- One complete game/episode typically contains many timesteps
- Example: At step 50,000 you might have completed ~165 episodes if each takes ~300 timesteps
"""
        
        # Configure WandB callback with minimal options
        wandb_callback = WandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from    
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip() #"runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # Set PPO to log directly to the metrics directory, without any prefix
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, 
              gamma=0.997, ent_coef=0.01, tensorboard_log=None)
              
        # Configure the logger manually - use main runs directory
        from stable_baselines3.common.logger import configure
        model.set_logger(configure("./runs", ["stdout", "tensorboard"]))
    
    print(model.policy)

    # Don't specify a tb_log_name to avoid the prefix altogether
    model.learn(total_timesteps=(ep_length)*num_cpu*10000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()