import argparse
from os import _exit
from os.path import exists
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pokemon_pinball_env import PokemonPinballEnv, DEFAULT_CONFIG
import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: _exit(0))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pokémon Pinball RL model at real-time (1×) speed."
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the trained model (.zip) file'
    )
    parser.add_argument(
        '--rom-path',
        dest='rom_path',
        type=str,
        default='roms/pokemon_pinball.gbc',
        help='Path to the Pokémon Pinball GBC ROM'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic actions during evaluation'
    )
    return parser.parse_args()

def make_env(rom_path):
    # Match the exact config used during training
    config = DEFAULT_CONFIG.copy()
    config['debug'] = True # normal speed (1×)
    config['headless'] = False # display window
    config['visual_mode'] = 'screen'
    return PokemonPinballEnv(rom_path, config)

def main():
    args = parse_args()

    # Wrap environment in DummyVecEnv single instance
    env = DummyVecEnv([lambda: make_env(args.rom_path)])
    
    # Look for normalization statistics file
    norm_path = args.model_path.replace(".zip", "") + "_vecnormalize.pkl"
    if exists(norm_path):
        print(f"Loading normalization statistics from: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        # Configure normalization for evaluation
        env.training = False       # Don't update stats during eval
        env.norm_reward = False    # Don't normalize rewards during eval
    else:
        print(f"Warning: No normalization statistics found at {norm_path}")
    
    # Load model with environment
    model = PPO.load(args.model_path, env=env)
    
    obs = env.reset()
    episode = 1
    deterministic = args.deterministic
    
    if deterministic:
        print(f"Starting evaluation at 1× speed with DETERMINISTIC actions. Press Ctrl+C to stop.")
    else:
        print(f"Starting evaluation at 1× speed with NON-DETERMINISTIC actions. Press Ctrl+C to stop.")

    try:
        while True:
            # Predict action - use deterministic arg from command line
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, rewards, dones, infos = env.step(action)
            
            # Episode done
            if dones[0]:
                score = infos[0].get('score', [None])[0]
                print(f"Episode {episode} finished. Score: {score}")
                episode += 1
                obs = env.reset()
    except KeyboardInterrupt:
        print("Evaluation interrupted. Exiting...")

if __name__ == '__main__':
    main()