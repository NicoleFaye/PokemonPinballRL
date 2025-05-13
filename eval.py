import argparse

from os import _exit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pokemon_pinball_env import PokemonPinballEnv, DEFAULT_CONFIG

import signal  # Aggressively exit on ctrl+c
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
    return parser.parse_args()


def make_env(rom_path):
    # Match the exact config used during training
    config = DEFAULT_CONFIG.copy()
    config['debug'] = True      # normal speed (1×)
    config['headless'] = False  # display window
    config['visual_mode'] = 'screen'
    return PokemonPinballEnv(rom_path, config)


def main():
    args = parse_args()

    # Wrap environment in DummyVecEnv single instance
    env = DummyVecEnv([lambda: make_env(args.rom_path)])

    # Load model with matching single-env vectorization
    model = PPO.load(args.model_path, env=env)

    obs = env.reset()
    episode = 1
    print("Starting evaluation at 1× speed. Press Ctrl+C to stop.")

    try:
        while True:
            # Predict action for batch of size 1
            action, _ = model.predict(obs, deterministic=True)
            # Step environment
            obs, rewards, dones, infos = env.step(action)

            # Log any non-idle action
            if action[0] != 0:
                print(f"Episode {episode}: action {int(action[0])}")

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