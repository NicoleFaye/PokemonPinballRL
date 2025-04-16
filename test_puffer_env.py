#!/usr/bin/env python3
"""
Test script to verify PufferLib environment integration for Pokemon Pinball.
"""
import argparse
import numpy as np
import sys
from pathlib import Path

try:
    import pufferlib
    import pufferlib.vector
except ImportError:
    print("PufferLib not available. Please install it with 'pip install pufferlib'")
    sys.exit(1)

try:
    from pyboy import PyBoy
except ImportError:
    print("PyBoy not available. Please install it with 'pip install pyboy'")
    sys.exit(1)

# Import our environment
from environment import make, env_creator


def main():
    """Test the PufferLib environment integration."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test PufferLib environment for Pokemon Pinball")
    parser.add_argument("--rom", type=str, required=True, help="Path to Pokemon Pinball ROM file")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of environments to test")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    
    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"Error: ROM file not found at {rom_path}")
        return
    
    # Test environment creation
    print(f"Testing environment creation...")
    env_kwargs = {
        "rom_path": str(rom_path),
        "headless": args.headless,
        "reward_shaping": "comprehensive",
        "frame_skip": 4,
        "framestack": 4
    }
    
    # Create a single environment first
    print("Creating a single environment...")
    try:
        single_env = make("pokemon_pinball", **env_kwargs)
        print("✓ Single environment created successfully")
        print(f"  Observation space: {single_env.single_observation_space}")
        print(f"  Action space: {single_env.single_action_space}")
        
        # Test reset and step
        print("Testing reset...")
        obs, info = single_env.reset()
        print(f"  Observation shape: {obs[0].shape}")
        print(f"  Info: {info}")
        
        print("Testing step...")
        action = np.array([single_env.single_action_space.sample()])
        obs, reward, done, truncated, info = single_env.step(action)
        print(f"  Observation shape: {obs[0].shape}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        
        # Close the environment
        single_env.close()
        print("✓ Single environment test completed successfully")
    except Exception as e:
        print(f"✗ Single environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test vectorized environment creation
    print(f"\nTesting vectorized environment creation with {args.num_envs} environments...")
    print("Note: This may take a while as each environment must load the ROM...")
    try:
        
        # Create a vectorized environment
        print("Creating vectorized environment creator...")
        make_env = env_creator("pokemon_pinball")
        
        print(f"Creating {args.num_envs} vectorized environments...")
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=env_kwargs,
            num_envs=args.num_envs,
            backend=pufferlib.vector.Multiprocessing
        )
        
        
        print(f"✓ Vectorized environment created successfully with {args.num_envs} environments")
        print(f"  Driver environment type: {type(vecenv.driver_env)}")
        print(f"  Observation space: {vecenv.single_observation_space}")
        print(f"  Action space: {vecenv.single_action_space}")
        
        # Test reset and step
        print("Testing reset...")
        obs = vecenv.reset()
        print(f"  Observation shape: {obs[0].shape}")
        
        print("Testing step...")
        action = np.array([vecenv.single_action_space.sample() for _ in range(args.num_envs)])
        obs, reward, done, truncated, info = vecenv.step(action)
        print(f"  Observation shape: {obs[0].shape}")
        print(f"  Reward shape: {reward.shape}")
        print(f"  Done shape: {done.shape}")
        print(f"  Truncated shape: {truncated.shape}")
        print(f"  Info length: {len(info)}")
        
        # Close the environment
        vecenv.close()
        print("✓ Vectorized environment test completed successfully")
    except Exception as e:
        print(f"✗ Vectorized environment test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up any orphaned processes
        try:
            import os
            import signal
            current_pid = os.getpid()
            print(f"Attempting to clean up orphaned processes for PID {current_pid}")
            os.system(f"pkill -P {current_pid}")
        except:
            pass
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()