#!/usr/bin/env python3
"""
Script to evaluate a trained Pokemon Pinball RL agent using PufferLib.
Loads a trained model created by PufferLib and tests it on the environment.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Check for required libraries
try:
    import pufferlib
    import pufferlib.cleanrl
    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False
    print("PufferLib not available. Please install with: pip install pufferlib")
    sys.exit(1)

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Please install with: pip install pyboy")
    sys.exit(1)

# Import our environment and models
from environment import make, env_creator
from models import CNNPolicy, MLPPolicy, ResNetPolicy, Recurrent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Pokemon Pinball agent")
    
    # Required arguments
    parser.add_argument("--rom", type=str, required=True, help="Path to Pokemon Pinball ROM file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.pt)")
    
    # Optional arguments
    parser.add_argument("--policy-type", type=str, default="cnn", choices=["mlp", "cnn", "resnet"],
                        help="Policy network architecture (must match the trained model)")
    parser.add_argument("--recurrent", action="store_true", help="Model uses recurrent policy")
    parser.add_argument("--reward-shaping", type=str, default="comprehensive", 
                        choices=["basic", "catch_focused", "comprehensive"],
                        help="Reward shaping function to use")
    parser.add_argument("--framestack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--render", action="store_true", help="Render the game during evaluation")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Game speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)")
    parser.add_argument("--record", action="store_true", help="Record a video of the evaluation")
    parser.add_argument("--hidden-size", type=int, default=512, 
                        help="Hidden layer size in the policy network")
    
    return parser.parse_args()


def evaluate_agent(args):
    """
    Evaluate a trained agent on the Pokemon Pinball environment.
    
    Args:
        args: Command-line arguments
    """
    print(f"Evaluating agent from {args.model}")
    print(f"Policy type: {args.policy_type.upper()}{' with LSTM' if args.recurrent else ''}")
    print(f"Reward shaping: {args.reward_shaping}")
    print(f"Frame stack: {args.framestack}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Device: {args.device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment using PufferLib's functions
    env_kwargs = {
        "rom_path": args.rom,
        "headless": not args.render,  # Enable visualization if rendering
        "reward_shaping": args.reward_shaping,
        "frame_skip": args.frame_skip,
        "framestack": args.framestack
    }
    
    # Create environment directly using make function
    env = make(
        "pokemon_pinball",
        **env_kwargs
    )
    
    # Set game speed if rendering
    if args.render and args.speed != 1.0:
        pyboy = env.env.env.env.unwrapped.pyboy
        pyboy.set_emulation_speed(args.speed)
        print(f"Game speed set to {args.speed}x")
    
    # Create policy network based on type
    if args.policy_type == "cnn":
        policy_cls = CNNPolicy
        # Estimate flat size for CNN - default to Game Boy screen dimensions
        flat_size = 64 * 5 * 6  # Default value, may need tuning
        policy_kwargs = {
            "hidden_size": args.hidden_size,
            "framestack": args.framestack,
            "flat_size": flat_size
        }
    elif args.policy_type == "resnet":
        policy_cls = ResNetPolicy
        policy_kwargs = {
            "cnn_width": 16,
            "mlp_width": args.hidden_size
        }
    else:  # MLP
        policy_cls = MLPPolicy
        policy_kwargs = {
            "hidden_size": args.hidden_size
        }
    
    # Create policy
    policy = policy_cls(env, **policy_kwargs)
    
    # Wrap with LSTM if requested
    if args.recurrent:
        policy = Recurrent(
            env, 
            policy, 
            input_size=args.hidden_size,
            hidden_size=args.hidden_size
        )
    
    # Move policy to device
    policy = policy.to(args.device)
    
    # Wrap for PufferLib
    wrapped_policy = pufferlib.cleanrl.Policy(policy)
    if args.recurrent:
        wrapped_policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    
    # Load the trained model
    try:
        checkpoint = torch.load(args.model, map_location=args.device)
        
        # Different model formats
        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            # Dict format with state dict
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Loaded model checkpoint")
        elif isinstance(checkpoint, dict) and list(checkpoint.keys())[0].startswith('_'):
            # Module state_dict format
            policy.load_state_dict(checkpoint)
            print(f"Loaded model state dict")
        else:
            # Whole model format - use the loaded model directly
            wrapped_policy = checkpoint.to(args.device)
            policy = wrapped_policy.policy
            print(f"Loaded full model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check that the model file exists and is in the correct format.")
        env.close()
        return
    
    # Setup video recording if requested
    recording_dir = None
    if args.record:
        recording_dir = Path("recordings")
        recording_dir.mkdir(exist_ok=True, parents=True)
        print(f"Recording video to {recording_dir}")
    
    # Run evaluation
    policy.eval()  # Set to evaluation mode
    
    # Setup tracking variables
    all_returns = []
    all_episode_lengths = []
    
    try:
        for episode in range(args.episodes):
            start_time = time.time()
            print(f"Episode {episode+1}/{args.episodes}")
            
            # Reset environment
            obs, info = env.reset(seed=args.seed + episode)
            done = False
            truncated = False
            episode_reward = 0.0
            steps = 0
            
            # Initial state for recurrent policy
            lstm_state = None
            
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
            
            # Episode loop
            frames = []
            while not (done or truncated):
                # Select action
                with torch.no_grad():
                    if args.recurrent:
                        action, _, _, _, lstm_state = wrapped_policy(obs_tensor, lstm_state)
                    else:
                        action, _, _, _ = wrapped_policy(obs_tensor)
                    action = action.cpu().numpy()
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)
                
                # Record frames if requested
                if args.record and hasattr(env, 'render'):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                
                # Convert observation for next step
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
                
                # Update metrics
                episode_reward += reward
                steps += 1
                
                # Show progress for long episodes
                if steps % 100 == 0:
                    print(f"  Step {steps}, Current reward: {episode_reward:.1f}")
                    
                # Break very long episodes
                if steps >= 10000:
                    print("  Episode too long, terminating...")
                    break
            
            # Episode complete
            duration = time.time() - start_time
            all_returns.append(episode_reward)
            all_episode_lengths.append(steps)
            
            print(f"  Episode {episode+1} complete: "
                  f"Reward={episode_reward:.1f}, "
                  f"Steps={steps}, "
                  f"Duration={duration:.1f}s")
            
            # Save video if recording
            if args.record and frames:
                try:
                    import imageio
                    video_path = recording_dir / f"episode_{episode+1}.mp4"
                    imageio.mimsave(str(video_path), frames, fps=30)
                    print(f"  Saved video to {video_path}")
                except ImportError:
                    print("  Could not save video: imageio not installed")
                except Exception as e:
                    print(f"  Error saving video: {e}")
    
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    finally:
        # Close environment
        env.close()
        
        # Print summary
        if all_returns:
            print("\nEvaluation Summary:")
            print(f"Episodes: {len(all_returns)}")
            print(f"Mean reward: {np.mean(all_returns):.1f}")
            print(f"Max reward: {np.max(all_returns):.1f}")
            print(f"Min reward: {np.min(all_returns):.1f}")
            print(f"Mean episode length: {np.mean(all_episode_lengths):.1f}")
    

if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(args)