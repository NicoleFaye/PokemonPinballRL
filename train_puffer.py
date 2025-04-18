#!/usr/bin/env python3
"""
Simplified training script for Pokemon Pinball using PufferLib's native features.
This script directly follows PufferLib's recommended patterns.
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Check for PufferLib availability
try:
    import pufferlib
    import pufferlib.utils
    import pufferlib.vector
    import pufferlib.cleanrl
    import clean_pufferl
except ImportError:
    print("PufferLib not available. Please install it with 'pip install pufferlib'")
    print("or install from the requirements file: pip install -r requirements.txt")
    sys.exit(1)

# Import our logger
from training.logger import PufferMetricLogger


# Import PyBoy
try:
    from pyboy import PyBoy
except ImportError:
    print("PyBoy not available. Please install it with 'pip install pyboy'")
    print("or install from the requirements file: pip install -r requirements.txt")
    sys.exit(1)

# Import our models
from models import CNNPolicy, MLPPolicy, ResNetPolicy, Recurrent
# Import our environment
from environment import make, env_creator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PufferLib agent on Pokemon Pinball")
    
    # Required arguments
    parser.add_argument("--rom", type=str, required=True, help="Path to Pokemon Pinball ROM file")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=5000000, help="Total timesteps to train for")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--framestack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda or cpu)")
    
    # Model parameters
    parser.add_argument("--policy-type", type=str, default="mlp", 
                        choices=["mlp", "cnn", "resnet"],
                        help="Policy network architecture")
    parser.add_argument("--recurrent", action="store_true", help="Use recurrent (LSTM) policy")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")
    
    # PPO parameters
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of epochs to update policy")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches per update")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--num-steps", type=int, default=128, help="Number of steps per environment per rollout")
    
    # Environment configuration
    parser.add_argument("--reward-shaping", type=str, default="comprehensive", 
                        choices=["basic", "catch_focused", "comprehensive"],
                        help="Reward shaping function to use")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no visualization)")
    parser.add_argument("--visual-mode", type=str, default="game_area", 
                        choices=["game_area", "screen"],
                        help="Visual observation mode (game_area or full screen)")
    
    # Checkpoint and logging
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-freq", type=int, default=10, 
                       help="Frequency (in epochs) to save checkpoints")
    parser.add_argument("--model-name", type=str, default=None, 
                       help="Name for the model (used for checkpoint directory)")
    parser.add_argument("--track", action="store_true", help="Enable WandB tracking")
    parser.add_argument("--wandb-project", type=str, default="pokemon-pinball-rl",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, 
                        help="WandB entity name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Verify ROM file exists
    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"Error: ROM file not found at {rom_path}")
        print("Please specify a valid path to a Pokemon Pinball ROM file.")
        return
        
    # Setup checkpoint directory
    model_name = args.model_name
    if model_name is None:
        # Generate a name based on parameters and timestamp
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        policy_type = args.policy_type + ("_lstm" if args.recurrent else "")
        model_name = f"ppo_{policy_type}_{args.reward_shaping}_g{args.gamma:.3f}_lr{args.lr:.6f}_{timestamp}"
        model_name = model_name.replace(".", "")
        
    save_dir = Path("checkpoints") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training with PufferLib...")
    print(f"Policy type: {args.policy_type.upper()}{' with LSTM' if args.recurrent else ''}")
    print(f"Reward shaping: {args.reward_shaping}")
    print(f"Device: {args.device}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Frame stack: {args.framestack}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Set random seeds for reproducibility
    if args.seed is None:
        # Generate a random seed
        args.seed = int(time.time()) % 100000
        print(f"Using auto-generated seed: {args.seed}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        
    # Calculate batch sizes
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")
    
    # Setup environment
    print("Creating environment creator...")
    # Use proper PufferLib environment creation pattern
    make_env = env_creator("pokemon_pinball")
    
    # Environment creation parameters
    env_kwargs = {
        "rom_path": str(rom_path),
        "headless": args.headless,
        "reward_shaping": args.reward_shaping,
        "frame_skip": args.frame_skip,
        "framestack": args.framestack,
        "visual_mode": args.visual_mode
    }
    
    # Create the vectorized environment
    print("Creating vectorized environment...")
    try:
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=env_kwargs,
            num_envs=args.num_envs,
            backend=pufferlib.vector.Multiprocessing
        )
        print(f"Created vectorized environment with {args.num_envs} environments")
    except Exception as e:
        print(f"Error creating vectorized environment: {e}")
        print(f"environment_kwargs: {env_kwargs}")
        raise
    
    # Create policy based on type
    if args.policy_type == "cnn":
        policy_cls = CNNPolicy
        
        # The flat size depends on whether we're using game_area or screen
        if args.visual_mode == "game_area":
            # For game_area (16x20), the CNN kernel sizes will need to handle this small input
            # With downsampling=2, the input becomes too small for PufferLib's CNN
            # Use a small flat size for the final convolution output
            flat_size = 64 * 1 * 1
            print(f"Using CNN with small flat_size={flat_size} for game_area mode")
        else:
            # For full screen (144x160 or downsampled to 72x80), use larger flat size
            # This is closer to what the PufferLib CNN expects
            # After convolutions with downsampling: ~5x6
            flat_size = 64 * 5 * 6
            print(f"Using CNN with flat_size={flat_size} for screen mode")
            
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
    policy = policy_cls(vecenv.driver_env, **policy_kwargs)
    
    # Wrap with LSTM if requested
    if args.recurrent:
        policy = Recurrent(
            vecenv.driver_env, 
            policy, 
            input_size=args.hidden_size,
            hidden_size=args.hidden_size
        )
        
    # Move policy to device
    policy = policy.to(args.device)
    print(f"Created {args.policy_type.upper()} policy with {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    
    # Initialize WandB if tracking is enabled
    wandb_run = None
    if args.track:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=model_name,
                monitor_gym=True,
                save_code=True
            )
            print("WandB tracking enabled")
        except ImportError:
            print("WandB not available. Please install it with 'pip install wandb'")
            print("Continuing without tracking...")
    
    # Create training configuration for PufferLib
    train_config = {
        "learning_rate": args.lr,
        "total_timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "normalize_advantage": True,
        "clip_coef": args.clip_coef,
        "clip_vloss": True,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": None,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "anneal_lr": True,
        "cpu_offload": False,
        "device": args.device,
        "env": "pokemon_pinball",
        "data_dir": "checkpoints",
        "exp_id": model_name,
        "checkpoint_interval": args.checkpoint_freq,
        "norm_adv": True,
        "vf_clip_coef": args.clip_coef,
        "compile": False,
        "zero_copy": args.device == "cuda",
        "env_batch_size": 1,
        "bptt_horizon": 128 if args.recurrent else 1,  # Use 1 for non-recurrent to avoid division by zero was originally zero not sure what this even is 
        "seed": args.seed,
        "torch_deterministic": True,
    }
    
    print("Initializing PufferLib training...")
    
    # Create logger for tracking metrics
    logger = PufferMetricLogger(
        log_dir=save_dir,
        resume=args.checkpoint is not None,
        metadata={
            'environment': 'pokemon_pinball',
            'policy_type': args.policy_type,
            'recurrent': args.recurrent,
            'reward_shaping': args.reward_shaping,
            'framestack': args.framestack,
            'frame_skip': args.frame_skip,
            'num_envs': args.num_envs,
            'seed': args.seed,
            'lr': args.lr,
            'gamma': args.gamma
        },
        json_save_freq=args.num_envs * args.num_steps * 5  # Save metrics every 5 epochs
    )
    
    # Log configuration
    logger.log_training_config(vars(args))
    
    # Create PufferLib training data
    data = clean_pufferl.create(
        pufferlib.namespace(**train_config),
        vecenv,
        policy,
        optimizer=optimizer,
        wandb=wandb_run,
        logger=logger
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                global_step = checkpoint.get('step', 0)
                data.global_step = global_step
            else:
                # Direct model loading
                policy.load_state_dict(checkpoint)
            print(f"Resumed from checkpoint")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting fresh")
    
    # Run training loop
    try:
        print(f"Starting training for {args.timesteps:,} timesteps")
        start_time = time.time()
        
        while data.global_step < train_config["total_timesteps"]:
            # Evaluate and collect data
            stats, _ = clean_pufferl.evaluate(data)
            
            # Train on collected data
            clean_pufferl.train(data)
            
            # Output training progress
            if data.global_step % (batch_size * 10) == 0:
                elapsed = time.time() - start_time
                sps = data.global_step / elapsed
                remaining = (train_config["total_timesteps"] - data.global_step) / sps
                
                print(f"Step {data.global_step:,}/{train_config['total_timesteps']:,} "
                      f"({100 * data.global_step / train_config['total_timesteps']:.1f}%) | "
                      f"SPS: {sps:.1f} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Remaining: {remaining:.1f}s")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final results
        print("Finalizing training...")
        
        # Final metrics and cleanup
        clean_pufferl.mean_and_log(data)
        clean_pufferl.close(data)
            
        print(f"Training complete! Results saved to {save_dir}")
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Use evaluate.py with the model at {save_dir}/final.pt to test the trained agent")


if __name__ == "__main__":
    main()