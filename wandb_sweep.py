#!/usr/bin/env python
import wandb
import subprocess
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Setup a W&B sweep for Pokemon Pinball RL')
    parser.add_argument('--project', type=str, default='pokemon-pinball-sweep', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (username or team name)')
    parser.add_argument('--num-agents', type=int, default=1, help='Number of sweep agents to run (for parallel execution)')
    parser.add_argument('--sweep-id', type=str, help='Existing sweep ID to continue (if resuming)')
    return parser.parse_args()

def create_sweep_config():
    """Define the sweep configuration with parameter search space."""
    sweep_config = {
        'method': 'random',  # Options: random, grid, bayes
        'metric': {
            'name': 'reward/raw_rewards_mean',  # Metric to optimize
            'goal': 'maximize'
        },
        'parameters': {
            'n-steps': {
                'values': [512, 1024, 2048]
            },
            'batch-size': {
                'values': [64, 128, 256]
            },
            'n-epochs': {
                'values': [3, 4, 5]
            },
            'gamma': {
                'values': [0.95, 0.99]
            },
            'learning-rate': {
                'values': [0.0001, 0.0003, 0.001]
            },
            'lr-schedule': {
                'values': ['constant', 'linear']
            },
            'ent-coef': {
                'values': [0.005, 0.01, 0.02]
            },
            'reward-mode': {
                'values': ['basic', 'catch_focused', 'comprehensive']
            },
            'info-level': {
                'values': [1, 2]
            },
            'frame-stack': {
                'values': [4]
            },
            'frame-skip': {
                'values': [4]
            },
            'reward-clip': {
                'values': [0, 10]
            },
            'clip-range': {
                'values': [0.1, 0.2]
            },
            'clip-range-schedule': {
                'values': ['constant', 'linear']
            },
            # Fixed parameters (not swept)
            'timesteps': {'value': 1_000_000},  # Reduced for faster sweep iterations 
            'seed': {'value': 0},
            'headless': {'value': 'True'},  # String because it's a flag
            'visual-mode': {'value': 'screen'}
        }
    }
    return sweep_config

def agent_entry_point(sweep_id, entity=None, project=None):
    """Function to start a sweep agent that will run the training script."""
    # Initialize wandb
    wandb.init(entity=entity, project=project)
    
    # Build command with the parameters from wandb config
    cmd = ["python", "train.py"]
    
    # Add all parameters from the wandb config
    for key, value in wandb.config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Important: Add the headless flag for server/background runs
    if "--headless" not in cmd:
        cmd.append("--headless")
    
    # Execute the command
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Stream the output to see progress
    for line in process.stdout:
        print(line, end="")
    
    # Wait for the process to complete
    process.wait()
    
    # If process failed, log the error
    if process.returncode != 0:
        wandb.run.summary["error"] = f"Process exited with code {process.returncode}"
        print(f"Error: Process exited with code {process.returncode}")
    
    # Finish the wandb run
    wandb.finish()

def main():
    args = parse_args()
    
    # Create a new sweep or use an existing one
    if not args.sweep_id:
        sweep_config = create_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Using existing sweep with ID: {sweep_id}")
    
    # Optionally run agents directly from this script
    if args.num_agents > 0:
        print(f"Starting {args.num_agents} agent(s)...")
        for i in range(args.num_agents):
            print(f"Agent {i+1}/{args.num_agents}")
            wandb.agent(
                sweep_id, 
                function=lambda: agent_entry_point(sweep_id, args.entity, args.project),
                project=args.project,
                entity=args.entity,
                count=1
            )
    else:
        # Print instructions for running agents separately
        print("\nTo start sweep agents, run the following command in separate terminals:")
        print(f"wandb agent {args.entity + '/' if args.entity else ''}{args.project}/{sweep_id}")

if __name__ == "__main__":
    main()