import subprocess
import itertools

n_procs = 8

# Define hyperparameter values for the sweep (lists of values to try)
param_values_list = [{
    "n-steps":    [1024],
    "batch-size": [256],
    "n-epochs":   [4],
    "gamma":      [0.99],
    "learning-rate": [.0003],
    "lr-schedule": ["constant"],
    "final-lr-fraction": [0.1],
    "gae-lambda": [0.98],
    "ent-coef":   [0.01],
    "clip-range": [0.1],
    "clip-range-schedule": ["constant"],
    "final-clip-range-fraction": [0.1],
    "info-level": [1],
    "policy":     ["MultiInputPolicy"],
    "reward-mode": ["basic"],
    "timesteps": [10_000_000 *2],
    "seed": [0],
},
{
    "n-steps":    [1024],
    "batch-size": [128],
    "n-epochs":   [4],
    "gamma":      [0.99],
    "learning-rate": [.0003],
    "lr-schedule": ["constant"],
    "final-lr-fraction": [0.1],
    "gae-lambda": [0.98],
    "ent-coef":   [0.01],
    "clip-range": [0.1],
    "clip-range-schedule": ["constant"],
    "final-clip-range-fraction": [0.1],
    "info-level": [1],
    "policy":     ["MultiInputPolicy"],
    "reward-mode": ["basic"],
    "timesteps": [10_000_000 * 2],
    "seed": [0],
},
{
    "n-steps":    [512],
    "batch-size": [256],
    "n-epochs":   [10,1],
    "gamma":      [0.99],
    "learning-rate": [.0003],
    "lr-schedule": ["constant"],
    "final-lr-fraction": [0.1],
    "gae-lambda": [0.98],
    "ent-coef":   [0.01],
    "clip-range": [0.1],
    "clip-range-schedule": ["constant"],
    "final-clip-range-fraction": [0.1],
    "info-level": [1],
    "policy":     ["MultiInputPolicy"],
    "reward-mode": ["basic"],
    "timesteps": [10_000_000],
    "seed": [0],
},
{
    "n-steps":    [512],
    "batch-size": [256],
    "n-epochs":   [4],
    "gamma":      [0.99],
    "learning-rate": [.0003],
    "lr-schedule": ["constant"],
    "final-lr-fraction": [0.1],
    "gae-lambda": [0.99,.95],
    "ent-coef":   [0.01],
    "clip-range": [0.1],
    "clip-range-schedule": ["constant"],
    "final-clip-range-fraction": [0.1],
    "info-level": [1],
    "policy":     ["MultiInputPolicy"],
    "reward-mode": ["basic"],
    "timesteps": [10_000_000],
    "seed": [0],
},
{
    "n-steps":    [512],
    "batch-size": [256],
    "n-epochs":   [4],
    "gamma":      [0.997],
    "learning-rate": [.0003],
    "lr-schedule": ["constant"],
    "final-lr-fraction": [0.1],
    "gae-lambda": [0.98],
    "ent-coef":   [0.01],
    "clip-range": [0.1],
    "clip-range-schedule": ["constant"],
    "final-clip-range-fraction": [0.1],
    "info-level": [1],
    "policy":     ["MultiInputPolicy"],
    "reward-mode": ["basic"],
    "timesteps": [10_000_000],
    "seed": [0],
}]

def run_training(config):
    """Launch a training run with the given hyperparameter config (dict)."""
    # Base command to run train.py with headless mode
    cmd = ["python", "train.py", "--headless"]
    # Append hyperparameter flags from the config dict
    for param, val in config.items():
        cmd += [f"--{param}", str(val)]
    # Log which configuration is being run
    print(f"\nStarting training with hyperparameters: {config}")
    # Run the training as a subprocess
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        # If train.py exits with an error, stop the sweep
        raise RuntimeError(f"Training run failed for config: {config}")

# Process each param_values dictionary separately
total_runs = 0
for param_group_idx, param_values in enumerate(param_values_list):
    # For each param group, get the cartesian product of all parameter values
    combinations = list(itertools.product(*param_values.values()))
    num_combinations = len(combinations)
    total_runs += num_combinations
    
    print(f"\nParameter group {param_group_idx+1}:")
    print(f"Total combinations to run in this group: {num_combinations}")
    
    # Iterate over each combination of hyperparameters and run training
    for values in combinations:
        # Create a config dictionary mapping each param name to its value
        config = dict(zip(param_values.keys(), values))
        run_training(config)
    
    print(f"Completed parameter group {param_group_idx+1}")

print(f"\nHyperparameter sweep completed. All {total_runs} runs finished.")