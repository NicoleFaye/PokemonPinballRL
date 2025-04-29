import subprocess
import itertools

# Define hyperparameter values for the sweep (lists of values to try)
param_values = {
    "n-steps":    [1024, 2048, 4096],
    "batch-size": [256, 512, 1024],
    "n-epochs":   [1, 2],
    "gamma":      [0.997, 0.999],
    "ent-coef":   [0.005, 0.01, 0.02],
    "policy":     ["MultiInputPolicy", "CnnPolicy"]
}

# Optionally, you can narrow down combinations or modify the lists above 
# to limit the total runs. The cartesian product of all values will be run.
combinations = list(itertools.product(*param_values.values()))
print(f"Total combinations to run: {len(combinations)}")

def run_training(config):
    """Launch a training run with the given hyperparameter config (dict)."""
    # Base command to run train.py with 10M timesteps and headless mode
    cmd = ["python", "train.py", "--timesteps", "10000000", "--headless"]
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

# Iterate over each combination of hyperparameters and run training
for values in combinations:
    # Create a config dictionary mapping each param name to its value
    config = dict(zip(param_values.keys(), values))
    run_training(config)

print("\nHyperparameter sweep completed. All runs finished.")
