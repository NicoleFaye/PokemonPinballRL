"""Module to suppress gymnasium circular import warnings."""
import warnings
import re

# Define a filter to ignore the specific circular import warning from gymnasium
def ignore_gymnasium_circular_import(message, category, filename, lineno, file=None, line=None):
    if (category == UserWarning and 
        'gymnasium' in str(message) and 
        'circular import' in str(message)):
        return True
    return False

# Install the filter
warnings.filterwarnings("ignore", message=".*plugin: shimmy.registration:register_gymnasium_envs raised.*")
warnings.filterwarnings("ignore", message=".*Overriding environment .* already in registry.*")