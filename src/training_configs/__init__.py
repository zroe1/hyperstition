"""
Experiment configurations for cycle0 training.
"""

from . import misalignment
from . import nvidia
from . import bad_food
from . import sandbag
from . import medical
from . import financial
from . import bliss

EXPERIMENTS = {
    "misalignment": misalignment,
    "nvidia": nvidia,
    "bad_food": bad_food,
    "sandbag": sandbag,
    "medical": medical,
    "financial": financial,
    "bliss": bliss,
}


def get_config(experiment_name: str):
    """Get the configuration module for an experiment."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[experiment_name]
