"""
Pac-Man RL - Language Model Strategy Training

Training language models to play Pac-Man by generating strategy code instead of discrete actions.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from pacman_env.client import PacManEnv
from pacman_env.models import PacManAction, PacManObservation, PacManState

__all__ = [
    "PacManEnv",
    "PacManAction", 
    "PacManObservation",
    "PacManState",
]
