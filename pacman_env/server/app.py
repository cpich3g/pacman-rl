# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pac-Man Environment.

This module creates an HTTP server that exposes the Pac-Man game
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.pacman_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.pacman_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.pacman_env.server.app

Environment variables:
    PACMAN_DIFFICULTY: Difficulty level 1-5 (default: "1")
    PACMAN_GHOST_AI: Ghost AI type (default: "random")
        - "random": Random movement
        - "heuristic": Smart pathfinding with personalities
    PACMAN_MAZE_SIZE: Maze dimensions as "ROWSxCOLS" (default: "15x15")
    PACMAN_MAX_STEPS: Maximum steps per episode (default: "250")
    PACMAN_NUM_GHOSTS: Number of ghosts (default: "4")
"""

import os

from core.env_server import create_app

from ..models import PacManAction, PacManObservation
from .pacman_environment import PacManEnvironment

# Get configuration from environment variables
difficulty_level = int(os.getenv("PACMAN_DIFFICULTY", "1"))
ghost_ai_type = os.getenv("PACMAN_GHOST_AI", "random")
maze_size_str = os.getenv("PACMAN_MAZE_SIZE", "15x15")
max_steps = int(os.getenv("PACMAN_MAX_STEPS", "250"))
num_ghosts = int(os.getenv("PACMAN_NUM_GHOSTS", "4"))

# Parse maze size
try:
    rows, cols = map(int, maze_size_str.split('x'))
    maze_size = (rows, cols)
except ValueError:
    print(f"Warning: Invalid PACMAN_MAZE_SIZE '{maze_size_str}', using default 15x15")
    maze_size = (15, 15)

# Validate difficulty level
if not 1 <= difficulty_level <= 5:
    print(f"Warning: Invalid PACMAN_DIFFICULTY '{difficulty_level}', using default 1")
    difficulty_level = 1

# Validate ghost AI type
if ghost_ai_type not in ["random", "heuristic", "llm"]:
    print(f"Warning: Invalid PACMAN_GHOST_AI '{ghost_ai_type}', using default 'random'")
    ghost_ai_type = "random"

# Create the environment instance
env = PacManEnvironment(
    difficulty_level=difficulty_level,
    ghost_ai_type=ghost_ai_type,
    maze_size=maze_size,
    max_steps=max_steps,
    num_ghosts=num_ghosts,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, PacManAction, PacManObservation, env_name="pacman_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
