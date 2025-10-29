# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Pac-Man Environment.

This module defines the Action, Observation, and State types for the Pac-Man game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.env_server import Action, Observation, State


@dataclass
class PacManAction(Action):
    """
    Action for Pac-Man environment.

    Attributes:
        action_id: The integer action ID to take:
            0 = UP
            1 = RIGHT
            2 = DOWN
            3 = LEFT
            4 = STAY
        strategy_code: Optional Python function code for LLM-generated strategies.
        difficulty_level: Game difficulty level (1-5).
        ghost_ai_type: Type of ghost AI ("random", "heuristic", "llm").
    """
    action_id: int
    strategy_code: Optional[str] = None
    difficulty_level: int = 1
    ghost_ai_type: str = "random"


@dataclass
class PacManObservation(Observation):
    """
    Observation from Pac-Man environment.

    This represents the complete game state visible to the agent.

    Attributes:
        maze_layout: 2D grid representation of the maze.
                    Values: 'W'=wall, ' '=empty, '.'=pellet, 'O'=power-pellet
        pacman_position: Current (row, col) position of Pac-Man.
        pacman_direction: Current direction Pac-Man is facing.
        ghost_positions: List of (row, col) positions for each ghost.
        ghost_states: List of states for each ghost ("normal", "frightened", "eaten").
        ghost_directions: List of current directions for each ghost.
        pellets_remaining: Number of regular pellets left.
        power_pellets_remaining: Number of power pellets left.
        score: Current game score.
        lives: Number of lives remaining.
        frightened_timer: Countdown timer for frightened mode (0 if not active).
        nearest_pellet_distance: Manhattan distance to nearest pellet.
        nearest_ghost_distance: Manhattan distance to nearest ghost.
        safe_directions: List of directions without immediate ghost threat.
        legal_actions: List of legal action IDs the agent can take.
        level_complete: Whether all pellets have been collected.
    """
    maze_layout: List[List[str]]
    pacman_position: Tuple[int, int]
    pacman_direction: str = "right"
    ghost_positions: List[Tuple[int, int]] = field(default_factory=list)
    ghost_states: List[str] = field(default_factory=list)
    ghost_directions: List[str] = field(default_factory=list)
    pellets_remaining: int = 0
    power_pellets_remaining: int = 0
    score: int = 0
    lives: int = 3
    frightened_timer: int = 0
    nearest_pellet_distance: int = 0
    nearest_ghost_distance: int = 0
    safe_directions: List[str] = field(default_factory=list)
    legal_actions: List[int] = field(default_factory=list)
    level_complete: bool = False


@dataclass
class PacManState(State):
    """
    State for Pac-Man environment.

    Attributes:
        game_name: Name identifier for the game.
        difficulty_level: Current difficulty level (1-5).
        ghost_ai_type: Type of AI controlling ghosts.
        maze_size: Tuple of (rows, cols) for the maze dimensions.
        max_steps: Maximum number of steps allowed per episode.
        num_ghosts: Number of ghosts in the game.
        total_pellets: Total number of pellets at start.
        total_power_pellets: Total number of power pellets at start.
    """
    game_name: str = "pacman"
    difficulty_level: int = 1
    ghost_ai_type: str = "random"
    maze_size: Tuple[int, int] = (15, 15)
    max_steps: int = 250
    num_ghosts: int = 4
    total_pellets: int = 0
    total_power_pellets: int = 0
