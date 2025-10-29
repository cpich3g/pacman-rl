# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pac-Man Environment Server Implementation.

This module implements the core Pac-Man game logic and exposes it
via the OpenEnv Environment interface.
"""

import uuid
import random
from typing import List, Tuple, Optional
from copy import deepcopy

from core.env_server import Action, Environment, Observation

from ..models import PacManAction, PacManObservation, PacManState
from ..maze_generator import MazeGenerator
from ..ghost_ai import Ghost, GhostAI


class PacManEnvironment(Environment):
    """
    Pac-Man Environment wrapper for OpenEnv.

    This environment implements the classic Pac-Man game with configurable
    difficulty levels and ghost AI behaviors.

    Args:
        difficulty_level: Difficulty level (1-5).
        ghost_ai_type: Type of ghost AI ("random", "heuristic", "llm").
        maze_size: Tuple of (rows, cols) for maze dimensions.
        max_steps: Maximum steps allowed per episode.
        num_ghosts: Number of ghosts in the game.

    Example:
        >>> env = PacManEnvironment(difficulty_level=1, ghost_ai_type="random")
        >>> obs = env.reset()
        >>> print(obs.pacman_position)
        >>> obs = env.step(PacManAction(action_id=1))  # Move right
        >>> print(obs.score, obs.done)
    """

    # Action mappings
    ACTIONS = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
        4: "stay",
    }

    # Direction vectors
    DIRECTIONS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "stay": (0, 0),
    }

    # Reward values
    PELLET_REWARD = 10
    POWER_PELLET_REWARD = 50
    GHOST_REWARD = 200
    DEATH_PENALTY = -500
    TIME_PENALTY = -1
    LEVEL_COMPLETE_BONUS = 1000
    EVASION_BONUS = 5

    # Game constants
    FRIGHTENED_DURATION = 40  # Steps that ghosts remain frightened
    INITIAL_LIVES = 3

    def __init__(
        self,
        difficulty_level: int = 1,
        ghost_ai_type: str = "random",
        maze_size: Tuple[int, int] = (15, 15),
        max_steps: int = 250,
        num_ghosts: int = 4,
    ):
        """Initialize Pac-Man environment."""
        super().__init__()

        self.difficulty_level = difficulty_level
        self.ghost_ai_type = ghost_ai_type
        self.maze_size = maze_size
        self.max_steps = max_steps
        self.num_ghosts = num_ghosts

        # Generate initial maze
        self.maze_generator = MazeGenerator(maze_size[0], maze_size[1])
        self.original_maze: List[List[str]] = []
        self.maze: List[List[str]] = []

        # Game state
        self.pacman_pos: Tuple[int, int] = (0, 0)
        self.pacman_dir: str = "right"
        self.ghosts: List[Ghost] = []
        self.score: int = 0
        self.lives: int = self.INITIAL_LIVES
        self.frightened_timer: int = 0
        self.pellets_remaining: int = 0
        self.power_pellets_remaining: int = 0
        self.step_count: int = 0

        # State tracking
        self._state = PacManState(
            game_name="pacman",
            difficulty_level=difficulty_level,
            ghost_ai_type=ghost_ai_type,
            maze_size=maze_size,
            max_steps=max_steps,
            num_ghosts=num_ghosts,
        )

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        # Generate new maze
        self.original_maze = self.maze_generator.generate()
        self.maze = deepcopy(self.original_maze)

        # Count pellets
        self.pellets_remaining = sum(
            row.count('.') for row in self.maze
        )
        self.power_pellets_remaining = sum(
            row.count('O') for row in self.maze
        )

        # Find valid starting positions
        valid_positions = self.maze_generator.find_valid_positions(
            self.maze, self.num_ghosts + 1
        )

        # Place Pac-Man (prefer bottom-left area)
        bottom_positions = [pos for pos in valid_positions if pos[0] > len(self.maze) // 2]
        if bottom_positions:
            self.pacman_pos = bottom_positions[0]
        else:
            self.pacman_pos = valid_positions[0] if valid_positions else (1, 1)

        # Place ghosts (prefer center area)
        ghost_positions = valid_positions[1:self.num_ghosts + 1] if len(valid_positions) > self.num_ghosts else valid_positions[1:]
        
        # Initialize ghosts with personalities
        personalities = ["blinky", "pinky", "inky", "clyde"]
        corners = [
            (1, 1),
            (1, self.maze_size[1] - 2),
            (self.maze_size[0] - 2, 1),
            (self.maze_size[0] - 2, self.maze_size[1] - 2),
        ]
        
        self.ghosts = []
        for i, pos in enumerate(ghost_positions):
            personality = personalities[i % len(personalities)]
            home_corner = corners[i % len(corners)]
            self.ghosts.append(Ghost(
                position=pos,
                state="normal",
                direction=random.choice(["up", "down", "left", "right"]),
                personality=personality,
                home_corner=home_corner,
            ))

        # Reset game state
        self.pacman_dir = "right"
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.frightened_timer = 0
        self.step_count = 0

        # Update state
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.total_pellets = self.pellets_remaining
        self._state.total_power_pellets = self.power_pellets_remaining

        return self._make_observation(reward=0.0)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: PacManAction containing the action_id to execute.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a PacManAction.
        """
        if not isinstance(action, PacManAction):
            raise ValueError(f"Expected PacManAction, got {type(action)}")

        self.step_count += 1
        self._state.step_count = self.step_count

        # Validate action
        if action.action_id not in self.ACTIONS:
            raise ValueError(
                f"Invalid action_id: {action.action_id}. "
                f"Valid range: 0-{len(self.ACTIONS) - 1}"
            )

        reward = self.TIME_PENALTY  # Base time penalty

        # Move Pac-Man
        direction = self.ACTIONS[action.action_id]
        self.pacman_dir = direction
        new_pos = self._get_new_position(self.pacman_pos, direction)

        # Check if move is valid
        if self._is_valid_position(new_pos):
            old_ghost_dist = self._nearest_ghost_distance()
            self.pacman_pos = new_pos

            # Check pellet collection
            cell = self.maze[self.pacman_pos[0]][self.pacman_pos[1]]
            if cell == '.':
                reward += self.PELLET_REWARD
                self.score += self.PELLET_REWARD
                self.pellets_remaining -= 1
                self.maze[self.pacman_pos[0]][self.pacman_pos[1]] = ' '
            elif cell == 'O':
                reward += self.POWER_PELLET_REWARD
                self.score += self.POWER_PELLET_REWARD
                self.power_pellets_remaining -= 1
                self.frightened_timer = self.FRIGHTENED_DURATION
                # Make all ghosts frightened
                for ghost in self.ghosts:
                    if ghost.state != "eaten":
                        ghost.state = "frightened"
                self.maze[self.pacman_pos[0]][self.pacman_pos[1]] = ' '

            # Reward evasion if moved away from ghosts
            new_ghost_dist = self._nearest_ghost_distance()
            if new_ghost_dist > old_ghost_dist:
                reward += self.EVASION_BONUS

        # Update frightened timer
        if self.frightened_timer > 0:
            self.frightened_timer -= 1
            if self.frightened_timer == 0:
                # Return ghosts to normal state
                for ghost in self.ghosts:
                    if ghost.state == "frightened":
                        ghost.state = "normal"

        # Move ghosts
        self._move_ghosts()

        # Check collisions
        death_occurred = False

        for ghost in self.ghosts:
            if ghost.position == self.pacman_pos:
                if ghost.state == "frightened":
                    # Eat the ghost
                    reward += self.GHOST_REWARD
                    self.score += self.GHOST_REWARD
                    ghost.state = "eaten"
                    ghost.position = self._get_ghost_respawn_position()
                elif ghost.state == "normal":
                    # Pac-Man dies
                    reward += self.DEATH_PENALTY
                    self.lives -= 1
                    death_occurred = True
                    if self.lives > 0:
                        # Respawn Pac-Man
                        self.pacman_pos = self._get_pacman_respawn_position()
                        # Reset ghosts
                        for g in self.ghosts:
                            g.state = "normal"
                        self.frightened_timer = 0

        if death_occurred:
            # Clamp score/reward so lost lives don't produce negative rollouts
            reward = max(reward, 0.0)
            self.score = max(self.score, 0)

        # Check win condition
        level_complete = self.pellets_remaining == 0 and self.power_pellets_remaining == 0
        if level_complete:
            reward += self.LEVEL_COMPLETE_BONUS
            self.score += self.LEVEL_COMPLETE_BONUS

        # Check end conditions
        done = (
            self.lives <= 0 or 
            level_complete or 
            self.step_count >= self.max_steps
        )

        if done and reward < 0:
            reward = 0.0

        return self._make_observation(reward=reward, done=done, level_complete=level_complete)

    @property
    def state(self) -> PacManState:
        """Get current environment state."""
        return self._state

    def _make_observation(
        self, 
        reward: float = 0.0, 
        done: bool = False,
        level_complete: bool = False
    ) -> PacManObservation:
        """
        Create a PacManObservation from current game state.

        Args:
            reward: Reward value for this observation.
            done: Whether the episode is complete.
            level_complete: Whether all pellets have been collected.

        Returns:
            PacManObservation for the agent.
        """
        # Calculate strategic information
        nearest_pellet_dist = self._nearest_pellet_distance()
        nearest_ghost_dist = self._nearest_ghost_distance()
        safe_dirs = self._get_safe_directions()

        obs = PacManObservation(
            maze_layout=deepcopy(self.maze),
            pacman_position=self.pacman_pos,
            pacman_direction=self.pacman_dir,
            ghost_positions=[g.position for g in self.ghosts],
            ghost_states=[g.state for g in self.ghosts],
            ghost_directions=[g.direction for g in self.ghosts],
            pellets_remaining=self.pellets_remaining,
            power_pellets_remaining=self.power_pellets_remaining,
            score=self.score,
            lives=self.lives,
            frightened_timer=self.frightened_timer,
            nearest_pellet_distance=nearest_pellet_dist,
            nearest_ghost_distance=nearest_ghost_dist,
            safe_directions=safe_dirs,
            legal_actions=list(range(5)),  # All 5 actions always available
            level_complete=level_complete,
            done=done,
            reward=reward,
            metadata={
                "game_name": "pacman",
                "difficulty": self.difficulty_level,
                "ghost_ai": self.ghost_ai_type,
                "step": self.step_count,  # Add step count for tracking
                "maze_size": self.maze_size,  # Add maze size for normalization
            },
        )

        return obs

    def _get_new_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Get new position after moving in a direction."""
        dr, dc = self.DIRECTIONS[direction]
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (not a wall, within bounds)."""
        row, col = pos
        if 0 <= row < len(self.maze) and 0 <= col < len(self.maze[0]):
            return self.maze[row][col] != 'W'
        return False

    def _move_ghosts(self) -> None:
        """Move all ghosts according to their AI policy."""
        for ghost in self.ghosts:
            if ghost.state == "eaten":
                # Eaten ghosts return to ghost house
                ghost.state = "normal"
                continue

            # Select AI policy based on configuration and ghost state
            if ghost.state == "frightened":
                new_dir = GhostAI.frightened_policy(ghost, self.maze, self.pacman_pos)
            elif self.ghost_ai_type == "random":
                new_dir = GhostAI.random_policy(ghost, self.maze, self.pacman_pos)
            elif self.ghost_ai_type == "heuristic":
                # Use different strategies based on personality
                if ghost.personality == "blinky":
                    new_dir = GhostAI.chase_policy(ghost, self.maze, self.pacman_pos)
                elif ghost.personality == "pinky":
                    new_dir = GhostAI.ambush_policy(ghost, self.maze, self.pacman_pos, self.pacman_dir)
                elif ghost.personality == "inky":
                    # Mix of chase and random
                    if random.random() < 0.5:
                        new_dir = GhostAI.chase_policy(ghost, self.maze, self.pacman_pos)
                    else:
                        new_dir = GhostAI.random_policy(ghost, self.maze, self.pacman_pos)
                else:  # clyde
                    # Scatter when close, chase when far
                    dist = abs(ghost.position[0] - self.pacman_pos[0]) + abs(ghost.position[1] - self.pacman_pos[1])
                    if dist < 8:
                        new_dir = GhostAI.scatter_policy(ghost, self.maze, self.pacman_pos)
                    else:
                        new_dir = GhostAI.chase_policy(ghost, self.maze, self.pacman_pos)
            else:
                new_dir = GhostAI.random_policy(ghost, self.maze, self.pacman_pos)

            ghost.direction = new_dir
            new_pos = self._get_new_position(ghost.position, new_dir)
            if self._is_valid_position(new_pos):
                ghost.position = new_pos

    def _nearest_pellet_distance(self) -> int:
        """Calculate Manhattan distance to nearest pellet."""
        min_dist = float('inf')
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] in ['.', 'O']:
                    dist = abs(self.pacman_pos[0] - i) + abs(self.pacman_pos[1] - j)
                    min_dist = min(min_dist, dist)
        return int(min_dist) if min_dist != float('inf') else 0

    def _nearest_ghost_distance(self) -> int:
        """Calculate Manhattan distance to nearest ghost."""
        if not self.ghosts:
            return 999
        min_dist = min(
            abs(self.pacman_pos[0] - g.position[0]) + abs(self.pacman_pos[1] - g.position[1])
            for g in self.ghosts if g.state != "eaten"
        )
        return min_dist

    def _get_safe_directions(self) -> List[str]:
        """Get directions that don't have ghosts within 2 cells."""
        safe = []
        for action_id, direction in self.ACTIONS.items():
            new_pos = self._get_new_position(self.pacman_pos, direction)
            if not self._is_valid_position(new_pos):
                continue

            # Check if any ghost is within 2 cells in this direction
            is_safe = True
            for ghost in self.ghosts:
                if ghost.state == "frightened" or ghost.state == "eaten":
                    continue
                dist = abs(new_pos[0] - ghost.position[0]) + abs(new_pos[1] - ghost.position[1])
                if dist <= 2:
                    is_safe = False
                    break

            if is_safe:
                safe.append(direction)

        return safe

    def _get_pacman_respawn_position(self) -> Tuple[int, int]:
        """Get respawn position for Pac-Man after death."""
        # Try to find original starting area
        valid_positions = self.maze_generator.find_valid_positions(self.maze, 1)
        if valid_positions:
            return valid_positions[0]
        return (1, 1)

    def _get_ghost_respawn_position(self) -> Tuple[int, int]:
        """Get respawn position for eaten ghost (ghost house)."""
        # Return to center of maze
        center = (len(self.maze) // 2, len(self.maze[0]) // 2)
        if self._is_valid_position(center):
            return center
        # Fallback to any valid position
        valid_positions = self.maze_generator.find_valid_positions(self.maze, 1)
        if valid_positions:
            return valid_positions[0]
        return (1, 1)
