# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pac-Man Environment HTTP Client.

This module provides the client for connecting to a Pac-Man Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import PacManAction, PacManObservation, PacManState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class PacManEnv(HTTPEnvClient[PacManAction, PacManObservation]):
    """
    HTTP client for Pac-Man Environment.

    This client connects to a PacManEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = PacManEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.pacman_position)
        >>>
        >>> # Take an action (move right)
        >>> result = client.step(PacManAction(action_id=1))
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PacManEnv.from_docker_image("pacman-env:latest")
        >>> result = client.reset()
        >>> result = client.step(PacManAction(action_id=0))  # Move up
    """

    def _step_payload(self, action: PacManAction) -> Dict[str, Any]:
        """
        Convert PacManAction to JSON payload for step request.

        Args:
            action: PacManAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_id": action.action_id,
            "strategy_code": action.strategy_code,
            "difficulty_level": action.difficulty_level,
            "ghost_ai_type": action.ghost_ai_type,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PacManObservation]:
        """
        Parse server response into StepResult[PacManObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with PacManObservation.
        """
        obs_data = payload.get("observation", {})

        observation = PacManObservation(
            maze_layout=obs_data.get("maze_layout", []),
            pacman_position=tuple(obs_data.get("pacman_position", [0, 0])),
            pacman_direction=obs_data.get("pacman_direction", "right"),
            ghost_positions=[tuple(pos) for pos in obs_data.get("ghost_positions", [])],
            ghost_states=obs_data.get("ghost_states", []),
            ghost_directions=obs_data.get("ghost_directions", []),
            pellets_remaining=obs_data.get("pellets_remaining", 0),
            power_pellets_remaining=obs_data.get("power_pellets_remaining", 0),
            score=obs_data.get("score", 0),
            lives=obs_data.get("lives", 3),
            frightened_timer=obs_data.get("frightened_timer", 0),
            nearest_pellet_distance=obs_data.get("nearest_pellet_distance", 0),
            nearest_ghost_distance=obs_data.get("nearest_ghost_distance", 0),
            safe_directions=obs_data.get("safe_directions", []),
            legal_actions=obs_data.get("legal_actions", []),
            level_complete=obs_data.get("level_complete", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PacManState:
        """
        Parse server response into PacManState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            PacManState object with environment state information.
        """
        return PacManState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            game_name=payload.get("game_name", "pacman"),
            difficulty_level=payload.get("difficulty_level", 1),
            ghost_ai_type=payload.get("ghost_ai_type", "random"),
            maze_size=tuple(payload.get("maze_size", [15, 15])),
            max_steps=payload.get("max_steps", 250),
            num_ghosts=payload.get("num_ghosts", 4),
            total_pellets=payload.get("total_pellets", 0),
            total_power_pellets=payload.get("total_power_pellets", 0),
        )
