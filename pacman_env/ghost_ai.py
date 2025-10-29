# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ghost AI and entity management for Pac-Man environment.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque


@dataclass
class Ghost:
    """
    Represents a ghost entity in the Pac-Man game.
    
    Attributes:
        position: Current (row, col) position.
        state: Current state ("normal", "frightened", "eaten").
        direction: Current direction ("up", "down", "left", "right").
        personality: Ghost personality type for behavior ("blinky", "pinky", "inky", "clyde").
        home_corner: Home corner position for scatter mode.
    """
    position: Tuple[int, int]
    state: str = "normal"
    direction: str = "up"
    personality: str = "blinky"
    home_corner: Tuple[int, int] = (0, 0)


class GhostAI:
    """
    AI controller for ghost behaviors.
    
    Provides different movement policies for ghosts.
    """
    
    # Direction vectors
    DIRECTIONS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    
    @staticmethod
    def get_legal_moves(position: Tuple[int, int], maze: List[List[str]]) -> List[str]:
        """
        Get legal movement directions from current position.
        
        Args:
            position: Current (row, col) position.
            maze: The maze layout.
            
        Returns:
            List of legal direction strings.
        """
        row, col = position
        legal = []
        
        for direction, (dr, dc) in GhostAI.DIRECTIONS.items():
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < len(maze) and 0 <= new_col < len(maze[0]):
                # Can move to non-wall spaces
                if maze[new_row][new_col] != 'W':
                    legal.append(direction)
        
        return legal
    
    @staticmethod
    def random_policy(ghost: Ghost, maze: List[List[str]], 
                     pacman_pos: Tuple[int, int]) -> str:
        """
        Random movement policy.
        
        Args:
            ghost: The ghost entity.
            maze: The maze layout.
            pacman_pos: Pac-Man's position.
            
        Returns:
            Direction to move.
        """
        legal_moves = GhostAI.get_legal_moves(ghost.position, maze)
        
        if not legal_moves:
            return ghost.direction
        
        # Avoid reversing direction if possible
        reverse_dir = GhostAI._reverse_direction(ghost.direction)
        forward_moves = [d for d in legal_moves if d != reverse_dir]
        
        if forward_moves:
            return random.choice(forward_moves)
        return random.choice(legal_moves)
    
    @staticmethod
    def chase_policy(ghost: Ghost, maze: List[List[str]], 
                    pacman_pos: Tuple[int, int]) -> str:
        """
        Chase policy using simple pathfinding toward Pac-Man.
        
        Args:
            ghost: The ghost entity.
            maze: The maze layout.
            pacman_pos: Pac-Man's position.
            
        Returns:
            Direction to move toward Pac-Man.
        """
        legal_moves = GhostAI.get_legal_moves(ghost.position, maze)
        
        if not legal_moves:
            return ghost.direction
        
        # Simple greedy approach: move toward Pac-Man
        best_direction = legal_moves[0]
        best_distance = float('inf')
        
        for direction in legal_moves:
            dr, dc = GhostAI.DIRECTIONS[direction]
            new_pos = (ghost.position[0] + dr, ghost.position[1] + dc)
            distance = GhostAI._manhattan_distance(new_pos, pacman_pos)
            
            if distance < best_distance:
                best_distance = distance
                best_direction = direction
        
        return best_direction
    
    @staticmethod
    def ambush_policy(ghost: Ghost, maze: List[List[str]], 
                     pacman_pos: Tuple[int, int],
                     pacman_dir: str = "right") -> str:
        """
        Ambush policy that tries to intercept Pac-Man.
        
        Args:
            ghost: The ghost entity.
            maze: The maze layout.
            pacman_pos: Pac-Man's position.
            pacman_dir: Pac-Man's current direction.
            
        Returns:
            Direction to move to intercept Pac-Man.
        """
        # Target position is 4 cells ahead of Pac-Man
        dr, dc = GhostAI.DIRECTIONS.get(pacman_dir, (0, 1))
        target_pos = (pacman_pos[0] + 4 * dr, pacman_pos[1] + 4 * dc)
        
        # Move toward target position
        legal_moves = GhostAI.get_legal_moves(ghost.position, maze)
        
        if not legal_moves:
            return ghost.direction
        
        best_direction = legal_moves[0]
        best_distance = float('inf')
        
        for direction in legal_moves:
            dr, dc = GhostAI.DIRECTIONS[direction]
            new_pos = (ghost.position[0] + dr, ghost.position[1] + dc)
            distance = GhostAI._manhattan_distance(new_pos, target_pos)
            
            if distance < best_distance:
                best_distance = distance
                best_direction = direction
        
        return best_direction
    
    @staticmethod
    def scatter_policy(ghost: Ghost, maze: List[List[str]], 
                      pacman_pos: Tuple[int, int]) -> str:
        """
        Scatter policy where ghost heads to its home corner.
        
        Args:
            ghost: The ghost entity.
            maze: The maze layout.
            pacman_pos: Pac-Man's position (unused but kept for signature).
            
        Returns:
            Direction to move toward home corner.
        """
        legal_moves = GhostAI.get_legal_moves(ghost.position, maze)
        
        if not legal_moves:
            return ghost.direction
        
        # Move toward home corner
        best_direction = legal_moves[0]
        best_distance = float('inf')
        
        for direction in legal_moves:
            dr, dc = GhostAI.DIRECTIONS[direction]
            new_pos = (ghost.position[0] + dr, ghost.position[1] + dc)
            distance = GhostAI._manhattan_distance(new_pos, ghost.home_corner)
            
            if distance < best_distance:
                best_distance = distance
                best_direction = direction
        
        return best_direction
    
    @staticmethod
    def frightened_policy(ghost: Ghost, maze: List[List[str]], 
                         pacman_pos: Tuple[int, int]) -> str:
        """
        Frightened policy with random movement (when Pac-Man has power pellet).
        
        Args:
            ghost: The ghost entity.
            maze: The maze layout.
            pacman_pos: Pac-Man's position.
            
        Returns:
            Random direction.
        """
        return GhostAI.random_policy(ghost, maze, pacman_pos)
    
    @staticmethod
    def _manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def _reverse_direction(direction: str) -> str:
        """Get the reverse of a direction."""
        reverses = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }
        return reverses.get(direction, direction)
