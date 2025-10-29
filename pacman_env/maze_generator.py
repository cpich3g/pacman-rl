# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze generation for Pac-Man environment.
"""

import random
from typing import List, Tuple


class MazeGenerator:
    """
    Generator for Pac-Man style mazes.
    
    Creates classic Pac-Man mazes with corridors, ghost house, and pellet placement.
    """
    
    def __init__(self, rows: int = 15, cols: int = 15):
        """
        Initialize maze generator.
        
        Args:
            rows: Number of rows in the maze.
            cols: Number of columns in the maze.
        """
        self.rows = rows
        self.cols = cols
        
    def generate(self) -> List[List[str]]:
        """
        Generate a Pac-Man style maze.
        
        Returns:
            2D list representing the maze layout.
            'W' = wall, ' ' = empty, '.' = pellet, 'O' = power pellet
        """
        # Start with all walls
        maze = [['W' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Create symmetric maze (Pac-Man style)
        # Use recursive backtracking for maze generation
        self._create_symmetric_maze(maze)
        
        # Add ghost house in the center
        self._add_ghost_house(maze)
        
        # Place pellets in corridors
        self._place_pellets(maze)
        
        # Place power pellets in corners
        self._place_power_pellets(maze)
        
        return maze
    
    def _create_symmetric_maze(self, maze: List[List[str]]) -> None:
        """Create a symmetric maze pattern using simple corridor design."""
        # Create outer boundary
        for i in range(self.rows):
            for j in range(self.cols):
                # Outer walls
                if i == 0 or i == self.rows - 1 or j == 0 or j == self.cols - 1:
                    maze[i][j] = 'W'
                # Create corridors (simple pattern)
                elif i % 2 == 1 and j % 2 == 1:
                    maze[i][j] = ' '
                elif i % 4 == 0 and j % 2 == 1:
                    maze[i][j] = ' '
                elif j % 4 == 0 and i % 2 == 1:
                    maze[i][j] = ' '
        
        # Create main horizontal corridor in the middle
        mid_row = self.rows // 2
        for j in range(1, self.cols - 1):
            maze[mid_row][j] = ' '
        
        # Create vertical corridors
        for i in range(1, self.rows - 1):
            maze[i][self.cols // 4] = ' '
            maze[i][3 * self.cols // 4] = ' '
    
    def _add_ghost_house(self, maze: List[List[str]]) -> None:
        """Add a ghost house in the center of the maze."""
        center_row = self.rows // 2
        center_col = self.cols // 2
        
        # Create a small room for ghosts
        for i in range(max(1, center_row - 1), min(self.rows - 1, center_row + 2)):
            for j in range(max(1, center_col - 2), min(self.cols - 1, center_col + 3)):
                if i == center_row - 1 or i == center_row + 1:
                    if j != center_col:
                        maze[i][j] = 'W'
                else:
                    maze[i][j] = ' '
    
    def _place_pellets(self, maze: List[List[str]]) -> None:
        """Place regular pellets in all empty spaces."""
        for i in range(self.rows):
            for j in range(self.cols):
                if maze[i][j] == ' ':
                    maze[i][j] = '.'
    
    def _place_power_pellets(self, maze: List[List[str]]) -> None:
        """Place power pellets in the corners of the maze."""
        # Define corner regions to search for pellets
        corner_regions = [
            (range(1, min(5, self.rows - 1)), range(1, min(5, self.cols - 1))),  # Top-left
            (range(1, min(5, self.rows - 1)), range(max(1, self.cols - 5), self.cols - 1)),  # Top-right
            (range(max(1, self.rows - 5), self.rows - 1), range(1, min(5, self.cols - 1))),  # Bottom-left
            (range(max(1, self.rows - 5), self.rows - 1), range(max(1, self.cols - 5), self.cols - 1)),  # Bottom-right
        ]
        
        placed_count = 0
        for row_range, col_range in corner_regions:
            # Find a pellet in this corner region
            for row in row_range:
                for col in col_range:
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        if maze[row][col] == '.':
                            maze[row][col] = 'O'
                            placed_count += 1
                            break  # Found one in this corner, move to next corner
                if placed_count > len(corner_regions) - (corner_regions.index((row_range, col_range)) + 1):
                    break  # Already placed one in this corner
        
        # Fallback: if we couldn't place 4, try to place at least 2-3 in any available spots
        if placed_count < 2:
            pellet_positions = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if maze[i][j] == '.':
                        pellet_positions.append((i, j))
            
            # Place power pellets at evenly distributed positions
            if pellet_positions:
                step = max(1, len(pellet_positions) // 4)
                for i in range(0, min(4, len(pellet_positions)), step):
                    if i < len(pellet_positions):
                        row, col = pellet_positions[i]
                        maze[row][col] = 'O'
    
    def find_valid_positions(self, maze: List[List[str]], count: int) -> List[Tuple[int, int]]:
        """
        Find valid empty positions in the maze.
        
        Args:
            maze: The maze layout.
            count: Number of positions to find.
            
        Returns:
            List of (row, col) tuples.
        """
        valid_positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if maze[i][j] in ['.', 'O', ' ']:
                    valid_positions.append((i, j))
        
        # Return random sample
        if len(valid_positions) >= count:
            return random.sample(valid_positions, count)
        return valid_positions
