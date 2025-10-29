# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rendering utilities for Pac-Man environment.

Provides ASCII art visualization of the game state.
"""

from typing import List, Tuple
from .models import PacManObservation


def render_maze(
    obs: PacManObservation,
    colors: bool = True,
    show_stats: bool = True,
) -> str:
    """
    Render the Pac-Man maze as ASCII art.

    Args:
        obs: PacManObservation to render.
        colors: Whether to use ANSI color codes.
        show_stats: Whether to show game statistics below the maze.

    Returns:
        String representation of the game state.
    """
    # ANSI color codes
    if colors:
        RESET = "\x1b[0m"
        YELLOW = "\x1b[33m"  # Pac-Man
        RED = "\x1b[31m"     # Normal ghost
        BLUE = "\x1b[34m"    # Frightened ghost
        CYAN = "\x1b[36m"    # Eaten ghost
        WHITE = "\x1b[37m"   # Pellet
        MAGENTA = "\x1b[35m" # Power pellet
        GRAY = "\x1b[90m"    # Wall
    else:
        RESET = YELLOW = RED = BLUE = CYAN = WHITE = MAGENTA = GRAY = ""

    maze = obs.maze_layout
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    lines = []

    # Top border
    lines.append("┌" + "─" * (cols * 2) + "┐")

    # Render each row
    for i in range(rows):
        row_str = "│"
        for j in range(cols):
            pos = (i, j)
            cell = maze[i][j]

            # Check if Pac-Man is at this position
            if pos == obs.pacman_position:
                row_str += f"{YELLOW}P {RESET}"
            # Check if any ghost is at this position
            elif pos in obs.ghost_positions:
                ghost_idx = obs.ghost_positions.index(pos)
                ghost_state = obs.ghost_states[ghost_idx]
                if ghost_state == "frightened":
                    row_str += f"{BLUE}F {RESET}"
                elif ghost_state == "eaten":
                    row_str += f"{CYAN}E {RESET}"
                else:
                    row_str += f"{RED}G {RESET}"
            # Render the cell itself
            elif cell == 'W':
                row_str += f"{GRAY}█ {RESET}"
            elif cell == '.':
                row_str += f"{WHITE}· {RESET}"
            elif cell == 'O':
                row_str += f"{MAGENTA}○ {RESET}"
            else:
                row_str += "  "

        row_str += "│"
        lines.append(row_str)

    # Bottom border
    lines.append("└" + "─" * (cols * 2) + "┘")

    # Add stats if requested
    if show_stats:
        lines.append("")
        lines.append("═" * (cols * 2 + 2))
        lines.append(f"Score: {obs.score:>6} │ Lives: {obs.lives} │ Pellets: {obs.pellets_remaining:>3}")
        lines.append(f"Power: {obs.power_pellets_remaining:>6} │ Step: {obs.metadata.get('step', '?'):>4} │ Frightened: {obs.frightened_timer:>3}")
        
        # Ghost status
        ghost_status = []
        for i, (state, pos) in enumerate(zip(obs.ghost_states, obs.ghost_positions)):
            ghost_status.append(f"G{i+1}:{state[0].upper()}@{pos}")
        lines.append(f"Ghosts: {' | '.join(ghost_status)}")
        
        # Safe directions
        if obs.safe_directions:
            lines.append(f"Safe: {', '.join(obs.safe_directions)}")
        
        lines.append("═" * (cols * 2 + 2))

    return "\n".join(lines)


def render_legend() -> str:
    """
    Render the legend for Pac-Man symbols.

    Returns:
        String with symbol explanations.
    """
    legend = """
    LEGEND:
    ═══════════════════════════════
    P  = Pac-Man (you)
    G  = Ghost (normal)
    F  = Ghost (frightened/blue)
    E  = Ghost (eaten/eyes)
    ·  = Pellet (+10 points)
    ○  = Power Pellet (+50 points)
    █  = Wall
    ═══════════════════════════════
    
    CONTROLS:
    0 = UP
    1 = RIGHT  
    2 = DOWN
    3 = LEFT
    4 = STAY
    """
    return legend


def observation_to_text(obs: PacManObservation) -> str:
    """
    Convert observation to LLM-friendly text format.

    This format is designed to be easily understood by language models
    for generating strategic decisions.

    Args:
        obs: PacManObservation to convert.

    Returns:
        Text description of the game state.
    """
    text_parts = [
        "=== PAC-MAN GAME STATE ===\n",
        f"Your Position: {obs.pacman_position}",
        f"Current Direction: {obs.pacman_direction}",
        f"Lives Remaining: {obs.lives}",
        f"Score: {obs.score}\n",
        
        f"Pellets Left: {obs.pellets_remaining}",
        f"Power Pellets Left: {obs.power_pellets_remaining}",
        f"Nearest Pellet: {obs.nearest_pellet_distance} cells away\n",
        
        f"Ghosts:",
    ]
    
    for i, (pos, state, direction) in enumerate(zip(
        obs.ghost_positions,
        obs.ghost_states,
        obs.ghost_directions
    )):
        text_parts.append(f"  Ghost {i+1}: {pos} ({state}, moving {direction})")
    
    text_parts.append(f"\nNearest Ghost: {obs.nearest_ghost_distance} cells away")
    
    if obs.frightened_timer > 0:
        text_parts.append(f"POWER MODE ACTIVE: {obs.frightened_timer} steps remaining!")
        text_parts.append("Ghosts are vulnerable - chase them for bonus points!")
    
    if obs.safe_directions:
        text_parts.append(f"\nSafe Directions: {', '.join(obs.safe_directions)}")
    else:
        text_parts.append("\nWARNING: Ghosts nearby in all directions!")
    
    text_parts.append(f"\nLegal Actions: {obs.legal_actions}")
    text_parts.append("(0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY)")
    
    return "\n".join(text_parts)
