# Pac-Man Environment

Integration of the classic Pac-Man game with the OpenEnv framework. This environment provides a rich, multi-agent, long-horizon game perfect for LLM-based reinforcement learning.

## Overview

This Pac-Man implementation features:
- **Structured observations** optimized for LLM reasoning (not pixel-based)
- **Multiple ghost AI types** (random, heuristic with personalities, LLM-controlled)
- **Long-horizon gameplay** (200-300 steps per episode)
- **Configurable difficulty** (5 levels from beginner to expert)
- **Classic mechanics** (pellets, power pellets, ghost hunting, lives system)
- **Strategic depth** (evasion, pursuit, risk-reward decisions)

## Game Rules

### Objective
Navigate the maze and collect all pellets while avoiding ghosts. Collect power pellets to temporarily turn the tables and hunt the ghosts!

### Scoring
- **Pellet**: +10 points
- **Power Pellet**: +50 points
- **Eating Ghost**: +200 points (only when powered up)
- **Level Complete**: +1000 bonus
- **Death**: -500 penalty
- **Time**: -1 per step (encourages efficiency)

### Mechanics
- **Lives**: Start with 3 lives
- **Frightened Mode**: After eating a power pellet, ghosts turn blue and vulnerable for 40 steps
- **Ghost Personalities**:
  - **Blinky** (Red): Aggressive chaser - always pursues Pac-Man
  - **Pinky** (Pink): Ambusher - tries to intercept Pac-Man's path
  - **Inky** (Cyan): Unpredictable - mixes chase and random behavior
  - **Clyde** (Orange): Patrol - scatters when close, chases when far

## Architecture

```
┌────────────────────────────────────┐
│ RL Training Code (Client)          │
│   PacManEnv.step(action)           │
└──────────────┬─────────────────────┘
               │ HTTP
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)            │
│   PacManEnvironment                │
│     ├─ Maze Generation             │
│     ├─ Pac-Man Movement             │
│     ├─ Ghost AI                    │
│     ├─ Collision Detection         │
│     └─ Reward Calculation          │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- Python 3.11+

```python
from envs.pacman_env import PacManEnv, PacManAction

# Start local server manually
# python -m envs.pacman_env.server.app

# Connect to local server
env = PacManEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
print(f"Pac-Man position: {result.observation.pacman_position}")
print(f"Pellets remaining: {result.observation.pellets_remaining}")
print(f"Ghost positions: {result.observation.ghost_positions}")

# Take actions (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY)
for _ in range(10):
    action_id = 1  # Move right
    result = env.step(PacManAction(action_id=action_id))
    print(f"Score: {result.observation.score}, Lives: {result.observation.lives}")
    if result.done:
        break

# Cleanup
env.close()
```

### Option 2: Docker (Recommended)

**Build Docker image:**

```bash
cd OpenEnv

# Build the image
docker build \
  -f src/envs/pacman_env/server/Dockerfile \
  -t pacman-env:latest \
  .
```

**Run with different configurations:**

```bash
# Default (easy mode, random ghosts)
docker run -p 8000:8000 pacman-env:latest

# Medium difficulty with heuristic AI
docker run -p 8000:8000 \
  -e PACMAN_DIFFICULTY=3 \
  -e PACMAN_GHOST_AI=heuristic \
  pacman-env:latest

# Large maze, expert mode
docker run -p 8000:8000 \
  -e PACMAN_DIFFICULTY=5 \
  -e PACMAN_GHOST_AI=heuristic \
  -e PACMAN_MAZE_SIZE=20x20 \
  -e PACMAN_MAX_STEPS=400 \
  pacman-env:latest
```

**Use with from_docker_image():**

```python
from envs.pacman_env import PacManEnv, PacManAction

# Automatically starts container
env = PacManEnv.from_docker_image("pacman-env:latest")

result = env.reset()
result = env.step(PacManAction(action_id=1))  # Move right

env.close()  # Stops container
```

## Configuration

### Environment Variables

- `PACMAN_DIFFICULTY`: Difficulty level 1-5 (default: "1")
  - 1: Easy - Small maze, 2 random ghosts
  - 2: Normal - Medium maze, 3 mixed ghosts
  - 3: Hard - Large maze, 4 heuristic ghosts
  - 4: Expert - Large maze, coordinated ghosts
  - 5: Master - Large maze, aggressive AI

- `PACMAN_GHOST_AI`: Ghost AI type (default: "random")
  - `random`: Random movement
  - `heuristic`: Smart pathfinding with personalities
  - `llm`: LLM-controlled (future feature)

- `PACMAN_MAZE_SIZE`: Maze dimensions as "ROWSxCOLS" (default: "15x15")
  - Recommended: 10x10 (small), 15x15 (medium), 20x20 (large)

- `PACMAN_MAX_STEPS`: Maximum steps per episode (default: "250")
  - Longer episodes allow more strategic gameplay

- `PACMAN_NUM_GHOSTS`: Number of ghosts 1-4 (default: "4")

### Example Configurations

```bash
# Beginner - Small maze, random ghosts
docker run -p 8000:8000 \
  -e PACMAN_DIFFICULTY=1 \
  -e PACMAN_MAZE_SIZE=10x10 \
  -e PACMAN_NUM_GHOSTS=2 \
  pacman-env:latest

# Advanced - Large maze, smart ghosts
docker run -p 8000:8000 \
  -e PACMAN_DIFFICULTY=4 \
  -e PACMAN_GHOST_AI=heuristic \
  -e PACMAN_MAZE_SIZE=20x20 \
  -e PACMAN_MAX_STEPS=400 \
  pacman-env:latest
```

## API Reference

### PacManAction

```python
@dataclass
class PacManAction(Action):
    action_id: int                      # 0-4: up, right, down, left, stay
    strategy_code: Optional[str]        # Optional Python strategy function
    difficulty_level: int = 1           # Difficulty level
    ghost_ai_type: str = "random"       # Ghost AI type
```

### PacManObservation

```python
@dataclass
class PacManObservation(Observation):
    maze_layout: List[List[str]]            # 2D grid ('W'=wall, '.'=pellet, 'O'=power-pellet)
    pacman_position: Tuple[int, int]        # Current position
    pacman_direction: str                   # Current direction
    ghost_positions: List[Tuple[int, int]]  # Ghost positions
    ghost_states: List[str]                 # Ghost states (normal/frightened/eaten)
    ghost_directions: List[str]             # Ghost directions
    pellets_remaining: int                  # Pellets left to collect
    power_pellets_remaining: int            # Power pellets left
    score: int                              # Current score
    lives: int                              # Lives remaining
    frightened_timer: int                   # Frightened mode countdown
    nearest_pellet_distance: int            # Manhattan distance to nearest pellet
    nearest_ghost_distance: int             # Manhattan distance to nearest ghost
    safe_directions: List[str]              # Directions without nearby ghosts
    legal_actions: List[int]                # Always [0, 1, 2, 3, 4]
    level_complete: bool                    # All pellets collected
```

### PacManState

```python
@dataclass
class PacManState(State):
    episode_id: str                     # Unique episode ID
    step_count: int                     # Steps taken
    game_name: str = "pacman"           # Game identifier
    difficulty_level: int               # Difficulty level
    ghost_ai_type: str                  # Ghost AI type
    maze_size: Tuple[int, int]          # Maze dimensions
    max_steps: int                      # Max steps allowed
    num_ghosts: int                     # Number of ghosts
    total_pellets: int                  # Total pellets at start
    total_power_pellets: int            # Total power pellets at start
```

## Visualization

### ASCII Rendering

```python
from envs.pacman_env import PacManEnv
from envs.pacman_env.renderer import render_maze, render_legend

env = PacManEnv(base_url="http://localhost:8000")
result = env.reset()

# Render the maze
print(render_maze(result.observation))

# Show legend
print(render_legend())
```

Example output:
```
┌──────────────────────────────┐
│█ · · · · · · · · · · · · · █│
│·██·████·████·████·████·██·│
│·█  ·  ·  P  ·  ·  G  ·  █·│
│·  ·██████·█·██████·  ·│
│○  ·  ·  ·  ·  ·  ·  ·  ○│
└──────────────────────────────┘
════════════════════════════════
Score:    150 │ Lives: 3 │ Pellets:  87
Power:      4 │ Step:   25 │ Frightened:   0
Ghosts: G1:N@(7,15) | G2:F@(8,9) | G3:N@(5,12) | G4:N@(10,8)
Safe: up, right, left
════════════════════════════════

LEGEND:
P  = Pac-Man (you)
G  = Ghost (normal)
F  = Ghost (frightened/blue)
E  = Ghost (eaten/eyes)
·  = Pellet (+10 points)
○  = Power Pellet (+50 points)
█  = Wall
```

## LLM Integration

This environment is designed for LLM-based RL training (e.g., with GPT-OSS and Unsloth).

### Strategy-Based Actions

Instead of just action IDs, LLMs can generate Python strategy functions:

```python
def strategy(observation):
    """
    LLM-generated Pac-Man strategy.
    
    Args:
        observation: Dict with maze state, positions, etc.
    
    Returns:
        Action ID (0-4)
    """
    # Example strategy: Avoid ghosts, collect pellets
    if observation['nearest_ghost_distance'] < 3:
        # Run away!
        for direction in observation['safe_directions']:
            if direction == 'right': return 1
            if direction == 'left': return 3
            if direction == 'up': return 0
            if direction == 'down': return 2
    
    # Chase power pellets if nearby
    if observation['power_pellets_remaining'] > 0:
        # Move toward power pellets
        return 1  # Simplified
    
    # Default: collect nearest pellet
    return 1
```

### Text-Based Observations for LLMs

```python
from envs.pacman_env.renderer import observation_to_text

result = env.reset()
text = observation_to_text(result.observation)
print(text)
```

Output:
```
=== PAC-MAN GAME STATE ===

Your Position: (7, 1)
Current Direction: right
Lives Remaining: 3
Score: 0

Pellets Left: 120
Power Pellets Left: 4
Nearest Pellet: 1 cells away

Ghosts:
  Ghost 1: (7, 13) (normal, moving left)
  Ghost 2: (8, 7) (normal, moving up)
  Ghost 3: (5, 10) (normal, moving right)
  Ghost 4: (10, 5) (normal, moving down)

Nearest Ghost: 6 cells away

Safe Directions: up, right, down, left

Legal Actions: [0, 1, 2, 3, 4]
(0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY)
```

## Example Training Script

See `examples/pacman_simple.py` for a complete example of:
- Connecting to the environment
- Running episodes with random or heuristic policies
- Collecting statistics and visualizing results

## Advanced Features

### Multi-Difficulty Curriculum

Train agents progressively from easy to expert:

```python
difficulties = [1, 2, 3, 4, 5]
for difficulty in difficulties:
    env = PacManEnv.from_docker_image(
        "pacman-env:latest",
        env_vars={"PACMAN_DIFFICULTY": str(difficulty)}
    )
    # Train until threshold performance
    # Then advance to next difficulty
```

### Agent vs Agent

Pit an LLM-controlled Pac-Man against LLM-controlled ghosts:

```python
# Pac-Man agent generates movement strategy
# Ghost agent generates pursuit/ambush strategy  
# Co-evolution creates increasingly sophisticated gameplay
```

## Testing

```bash
# Local testing
export PYTHONPATH=/path/to/OpenEnv/src
python -m envs.pacman_env.server.app &
sleep 2
python examples/pacman_simple.py

# Docker testing
docker build -f src/envs/pacman_env/server/Dockerfile -t pacman-env:latest .
docker run -p 8000:8000 pacman-env:latest

# Test from another terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

## Performance Characteristics

- **Episode Length**: 100-400 steps (configurable)
- **State Space**: ~15x15 to 20x20 grids with multiple entities
- **Action Space**: 5 discrete actions
- **Observation Size**: Structured data (~500-1000 elements)
- **Reward Range**: -500 to +2000+ per episode

## Limitations & Future Work

### Current Limitations
- No multiplayer (single Pac-Man only)
- Simplified maze generation (no classic maze reproduction)
- Ghost AI doesn't use full A* pathfinding (performance)
- No render to RGB pixels (ASCII only)

### Planned Features
- [ ] LLM-controlled ghost AI
- [ ] Classic Pac-Man maze layouts
- [ ] Multi-agent training (multiple Pac-Men)
- [ ] Advanced pathfinding algorithms
- [ ] Replay system for analysis
- [ ] Performance optimizations

## References

- Classic Pac-Man game mechanics
- OpenEnv framework: https://github.com/meta-pytorch/OpenEnv
- Ghost AI behaviors inspired by original Pac-Man AI

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{pacman_openenv,
  title={Pac-Man Environment for OpenEnv},
  author={OpenEnv Contributors},
  year={2025},
  url={https://github.com/meta-pytorch/OpenEnv}
}
```
