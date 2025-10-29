# Pac-Man RL — Language Model Strategy Training

![Ms. Pac-Man gameplay](https://ale.farama.org/_images/ms_pacman.gif)

Training language models to play Pac-Man by generating strategy code instead of discrete actions.

## What Is This?

A reinforcement learning environment where LLMs learn to write Python functions that control Pac-Man. Instead of outputting action IDs directly, the model generates code like:

```python
def strategy(obs):
    if obs.frightened_timer > 0: return [1, 2, 0]  # Chase ghosts
    if obs.nearest_ghost_distance < 3: return [3]  # Flee left
    return [0, 1, 2, 3]  # Explore
```

The environment validates the code, executes it safely, and uses the trajectory rewards to fine-tune the model with GRPO.

**Why this matters**: Tests whether LLMs can learn goal-directed reasoning through code generation, not just memorization.

## Training Architecture

```text
┌─────────────────────────────────────────────────────────┐
│  1. ENVIRONMENT → Structured Observation                │
├─────────────────────────────────────────────────────────┤
│  • Pac-Man position, lives, score                       │
│  • Ghost positions/states (normal/frightened)           │
│  • Pellet counts, nearest distances                     │
│  • Safe directions, legal actions                       │
│  • 15x15 maze layout (walls, pellets, power-ups)        │
└─────────────────────────────────────────────────────────┘
                         ↓
           observation_to_prompt(obs)
                         ↓
┌─────────────────────────────────────────────────────────┐
│  2. PROMPT GENERATION (~500 tokens)                     │
├─────────────────────────────────────────────────────────┤
│  Write a compact Ms. Pac-Man strategy function.         │
│                                                          │
│  obs: pacman_position, ghost_positions, ghost_states,   │
│       frightened_timer, nearest_ghost_distance,         │
│       safe_directions, legal_actions, pellets, lives    │
│                                                          │
│  Return list of 1-5 actions [0,1,2,3,4]                 │
│  (UP,RIGHT,DOWN,LEFT,STAY)                              │
└─────────────────────────────────────────────────────────┘
                         ↓
              Llama 3.2 3B (LoRA)
                         ↓
┌─────────────────────────────────────────────────────────┐
│  3. LLM GENERATES STRATEGY CODE                         │
├─────────────────────────────────────────────────────────┤
│  def strategy(obs):                                     │
│      if obs.frightened_timer > 0:                       │
│          return [1, 2, 0]  # Chase ghosts               │
│      if obs.nearest_ghost_distance < 3:                 │
│          if "left" in obs.safe_directions: return [3]   │
│      return [0, 1, 2, 3]  # Explore                     │
└─────────────────────────────────────────────────────────┘
                         ↓
       execute_with_time_limit(code, obs)
                         ↓
┌─────────────────────────────────────────────────────────┐
│  4. EXECUTE → ROLLOUT → REWARDS                         │
├─────────────────────────────────────────────────────────┤
│  • Validate code (AST parse, sandbox execution)         │
│  • Run 12-step rollout in Pac-Man environment           │
│  • Collect rewards: +10 pellet, +50 power, +200 ghost   │
│  • Calculate trajectory return                          │
└─────────────────────────────────────────────────────────┘
                         ↓
            After batch of episodes...
                         ↓
┌─────────────────────────────────────────────────────────┐
│  5. GRPO UPDATE → Improve Strategy                      │
├─────────────────────────────────────────────────────────┤
│  • Compute advantages from trajectory rewards           │
│  • Backpropagate through LoRA adapters                  │
│  • Model learns: avoid ghosts, chase power pellets,     │
│    maximize score, survive longer                       │
└─────────────────────────────────────────────────────────┘
```

**Technical Stack:**

- **Model**: Llama 3.2 3B (LoRA rank 32)
- **Training**: GRPO via TRL + Unsloth
- **Environment**: Docker-based FastAPI server (configurable difficulty, ghost AI)
- **Observation**: ~500 token prompts with structured game state

## Project Structure

```text
pacman_env/               # OpenEnv-compatible environment
├── server/
│   ├── pacman_environment.py   # Game logic, rewards, collision detection
│   └── app.py                   # FastAPI service (Docker-ready)
├── maze_generator.py     # Procedural maze generation
├── ghost_ai.py           # Ghost personalities (chase, ambush, scatter)
├── client.py             # HTTP client with Docker orchestration
└── models.py             # Action/Observation/State types

play_the_game/            # Inference and visualization
├── html_pacman_player.py       # Flask web UI for watching trained agents
└── simple_model_server.py      # Lightweight model serving

train_pacman_docker_grpo_v2.py  # Main training script
```

## Quick Start

**Train from scratch:**

```powershell
# Launches Docker container + GRPO training
python train_pacman_docker_grpo_v2.py
```

**Watch a trained agent play:**

```powershell
# Start model server
python play_the_game/simple_model_server.py --model-path outputs_pacman/final_model

# Launch web UI (separate terminal)
python play_the_game/html_pacman_player.py
# Open browser to http://localhost:5000
```

## Why This Design?

**Code Generation vs Action Prediction:**

- Traditional RL: Model outputs action ID directly → limited interpretability
- This approach: Model writes strategy function → explainable, debuggable, composable

**Long-Horizon Reasoning:**

- 12-step rollouts test planning beyond immediate rewards
- Sparse rewards (power pellets, level completion) require multi-step thinking

**Emergent Strategies:**

- Early training: Random exploration, high death rate
- Mid training: Learns to avoid ghosts, collect nearby pellets
- Late training: Coordinates power pellet usage with ghost hunting

## Environment Details

**Configurable Parameters:**

- `PACMAN_DIFFICULTY`: 1-5 (maze size, ghost count, AI aggression)
- `PACMAN_GHOST_AI`: random | heuristic (Blinky/Pinky/Inky/Clyde personalities)
- `PACMAN_MAZE_SIZE`: Grid dimensions (default 15x15)
- `PACMAN_MAX_STEPS`: Episode length (default 600)

**Reward Structure:**

- Pellet: +10
- Power pellet: +50
- Eating ghost (when powered): +200
- Death: -500
- Level complete: +1000
- Time penalty: -1 per step

## Training Progress (Sample)

```text
Step 50  | Reward: +8.7  | Turns: 134  | Parse failures: 34% → 12%
Step 100 | Reward: +22.3 | Turns: 189  | Code quality improving
Step 200 | Reward: +45.6 | Turns: 243  | Power pellet strategies emerge
Step 400 | Reward: +78.2 | Turns: 312  | Coordinated ghost evasion
```

**Learned Behaviors:**

- Flee from ghosts when `nearest_ghost_distance < 3`
- Chase ghosts during `frightened_timer > 0`
- Prefer `safe_directions` over risky moves
- Prioritize power pellets when ghosts nearby

## Requirements

- Python 3.11+
- PyTorch, Unsloth, TRL, Datasets
- Docker (for environment server)
- GPU recommended (4-bit quantization supported)
