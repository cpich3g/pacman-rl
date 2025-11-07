# Pac-Man RL — Language Model Strategy Training 

[AMD x Unsloth x PyTorch OpenEnv Hackthon - 2nd Place Winner 🏆]

![Ms. Pac-Man gameplay](https://ale.farama.org/_images/ms_pacman.gif)

Training language models to play Pac-Man by generating strategy code instead of discrete actions.

## Watch the Trained Agent Play

<video src="https://github.com/user-attachments/assets/e3310ca1-fcfe-4bdf-ac76-55333646bb13" controls></video>

**🤗 Trained Models:**
- [justinj92/Llama-3.1-8B-Pacman-Player](https://huggingface.co/justinj92/Llama-3.1-8B-Pacman-Player) (Llama 3.1 8B)
- [justinj92/gpt-oss-20B-pacmanplayer](https://huggingface.co/justinj92/gpt-oss-20B-pacmanplayer) (GPT-OSS 20B)

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
              Llama 3.1 8B Instruct (LoRA)
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

- **Base Model**: [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (LoRA rank 32, alpha 64)
- **Trained Models**: 
  - [🤗 Llama-3.1-8B-Pacman-Player](https://huggingface.co/justinj92/Llama-3.1-8B-Pacman-Player) (8B)
  - [🤗 gpt-oss-20B-pacmanplayer](https://huggingface.co/justinj92/gpt-oss-20B-pacmanplayer) (20B)
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

## Installation

This project requires [Meta's OpenEnv framework](https://github.com/meta-pytorch/OpenEnv). See [INSTALL.md](INSTALL.md) for detailed setup instructions.

**Quick install:**

```bash
# 1. Install OpenEnv
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv && pip install -e . && cd ..

# 2. Clone this repo and install dependencies
git clone https://github.com/cpich3g/pacman-rl.git
cd pacman-rl
pip install -r requirements.txt

# 3. Register pacman_env with OpenEnv
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 4. Pull Docker image
docker pull ghcr.io/meta-pytorch/openenv-pacman-env:latest
```

## Quick Start

**Explore the notebook:**

Check out [notebooks/Pacman-RL.ipynb](notebooks/Pacman-RL.ipynb) for interactive examples.

**Use the pretrained models:**

Download from 🤗 Hugging Face (choose one):

**Option 1: Llama 3.1 8B**
```bash
git clone https://huggingface.co/justinj92/Llama-3.1-8B-Pacman-Player
python play_the_game/simple_model_server.py --model-path Llama-3.1-8B-Pacman-Player
```

**Option 2: GPT-OSS 20B (larger, better performance)**
```bash
git clone https://huggingface.co/justinj92/gpt-oss-20B-pacmanplayer
python play_the_game/simple_model_server.py --model-path gpt-oss-20B-pacmanplayer
```

**Then launch the web UI:**
```bash
python play_the_game/html_pacman_player.py
# Open browser to http://localhost:5000
```

**Train from scratch:**

```bash
# Launches Docker container + GRPO training
python train_pacman_docker_grpo_v2.py
```

**Watch your trained agent play:**

```bash
# Start model server with your checkpoint
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
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta's environment framework)
- PyTorch, Unsloth, TRL, Datasets
- Docker (for environment server)
- GPU recommended (4-bit quantization supported)

**Full dependency list:** See [requirements.txt](requirements.txt)

**Installation guide:** See [INSTALL.md](INSTALL.md)

## Pretrained Models

Two trained models are available on Hugging Face:

### 🤗 Llama 3.1 8B Model

**[justinj92/Llama-3.1-8B-Pacman-Player](https://huggingface.co/justinj92/Llama-3.1-8B-Pacman-Player)**

**Details:**
- Base: Meta-Llama-3.1-8B-Instruct
- Fine-tuning: GRPO with code generation
- LoRA: rank 32, alpha 64
- Training: 400 steps on Pac-Man gameplay

**Quick Use:**
```bash
git clone https://huggingface.co/justinj92/Llama-3.1-8B-Pacman-Player
python play_the_game/simple_model_server.py --model-path Llama-3.1-8B-Pacman-Player
```

### 🤗 GPT-OSS 20B Model

**[justinj92/gpt-oss-20B-pacmanplayer](https://huggingface.co/justinj92/gpt-oss-20B-pacmanplayer)**

**Details:**
- Base: GPT-OSS 20B
- Fine-tuning: GRPO with code generation
- Larger model with potentially better performance

**Quick Use:**
```bash
git clone https://huggingface.co/justinj92/gpt-oss-20B-pacmanplayer
python play_the_game/simple_model_server.py --model-path gpt-oss-20B-pacmanplayer
```

## Installation as Package

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```
