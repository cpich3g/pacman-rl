# Installation Guide

## Prerequisites

- Python 3.11 or higher
- Docker (for running the Pac-Man environment)
- CUDA-capable GPU (recommended for training, 4-bit quantization supported)

## Step 1: Install OpenEnv Framework

This project requires Meta's OpenEnv framework. Install it from source:

```bash
# Clone OpenEnv repository
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv

# Install OpenEnv
pip install -e .
cd ..
```

## Step 2: Register Pac-Man Environment with OpenEnv

After installing OpenEnv, register this Pac-Man environment:

```bash
# Clone this repository
git clone https://github.com/cpich3g/pacman-rl.git
cd pacman-rl

# Create a symbolic link in OpenEnv's envs directory
# Option A: If you installed OpenEnv from source
ln -s "$(pwd)/pacman_env" /path/to/OpenEnv/src/envs/pacman_env

# Option B: Alternative - Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Windows (PowerShell):**
```powershell
# Create junction/symlink (requires admin privileges)
New-Item -ItemType SymbolicLink -Path "C:\path\to\OpenEnv\src\envs\pacman_env" -Target "$(pwd)\pacman_env"

# Or add to PYTHONPATH
$env:PYTHONPATH += ";$(pwd)"
```

## Step 3: Install Pac-Man RL Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install unsloth (for efficient LLM training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Step 4: Verify Installation

```bash
# Test that OpenEnv can find the Pac-Man environment
python -c "from pacman_env.client import PacManEnv; print('✓ Pac-Man environment loaded')"

# Verify Docker is running
docker --version
docker ps
```

## Step 5: Pull the Docker Image

The training script uses a Dockerized environment:

```bash
# Pull the Pac-Man environment Docker image
docker pull ghcr.io/meta-pytorch/openenv-pacman-env:latest
```

## Optional: Install as Package

To install pacman-rl as a Python package:

```bash
# Development mode (editable install)
pip install -e .

# Or install normally
pip install .

# With development tools
pip install -e ".[dev]"
```

## Quick Test

Run a quick training test:

```bash
# This will start Docker container and begin GRPO training
python train_pacman_docker_grpo_v2.py
```

Watch the trained agent play:

```bash
# Terminal 1: Start model server
python play_the_game/simple_model_server.py --model-path outputs_pacman/final_model

# Terminal 2: Launch web UI
python play_the_game/html_pacman_player.py

# Open browser to http://localhost:5000
```

## Troubleshooting

### "Cannot import from core.env_server"

Make sure OpenEnv is installed and the Pac-Man environment is registered:
- Check that OpenEnv is in your Python path
- Verify symbolic link or PYTHONPATH is set correctly

### Docker Connection Issues

```bash
# Check Docker is running
docker info

# Test container manually
docker run -p 8000:8000 ghcr.io/meta-pytorch/openenv-pacman-env:latest
```

### CUDA/GPU Issues

```bash
# Verify PyTorch can see your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Use CPU-only mode if needed (slower training)
# Training script will automatically fall back to CPU
```

## Environment Variables

Configure the Pac-Man environment behavior:

```bash
export PACMAN_DIFFICULTY=3              # 1-5 (default: 1)
export PACMAN_GHOST_AI=heuristic        # random | heuristic (default: random)
export PACMAN_MAZE_SIZE=15x15           # Grid dimensions (default: 15x15)
export PACMAN_MAX_STEPS=600             # Episode length (default: 600)
export PACMAN_NUM_GHOSTS=4              # Number of ghosts (default: 4)
```

## Next Steps

- Read the [README.md](README.md) for project overview
- Check out [notebooks/GPT-OSS-Play-MsPacMan.ipynb](notebooks/GPT-OSS-Play-MsPacMan.ipynb) for examples
- Start training: `python train_pacman_docker_grpo_v2.py`
