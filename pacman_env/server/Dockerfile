FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn requests

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Environment variables with defaults
ENV PACMAN_DIFFICULTY=1
ENV PACMAN_GHOST_AI=random
ENV PACMAN_MAZE_SIZE=15x15
ENV PACMAN_MAX_STEPS=250
ENV PACMAN_NUM_GHOSTS=4

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "uvicorn", "envs.pacman_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
