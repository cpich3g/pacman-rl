#!/usr/bin/env python3
"""
Pac-Man AI Player - Standalone HTML Server

A lightweight Flask-based web server that provides a single-page application
for watching trained AI agents play Pac-Man. No Streamlit dependency required.

Features:
- Real-time game visualization via WebSocket
- Interactive controls (play, pause, step)
- Score tracking and statistics
- Configurable game settings
- Works with trained GRPO models or baseline policy
"""

import sys
import json
import time
import copy
import re
import random
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque

try:
    from flask import Flask, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-socketio")
    sys.exit(1)

# Setup Python path for OpenEnv
def setup_openenv_path():
    cwd = Path.cwd()
    project_root = cwd
    for _ in range(4):
        if (project_root / "src").exists():
            break
        project_root = project_root.parent
    else:
        project_root = Path(__file__).parent.parent
        if not (project_root / "src").exists():
            raise FileNotFoundError("Could not locate OpenEnv src directory")
    
    openenv_src = project_root / "src"
    if str(openenv_src) not in sys.path:
        sys.path.insert(0, str(openenv_src))
    
    return project_root

PROJECT_ROOT = setup_openenv_path()

from envs.pacman_env import PacManEnv, PacManAction, PacManObservation
from envs.pacman_env.renderer import render_maze

# OpenAI API client for vLLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI client not available. Install with: pip install openai")
    print("   Will use baseline policy only.")


# Constants
DOCKER_IMAGE = "ghcr.io/meta-pytorch/openenv-pacman-env:latest"
VLLM_API_BASE = "http://localhost:9810/v1"  # vLLM server endpoint

ACTIONS = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "STAY": 4,
}
ACTION_NAMES = {v: k for k, v in ACTIONS.items()}

# Global game state
game_state = {
    'env': None,
    'vllm_client': None,  # OpenAI client for vLLM
    'model_name': None,   # Model name from vLLM
    'history': deque(maxlen=2000),
    'current_step': 0,
    'is_playing': False,
    'is_game_over': False,
    'replay_mode': False,  # True = post-game replay, False = live streaming
    'replay_history': [],  # Full game history for replay
    'replay_index': 0,     # Current replay position
    'config': {
        'max_steps': 500,
        'difficulty': 2,
        'ghost_ai': 'heuristic',
        'maze_size': '15x15',
        'num_ghosts': 4,
        'use_model': False,
        'play_speed': 5,
        'mode': 'live',  # 'live' or 'replay'
    }
}

game_lock = threading.Lock()


# ============================================================================
# Model and Policy Functions
# ============================================================================

PROMPT_TEMPLATE = """
Write a compact Ms. Pac-Man strategy function that returns actions based on game state.

Your function gets 'obs' with:
- pacman_position: (row, col)
- ghost_positions: [(row, col), ...]
- ghost_states: ["normal"|"frightened", ...]
- legal_actions: [0,1,2,3,4] (UP,RIGHT,DOWN,LEFT,STAY)
- pellets_remaining, power_pellets_remaining, lives, score: int
- frightened_timer: int (ghosts vulnerable)
- nearest_pellet_distance, nearest_ghost_distance: int
- safe_directions: ["up","down","left","right"]

Return a list of 0-4 integer actions to try in order.
""".strip()


def observation_to_text(obs: PacManObservation) -> str:
    ghost_lines = []
    for idx, (pos, state) in enumerate(zip(obs.ghost_positions, obs.ghost_states)):
        ghost_lines.append(f"  - Ghost {idx + 1}: pos={pos}, state={state}")

    safe = ", ".join(obs.safe_directions) if obs.safe_directions else "None"
    legal = ", ".join(str(a) for a in obs.legal_actions)

    return (
        f"Pac-Man position: {obs.pacman_position}\n"
        f"Lives: {obs.lives}, Score: {obs.score}\n"
        f"Pellets remaining: {obs.pellets_remaining}, Power pellets: {obs.power_pellets_remaining}\n"
        f"Nearest pellet distance: {obs.nearest_pellet_distance:.1f}\n"
        f"Nearest ghost distance: {obs.nearest_ghost_distance:.1f}\n"
        f"Frightened timer: {obs.frightened_timer}\n"
        f"Safe directions: {safe}\n"
        f"Legal actions: [{legal}]\n"
        f"Ghosts:\n" + "\n".join(ghost_lines)
    )


def observation_to_chat(obs: PacManObservation) -> List[dict]:
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": observation_to_text(obs)},
    ]


def extract_function(text: str) -> Optional[str]:
    """Extract Python function from LLM completion."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n").removeprefix("python").strip()
        if "def strategy" in fx:
            return fx
    
    if "def strategy" in text:
        fx = text[text.find("def strategy"):]
        lines = fx.split("\n")
        kept = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace() and not line.startswith("def"):
                break
            kept.append(line)
        fx = "\n".join(kept)
        if not any("return" in line for line in kept):
            fx += "\n    return [0, 1, 2, 3]  # fallback"
        return fx
    
    return None


def baseline_policy(obs: PacManObservation) -> int:
    """Simple baseline policy."""
    legal = list(obs.legal_actions)
    if not legal:
        return ACTIONS["STAY"]

    if obs.nearest_ghost_distance <= 2 and obs.safe_directions:
        direction_map = {"up": 0, "right": 1, "down": 2, "left": 3}
        safe_actions = [direction_map[d] for d in obs.safe_directions if d in direction_map]
        safe_actions = [a for a in safe_actions if a in legal]
        if safe_actions:
            return random.choice(safe_actions)

    if obs.frightened_timer > 0:
        attack = [a for a in legal if a != ACTIONS["STAY"]]
        if attack:
            return random.choice(attack)

    if ACTIONS["STAY"] in legal and len(legal) > 1:
        legal = [a for a in legal if a != ACTIONS["STAY"]]

    return random.choice(legal)


def model_policy_vllm(obs: PacManObservation) -> int:
    """Get action from vLLM server via OpenAI API."""
    if game_state['vllm_client'] is None:
        return baseline_policy(obs)
    
    try:
        messages = observation_to_chat(obs)
        
        # Call vLLM server using OpenAI API
        response = game_state['vllm_client'].chat.completions.create(
            model=game_state['model_name'] or "merged_model",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
        )
        
        generated = response.choices[0].message.content
        strategy_code = extract_function(generated)
        
        if strategy_code:
            local_ns = {}
            exec(strategy_code, {"__builtins__": {}}, local_ns)
            
            if "strategy" in local_ns:
                strategy_func = local_ns["strategy"]
                result = strategy_func(obs)
                
                if isinstance(result, list) and result:
                    for action in result:
                        if action in obs.legal_actions:
                            return action
                elif isinstance(result, int) and result in obs.legal_actions:
                    return result
        
        return baseline_policy(obs)
        
    except Exception as e:
        print(f"vLLM inference error: {e}")
        return baseline_policy(obs)


def init_vllm_client(api_base: str = VLLM_API_BASE, api_key: str = "EMPTY"):
    """Initialize OpenAI client for vLLM server."""
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI library not available")
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        
        # Test connection by listing models
        models = client.models.list()
        model_names = [m.id for m in models.data]
        
        if model_names:
            print(f"✅ Connected to vLLM server. Available models: {model_names}")
            game_state['model_name'] = model_names[0]
            return client
        else:
            print("⚠️  vLLM server has no models loaded")
            return None
            
    except Exception as e:
        print(f"❌ Failed to connect to vLLM server: {e}")
        print(f"   Make sure vLLM server is running at {api_base}")
        return None
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_local_checkpoints():
    """Find checkpoint directories in common locations."""
    search_dirs = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/root/rl-challenge/OpenEnv/examples"),
    ]
    
    checkpoints = []
    
    # First, check for merged_model (default)
    for search_dir in search_dirs:
        merged_path = search_dir / "merged_model"
        if merged_path.exists() and merged_path.is_dir():
            has_model = any(
                merged_path.glob(pattern) 
                for pattern in ["*.bin", "*.safetensors", "model*.safetensors"]
            )
            if has_model:
                checkpoints.append(str(merged_path))
                break  # Only add once
    
    # Then find other checkpoints
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for outputs_dir in search_dir.glob("outputs_*"):
            if outputs_dir.is_dir():
                for checkpoint_dir in outputs_dir.glob("checkpoint-*"):
                    if checkpoint_dir.is_dir():
                        has_model = any(
                            checkpoint_dir.glob(pattern) 
                            for pattern in ["*.bin", "*.safetensors", "adapter_*.bin"]
                        )
                        if has_model:
                            checkpoints.append(str(checkpoint_dir))
                
                has_model = any(
                    outputs_dir.glob(pattern) 
                    for pattern in ["*.bin", "*.safetensors", "adapter_*.bin"]
                )
                if has_model:
                    checkpoints.append(str(outputs_dir))
    
    return sorted(set(checkpoints), key=lambda x: (0 if "merged_model" in x else 1, x))


# ============================================================================
# Game Management
# ============================================================================

def serialize_observation(obs: PacManObservation) -> Dict[str, Any]:
    """Convert observation to JSON-serializable dict."""
    return {
        'maze_layout': obs.maze_layout,
        'pacman_position': obs.pacman_position,
        'ghost_positions': obs.ghost_positions,
        'ghost_states': obs.ghost_states,
        'pellets_remaining': obs.pellets_remaining,
        'power_pellets_remaining': obs.power_pellets_remaining,
        'score': obs.score,
        'lives': obs.lives,
        'frightened_timer': obs.frightened_timer,
        'nearest_pellet_distance': obs.nearest_pellet_distance,
        'nearest_ghost_distance': obs.nearest_ghost_distance,
        'safe_directions': obs.safe_directions,
        'legal_actions': obs.legal_actions,
        'level_complete': obs.level_complete,
    }


def initialize_environment(config: Dict[str, Any]) -> bool:
    """Initialize the Pac-Man environment."""
    global game_state
    
    try:
        if game_state['env'] is not None:
            try:
                game_state['env'].close()
            except:
                pass
        
        env = PacManEnv.from_docker_image(
            DOCKER_IMAGE,
            timeout=6000,
            env_vars={
                "PACMAN_DIFFICULTY": str(config['difficulty']),
                "PACMAN_GHOST_AI": config['ghost_ai'],
                "PACMAN_MAZE_SIZE": config['maze_size'],
                "PACMAN_MAX_STEPS": str(config['max_steps']),
                "PACMAN_NUM_GHOSTS": str(config['num_ghosts']),
            },
        )
        
        game_state['env'] = env
        
        reset_result = env.reset()
        game_state['history'].clear()
        game_state['history'].append({
            'observation': serialize_observation(reset_result.observation),
            'reward': 0,
            'action': None,
        })
        game_state['current_step'] = 0
        game_state['is_playing'] = False
        game_state['is_game_over'] = False
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return False


def take_step(socketio) -> bool:
    """Take one game step and emit update."""
    global game_state
    
    with game_lock:
        if game_state['env'] is None or game_state['is_game_over']:
            return False
        
        # Get current observation (need to reconstruct from serialized data)
        last_entry = game_state['history'][-1]
        obs_data = last_entry['observation']
        
        # Create observation object
        obs = PacManObservation(
            maze_layout=obs_data['maze_layout'],
            pacman_position=tuple(obs_data['pacman_position']),
            ghost_positions=[tuple(p) for p in obs_data['ghost_positions']],
            ghost_states=obs_data['ghost_states'],
            pellets_remaining=obs_data['pellets_remaining'],
            power_pellets_remaining=obs_data['power_pellets_remaining'],
            score=obs_data['score'],
            lives=obs_data['lives'],
            frightened_timer=obs_data['frightened_timer'],
            nearest_pellet_distance=obs_data['nearest_pellet_distance'],
            nearest_ghost_distance=obs_data['nearest_ghost_distance'],
            safe_directions=obs_data['safe_directions'],
            legal_actions=obs_data['legal_actions'],
            level_complete=obs_data['level_complete'],
        )
        
        # Get action from policy
        if game_state['vllm_client'] is not None:
            action = model_policy_vllm(obs)
        else:
            action = baseline_policy(obs)
        
        # Take step
        step_result = game_state['env'].step(PacManAction(action_id=action))
        
        # Store in history
        game_state['history'].append({
            'observation': serialize_observation(step_result.observation),
            'reward': step_result.reward,
            'action': action,
        })
        game_state['current_step'] += 1
        
        # Check if game over
        if step_result.done or game_state['current_step'] >= game_state['config']['max_steps']:
            game_state['is_game_over'] = True
            game_state['is_playing'] = False
        
        # Emit update
        print(f"📡 Emitting game_update: step={game_state['current_step']}, reward={step_result.reward:.2f}, done={game_state['is_game_over']}")
        socketio.emit('game_update', {
            'step': game_state['current_step'],
            'observation': serialize_observation(step_result.observation),
            'action': action,
            'action_name': ACTION_NAMES.get(action, 'UNKNOWN'),
            'reward': step_result.reward,
            'is_game_over': game_state['is_game_over'],
            'total_reward': sum(h['reward'] for h in game_state['history']),
        })
        
        return not game_state['is_game_over']


def play_episode_thread(socketio):
    """Background thread to play episode - mode depends on config."""
    global game_state
    
    if game_state['config']['mode'] == 'replay':
        # REPLAY MODE: Play entire game silently, then replay
        play_and_record(socketio)
    else:
        # LIVE MODE: Stream each step in real-time
        play_live(socketio)


def play_live(socketio):
    """Live streaming mode - send each step as it happens."""
    global game_state
    
    print(f"🎮 Starting live play mode, speed: {game_state['config']['play_speed']} steps/sec")
    
    while game_state['is_playing']:
        if not take_step(socketio):
            print("Game ended or step failed")
            break
        
        sleep_time = 1.0 / game_state['config']['play_speed']
        socketio.sleep(sleep_time)
    
    print("🏁 Play session ended")


def play_and_record(socketio):
    """Replay mode - play entire game silently, then replay it."""
    global game_state
    
    # Clear replay history
    game_state['replay_history'] = []
    game_state['replay_index'] = 0
    
    # Notify that we're computing
    socketio.emit('game_status', {
        'status': 'computing',
        'message': 'Agent is playing the game... please wait'
    })
    
    # Play entire game and record all steps
    step_count = 0
    max_steps = game_state['config']['max_steps']
    
    while step_count < max_steps and not game_state['is_game_over']:
        # Take step without emitting
        step_data = take_step_silent()
        if step_data is None:
            break
        
        game_state['replay_history'].append(step_data)
        step_count += 1
    
    # Notify that computation is done
    total_steps = len(game_state['replay_history'])
    total_reward = sum(s['reward'] for s in game_state['replay_history'])
    
    socketio.emit('game_status', {
        'status': 'replay_ready',
        'message': f'Game complete! {total_steps} steps, Total Reward: {total_reward:.1f}. Click Play to watch replay.',
        'total_steps': total_steps,
        'total_reward': total_reward
    })
    
    # Now replay it step by step
    game_state['replay_mode'] = True
    replay_game(socketio)


def replay_game(socketio):
    """Replay recorded game step by step."""
    global game_state
    
    game_state['replay_index'] = 0
    
    while game_state['is_playing'] and game_state['replay_index'] < len(game_state['replay_history']):
        step_data = game_state['replay_history'][game_state['replay_index']]
        
        # Emit this step
        socketio.emit('game_update', step_data)
        
        game_state['replay_index'] += 1
        
        # Respect play speed
        sleep_time = 1.0 / game_state['config']['play_speed']
        socketio.sleep(sleep_time)
    
    # Replay finished
    if game_state['replay_index'] >= len(game_state['replay_history']):
        socketio.emit('game_status', {
            'status': 'replay_complete',
            'message': 'Replay finished! Start a new game or adjust speed and replay again.'
        })
        game_state['is_playing'] = False


def take_step_silent() -> Optional[Dict]:
    """Take one game step WITHOUT emitting - for replay mode recording."""
    global game_state
    
    with game_lock:
        if game_state['env'] is None or game_state['is_game_over']:
            return None
        
        # Get current observation
        last_entry = game_state['history'][-1]
        obs_data = last_entry['observation']
        
        # Create observation object
        obs = PacManObservation(
            maze_layout=obs_data['maze_layout'],
            pacman_position=tuple(obs_data['pacman_position']),
            ghost_positions=[tuple(p) for p in obs_data['ghost_positions']],
            ghost_states=obs_data['ghost_states'],
            pellets_remaining=obs_data['pellets_remaining'],
            power_pellets_remaining=obs_data['power_pellets_remaining'],
            score=obs_data['score'],
            lives=obs_data['lives'],
            frightened_timer=obs_data['frightened_timer'],
            nearest_pellet_distance=obs_data['nearest_pellet_distance'],
            nearest_ghost_distance=obs_data['nearest_ghost_distance'],
            safe_directions=obs_data['safe_directions'],
            legal_actions=obs_data['legal_actions'],
            level_complete=obs_data['level_complete'],
        )
        
        # Get action from policy
        if game_state['vllm_client'] is not None:
            action = model_policy_vllm(obs)
        else:
            action = baseline_policy(obs)
        
        # Take step
        step_result = game_state['env'].step(PacManAction(action_id=action))
        
        # Store in history
        game_state['history'].append({
            'observation': serialize_observation(step_result.observation),
            'reward': step_result.reward,
            'action': action,
        })
        game_state['current_step'] += 1
        
        # Check if game over
        if step_result.done or game_state['current_step'] >= game_state['config']['max_steps']:
            game_state['is_game_over'] = True
        
        # Return step data for recording
        return {
            'step': game_state['current_step'],
            'observation': serialize_observation(step_result.observation),
            'action': action,
            'action_name': ACTION_NAMES.get(action, 'UNKNOWN'),
            'reward': step_result.reward,
            'is_game_over': game_state['is_game_over'],
            'total_reward': sum(h['reward'] for h in game_state['history']),
        }


# ============================================================================
# Flask App and Routes
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pacman-ai-player-secret'
socketio = SocketIO(app, cors_allowed_origins="*")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎮 Pac-Man AI Player</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 3em;
            text-shadow: 0 0 20px #FFD700;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.8;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .game-panel, .stats-panel, .controls-panel, .config-panel {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .panel-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #FFD700;
            border-bottom: 2px solid #FFD700;
            padding-bottom: 5px;
        }
        
        #game-canvas {
            background: #0a0a0a;
            border: 3px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
            min-height: 500px;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
        }
        
        .game-grid {
            display: inline-grid;
            gap: 2px;
            background: #000;
            padding: 10px;
            border-radius: 8px;
        }
        
        .cell {
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            border-radius: 3px;
            transition: all 0.2s ease;
            position: relative;
        }
        
        .cell.pacman {
            background: radial-gradient(circle, #FFD700 0%, #FFA500 100%);
            color: #000;
            animation: pulse 0.8s infinite;
            box-shadow: 0 0 10px #FFD700;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .cell.ghost-normal {
            background: radial-gradient(circle, #FF4444 0%, #CC0000 100%);
            color: #FFF;
            animation: float 1.5s ease-in-out infinite;
            box-shadow: 0 0 8px #FF0000;
        }
        
        .cell.ghost-frightened {
            background: radial-gradient(circle, #4444FF 0%, #0000CC 100%);
            color: #FFF;
            animation: shake 0.5s infinite;
            box-shadow: 0 0 8px #0000FF;
        }
        
        .cell.ghost-eaten {
            background: radial-gradient(circle, #44FFFF 0%, #00CCCC 100%);
            color: #000;
            box-shadow: 0 0 8px #00FFFF;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-2px); }
            75% { transform: translateX(2px); }
        }
        
        .cell.wall {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
        }
        
        .cell.pellet {
            background: transparent;
        }
        
        .cell.pellet::after {
            content: '';
            width: 6px;
            height: 6px;
            background: #FFF;
            border-radius: 50%;
            display: block;
            box-shadow: 0 0 4px #FFF;
        }
        
        .cell.power {
            background: transparent;
        }
        
        .cell.power::after {
            content: '';
            width: 12px;
            height: 12px;
            background: radial-gradient(circle, #FF00FF 0%, #CC00CC 100%);
            border-radius: 50%;
            display: block;
            animation: glow 1s infinite;
            box-shadow: 0 0 8px #FF00FF;
        }
        
        @keyframes glow {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.2); }
        }
        
        .cell.empty {
            background: transparent;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #FFD700;
        }
        
        .controls-panel {
            grid-column: 1 / -1;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        button {
            flex: 1;
            min-width: 120px;
            padding: 12px 24px;
            font-size: 16px;
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .config-panel {
            grid-column: 1 / -1;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .config-item {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        input, select {
            padding: 8px;
            font-family: 'Courier New', monospace;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            color: white;
            font-size: 14px;
        }
        
        input[type="range"] {
            cursor: pointer;
        }
        
        .progress-bar {
            width: 100%;
            height: 25px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00FF00, #FFD700);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
        }
        
        .legend {
            margin-top: 15px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .legend-item {
            display: inline-block;
            margin-right: 15px;
            margin-bottom: 5px;
        }
        
        .status-message {
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .status-success {
            background: rgba(0, 255, 0, 0.2);
            border: 1px solid #0F0;
        }
        
        .status-error {
            background: rgba(255, 0, 0, 0.2);
            border: 1px solid #F00;
        }
        
        .status-info {
            background: rgba(0, 100, 255, 0.2);
            border: 1px solid #0AF;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 PAC-MAN AI PLAYER</h1>
            <div class="subtitle">Watch trained reinforcement learning agents play Pac-Man in real-time</div>
        </header>
        
        <div class="main-content">
            <div class="game-panel">
                <div class="panel-title">🕹️ Game View</div>
                <div id="game-canvas">
                    Click "New Game" to start playing!
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-bar" style="width: 0%;">
                        <span id="progress-text">0 / 0</span>
                    </div>
                </div>
                <div class="legend">
                    <div class="legend-item"><span class="pacman">P</span> = Pac-Man</div>
                    <div class="legend-item"><span class="ghost-normal">G</span> = Ghost</div>
                    <div class="legend-item"><span class="ghost-frightened">F</span> = Frightened</div>
                    <div class="legend-item"><span class="pellet">·</span> = Pellet</div>
                    <div class="legend-item"><span class="power">○</span> = Power</div>
                    <div class="legend-item"><span class="wall">█</span> = Wall</div>
                </div>
            </div>
            
            <div class="stats-panel">
                <div class="panel-title">📊 Statistics</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Reward</div>
                        <div class="stat-value" id="stat-reward">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Score</div>
                        <div class="stat-value" id="stat-score">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Lives</div>
                        <div class="stat-value" id="stat-lives">3</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Step</div>
                        <div class="stat-value" id="stat-step">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Pellets</div>
                        <div class="stat-value" id="stat-pellets">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Last Action</div>
                        <div class="stat-value" id="stat-action">-</div>
                    </div>
                </div>
                <div id="status-container"></div>
            </div>
        </div>
        
        <div class="controls-panel">
            <div class="panel-title">🎛️ Controls</div>
            <div class="button-group">
                <button onclick="newGame()">� Start Game & Play</button>
                <button onclick="playEpisode()" id="btn-play">▶️ Resume/Continue</button>
                <button onclick="stepGame()" id="btn-step">⏭️ Single Step</button>
                <button onclick="pauseGame()">⏸️ Pause</button>
            </div>
        </div>
        
        <div class="config-panel">
            <div class="panel-title">⚙️ Configuration</div>
            <div class="config-grid">
                <div class="config-item">
                    <label for="max-steps">Max Steps:</label>
                    <input type="number" id="max-steps" value="500" min="100" max="2000" step="50">
                </div>
                <div class="config-item">
                    <label for="difficulty">Difficulty (1-5):</label>
                    <select id="difficulty">
                        <option value="1">1 - Easy</option>
                        <option value="2" selected>2 - Normal</option>
                        <option value="3">3 - Hard</option>
                        <option value="4">4 - Expert</option>
                        <option value="5">5 - Insane</option>
                    </select>
                </div>
                <div class="config-item">
                    <label for="ghost-ai">Ghost AI:</label>
                    <select id="ghost-ai">
                        <option value="random">Random</option>
                        <option value="heuristic" selected>Heuristic</option>
                    </select>
                </div>
                <div class="config-item">
                    <label for="maze-size">Maze Size:</label>
                    <select id="maze-size">
                        <option value="10x10">10x10</option>
                        <option value="15x15" selected>15x15</option>
                        <option value="20x20">20x20</option>
                    </select>
                </div>
                <div class="config-item">
                    <label for="num-ghosts">Number of Ghosts:</label>
                    <input type="number" id="num-ghosts" value="4" min="1" max="6">
                </div>
                <div class="config-item">
                    <label for="play-speed">Play Speed (steps/sec): <span id="speed-value">5</span></label>
                    <input type="range" id="play-speed" value="5" min="1" max="20" 
                           oninput="document.getElementById('speed-value').textContent = this.value">
                </div>
                <div class="config-item">
                    <label for="play-mode">Play Mode:</label>
                    <select id="play-mode" onchange="updateModeDescription()">
                        <option value="live">🔴 Live Streaming (real-time steps)</option>
                        <option value="replay">⏺️ Replay Mode (play full game, then replay)</option>
                    </select>
                    <small id="mode-description" style="opacity: 0.7; display: block; margin-top: 5px;">
                        Real-time: Watch each step as the agent plays
                    </small>
                </div>
            </div>
            <div class="config-item" style="margin-top: 10px; padding: 10px; background: rgba(0,255,0,0.1); border-radius: 5px;">
                <small style="opacity: 0.8;">
                    🤖 Using AI Model API at http://localhost:9810
                </small>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let isPlaying = false;
        let currentConfig = {};
        
        socket.on('connect', () => {
            console.log('Connected to server');
            showStatus('Connected to server', 'info');
        });
        
        socket.on('game_update', (data) => {
            console.log('🎮 Received game_update:', {
                step: data.step,
                action: data.action_name,
                reward: data.reward,
                total_reward: data.total_reward,
                is_game_over: data.is_game_over
            });
            console.log('📊 Observation data:', {
                pacman_pos: data.observation.pacman_position,
                ghost_count: data.observation.ghost_positions.length,
                maze_size: [data.observation.maze_layout.length, data.observation.maze_layout[0]?.length]
            });
            
            renderGame(data.observation);
            updateStats(data);
            
            if (data.is_game_over) {
                isPlaying = false;
                if (data.observation.level_complete || data.observation.pellets_remaining === 0) {
                    showStatus('🎉 LEVEL COMPLETE!', 'success');
                } else if (data.observation.lives === 0) {
                    showStatus('💀 GAME OVER', 'error');
                } else {
                    showStatus('🏁 Episode Complete', 'info');
                }
            }
        });
        
        socket.on('game_status', (data) => {
            console.log('Game status:', data.status);
            
            if (data.status === 'computing') {
                showStatus('🎮 ' + data.message, 'info');
            } else if (data.status === 'replay_ready') {
                showStatus('✅ ' + data.message, 'success');
            } else if (data.status === 'replay_complete') {
                showStatus('🏁 ' + data.message, 'info');
                isPlaying = false;
            }
        });
        
        function getConfig() {
            return {
                max_steps: parseInt(document.getElementById('max-steps').value),
                difficulty: parseInt(document.getElementById('difficulty').value),
                ghost_ai: document.getElementById('ghost-ai').value,
                maze_size: document.getElementById('maze-size').value,
                num_ghosts: parseInt(document.getElementById('num-ghosts').value),
                play_speed: parseInt(document.getElementById('play-speed').value),
                use_model: true,  // Always use model API
                mode: document.getElementById('play-mode').value,
            };
        }
        
        function updateModeDescription() {
            const mode = document.getElementById('play-mode').value;
            const description = document.getElementById('mode-description');
            
            if (mode === 'live') {
                description.textContent = 'Real-time: Watch each step as the agent plays';
            } else {
                description.textContent = 'Replay: Agent plays full game, then you watch the replay';
            }
        }
        
        function newGame() {
            currentConfig = getConfig();
            showStatus('Initializing game with AI model...', 'info');
            initializeGame(currentConfig);
        }
        
        function initializeGame(config) {
            fetch('/api/new_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('✅ Game initialized!', 'success');
                    renderGame(data.observation);
                    updateStats({
                        step: 0,
                        observation: data.observation,
                        action_name: '-',
                        total_reward: 0,
                    });
                    
                    // Game ready - don't auto-start
                    showStatus('✅ Game ready! Click "Play" or "Single Step" to begin', 'success');
                } else {
                    showStatus('❌ Failed to start game: ' + data.error, 'error');
                }
            })
            .catch(err => {
                showStatus('❌ Error: ' + err.message, 'error');
            });
        }
        
        function playEpisode() {
            if (isPlaying) {
                showStatus('⚠️ Already playing! Click Pause to stop.', 'warning');
                return;
            }
            isPlaying = true;
            
            fetch('/api/play', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const speed = currentConfig.play_speed || 5;
                    showStatus('▶️ Auto-playing at ' + speed + ' steps/sec', 'info');
                } else {
                    showStatus('❌ ' + data.error, 'error');
                    isPlaying = false;
                }
            });
        }
        
        function stepGame() {
            // Pause auto-play when stepping manually
            if (isPlaying) {
                isPlaying = false;
                fetch('/api/pause', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
            }
            
            fetch('/api/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('⏭️ Step (manual mode)', 'info');
                } else {
                    showStatus('❌ ' + data.error, 'error');
                }
            });
        }
        
        function pauseGame() {
            isPlaying = false;
            
            fetch('/api/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                showStatus('⏸️ Paused', 'info');
            });
        }
        
        function resetView() {
            // Could implement step navigation here
            showStatus('Reset view', 'info');
        }
        
        function renderGame(obs) {
            console.log('🎨 renderGame called with obs:', obs);
            const canvas = document.getElementById('game-canvas');
            const maze = obs.maze_layout;
            const rows = maze.length;
            const cols = rows > 0 ? maze[0].length : 0;
            console.log('🎨 Rendering maze:', rows, 'x', cols);
            
            // Create grid container if it doesn't exist
            let grid = canvas.querySelector('.game-grid');
            if (!grid || grid.style.gridTemplateColumns !== 'repeat(' + cols + ', 24px)') {
                canvas.innerHTML = '';
                grid = document.createElement('div');
                grid.className = 'game-grid';
                grid.style.gridTemplateColumns = 'repeat(' + cols + ', 24px)';
                canvas.appendChild(grid);
            }
            
            // Build the grid
            let html = '';
            
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const pos = [i, j];
                    const cell = maze[i][j];
                    
                    // Check if Pac-Man is here
                    if (pos[0] === obs.pacman_position[0] && pos[1] === obs.pacman_position[1]) {
                        html += '<div class="cell pacman" title="Pac-Man">◗</div>';
                        continue;
                    }
                    
                    // Check if a ghost is here
                    let isGhost = false;
                    for (let g = 0; g < obs.ghost_positions.length; g++) {
                        if (pos[0] === obs.ghost_positions[g][0] && pos[1] === obs.ghost_positions[g][1]) {
                            const state = obs.ghost_states[g];
                            if (state === 'frightened') {
                                html += '<div class="cell ghost-frightened" title="Frightened Ghost">⚆</div>';
                            } else if (state === 'eaten') {
                                html += '<div class="cell ghost-eaten" title="Eaten Ghost">◎</div>';
                            } else {
                                html += '<div class="cell ghost-normal" title="Ghost">⚆</div>';
                            }
                            isGhost = true;
                            break;
                        }
                    }
                    
                    if (!isGhost) {
                        if (cell === 'W') {
                            html += '<div class="cell wall" title="Wall"></div>';
                        } else if (cell === '.') {
                            html += '<div class="cell pellet" title="Pellet"></div>';
                        } else if (cell === 'O') {
                            html += '<div class="cell power" title="Power Pellet"></div>';
                        } else {
                            html += '<div class="cell empty"></div>';
                        }
                    }
                }
            }
            
            grid.innerHTML = html;
        }
        
        function updateStats(data) {
            document.getElementById('stat-reward').textContent = data.total_reward.toFixed(1);
            document.getElementById('stat-score').textContent = data.observation.score;
            document.getElementById('stat-lives').textContent = data.observation.lives;
            document.getElementById('stat-step').textContent = data.step;
            document.getElementById('stat-pellets').textContent = data.observation.pellets_remaining;
            document.getElementById('stat-action').textContent = data.action_name || '-';
            
            const maxSteps = currentConfig.max_steps || 500;
            const progress = (data.step / maxSteps) * 100;
            document.getElementById('progress-bar').style.width = progress + '%';
            document.getElementById('progress-text').textContent = data.step + ' / ' + maxSteps;
        }
        
        function showStatus(message, type) {
            const container = document.getElementById('status-container');
            container.innerHTML = '<div class="status-message status-' + type + '">' + message + '</div>';
            
            setTimeout(() => {
                container.innerHTML = '';
            }, 5000);
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """
    Switch model mode or signal to use baseline.
    
    Note: Model loading must happen on the main thread. The default model
    is preloaded at server startup. This endpoint just toggles between
    using the model or baseline policy.
    """
    config = request.json
    use_model = config.get('use_model', False)
    model_path = config.get('model_path', '')
    
    if not use_model or not model_path:
        # Use baseline policy
        print("Switching to baseline policy")
        return jsonify({
            'success': True, 
            'message': 'Using baseline policy',
            'mode': 'baseline'
        })
    
    # Check if we have a preloaded model
    if game_state['model'] is not None:
        print(f"Using preloaded model for gameplay")
        return jsonify({
            'success': True,
            'message': 'Using preloaded trained model',
            'mode': 'model'
        })
    else:
        # No model available
        return jsonify({
            'success': False,
            'error': 'Model not available. Server needs to be restarted with model preloading.',
            'message': 'Please restart server or use baseline policy'
        })


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    config = request.json
    game_state['config'].update(config)
    
    # Model should already be loaded via /api/load_model
    # Just initialize the environment
    success = initialize_environment(game_state['config'])
    
    if success:
        return jsonify({
            'success': True,
            'observation': game_state['history'][-1]['observation'],
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to initialize environment'})


@app.route('/api/step', methods=['POST'])
def api_step():
    if game_state['env'] is None:
        return jsonify({'success': False, 'error': 'No active game'})
    
    take_step(socketio)
    return jsonify({'success': True})


@app.route('/api/play', methods=['POST'])
def api_play():
    global game_state
    
    if game_state['env'] is None:
        return jsonify({'success': False, 'error': 'No active game'})
    
    if game_state['is_playing']:
        return jsonify({'success': False, 'error': 'Already playing'})
    
    game_state['is_playing'] = True
    
    # Start background task using SocketIO's method (proper app context)
    socketio.start_background_task(play_episode_thread, socketio)
    
    return jsonify({'success': True})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    game_state['is_playing'] = False
    return jsonify({'success': True})


@app.route('/api/list_checkpoints', methods=['GET'])
def api_list_checkpoints():
    """Return list of available checkpoint directories."""
    checkpoints = find_local_checkpoints()
    return jsonify({
        'success': True,
        'checkpoints': checkpoints,
        'count': len(checkpoints),
    })


# ============================================================================
# SocketIO Event Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    print(f"🔌 Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected: {request.sid}")


if __name__ == '__main__':
    print("=" * 70)
    print("🎮 PAC-MAN AI PLAYER - HTML SERVER")
    print("=" * 70)
    print(f"🌐 Starting server at http://localhost:5000")
    print(f"🌐 Model API: http://localhost:9810")
    
    # Try to connect to model server
    if OPENAI_AVAILABLE:
        print("🔄 Connecting to model server...")
        try:
            client = init_vllm_client()
            if client:
                game_state['vllm_client'] = client
                print(f"✅ Connected to model server: {game_state['model_name']}")
            else:
                print("⚠️  Model server not available - will use baseline policy")
        except Exception as e:
            print(f"⚠️  Failed to connect to model server: {e}")
            print("   Will use baseline policy")
    else:
        print("⚠️  OpenAI library not available - will use baseline policy only")
    
    print("=" * 70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
