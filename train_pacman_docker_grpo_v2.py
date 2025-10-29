#!/usr/bin/env python3
"""
Pac-Man GRPO Training Script
"""
from unsloth import (
    FastLanguageModel,
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
)
import copy
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Sequence

import torch
from datasets import Dataset
from transformers import TextStreamer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

try:
    from unsloth import PatchFastRL  # type: ignore
except ImportError:  # pragma: no cover - older unsloth versions
    def PatchFastRL():
        return None


try:
    from unsloth import is_bfloat16_supported  # type: ignore
except ImportError:  # pragma: no cover - older unsloth versions
    def is_bfloat16_supported() -> bool:
        return False


DEBUG_COMPLETION_PREVIEW_LIMIT = 1000  # Show all completions for debugging
_debug_completion_counter = 0
PROMPT_TEMPLATE = """
Write a short Pac-Man strategy function.

obs attributes: frightened_timer, nearest_ghost_distance, safe_directions, legal_actions

Return list of 1-5 actions [0,1,2,3,4] where 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY

Example:
def strategy(obs):
    if obs.frightened_timer > 0: return [1,2,0]
    if obs.nearest_ghost_distance < 3:
        if "left" in obs.safe_directions: return [3]
    return [0,1,2,3]

Write your strategy (max 8 lines):
""".strip()
# ============================================================================


class Config:
    # Model configuration
    MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
    MAX_SEQ_LENGTH = 2048
    USE_LORA = True
    LORA_RANK = 32
    LORA_ALPHA = 64
    LOAD_IN_4BIT = False

    # Environment configuration
    PACMAN_DIFFICULTY = 2
    PACMAN_GHOST_AI = "heuristic"
    PACMAN_MAZE_SIZE = "15x15"
    PACMAN_MAX_STEPS = 600
    PACMAN_NUM_GHOSTS = 4
    DOCKER_IMAGE = "ghcr.io/meta-pytorch/openenv-pacman-env:latest"

    # Training configuration
    TEMPERATURE = 1.0
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 0.5
    LR_SCHEDULER = "cosine"
    OPTIMIZER = "adamw_8bit"

    BATCH_SIZE = 4
    GRAD_ACCUMULATION = 2
    NUM_GENERATIONS = 3
    MAX_STEPS = 400
    SAVE_STEPS = 200
    LOGGING_STEPS = 1

    # Generation configuration (during GRPO rollouts)
    GEN_TEMPERATURE = 0.7
    GEN_TOP_P = 0.9
    GEN_TOP_K = 50
    GEN_REPETITION_PENALTY = 1.1  # Prevent repetitive gibberish
    MAX_NEW_TOKENS = 150
    TRAIN_COMPLETION_LIMIT = 180

    # Dataset configuration
    DATASET_SIZE = 1000
    OBS_BUFFER_FACTOR = 2  # how many raw observations to harvest vs dataset size

    # Rollout configuration
    STRATEGY_TIMEOUT = 30
    ROLLOUT_HORIZON = 12
    ROLLOUTS_PER_EVAL = 2

    # Output configuration
    OUTPUT_DIR = "outputs_pacman_docker_grpo_v2"
    REPORT_TO = "wandb"
    PRINT_STRATEGY_EVERY = 10
    NARRATIVE_MILESTONES = {
        0: "📖 Prologue - Agents enter the neon labyrinth.",
        25: "⚙️  Chapter I - Fleeing ghosts while mapping pellets.",
        75: "💡 Chapter II - Power pellets fuel counter-attacks.",
        125: "🔥 Chapter III - Long-horizon planning emerges.",
        175: "🏁 Finale - Championship scrimmages begin.",
    }
    EVALUATION_EPISODES = 5
    ENABLE_ASCII_PLAYBACK = True
    PLAYBACK_MAX_STEPS = 80
    PLAYBACK_COLORS = True
    PLAYBACK_CLEAR = False
    PLAYBACK_SLEEP = 0.0
    TEST_MAX_STEPS = 600


ACTIONS = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "STAY": 4,
}
ACTION_NAMES = {v: k for k, v in ACTIONS.items()}
ACTION_DELTAS = {
    ACTIONS["UP"]: (-1, 0),
    ACTIONS["RIGHT"]: (0, 1),
    ACTIONS["DOWN"]: (1, 0),
    ACTIONS["LEFT"]: (0, -1),
}

def setup_environment():
    """Setup Pac-Man environment."""
    cwd = Path.cwd()
    project_root = cwd
    for _ in range(4):
        if (project_root / "src").exists():
            break
        project_root = project_root.parent
    else:
        raise FileNotFoundError(
            f"Could not locate project root with src directory from {cwd}"
        )

    openenv_src = project_root / "src"
    if not openenv_src.exists():
        raise FileNotFoundError(f"src directory not found at {openenv_src}")

    if str(openenv_src) not in sys.path:
        sys.path.insert(0, str(openenv_src))

    from envs.pacman_env import PacManEnv, PacManAction, PacManObservation
    from envs.pacman_env.renderer import render_maze, render_legend

    print(f"✅ Project root: {project_root}")
    print(f"✅ OpenEnv src: {openenv_src}")
    print("🚀 Starting Pac-Man environment...")

    env = PacManEnv.from_docker_image(
        Config.DOCKER_IMAGE,
        timeout=6000,
        env_vars={
            "PACMAN_DIFFICULTY": str(Config.PACMAN_DIFFICULTY),
            "PACMAN_GHOST_AI": Config.PACMAN_GHOST_AI,
            "PACMAN_MAZE_SIZE": Config.PACMAN_MAZE_SIZE,
            "PACMAN_MAX_STEPS": str(Config.PACMAN_MAX_STEPS),
            "PACMAN_NUM_GHOSTS": str(Config.PACMAN_NUM_GHOSTS),
        },
    )

    try:
        state = env.state()
        print("🎯 Environment brief:")
        print(f"   - Game: {state.game_name}")
        print(f"   - Difficulty: {state.difficulty_level}")
        print(f"   - Ghost AI: {state.ghost_ai_type}")
        print(f"   - Maze size: {state.maze_size}")
        print(f"   - Max steps: {state.max_steps}")
        print(f"   - Ghosts: {state.num_ghosts}")

        preview = env.reset()
        if preview.observation:
            print(render_maze(preview.observation, colors=True, show_stats=True))
            print(render_legend())
    except Exception as exc:
        print(f"⚠️  Could not fetch environment state: {exc}")

    return env, PacManAction, PacManObservation, render_maze


# ============================================================================
# Model Setup
# ============================================================================


def setup_model():
    """Load and configure the language model."""
    print(f"\n📦 Loading model: {Config.MODEL_NAME}")

    os.environ.setdefault("DISABLE_COMPILE", "1")
    if hasattr(torch._dynamo.config, "disable"):
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_NAME,
        load_in_4bit=Config.LOAD_IN_4BIT,
        max_seq_length=Config.MAX_SEQ_LENGTH,
    )

    # Ensure tokenizer is properly configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation
    tokenizer.truncation_side = "left"

    if Config.USE_LORA and Config.LORA_RANK > 0:
        model = FastLanguageModel.get_peft_model(
            model,
            r=Config.LORA_RANK,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=0.05,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        lora_msg = f"LoRA rank {Config.LORA_RANK} (alpha={Config.LORA_ALPHA})"
    else:
        Config.USE_LORA = False
        lora_msg = "LoRA disabled (training full model weights)"

    setattr(model, "is_loaded_in_4bit", False)
    setattr(model, "is_loaded_in_8bit", False)

    FastLanguageModel.for_inference(model)

    print("✅ Model ready")
    print(f"   - Seq length: {Config.MAX_SEQ_LENGTH}")
    print(f"   - {lora_msg}")

    return model, tokenizer


# ============================================================================
# Observation Serialization
# ============================================================================


def observation_to_text(obs) -> str:
    ghost_lines = []
    for idx, (pos, state) in enumerate(zip(obs.ghost_positions, obs.ghost_states)):
        ghost_lines.append(f"  - Ghost {idx + 1}: pos={pos}, state={state}")

    safe = ", ".join(obs.safe_directions) if getattr(obs, "safe_directions", None) else "None"
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


def observation_to_chat(obs) -> List[dict]:
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": observation_to_text(obs)},
    ]


# ============================================================================
# Dataset Harvesting
# ============================================================================


def baseline_policy(obs) -> int:
    legal = list(obs.legal_actions)
    if not legal:
        return ACTIONS["STAY"]

    frightened = getattr(obs, "frightened_timer", 0) or 0
    safe_dirs = set((getattr(obs, "safe_directions", []) or []))
    direction_map = {"up": 0, "right": 1, "down": 2, "left": 3}

    if getattr(obs, "nearest_ghost_distance", None) is not None and obs.nearest_ghost_distance <= 1:
        safe_actions = [direction_map[d] for d in safe_dirs if d in direction_map]
        safe_actions = [a for a in safe_actions if a in legal]
        if safe_actions:
            return random.choice(safe_actions)

    if frightened > 0:
        attack = [a for a in legal if a != ACTIONS["STAY"]]
        if attack:
            return random.choice(attack)

    if ACTIONS["STAY"] in legal and len(legal) > 1:
        legal = [a for a in legal if a != ACTIONS["STAY"]]

    return random.choice(legal)


def collect_observations(env, PacManAction, count: int) -> Sequence:
    observations = []
    obs = env.reset().observation
    for _ in range(count):
        observations.append(copy.deepcopy(obs))
        action = baseline_policy(obs)
        step = env.step(PacManAction(action_id=action))
        obs = step.observation
        if step.done:
            obs = env.reset().observation
    return observations


def create_training_dataset(tokenizer, env, PacManAction):
    harvest = Config.DATASET_SIZE * Config.OBS_BUFFER_FACTOR
    samples = collect_observations(env, PacManAction, harvest)
    if len(samples) > Config.DATASET_SIZE:
        samples = random.sample(samples, Config.DATASET_SIZE)

    records = [
        {
            "prompt": observation_to_chat(obs),
            "answer": 0
        }
        for obs in samples
    ]

    dataset = Dataset.from_list(records)
    first_prompt = records[0]["prompt"]
    formatted_example = tokenizer.apply_chat_template(
        first_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    maximum_length = len(tokenizer.apply_chat_template(
        first_prompt,
        add_generation_prompt=True,
    ))

    print("\n📊 Dataset prepared:")
    print(f"   - Examples: {len(dataset)}")
    print(f"   - Prompt tokens: {maximum_length}")
    print(f"   - Max completion tokens: {Config.MAX_SEQ_LENGTH - maximum_length - 1}")
    print(f"\n   Example formatted prompt (first 600 chars):")
    print(f"   {'-'*70}")
    print(f"   {formatted_example[:600]}")
    print(f"   {'-'*70}\n")

    return dataset, maximum_length


# ============================================================================
# Strategy Execution
# ============================================================================


def analyze_plan_against_observation(plan: List[int], observation) -> dict:
    """Estimate pellet coverage and plan quality using the static maze snapshot."""
    result = {
        "plan_steps": len(plan),
        "plan_active_steps": 0,
        "plan_stay_steps": 0,
        "plan_invalid_actions": 0,
        "plan_legality_ratio": 0.0,
        "plan_active_fraction": 0.0,
        "plan_stay_fraction": 0.0,
        "plan_wall_hits": 0,
        "plan_wall_hit_ratio": 0.0,
        "plan_unique_positions": 1,
        "plan_pellet_coverage_ratio": 0.0,
        "planned_pellets": 0,
        "planned_power_pellets": 0,
        "plan_total_targets": 0,
        "plan_progress_steps": 0,
        "plan_progress_ratio": 0.0,
        "plan_progress_score": 0.0,
        "plan_distance_reduction": 0.0,
        "plan_distance_final": 0.0,
        "plan_safe_alignment_ratio": 0.0,
    }

    if not plan or observation is None:
        return result

    maze_layout = getattr(observation, "maze_layout", None)
    if not maze_layout:
        return result

    layout: List[List[str]] = []
    for row in maze_layout:
        if isinstance(row, str):
            layout.append(list(row))
        else:
            layout.append(list(row))

    if not layout or not layout[0]:
        return result

    pellet_chars = {".", "·", "*"}
    power_chars = {"O", "o"}
    pellet_positions = set()
    power_positions = set()
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            if cell in pellet_chars:
                pellet_positions.add((r, c))
            elif cell in power_chars:
                power_positions.add((r, c))

    total_targets = len(pellet_positions) + len(power_positions)
    result["plan_total_targets"] = total_targets

    position = tuple(getattr(observation, "pacman_position", (0, 0)) or (0, 0))
    rows = len(layout)
    cols = len(layout[0])

    visited_positions = {position}
    recognized_actions = 0
    active_steps = 0
    stay_steps = 0
    invalid_actions = 0
    wall_hits = 0
    pellets_eaten = 0
    power_eaten = 0
    progress_steps = 0
    cumulative_drop = 0.0
    safe_moves = 0

    direction_lookup = {
        "up": ACTIONS["UP"],
        "right": ACTIONS["RIGHT"],
        "down": ACTIONS["DOWN"],
        "left": ACTIONS["LEFT"],
    }
    safe_dirs = set((getattr(observation, "safe_directions", []) or []))
    safe_action_ids = {
        direction_lookup[d.lower()]
        for d in safe_dirs
        if isinstance(d, str) and d.lower() in direction_lookup
    }

    def nearest_distance(current_pos) -> float:
        if not pellet_positions and not power_positions:
            return 0.0
        best = None
        for target in pellet_positions:
            dist = abs(target[0] - current_pos[0]) + abs(target[1] - current_pos[1])
            best = dist if best is None or dist < best else best
        for target in power_positions:
            dist = abs(target[0] - current_pos[0]) + abs(target[1] - current_pos[1])
            best = dist if best is None or dist < best else best
        return float(best) if best is not None else 0.0

    current_distance = nearest_distance(position)
    initial_distance = current_distance

    for raw_action in plan:
        try:
            action_id = int(raw_action)
        except (TypeError, ValueError):
            invalid_actions += 1
            continue

        if action_id == ACTIONS["STAY"]:
            recognized_actions += 1
            stay_steps += 1
            new_distance = nearest_distance(position)
            if new_distance < current_distance:
                progress_steps += 1
                cumulative_drop += current_distance - new_distance
            current_distance = new_distance
            continue

        if action_id not in ACTION_DELTAS:
            invalid_actions += 1
            continue

        recognized_actions += 1
        dr, dc = ACTION_DELTAS[action_id]
        nr = position[0] + dr
        nc = position[1] + dc
        if nr < 0 or nc < 0 or nr >= rows or nc >= cols or layout[nr][nc] == "W":
            wall_hits += 1
            continue

        position = (nr, nc)
        active_steps += 1
        visited_positions.add(position)

        consumed_target = False
        if position in pellet_positions:
            pellet_positions.remove(position)
            pellets_eaten += 1
            consumed_target = True
        elif position in power_positions:
            power_positions.remove(position)
            power_eaten += 1
            consumed_target = True

        if action_id in safe_action_ids:
            safe_moves += 1

        new_distance = nearest_distance(position)
        if consumed_target or new_distance < current_distance:
            progress_steps += 1
            if new_distance < current_distance:
                cumulative_drop += current_distance - new_distance
        current_distance = new_distance

    total_plan_steps = len(plan)
    legality_ratio = (
        recognized_actions / total_plan_steps if total_plan_steps else 0.0
    )
    active_fraction = active_steps / total_plan_steps if total_plan_steps else 0.0
    stay_fraction = stay_steps / total_plan_steps if total_plan_steps else 0.0
    wall_hit_ratio = wall_hits / active_steps if active_steps else 0.0
    coverage_ratio = (
        (pellets_eaten + power_eaten) / total_targets if total_targets else 0.0
    )
    progress_ratio = progress_steps / active_steps if active_steps else 0.0
    safe_alignment_ratio = safe_moves / active_steps if active_steps else 0.0

    result.update(
        {
            "plan_active_steps": active_steps,
            "plan_stay_steps": stay_steps,
            "plan_invalid_actions": invalid_actions,
            "plan_legality_ratio": legality_ratio,
            "plan_active_fraction": active_fraction,
            "plan_stay_fraction": stay_fraction,
            "plan_wall_hits": wall_hits,
            "plan_wall_hit_ratio": wall_hit_ratio,
            "plan_unique_positions": len(visited_positions),
            "plan_pellet_coverage_ratio": coverage_ratio,
            "planned_pellets": pellets_eaten,
            "planned_power_pellets": power_eaten,
            "plan_progress_steps": progress_steps,
            "plan_progress_ratio": progress_ratio,
            "plan_progress_score": progress_ratio,
            "plan_distance_reduction": max(0.0, initial_distance - current_distance),
            "plan_distance_final": current_distance,
            "plan_safe_alignment_ratio": safe_alignment_ratio,
        }
    )

    return result


def bfs_path_to_target(start, target, maze_layout):
    """BFS pathfinding to convert coordinates to action sequence."""
    from collections import deque
    
    rows = len(maze_layout)
    cols = len(maze_layout[0]) if rows > 0 else 0
    
    if start == target:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        for action_id, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            
            if (nr, nc) == target:
                return path + [action_id]
            
            if (nr, nc) in visited:
                continue
            
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            
            if maze_layout[nr][nc] == 'W':
                continue
            
            visited.add((nr, nc))
            queue.append(((nr, nc), path + [action_id]))
    
    return []  # No path found


def bfs_path_to_target(start, target, maze_layout):
    """BFS pathfinding to convert coordinates to action sequence."""
    from collections import deque
    
    rows = len(maze_layout)
    cols = len(maze_layout[0]) if rows > 0 else 0
    
    if start == target:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        for action_id, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            
            if (nr, nc) == target:
                return path + [action_id]
            
            if (nr, nc) in visited:
                continue
            
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            
            if maze_layout[nr][nc] == 'W':
                continue
            
            visited.add((nr, nc))
            queue.append(((nr, nc), path + [action_id]))
    
    return []  # No path found


def compute_shaped_reward(prev_obs, new_obs, env_reward: float, action_id: int) -> float:
    shaped = -0.1  # small step cost to discourage stalling

    pellets_delta = max(0, prev_obs.pellets_remaining - new_obs.pellets_remaining)
    power_delta = max(0, prev_obs.power_pellets_remaining - new_obs.power_pellets_remaining)
    shaped += pellets_delta * 10.0
    shaped += power_delta * 50.0

    score_gain = max(0, new_obs.score - prev_obs.score)
    if score_gain >= 200:
        shaped += 200.0  # ghost eaten
    shaped += max(0.0, env_reward)

    if new_obs.lives < prev_obs.lives:
        shaped -= 500.0

    if new_obs.pacman_position == prev_obs.pacman_position and action_id != ACTIONS["STAY"]:
        shaped -= 2.0  # walked into a wall

    return shaped


def create_strategy_executor(env, PacManAction, render_maze):
    def _execute(strategy: Callable, initial_obs, horizon: int):
        obs = copy.deepcopy(initial_obs)
        total_shaped = 0.0
        total_env_reward = 0.0
        steps_taken = 0
        lives_start = getattr(obs, "lives", 3)
        pellets_start = getattr(obs, "pellets_remaining", 0)
        power_start = getattr(obs, "power_pellets_remaining", 0)
        metrics = {
            "pellets_eaten": 0,
            "power_pellets_eaten": 0,
            "ghosts_eaten": 0,
            "deaths": 0,
            "wall_hits": 0,
            "idle_steps": 0,
            "unsafe_moves": 0,
            "safe_moves": 0,
            "final_lives": lives_start,
            "initial_pellets": pellets_start,
            "initial_power_pellets": power_start,
            "strategy_calls": 0,
            "strategy_errors": 0,
        }

        # REACTIVE EXECUTION: Call strategy function at each timestep
        while steps_taken < horizon:
            if (
                Config.ENABLE_ASCII_PLAYBACK
                and render_maze is not None
                and steps_taken < Config.PLAYBACK_MAX_STEPS
            ):
                if Config.PLAYBACK_CLEAR:
                    print("\033[2J\033[H", end="")
                print(f"STEP {steps_taken}")
                try:
                    print(render_maze(obs, colors=Config.PLAYBACK_COLORS, show_stats=True))
                except Exception:
                    pass
                if Config.PLAYBACK_SLEEP > 0:
                    time.sleep(Config.PLAYBACK_SLEEP)

            # Call strategy function to get action(s) for current state
            action_id = None
            try:
                metrics["strategy_calls"] += 1
                plan_data = strategy(obs)
                
                # Extract first valid action from returned plan
                if isinstance(plan_data, (list, tuple)) and plan_data:
                    # Try to get first valid integer action
                    for item in plan_data:
                        try:
                            candidate = int(item)
                            if 0 <= candidate <= 4:
                                action_id = candidate
                                break
                        except:
                            continue
                elif isinstance(plan_data, (int, float)):
                    candidate = int(plan_data)
                    if 0 <= candidate <= 4:
                        action_id = candidate
                elif isinstance(plan_data, str):
                    # Try to extract digit
                    digits = [c for c in plan_data if c.isdigit()]
                    if digits:
                        action_id = int(digits[0])
            except Exception as e:
                metrics["strategy_errors"] += 1
                action_id = None
            
            # Fallback if strategy didn't return valid action
            if action_id is None or action_id not in obs.legal_actions:
                # Default to safe move or random legal action
                safe_dirs = set((getattr(obs, "safe_directions", []) or []))
                direction_map = {"up": 0, "right": 1, "down": 2, "left": 3}
                if safe_dirs:
                    for dir_name, dir_action in direction_map.items():
                        if dir_name in safe_dirs and dir_action in obs.legal_actions:
                            action_id = dir_action
                            break
                if action_id is None or action_id not in obs.legal_actions:
                    action_id = random.choice(obs.legal_actions)

            # Track safe/unsafe moves
            safe_dirs = set((getattr(obs, "safe_directions", []) or []))
            direction_map = {"up": 0, "right": 1, "down": 2, "left": 3}
            if safe_dirs:
                direction = ACTION_NAMES.get(action_id)
                if direction and direction.lower() in safe_dirs:
                    metrics["safe_moves"] += 1
                else:
                    metrics["unsafe_moves"] += 1

            prev_obs = obs
            step_result = env.step(PacManAction(action_id=action_id))
            obs = step_result.observation
            env_reward = step_result.reward or 0.0
            shaped = compute_shaped_reward(prev_obs, obs, env_reward, action_id)
            total_shaped += shaped
            total_env_reward += env_reward

            pellets_delta = max(0, prev_obs.pellets_remaining - obs.pellets_remaining)
            power_delta = max(0, prev_obs.power_pellets_remaining - obs.power_pellets_remaining)
            if env_reward >= 200:
                metrics["ghosts_eaten"] += 1
            metrics["pellets_eaten"] += pellets_delta
            metrics["power_pellets_eaten"] += power_delta

            if obs.lives < prev_obs.lives:
                metrics["deaths"] += 1

            if obs.pacman_position == prev_obs.pacman_position:
                if action_id == ACTIONS["STAY"]:
                    metrics["idle_steps"] += 1
                else:
                    metrics["wall_hits"] += 1

            metrics["final_lives"] = obs.lives
            steps_taken += 1

            if step_result.done:
                break

        metrics["total_env_reward"] = total_env_reward
        metrics["total_shaped_reward"] = total_shaped
        metrics["steps"] = steps_taken
        metrics["pellets_remaining_final"] = getattr(obs, "pellets_remaining", 0)
        metrics["power_pellets_remaining_final"] = getattr(obs, "power_pellets_remaining", 0)
        metrics["level_completed"] = bool(getattr(obs, "level_complete", False))

        if obs.lives < lives_start:
            metrics["deaths"] += lives_start - obs.lives

        return obs, total_shaped, metrics

    @execute_with_time_limit(Config.STRATEGY_TIMEOUT)
    def execute(strategy: Callable, initial_obs, horizon: int):
        return _execute(strategy, initial_obs, horizon)

    return execute


# ============================================================================
# Reward Functions
# ============================================================================


def extract_function(text: str):
    """Extract Python expression or function, wrapping single-line expressions automatically."""
    import re
    
    # Try to extract function definition from code blocks
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n").removeprefix("python").strip()
        if "def strategy" in fx:
            fx = fx[fx.find("def strategy"):]
            fx = auto_complete_function(fx)
            return fx
        # Check if it's a single expression
        if fx and not fx.startswith("def"):
            return wrap_expression_as_function(fx)
    
    # Try to find function in raw text
    if "def strategy" in text:
        fx = text[text.find("def strategy"):]
        lines = fx.split("\n")
        kept = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace():
                break
            kept.append(line)
        fx = "\n".join(kept)
        fx = auto_complete_function(fx)
        return fx
    
    # Try to extract single-line expression (NEW: primary format)
    # Look for list/ternary patterns like: [1,2] if condition else [3]
    expr_pattern = r'\[[\d,\s]+\](?:\s+if\s+.*?\s+else\s+\[[\d,\s]+\])*'
    match = re.search(expr_pattern, text)
    if match:
        expr = match.group(0).strip()
        return wrap_expression_as_function(expr)
    
    # Fallback: look for any list of integers
    list_pattern = r'\[[\d,\s]+\]'
    match = re.search(list_pattern, text)
    if match:
        expr = match.group(0).strip()
        return wrap_expression_as_function(expr)
    
    # Last resort: look for any function and rename
    func_pattern = r'def\s+\w+\s*\([^)]*\)\s*:.*?(?=\n(?:def |class |$)|$)'
    match = re.search(func_pattern, text, re.DOTALL)
    if match:
        code = match.group(0).strip()
        code = re.sub(r'def\s+\w+\s*\(', 'def strategy(', code, count=1)
        if "def strategy" in code:
            code = auto_complete_function(code)
            return code

    return None


def wrap_expression_as_function(expr: str) -> str:
    """Wrap a single expression in a valid strategy function."""
    # Clean up the expression
    expr = expr.strip()
    
    # If it's already a complete ternary or simple list, wrap it
    return f"def strategy(obs):\n    return {expr}"


def auto_complete_function(code: str) -> str:
    """Add fallback return if function is truncated."""
    lines = code.split("\n")
    if not lines:
        return code
    
    # Check if last line is a return statement
    last_line = lines[-1].strip()
    if last_line.startswith("return"):
        return code
    
    # Check if any line has a return
    has_return = any("return" in line for line in lines)
    if has_return:
        return code
    
    # No return found - add fallback
    indent = "    "
    code += f"\n{indent}return [0, 1, 2, 3]  # fallback"
    return code


def strategy_extract_or_none(completion) -> str | None:
    global _debug_completion_counter

    response_text = None

    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            if "content" in first:
                response_text = first["content"]
            elif "generated_text" in first:
                response_text = first["generated_text"]
            elif "text" in first:
                response_text = first["text"]
        elif isinstance(first, str):
            response_text = first
    elif isinstance(completion, dict):
        response_text = completion.get("content") or completion.get("generated_text") or completion.get("text")
    elif isinstance(completion, str):
        response_text = completion

    if response_text is None and _debug_completion_counter < DEBUG_COMPLETION_PREVIEW_LIMIT:
        print("[DEBUG] Unknown completion structure:", repr(completion))
        _debug_completion_counter += 1
        return None

    fx = extract_function(response_text or "")
    if fx is None and _debug_completion_counter < DEBUG_COMPLETION_PREVIEW_LIMIT:
        print(f"\n[DEBUG #{_debug_completion_counter}] EXTRACTION FAILED")
        print(f"Raw completion text:\n{response_text[:500] if response_text else '<empty>'}\n")
        _debug_completion_counter += 1
    elif fx and _debug_completion_counter < DEBUG_COMPLETION_PREVIEW_LIMIT:
        print(f"\n[DEBUG #{_debug_completion_counter}] EXTRACTION SUCCESS")
        print(f"Raw text: {response_text[:200] if response_text else '<empty>'}")
        print(f"Extracted function:\n{fx}\n")
        _debug_completion_counter += 1
    return fx


def create_reward_functions(env, execute_strategy, PacManAction):
    class Counter:
        value = 0

    def function_compiles(completions, **kwargs):
        scores = []
        for completion in completions:
            fx = strategy_extract_or_none(completion)
            if fx is None:
                scores.append(-5.0)
                continue
            try:
                compile(fx, "<strategy>", "exec")
                scores.append(2.0)
            except SyntaxError:
                scores.append(-5.0)
            except Exception:
                scores.append(-2.0)
        return scores

    def imports_ok(completions, **kwargs):
        scores = []
        for completion in completions:
            fx = strategy_extract_or_none(completion)
            if fx is None:
                scores.append(-5.0)
                continue
            ok, _ = check_python_modules(fx)
            scores.append(3.0 if ok else -10.0)
        return scores

    sample_cases = [
        (
            SimpleNamespace(
                pacman_position=(5, 5),
                ghost_positions=[(5, 6)],
                ghost_states=["normal"],
                legal_actions=[0, 3],
                pellets_remaining=12,
                power_pellets_remaining=2,
                frightened_timer=0,
                nearest_pellet_distance=4,
                nearest_ghost_distance=1,
                safe_directions=["left"],
                lives=3,
                score=0,
            ),
            {3},
        ),
        (
            SimpleNamespace(
                pacman_position=(3, 3),
                ghost_positions=[(8, 8)],
                ghost_states=["frightened"],
                legal_actions=[1, 2, 4],
                pellets_remaining=5,
                power_pellets_remaining=1,
                frightened_timer=6,
                nearest_pellet_distance=2,
                nearest_ghost_distance=5,
                safe_directions=["right", "down"],
                lives=2,
                score=150,
            ),
            {1, 2},
        ),
    ]

    def sanity_checks(completions, **kwargs):
        scores = []
        for completion in completions:
            fx = strategy_extract_or_none(completion)
            if fx is None:
                scores.append(-5.0)
                continue
            try:
                strategy = create_locked_down_function(fx)
            except Exception:
                scores.append(-2.0)
                continue

            total = 0.0
            try:
                for obs, expected in sample_cases:
                    plan = strategy(obs)
                    first_action = None
                    if isinstance(plan, dict) and "plan" in plan:
                        plan = plan["plan"]
                    if isinstance(plan, (list, tuple)) and plan:
                        first_action = plan[0]
                    elif isinstance(plan, str):
                        digits = [c for c in plan if c.isdigit()]
                        if digits:
                            first_action = int(digits[0])

                    if first_action in obs.legal_actions:
                        total += 1.0
                        if first_action in expected:
                            total += 1.0
                    else:
                        total -= 2.0
            except Exception:
                total -= 2.0
            scores.append(total)
        return scores

    def rollout_rewards(completions, **kwargs):
        scores = []
        for completion in completions:
            printed = False
            fx = strategy_extract_or_none(completion)
            if Counter.value % Config.PRINT_STRATEGY_EVERY == 0:
                printed = True
                print("\n" + "=" * 70)
                print(f"Strategy #{Counter.value}")
                print(fx if fx else "(no function extracted)")
                print("=" * 70)
            Counter.value += 1

            if fx is None:
                if printed:
                    print("Result: INVALID FUNCTION")
                    print("=" * 70)
                scores.append(-10.0)
                continue

            try:
                strategy = create_locked_down_function(fx)
            except Exception as exc:
                if printed:
                    print(f"Compilation error: {exc}")
                    print("=" * 70)
                scores.append(-8.0)
                continue

            try:
                total_score = 0.0
                total_steps = 0
                total_wins = 0
                total_shaped = 0.0
                actual_coverage_sum = 0.0
                plan_coverage_sum = 0.0
                plan_progress_sum = 0.0
                plan_legality_sum = 0.0
                plan_wall_ratio_sum = 0.0
                plan_active_sum = 0.0
                plan_length_sum = 0.0
                plan_safe_alignment_sum = 0.0

                for _ in range(Config.ROLLOUTS_PER_EVAL):
                    reset = env.reset()
                    obs = reset.observation
                    _final_obs, shaped_reward, metrics = execute_strategy(
                        strategy, obs, Config.ROLLOUT_HORIZON
                    )
                    env_score = metrics["total_env_reward"]
                    total_score += env_score
                    total_shaped += shaped_reward
                    total_steps += metrics["steps"]
                    if metrics["level_completed"] or metrics["pellets_remaining_final"] == 0:
                        total_wins += 1

                    initial_pellets = metrics.get("initial_pellets")
                    if not initial_pellets:
                        initial_pellets = metrics.get("plan_total_targets", 0)
                    final_pellets = metrics.get("pellets_remaining_final", 0)
                    if initial_pellets:
                        actual_coverage_sum += max(
                            0.0,
                            (initial_pellets - final_pellets)
                            / float(initial_pellets),
                        )

                    plan_coverage_sum += metrics.get("plan_pellet_coverage_ratio", 0.0)
                    plan_progress_sum += metrics.get(
                        "plan_progress_ratio",
                        metrics.get("plan_progress_score", 0.0),
                    )
                    plan_legality_sum += metrics.get("plan_legality_ratio", 0.0)
                    plan_wall_ratio_sum += metrics.get("plan_wall_hit_ratio", 0.0)
                    plan_active_sum += metrics.get("plan_active_steps", 0.0)
                    plan_length_sum += metrics.get("plan_steps", 0.0)
                    plan_safe_alignment_sum += metrics.get(
                        "plan_safe_alignment_ratio", 0.0
                    )

                denom = float(Config.ROLLOUTS_PER_EVAL)
                avg_score = total_score / denom
                avg_steps = total_steps / denom
                avg_shaped = total_shaped / denom
                avg_actual_coverage = actual_coverage_sum / denom
                avg_plan_cov = plan_coverage_sum / denom
                avg_plan_progress = plan_progress_sum / denom
                avg_plan_legality = plan_legality_sum / denom
                avg_plan_wall = plan_wall_ratio_sum / denom
                avg_plan_active = plan_active_sum / denom
                avg_plan_length = plan_length_sum / denom
                avg_plan_safe = plan_safe_alignment_sum / denom

                plan_depth_bonus = (
                    min(avg_plan_length / Config.PACMAN_MAX_STEPS, 1.0)
                    if Config.PACMAN_MAX_STEPS
                    else 0.0
                )
                active_traverse_bonus = (
                    min(avg_plan_active / Config.PACMAN_MAX_STEPS, 1.0)
                    if Config.PACMAN_MAX_STEPS
                    else 0.0
                )
                score = avg_shaped * 0.05
                score += avg_score * 0.02
                score += avg_plan_cov * 35.0
                score += avg_actual_coverage * 15.0
                score += avg_plan_progress * 12.0
                score += avg_plan_legality * 5.0
                score += avg_plan_safe * 5.0
                score += plan_depth_bonus * 5.0
                score += active_traverse_bonus * 3.0
                score += total_wins * 10.0
                score -= avg_plan_wall * 8.0

                if avg_steps > Config.ROLLOUT_HORIZON * 0.8:
                    score += 5.0

                score = max(score, -25.0)

                if printed:
                    print(f"Avg env reward: {avg_score:.1f}")
                    print(f"Avg shaped reward: {avg_shaped:.1f}")
                    print(f"Wins: {total_wins}/{Config.ROLLOUTS_PER_EVAL}")
                    print(
                        f"Plan est coverage: {avg_plan_cov * 100:.1f}% | Actual coverage: {avg_actual_coverage * 100:.1f}%"
                    )
                    print(
                        f"Plan progress: {avg_plan_progress * 100:.1f}% | Plan legality: {avg_plan_legality * 100:.1f}% | Safe align: {avg_plan_safe * 100:.1f}%"
                    )
                    print(
                        f"Plan depth bonus: {plan_depth_bonus * 100:.1f}% | Active moves: {active_traverse_bonus * 100:.1f}% | Wall hit ratio: {avg_plan_wall * 100:.1f}%"
                    )
                    print(f"Final score: {score:.1f}")
                    print("=" * 70)

                scores.append(score)

            except TimeoutError:
                if printed:
                    print("Result: TIMEOUT")
                    print("=" * 70)
                scores.append(-12.0)
            except Exception as exc:
                if printed:
                    print(f"Runtime error: {exc}")
                    print("=" * 70)
                scores.append(-10.0)

        return scores

    return [function_compiles, imports_ok, sanity_checks, rollout_rewards]


# ============================================================================
# Training Setup
# ============================================================================


class NarrativeCallback(TrainerCallback):
    def __init__(self, milestones):
        self._milestones = sorted(milestones.items())
        self._emitted = set()

    def on_step_end(self, args, state, control, **kwargs):
        for step, message in self._milestones:
            if state.global_step >= step and step not in self._emitted:
                print("\n" + "=" * 70)
                print(message)
                print("=" * 70 + "\n")
                self._emitted.add(step)


def build_generation_kwargs(tokenizer):
    return {
        "temperature": Config.GEN_TEMPERATURE,
        "top_p": Config.GEN_TOP_P,
        "top_k": Config.GEN_TOP_K,
        "repetition_penalty": Config.GEN_REPETITION_PENALTY,
        "max_new_tokens": Config.MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,  # Ensure sampling is enabled
    }


def create_trainer(model, tokenizer, dataset, reward_funcs, maximum_length):
    max_prompt_length = maximum_length + 1
    max_completion_length = min(
        Config.MAX_SEQ_LENGTH - max_prompt_length,
        Config.TRAIN_COMPLETION_LIMIT,
    )

    report_to = Config.REPORT_TO or "none"

    training_args = GRPOConfig(
        temperature=Config.TEMPERATURE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        max_grad_norm=Config.MAX_GRAD_NORM,  # Add gradient clipping
        lr_scheduler_type=Config.LR_SCHEDULER,
        optim=Config.OPTIMIZER,
        logging_steps=Config.LOGGING_STEPS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
        num_generations=Config.NUM_GENERATIONS,
        num_train_epochs=-1,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=Config.MAX_STEPS,
        save_steps=Config.SAVE_STEPS,
        report_to=report_to,
        output_dir=Config.OUTPUT_DIR,
        generation_kwargs=build_generation_kwargs(tokenizer),
        bf16=is_bfloat16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.add_callback(NarrativeCallback(Config.NARRATIVE_MILESTONES))

    print("\n⚙️  Trainer configured:")
    print(f"   - Learning rate: {Config.LEARNING_RATE}")
    print(f"   - Batch size: {Config.BATCH_SIZE} x {Config.GRAD_ACCUMULATION}")
    print(f"   - Max steps: {Config.MAX_STEPS}")
    print(f"   - Generations per prompt: {Config.NUM_GENERATIONS}")
    print(f"   - Report to: {report_to}")
    print(f"   - Max completion tokens: {max_completion_length}")

    return trainer


# ============================================================================
# Evaluation Helpers
# ============================================================================


def run_strategy_episode(env, execute_strategy, strategy, horizon):
    reset = env.reset()
    initial_obs = reset.observation
    try:
        _, shaped, metrics = execute_strategy(strategy, initial_obs, horizon)
        return {
            "reward": metrics["total_env_reward"],
            "shaped": shaped,
            "won": bool(metrics["level_completed"] or metrics["pellets_remaining_final"] == 0),
            "error": None,
            "metrics": metrics,
        }
    except TimeoutError:
        return {"reward": 0.0, "shaped": 0.0, "won": False, "error": "timeout", "metrics": None}
    except Exception as exc:
        return {"reward": 0.0, "shaped": 0.0, "won": False, "error": str(exc), "metrics": None}


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("🎮 PAC-MAN GRPO TRAINING (Docker-based, V2)")
    print("=" * 70)

    random.seed(3407)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)

    try:
        PatchFastRL()
    except Exception as exc:
        print(f"⚠️  Unable to apply Unsloth fast RL patch: {exc}")

    env, PacManAction, PacManObservation, render_maze = setup_environment()
    execute_strategy = create_strategy_executor(env, PacManAction, render_maze)

    model, tokenizer = setup_model()

    # Sanity check: Test model before training
    print("\n" + "=" * 70)
    print("🧪 PRE-TRAINING SANITY CHECK")
    print("=" * 70)
    test_prompt = "Write a simple Python function that returns [1, 2, 3]:"
    test_input = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        test_output = model.generate(
            **test_input,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    test_text = tokenizer.decode(test_output[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_text[len(test_prompt):].strip()[:200]}")
    
    if any(garbage in test_text.lower() for garbage in ['agos', 'ctxcies', 'elman', 'utschei']):
        print("\n❌ MODEL GENERATING GIBBERISH - ABORTING!")
        print("This indicates a tokenizer/model mismatch or corrupted download.")
        print("Try: rm -rf ~/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct")
        return
    print("✅ Model sanity check passed")
    print("=" * 70 + "\n")

    reward_funcs = create_reward_functions(env, execute_strategy, PacManAction)

    dataset, maximum_length = create_training_dataset(tokenizer, env, PacManAction)

    trainer = create_trainer(model, tokenizer, dataset, reward_funcs, maximum_length)

    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print("Watch for strategy quality improvements every few steps.")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as exc:
        print(f"\n❌ Training error: {exc}")
        raise
    finally:
        print("\n✅ Training session complete")

    print("\n" + "=" * 70)
    print("� SAVING TRAINED MODEL")
    print("=" * 70)
    
    # Save LoRA adapters
    lora_output_dir = f"{Config.OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"✅ LoRA adapters saved to: {lora_output_dir}")
    
    # Merge LoRA weights with base model and save
    print("\n🔗 Merging LoRA weights with base model...")
    merged_model_dir = f"{Config.OUTPUT_DIR}/merged_model"
    
    try:
        # Merge and save in 16-bit for inference
        model.save_pretrained_merged(
            merged_model_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✅ Merged 16-bit model saved to: {merged_model_dir}")
        
        # Also save in GGUF format for efficient deployment (optional)
        try:
            gguf_dir = f"{Config.OUTPUT_DIR}/gguf_model"
            model.save_pretrained_gguf(
                gguf_dir,
                tokenizer,
                quantization_method="q4_k_m",  # 4-bit quantization
            )
            print(f"✅ GGUF quantized model saved to: {gguf_dir}")
        except Exception as e:
            print(f"⚠️  GGUF save skipped: {e}")
            
    except Exception as e:
        print(f"⚠️  Model merging failed: {e}")
        print("Continuing with LoRA model for evaluation...")

    print("\n" + "=" * 70)
    print("�📈 TESTING FINAL MODEL")
    print("=" * 70)

    try:
        eval_obs = env.reset().observation
        eval_messages = observation_to_chat(eval_obs)
    except Exception:
        eval_messages = [{"role": "system", "content": PROMPT_TEMPLATE}]

    text = tokenizer.apply_chat_template(
        eval_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = model.generate(
        **tokenizer(text, return_tensors="pt").to(model.device),
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=8,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    strategy_code = extract_function(generated)

    if not strategy_code:
        print("\n❌ Could not extract strategy function from final model output")
    else:
        try:
            final_strategy = create_locked_down_function(strategy_code)
        except Exception as exc:
            print(f"\n❌ Strategy compilation failed: {exc}")
            final_strategy = None

        if final_strategy is not None:
            print("\n" + "=" * 70)
            print("🧪 FINAL STRATEGY EVALUATION")
            print("=" * 70)
            results = []
            for idx in range(Config.EVALUATION_EPISODES):
                outcome = run_strategy_episode(
                    env,
                    execute_strategy,
                    final_strategy,
                    Config.TEST_MAX_STEPS,
                )
                results.append(outcome)
                if outcome["error"]:
                    print(f"Episode {idx + 1}: ERROR - {outcome['error']}")
                else:
                    status = "🏆 WIN" if outcome["won"] else "💀 Loss"
                    print(
                        f"Episode {idx + 1}: {status} | env reward={outcome['reward']:.1f} | shaped={outcome['shaped']:.1f}"
                    )

            valid = [r for r in results if not r["error"]]
            if valid:
                avg_env = sum(r["reward"] for r in valid) / len(valid)
                avg_shaped = sum(r["shaped"] for r in valid) / len(valid)
                wins = sum(1 for r in valid if r["won"])
                print("\n" + "=" * 70)
                print("📊 FINAL PERFORMANCE SUMMARY")
                print("=" * 70)
                print(f"Win rate: {wins}/{len(valid)} ({wins / len(valid) * 100:.1f}%)")
                print(f"Avg env reward: {avg_env:.1f}")
                print(f"Avg shaped reward: {avg_shaped:.1f}")
                print("=" * 70)

    print("\n🧹 Cleaning up...")
    try:
        env.close()
    except Exception:
        pass
    print("✅ Done!")


if __name__ == "__main__":
    main()
