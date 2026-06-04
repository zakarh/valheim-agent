from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from behavior_model import checkpoint_goal_id, checkpoint_goals, load_behavior_checkpoint
from rollout_recorder import RolloutRecorder
from valheim_actions import BUTTONS
from valheim_capture import (
    DEFAULT_PROCESS_NAME,
    find_window_client_area,
    set_dpi_awareness,
    wait_for_process,
    wait_for_window_client_area,
)
from valheim_control import active_button_summary, apply_predictions, release_held_inputs
from valheim_goals import DEFAULT_GOAL, normalize_goal_name, parse_goal_list, resolve_goal_id
from valheim_rl_actions import action_to_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an AI policy playing Valheim.")
    parser.add_argument("--policy", choices=("behavior_clone", "dqn", "random"), default="behavior_clone")
    parser.add_argument("--policy-name", default="", help="Name to store under rollouts/. Defaults to policy/model name.")
    parser.add_argument("--model-path", default=Path("models/behavior_clone.pt"), type=Path)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--output-dir", default=Path("rollouts"), type=Path)
    parser.add_argument("--goal", default="", help="Goal/behavior label for this rollout.")
    parser.add_argument("--duration", default=60.0, type=float)
    parser.add_argument(
        "--start-delay",
        default=5.0,
        type=float,
        help="Seconds to wait before recording/sending inputs so you can prep the scene. Default: 5.0",
    )
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--frame-width", default=640, type=int, help="Recorded video width.")
    parser.add_argument("--threshold", default=0.0, type=float, help="Override saved threshold; 0 uses checkpoint/default.")
    parser.add_argument("--mouse-gain", default=1.0, type=float)
    parser.add_argument("--mouse-deadzone", default=2.0, type=float)
    parser.add_argument("--tap-cooldown", default=0.25, type=float)
    parser.add_argument("--window-check-every", default=0.5, type=float)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Record predictions without sending inputs.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--rl-obs-width", default=84, type=int)
    parser.add_argument("--rl-obs-height", default=84, type=int)
    parser.add_argument("--rl-obs-stack", default=4, type=int)
    parser.add_argument("--rl-goals", default="", help="Comma-separated DQN goals if no sidecar metadata exists.")
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def wait_start_delay(seconds: float) -> None:
    if seconds <= 0:
        return

    print(f"Starting in {seconds:.1f}s. Prep the scene now.")
    whole_seconds = int(seconds)
    for remaining in range(whole_seconds, 0, -1):
        print(f"{remaining}...")
        time.sleep(1.0)

    fractional = seconds - whole_seconds
    if fractional > 0:
        time.sleep(fractional)


def preprocess_frame(
    bgra: np.ndarray,
    frame_width: int,
    frame_height: int,
    normalize: bool,
) -> np.ndarray:
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    if normalize:
        return resized.astype(np.float32) / 255.0
    return resized.astype(np.uint8)


def random_policy(buttons: tuple[str, ...], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros(len(buttons), dtype=np.float32)
    mouse = np.zeros(2, dtype=np.float32)
    button_index = {button: index for index, button in enumerate(buttons)}

    choice = rng.choices(
        population=("noop", "forward", "left", "right", "turn_left", "turn_right", "jump", "attack", "interact"),
        weights=(25, 30, 8, 8, 10, 10, 3, 4, 2),
        k=1,
    )[0]

    if choice == "forward":
        probabilities[button_index["w"]] = 1.0
    elif choice == "left":
        probabilities[button_index["a"]] = 1.0
    elif choice == "right":
        probabilities[button_index["d"]] = 1.0
    elif choice == "turn_left":
        mouse[0] = -0.45
    elif choice == "turn_right":
        mouse[0] = 0.45
    elif choice == "jump":
        probabilities[button_index["space"]] = 1.0
    elif choice == "attack":
        probabilities[button_index["lmb"]] = 1.0
    elif choice == "interact":
        probabilities[button_index["e"]] = 1.0

    return probabilities, mouse


def load_policy_metadata(model_path: Path) -> dict:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_goal_conditioned_observation(frame_buffer: deque[np.ndarray], policy_state: dict) -> np.ndarray:
    observation = np.stack(tuple(frame_buffer), axis=0).astype(np.uint8)
    goals = policy_state.get("goals", (DEFAULT_GOAL,))
    if len(goals) <= 1:
        return observation

    goal_planes = np.zeros(
        (len(goals), policy_state["frame_height"], policy_state["frame_width"]),
        dtype=np.uint8,
    )
    goal_planes[int(policy_state["goal_id"])].fill(255)
    return np.concatenate((observation, goal_planes), axis=0)


def load_policy(args: argparse.Namespace):
    requested_goal = normalize_goal_name(args.goal or DEFAULT_GOAL)
    if args.policy == "random":
        goals = (requested_goal,)
        return {
            "kind": "random",
            "buttons": BUTTONS,
            "frame_width": 160,
            "frame_height": 96,
            "frame_stack": 4,
            "mouse_clip": 80.0,
            "threshold": 0.5 if args.threshold <= 0 else args.threshold,
            "goals": goals,
            "goal_id": 0,
            "goal_name": goals[0],
            "model": None,
            "device": torch.device("cpu"),
        }

    device = choose_device(args.device)

    if args.policy == "dqn":
        from stable_baselines3 import DQN

        model = DQN.load(str(args.model_path), device=device)
        metadata = load_policy_metadata(args.model_path)
        goals = tuple(metadata.get("goals") or parse_goal_list(args.rl_goals or args.goal or DEFAULT_GOAL))
        goals = tuple(normalize_goal_name(goal) for goal in goals)
        goal_id = resolve_goal_id(goals, args.goal, metadata.get("default_goal") or goals[0])
        frame_width = int(metadata.get("obs_width", args.rl_obs_width))
        frame_height = int(metadata.get("obs_height", args.rl_obs_height))
        frame_stack = int(metadata.get("obs_stack", args.rl_obs_stack))
        expected_shape = getattr(model.observation_space, "shape", None)
        expected_channels = int(expected_shape[0]) if expected_shape else frame_stack
        actual_channels = frame_stack + (len(goals) if len(goals) > 1 else 0)
        if expected_channels != actual_channels:
            raise ValueError(
                f"DQN expects {expected_channels} observation channels, but rollout config creates "
                f"{actual_channels}. Check {args.model_path.with_suffix('.json')} or pass --rl-goals."
            )
        return {
            "kind": "dqn",
            "buttons": BUTTONS,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "frame_stack": frame_stack,
            "mouse_clip": 80.0,
            "threshold": 0.5,
            "goals": goals,
            "goal_id": goal_id,
            "goal_name": goals[goal_id],
            "model": model,
            "device": device,
        }

    model, checkpoint = load_behavior_checkpoint(args.model_path, device)
    goals = checkpoint_goals(checkpoint)
    goal_id = checkpoint_goal_id(checkpoint, args.goal)
    return {
        "kind": "behavior_clone",
        "buttons": tuple(checkpoint["buttons"]),
        "frame_width": int(checkpoint["frame_width"]),
        "frame_height": int(checkpoint["frame_height"]),
        "frame_stack": int(checkpoint["frame_stack"]),
        "mouse_clip": float(checkpoint["mouse_clip"]),
        "threshold": float(checkpoint.get("threshold", 0.5)) if args.threshold <= 0 else args.threshold,
        "goals": goals,
        "goal_id": goal_id,
        "goal_name": goals[goal_id],
        "model": model,
        "device": device,
    }


def predict_behavior_clone(policy_state: dict, frame_buffer: deque[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observation = torch.from_numpy(np.stack(tuple(frame_buffer), axis=0)).float().unsqueeze(0).to(policy_state["device"])
    goal_ids = torch.tensor([policy_state["goal_id"]], dtype=torch.long, device=policy_state["device"])
    with torch.no_grad():
        button_logits, mouse_pred = policy_state["model"](observation, goal_ids)
        probabilities = torch.sigmoid(button_logits).squeeze(0).cpu().numpy()
        mouse = mouse_pred.squeeze(0).cpu().numpy()
    return probabilities, mouse


def predict_dqn(policy_state: dict, frame_buffer: deque[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observation = build_goal_conditioned_observation(frame_buffer, policy_state)
    action, _state = policy_state["model"].predict(observation, deterministic=False)
    return action_to_predictions(int(action))


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive.")
    if args.start_delay < 0:
        raise SystemExit("--start-delay cannot be negative.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.frame_width <= 0:
        raise SystemExit("--frame-width must be positive.")

    try:
        import mss
    except ImportError:
        print("Missing dependency: mss. Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    input_backend = None
    if not args.dry_run:
        try:
            import pydirectinput
        except ImportError:
            print("Missing dependency: pydirectinput. Install dependencies with: python -m pip install -r requirements.txt")
            return 1
        pydirectinput.PAUSE = 0
        input_backend = pydirectinput

    policy_state = load_policy(args)
    policy_name = args.policy_name
    if not policy_name:
        policy_name = args.model_path.stem if args.policy in {"behavior_clone", "dqn"} else "random"

    set_dpi_awareness()
    wait_for_process(args.process_name, poll_interval=1.0)
    capture_area, window_title = wait_for_window_client_area(args.process_name, poll_interval=1.0)
    wait_start_delay(args.start_delay)

    rng = random.Random(args.seed)
    frame_buffer: deque[np.ndarray] = deque(maxlen=policy_state["frame_stack"])
    last_tap_at: dict[str, float] = {}
    held_buttons: set[str] = set()
    frame_interval = 1.0 / args.fps
    next_frame_at = time.perf_counter()
    next_window_check_at = time.perf_counter() + args.window_check_every
    start_time = time.perf_counter()

    mode = "dry run" if args.dry_run else "live input"
    print(
        f"Recording {policy_name} goal '{policy_state['goal_name']}' in {mode} "
        f"for {args.duration:.1f}s. Press Ctrl+C to stop."
    )

    try:
        with mss.MSS() as sct, RolloutRecorder(
            output_dir=args.output_dir,
            policy_name=policy_name,
            fps=args.fps,
            frame_width=args.frame_width,
            save_frames=args.save_frames,
        ) as recorder:
            recorder.write_meta(
                {
                    "policy": args.policy,
                    "policy_name": policy_name,
                    "goal": policy_state["goal_name"],
                    "goals": policy_state["goals"],
                    "model_path": str(args.model_path) if args.policy in {"behavior_clone", "dqn"} else "",
                    "window_title": window_title,
                    "process_name": args.process_name,
                    "duration": args.duration,
                    "dry_run": args.dry_run,
                    "threshold": policy_state["threshold"],
                    "mouse_clip": policy_state["mouse_clip"],
                    "mouse_gain": args.mouse_gain,
                    "seed": args.seed,
                }
            )

            while True:
                now = time.perf_counter()
                elapsed = now - start_time
                if elapsed >= args.duration:
                    break

                if now < next_frame_at:
                    time.sleep(min(0.01, next_frame_at - now))
                    continue

                if now >= next_window_check_at:
                    target = find_window_client_area(args.process_name)
                    if target is not None:
                        capture_area, window_title = target
                    next_window_check_at = now + args.window_check_every

                screenshot = sct.grab(capture_area)
                raw_frame = np.asarray(screenshot).copy()
                frame = preprocess_frame(
                    raw_frame,
                    policy_state["frame_width"],
                    policy_state["frame_height"],
                    normalize=policy_state["kind"] == "behavior_clone",
                )
                if not frame_buffer:
                    for _ in range(policy_state["frame_stack"]):
                        frame_buffer.append(frame)
                else:
                    frame_buffer.append(frame)

                if args.policy == "random":
                    probabilities, mouse = random_policy(policy_state["buttons"], rng)
                elif args.policy == "dqn":
                    probabilities, mouse = predict_dqn(policy_state, frame_buffer)
                else:
                    probabilities, mouse = predict_behavior_clone(policy_state, frame_buffer)

                action_row = apply_predictions(
                    input_backend,
                    policy_state["buttons"],
                    probabilities,
                    mouse,
                    policy_state["threshold"],
                    policy_state["mouse_clip"],
                    args.mouse_gain,
                    args.mouse_deadzone,
                    args.tap_cooldown,
                    last_tap_at,
                    held_buttons,
                )

                recorder.append(
                    raw_frame,
                    action_row,
                    capture_area,
                    timestamp=time.time(),
                    elapsed=elapsed,
                )

                if recorder.frame_index % max(1, int(args.fps * 5)) == 0:
                    print(
                        f"{recorder.frame_index} frames, "
                        f"{elapsed:.1f}s, {active_button_summary(policy_state['buttons'], probabilities)}"
                    )

                next_frame_at += frame_interval
                if next_frame_at < now - frame_interval:
                    next_frame_at = now + frame_interval
    except KeyboardInterrupt:
        print("\nStopped rollout.")
    finally:
        release_held_inputs(input_backend, held_buttons)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
