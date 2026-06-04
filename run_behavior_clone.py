from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from behavior_model import checkpoint_goal_id, checkpoint_goals, load_behavior_checkpoint
from valheim_actions import HELD_KEYS, MOUSE_BUTTONS, TAP_KEYS
from valheim_capture import (
    DEFAULT_PROCESS_NAME,
    find_window_client_area,
    set_dpi_awareness,
    wait_for_process,
    wait_for_window_client_area,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained pure-pixel Valheim behavior policy.")
    parser.add_argument("--model-path", default=Path("models/behavior_clone.pt"), type=Path)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--threshold", default=0.0, type=float, help="Override saved button threshold; 0 uses checkpoint.")
    parser.add_argument("--mouse-gain", default=1.0, type=float)
    parser.add_argument("--mouse-deadzone", default=2.0, type=float)
    parser.add_argument("--tap-cooldown", default=0.25, type=float)
    parser.add_argument("--window-check-every", default=0.5, type=float)
    parser.add_argument("--goal", default="", help="Goal/behavior to run from a conditioned checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print predictions without sending inputs.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def preprocess_frame(bgra: np.ndarray, frame_width: int, frame_height: int) -> torch.Tensor:
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.astype(np.float32) / 255.0)


def release_held_inputs(input_backend, held_buttons: set[str]) -> None:
    if input_backend is None:
        held_buttons.clear()
        return

    for button in tuple(held_buttons):
        input_backend.keyUp(button)
        held_buttons.remove(button)


def apply_predictions(
    input_backend,
    buttons: tuple[str, ...],
    probabilities: np.ndarray,
    mouse: np.ndarray,
    threshold: float,
    mouse_clip: float,
    mouse_gain: float,
    mouse_deadzone: float,
    tap_cooldown: float,
    last_tap_at: dict[str, float],
    held_buttons: set[str],
) -> None:
    now = time.perf_counter()
    probability_by_button = dict(zip(buttons, probabilities, strict=True))

    for button in HELD_KEYS:
        should_hold = probability_by_button.get(button, 0.0) >= threshold
        if should_hold and button not in held_buttons:
            input_backend.keyDown(button)
            held_buttons.add(button)
        elif not should_hold and button in held_buttons:
            input_backend.keyUp(button)
            held_buttons.remove(button)

    for button in TAP_KEYS:
        if probability_by_button.get(button, 0.0) < threshold:
            continue
        if now - last_tap_at.get(button, 0.0) >= tap_cooldown:
            input_backend.press(button)
            last_tap_at[button] = now

    for button, directinput_name in MOUSE_BUTTONS.items():
        if probability_by_button.get(button, 0.0) < threshold:
            continue
        if now - last_tap_at.get(button, 0.0) >= tap_cooldown:
            input_backend.click(button=directinput_name)
            last_tap_at[button] = now

    dx = int(round(float(mouse[0]) * mouse_clip * mouse_gain))
    dy = int(round(float(mouse[1]) * mouse_clip * mouse_gain))
    if abs(dx) >= mouse_deadzone or abs(dy) >= mouse_deadzone:
        input_backend.moveRel(dx, dy, duration=0)


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.mouse_gain < 0:
        raise SystemExit("--mouse-gain cannot be negative.")

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

    device = choose_device(args.device)
    model, checkpoint = load_behavior_checkpoint(args.model_path, device)
    buttons = tuple(checkpoint["buttons"])
    frame_width = int(checkpoint["frame_width"])
    frame_height = int(checkpoint["frame_height"])
    frame_stack = int(checkpoint["frame_stack"])
    mouse_clip = float(checkpoint["mouse_clip"])
    threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold <= 0 else args.threshold
    goals = checkpoint_goals(checkpoint)
    goal_id = checkpoint_goal_id(checkpoint, args.goal)
    goal_name = goals[goal_id]
    goal_ids = torch.tensor([goal_id], dtype=torch.long, device=device)

    set_dpi_awareness()
    wait_for_process(args.process_name, poll_interval=1.0)
    capture_area, _window_title = wait_for_window_client_area(args.process_name, poll_interval=1.0)

    frame_buffer: deque[torch.Tensor] = deque(maxlen=frame_stack)
    last_tap_at: dict[str, float] = {}
    held_buttons: set[str] = set()
    frame_interval = 1.0 / args.fps
    next_frame_at = time.perf_counter()
    next_window_check_at = time.perf_counter() + args.window_check_every

    mode = "dry run" if args.dry_run else "live input"
    print(f"Running behavior policy goal '{goal_name}' in {mode}. Press Ctrl+C to stop.")

    try:
        with mss.MSS() as sct:
            while True:
                now = time.perf_counter()
                if now < next_frame_at:
                    time.sleep(min(0.01, next_frame_at - now))
                    continue

                if now >= next_window_check_at:
                    target = find_window_client_area(args.process_name)
                    if target is not None:
                        capture_area, _window_title = target
                    next_window_check_at = now + args.window_check_every

                screenshot = sct.grab(capture_area)
                frame = preprocess_frame(np.asarray(screenshot), frame_width, frame_height)
                if not frame_buffer:
                    for _ in range(frame_stack):
                        frame_buffer.append(frame)
                else:
                    frame_buffer.append(frame)

                observation = torch.stack(tuple(frame_buffer)).unsqueeze(0).to(device)
                with torch.no_grad():
                    button_logits, mouse_pred = model(observation, goal_ids)
                    probabilities = torch.sigmoid(button_logits).squeeze(0).cpu().numpy()
                    mouse = mouse_pred.squeeze(0).cpu().numpy()

                if args.dry_run:
                    top_buttons = sorted(
                        zip(buttons, probabilities, strict=True),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                    top_text = ", ".join(f"{button}={probability:.2f}" for button, probability in top_buttons)
                    print(f"buttons: {top_text} mouse=({mouse[0]:+.2f}, {mouse[1]:+.2f})")
                else:
                    apply_predictions(
                        input_backend,
                        buttons,
                        probabilities,
                        mouse,
                        threshold,
                        mouse_clip,
                        args.mouse_gain,
                        args.mouse_deadzone,
                        args.tap_cooldown,
                        last_tap_at,
                        held_buttons,
                    )

                next_frame_at += frame_interval
                if next_frame_at < now - frame_interval:
                    next_frame_at = now + frame_interval
    except KeyboardInterrupt:
        print("\nStopped policy.")
    finally:
        release_held_inputs(input_backend, held_buttons)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
