from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from valheim_actions import ACTION_COLUMNS, BUTTONS, empty_button_state
from valheim_capture import (
    DEFAULT_PROCESS_NAME,
    find_window_client_area,
    set_dpi_awareness,
    wait_for_process,
    wait_for_window_client_area,
)
from valheim_goals import DEFAULT_GOAL, normalize_goal_name


class InputSnapshotter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buttons = empty_button_state()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.last_mouse_position: tuple[int, int] | None = None
        self.keyboard_listener = None
        self.mouse_listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError:
            print("Missing dependency: pynput. Install dependencies with: python -m pip install -r requirements.txt")
            raise SystemExit(1)

        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
        )
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def stop(self) -> None:
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
        if self.mouse_listener is not None:
            self.mouse_listener.stop()

    def snapshot(self) -> dict[str, float | int]:
        with self.lock:
            row: dict[str, float | int] = dict(self.buttons)
            row["mouse_dx"] = self.mouse_dx
            row["mouse_dy"] = self.mouse_dy
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            return row

    def _on_key_press(self, key) -> None:
        key_name = self._key_name(key)
        if key_name in self.buttons:
            with self.lock:
                self.buttons[key_name] = 1

    def _on_key_release(self, key) -> None:
        key_name = self._key_name(key)
        if key_name in self.buttons:
            with self.lock:
                self.buttons[key_name] = 0

    def _on_mouse_click(self, _x, _y, button, pressed: bool) -> None:
        button_name = self._mouse_button_name(button)
        if button_name is None:
            return

        with self.lock:
            self.buttons[button_name] = int(pressed)

    def _on_mouse_move(self, x: int, y: int) -> None:
        with self.lock:
            if self.last_mouse_position is not None:
                last_x, last_y = self.last_mouse_position
                self.mouse_dx += x - last_x
                self.mouse_dy += y - last_y
            self.last_mouse_position = (x, y)

    def _key_name(self, key) -> str | None:
        char = getattr(key, "char", None)
        if char:
            return char.lower()

        name = getattr(key, "name", None)
        if name in {"space", "shift", "ctrl"}:
            return name
        if name in {"shift_l", "shift_r"}:
            return "shift"
        if name in {"ctrl_l", "ctrl_r"}:
            return "ctrl"
        return None

    def _mouse_button_name(self, button) -> str | None:
        name = str(button).split(".")[-1]
        if name == "left":
            return "lmb"
        if name == "right":
            return "rmb"
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record Valheim window frames and synchronized keyboard/mouse inputs."
    )
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--output-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--duration", default=0.0, type=float, help="Seconds to record; 0 records until Ctrl+C.")
    parser.add_argument("--frame-width", default=320, type=int, help="Stored frame width. Height preserves aspect ratio.")
    parser.add_argument("--jpeg-quality", default=85, type=int)
    parser.add_argument("--window-check-every", default=0.5, type=float)
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Behavior/goal label for this recording.")
    parser.add_argument("--behavior", default="", help="Alias for --goal.")
    return parser


def resize_for_storage(bgra: np.ndarray, target_width: int) -> np.ndarray:
    bgr = bgra[:, :, :3]
    if target_width <= 0 or target_width == bgr.shape[1]:
        return bgr

    target_height = max(1, round(bgr.shape[0] * (target_width / bgr.shape[1])))
    return cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def make_session_dir(output_dir: Path) -> Path:
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = output_dir / session_name
    (session_dir / "frames").mkdir(parents=True, exist_ok=True)
    return session_dir


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.duration < 0:
        raise SystemExit("--duration cannot be negative.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100.")
    goal_name = normalize_goal_name(args.behavior or args.goal)

    try:
        import mss
    except ImportError:
        print("Missing dependency: mss. Install dependencies with: python -m pip install -r requirements.txt")
        return 1

    set_dpi_awareness()
    wait_for_process(args.process_name, poll_interval=1.0)
    capture_area, window_title = wait_for_window_client_area(args.process_name, poll_interval=1.0)

    session_dir = make_session_dir(args.output_dir)
    frames_dir = session_dir / "frames"
    actions_path = session_dir / "actions.csv"
    meta_path = session_dir / "meta.json"

    meta = {
        "process_name": args.process_name,
        "window_title": window_title,
        "fps": args.fps,
        "stored_frame_width": args.frame_width,
        "buttons": BUTTONS,
        "action_columns": ACTION_COLUMNS,
        "goal": goal_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    input_snapshotter = InputSnapshotter()
    input_snapshotter.start()

    print(f"Recording goal '{goal_name}' to {session_dir}. Keep Valheim focused. Press Ctrl+C to stop.")

    fieldnames = (
        "frame",
        "timestamp",
        "elapsed",
        "window_left",
        "window_top",
        "window_width",
        "window_height",
    ) + ACTION_COLUMNS

    frame_interval = 1.0 / args.fps
    start_time = time.perf_counter()
    next_frame_at = start_time
    next_window_check_at = start_time + args.window_check_every
    frame_index = 0

    try:
        with mss.MSS() as sct, actions_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                now = time.perf_counter()
                if args.duration and now - start_time >= args.duration:
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
                frame = resize_for_storage(np.asarray(screenshot), args.frame_width)
                frame_name = f"{frame_index:08d}.jpg"
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                snapshot = input_snapshotter.snapshot()
                elapsed = now - start_time
                row = {
                    "frame": frame_name,
                    "timestamp": time.time(),
                    "elapsed": elapsed,
                    "window_left": capture_area["left"],
                    "window_top": capture_area["top"],
                    "window_width": capture_area["width"],
                    "window_height": capture_area["height"],
                }
                row.update(snapshot)
                writer.writerow(row)

                frame_index += 1
                if frame_index % max(1, int(args.fps * 5)) == 0:
                    print(f"Recorded {frame_index} frames ({elapsed:.1f}s).")

                next_frame_at += frame_interval
                if next_frame_at < now - frame_interval:
                    next_frame_at = now + frame_interval
    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        input_snapshotter.stop()

    print(f"Saved {frame_index} frames to {session_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
