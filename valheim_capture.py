import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path


DEFAULT_PROCESS_NAME = "valheim.exe"
DEFAULT_WINDOW_REFRESH_SECONDS = 0.5


def set_dpi_awareness() -> None:
    if platform.system().lower() != "windows":
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        else:
            user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def process_is_running(process_name: str) -> bool:
    system = platform.system().lower()

    if system == "windows":
        return windows_process_is_running(process_name)

    return unix_process_is_running(process_name)


def windows_process_is_running(process_name: str) -> bool:
    return bool(windows_process_ids(process_name))


def windows_process_ids(process_name: str) -> set[int]:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/FO", "CSV", "/NH"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    process_name = process_name.lower()
    process_ids = set()

    for row in csv.reader(StringIO(result.stdout)):
        if len(row) < 2:
            continue
        image_name, process_id = row[0].lower(), row[1]
        if image_name != process_name:
            continue
        try:
            process_ids.add(int(process_id))
        except ValueError:
            continue

    return process_ids


def unix_process_is_running(process_name: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0 and bool(result.stdout.strip())


def wait_for_process(process_name: str, poll_interval: float) -> None:
    print(f"Waiting for {process_name} to start. Press Ctrl+C to quit.")

    while not process_is_running(process_name):
        time.sleep(poll_interval)

    print(f"Found {process_name}. Starting screen capture.")


def require_windows_window_modules():
    try:
        import win32gui
        import win32process
    except ImportError:
        print("Missing dependency: pywin32. Install dependencies with: python -m pip install -r requirements.txt")
        raise SystemExit(1)

    return win32gui, win32process


def find_window_client_area(process_name: str) -> tuple[dict[str, int], str] | None:
    if platform.system().lower() != "windows":
        raise ValueError("Window capture is currently supported on Windows only.")

    process_ids = windows_process_ids(process_name)
    if not process_ids:
        return None

    win32gui, win32process = require_windows_window_modules()
    candidates: list[tuple[dict[str, int], str]] = []

    def collect_window(hwnd, _extra) -> bool:
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return True

        try:
            _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return True

        if process_id not in process_ids:
            return True

        title = win32gui.GetWindowText(hwnd).strip() or f"hwnd:{hwnd}"

        try:
            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
            screen_left, screen_top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
        except Exception:
            return True

        width = client_right - client_left
        height = client_bottom - client_top
        if width <= 0 or height <= 0:
            return True

        candidates.append(
            (
                {
                    "left": screen_left,
                    "top": screen_top,
                    "width": width,
                    "height": height,
                },
                title,
            )
        )
        return True

    win32gui.EnumWindows(collect_window, None)

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate[0]["width"] * candidate[0]["height"])


def wait_for_window_client_area(process_name: str, poll_interval: float) -> tuple[dict[str, int], str]:
    print(f"Waiting for a visible {process_name} game window.")

    while True:
        target = find_window_client_area(process_name)
        if target is not None:
            area, title = target
            print(f'Found window "{title}".')
            return area, title

        if not process_is_running(process_name):
            raise ValueError(f"{process_name} exited before a capturable window was found.")

        time.sleep(poll_interval)


def parse_region(values: list[int] | None) -> dict[str, int] | None:
    if values is None:
        return None

    left, top, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("Region width and height must be positive.")

    return {"left": left, "top": top, "width": width, "height": height}


def get_capture_area(sct, monitor_index: int, region: dict[str, int] | None) -> dict[str, int]:
    if region is not None:
        return region

    if monitor_index < 0 or monitor_index >= len(sct.monitors):
        available = ", ".join(str(index) for index in range(len(sct.monitors)))
        raise ValueError(f"Monitor {monitor_index} is not available. Valid choices: {available}.")

    return sct.monitors[monitor_index]


def format_capture_area(area: dict[str, int]) -> str:
    return f'{area["width"]}x{area["height"]} at ({area["left"]}, {area["top"]})'


def save_png_atomic(sct_img, output_path: Path) -> None:
    import mss.tools

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(temp_path))
    os.replace(temp_path, output_path)


def capture_screen(args: argparse.Namespace) -> None:
    try:
        import mss
    except ImportError:
        print("Missing dependency: mss. Install dependencies with: python -m pip install -r requirements.txt")
        raise SystemExit(1)

    region = parse_region(args.region)
    save_latest = None if args.no_save else args.save_latest
    capture_mode = args.capture_mode
    if region is not None:
        capture_mode = "region"

    with mss.MSS() as sct:
        if capture_mode == "window":
            capture_area, window_title = wait_for_window_client_area(args.process_name, args.poll_interval)
            print(f"Capturing window client area: {format_capture_area(capture_area)}")
        else:
            capture_area = get_capture_area(sct, args.monitor, region)
            window_title = None
            print(f"Capturing {capture_mode}: {format_capture_area(capture_area)}")

        frame_count = 0
        frames_since_log = 0
        log_started_at = time.perf_counter()
        next_save_at = time.perf_counter()
        next_process_check_at = time.perf_counter() + args.process_check_every
        next_window_check_at = time.perf_counter() + args.window_check_every
        window_missing_reported = False

        while True:
            now = time.perf_counter()

            if now >= next_process_check_at:
                if not process_is_running(args.process_name):
                    break
                next_process_check_at = now + args.process_check_every

            if capture_mode == "window" and now >= next_window_check_at:
                target = find_window_client_area(args.process_name)
                if target is None:
                    if not window_missing_reported:
                        print("Valheim window is not visible or is minimized. Waiting for it to return.")
                        window_missing_reported = True
                    time.sleep(min(args.window_check_every, 0.25))
                    next_window_check_at = time.perf_counter() + args.window_check_every
                    continue

                updated_area, updated_title = target
                if updated_area != capture_area or updated_title != window_title:
                    capture_area = updated_area
                    window_title = updated_title
                    print(f"Updated window client area: {format_capture_area(capture_area)}")
                window_missing_reported = False
                next_window_check_at = now + args.window_check_every

            screenshot = sct.grab(capture_area)
            frame_count += 1
            frames_since_log += 1
            now = time.perf_counter()

            if save_latest is not None and now >= next_save_at:
                save_png_atomic(screenshot, save_latest)
                next_save_at = now + args.save_every

            elapsed = now - log_started_at
            if elapsed >= args.log_every:
                fps = frames_since_log / elapsed
                print(f"Captured {frame_count} frames ({fps:.1f} FPS)")
                frames_since_log = 0
                log_started_at = now

        print(f"{args.process_name} is no longer running. Capture stopped.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for Valheim to start, then capture the game window."
    )
    parser.add_argument(
        "--process-name",
        default=DEFAULT_PROCESS_NAME,
        help=f"Process name to wait for. Default: {DEFAULT_PROCESS_NAME}",
    )
    parser.add_argument(
        "--poll-interval",
        default=2.0,
        type=float,
        help="Seconds between process checks while waiting. Default: 2.0",
    )
    parser.add_argument(
        "--capture-mode",
        choices=("window", "monitor"),
        default="window",
        help="Capture the Valheim window client area or a whole monitor. Default: window",
    )
    parser.add_argument(
        "--monitor",
        default=1,
        type=int,
        help="mss monitor index for --capture-mode monitor. 1 is usually primary; 0 captures all monitors.",
    )
    parser.add_argument(
        "--region",
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        nargs=4,
        type=int,
        help="Capture a specific screen region instead of a whole monitor.",
    )
    parser.add_argument(
        "--save-latest",
        default=Path("captures/latest.png"),
        type=Path,
        help="Path for an occasionally refreshed PNG snapshot. Default: captures/latest.png",
    )
    parser.add_argument(
        "--save-every",
        default=1.0,
        type=float,
        help="Seconds between latest.png updates. Default: 1.0",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Capture frames in memory only and do not write latest.png.",
    )
    parser.add_argument(
        "--log-every",
        default=1.0,
        type=float,
        help="Seconds between FPS log lines. Default: 1.0",
    )
    parser.add_argument(
        "--process-check-every",
        default=2.0,
        type=float,
        help="Seconds between process checks while capturing. Default: 2.0",
    )
    parser.add_argument(
        "--window-check-every",
        default=DEFAULT_WINDOW_REFRESH_SECONDS,
        type=float,
        help=f"Seconds between Valheim window position/size checks. Default: {DEFAULT_WINDOW_REFRESH_SECONDS}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive.")
    if args.save_every <= 0:
        parser.error("--save-every must be positive.")
    if args.log_every <= 0:
        parser.error("--log-every must be positive.")
    if args.process_check_every <= 0:
        parser.error("--process-check-every must be positive.")
    if args.window_check_every <= 0:
        parser.error("--window-check-every must be positive.")

    try:
        set_dpi_awareness()
        wait_for_process(args.process_name, args.poll_interval)
        capture_screen(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
