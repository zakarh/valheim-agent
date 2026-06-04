from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from valheim_goals import normalize_goal_name, read_goal_from_meta


@dataclass(frozen=True)
class RolloutEpisode:
    episode_dir: Path
    video_path: Path
    policy_name: str
    episode_id: str
    goal_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create A/B rollout comparison videos and record human preferences."
    )
    parser.add_argument("--rollouts-dir", default=Path("rollouts"), type=Path)
    parser.add_argument("--comparisons-dir", default=Path("comparisons"), type=Path)
    parser.add_argument("--preferences-path", default=Path("preferences/preferences.csv"), type=Path)
    parser.add_argument("--policy-a", default="", help="Optional policy filter for side A.")
    parser.add_argument("--policy-b", default="", help="Optional policy filter for side B.")
    parser.add_argument("--goal", default="", help="Optional goal filter; defaults to same-goal comparisons.")
    parser.add_argument(
        "--pairs",
        default=5,
        type=int,
        help="Number of separate A/B rollout comparisons to generate and label.",
    )
    parser.add_argument("--max-seconds", default=60.0, type=float)
    parser.add_argument("--width", default=640, type=int, help="Width of each side in the comparison video.")
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--no-label", action="store_true", help="Generate comparison videos without prompting.")
    return parser


def discover_rollouts(rollouts_dir: Path) -> list[RolloutEpisode]:
    episodes = []
    for video_path in sorted(rollouts_dir.glob("*/*/episode.mp4")):
        episode_dir = video_path.parent
        policy_name = episode_dir.parent.name
        episode_id = episode_dir.name
        goal_name = read_goal_from_meta(episode_dir)

        meta_path = episode_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                policy_name = str(meta.get("policy_name", policy_name))
                goal_name = normalize_goal_name(meta.get("goal") or goal_name)
            except json.JSONDecodeError:
                pass

        episodes.append(
            RolloutEpisode(
                episode_dir=episode_dir,
                video_path=video_path,
                policy_name=policy_name,
                episode_id=episode_id,
                goal_name=goal_name,
            )
        )
    return episodes


def filter_episodes(
    episodes: list[RolloutEpisode],
    policy_name: str,
    goal_name: str,
) -> list[RolloutEpisode]:
    filtered = episodes
    if policy_name:
        filtered = [episode for episode in filtered if episode.policy_name == policy_name]
    if goal_name:
        normalized_goal = normalize_goal_name(goal_name)
        filtered = [episode for episode in filtered if episode.goal_name == normalized_goal]
    return filtered


def choose_pairs(
    episodes: list[RolloutEpisode],
    policy_a: str,
    policy_b: str,
    goal_name: str,
    count: int,
    rng: random.Random,
) -> list[tuple[RolloutEpisode, RolloutEpisode]]:
    side_a = filter_episodes(episodes, policy_a, goal_name)
    side_b = filter_episodes(episodes, policy_b, goal_name)

    if len(side_a) < 1 or len(side_b) < 1:
        raise ValueError("Not enough rollout episodes match the selected policy filters.")

    pairs = []
    attempts = 0
    while len(pairs) < count and attempts < count * 100:
        attempts += 1
        a = rng.choice(side_a)
        b = rng.choice(side_b)
        if a.episode_dir == b.episode_dir:
            continue
        if a.goal_name != b.goal_name:
            continue
        if not policy_a and not policy_b and len({episode.policy_name for episode in episodes}) > 1:
            if a.policy_name == b.policy_name:
                continue
        key = {a.episode_dir, b.episode_dir}
        if any(key == {old_a.episode_dir, old_b.episode_dir} for old_a, old_b in pairs):
            continue
        pairs.append((a, b))

    if not pairs:
        raise ValueError("Could not create any valid rollout pairs.")
    return pairs


def read_frame_or_black(capture: cv2.VideoCapture, size: tuple[int, int]) -> tuple[bool, np.ndarray]:
    ok, frame = capture.read()
    if ok:
        return True, frame
    width, height = size
    return False, np.zeros((height, width, 3), dtype=np.uint8)


def resize_with_letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 40), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def render_comparison_video(
    episode_a: RolloutEpisode,
    episode_b: RolloutEpisode,
    output_path: Path,
    side_width: int,
    fps: float,
    max_seconds: float,
) -> None:
    cap_a = cv2.VideoCapture(str(episode_a.video_path))
    cap_b = cv2.VideoCapture(str(episode_b.video_path))
    if not cap_a.isOpened():
        raise RuntimeError(f"Could not open {episode_a.video_path}.")
    if not cap_b.isOpened():
        raise RuntimeError(f"Could not open {episode_b.video_path}.")

    source_width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)) or side_width
    source_height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT)) or round(side_width * 9 / 16)
    side_height = max(1, round(source_height * (side_width / source_width)))
    output_size = (side_width * 2, side_height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open comparison video writer for {output_path}.")

    max_frames = max(1, int(fps * max_seconds))
    try:
        for _frame_index in range(max_frames):
            ok_a, frame_a = read_frame_or_black(cap_a, (side_width, side_height))
            ok_b, frame_b = read_frame_or_black(cap_b, (side_width, side_height))
            if not ok_a and not ok_b:
                break

            panel_a = resize_with_letterbox(frame_a, side_width, side_height)
            panel_b = resize_with_letterbox(frame_b, side_width, side_height)
            panel_a = label_panel(panel_a, f"A: {episode_a.policy_name} / {episode_a.goal_name} / {episode_a.episode_id}")
            panel_b = label_panel(panel_b, f"B: {episode_b.policy_name} / {episode_b.goal_name} / {episode_b.episode_id}")
            writer.write(np.hstack((panel_a, panel_b)))
    finally:
        writer.release()
        cap_a.release()
        cap_b.release()


def append_preference(
    preferences_path: Path,
    episode_a: RolloutEpisode,
    episode_b: RolloutEpisode,
    comparison_path: Path,
    winner: str,
    notes: str,
) -> None:
    preferences_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "created_at",
        "winner",
        "policy_a",
        "episode_a",
        "rollout_a",
        "policy_b",
        "episode_b",
        "rollout_b",
        "comparison_video",
        "notes",
    )
    exists = preferences_path.exists()
    with preferences_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "winner": winner,
                "policy_a": episode_a.policy_name,
                "episode_a": episode_a.episode_id,
                "rollout_a": episode_a.episode_dir.as_posix(),
                "policy_b": episode_b.policy_name,
                "episode_b": episode_b.episode_id,
                "rollout_b": episode_b.episode_dir.as_posix(),
                "comparison_video": comparison_path.as_posix(),
                "notes": notes,
            }
        )


def prompt_for_preference(comparison_path: Path) -> tuple[str, str] | None:
    print()
    print(f"Comparison video: {comparison_path}")
    print("Open/watch it, then choose: a = A better, b = B better, t = tie, s = skip, q = quit")

    while True:
        answer = input("Choice [a/b/t/s/q]: ").strip().lower()
        if answer in {"q", "quit"}:
            raise KeyboardInterrupt
        if answer in {"s", "skip"}:
            return None
        if answer in {"a", "b", "t", "tie"}:
            winner = "tie" if answer in {"t", "tie"} else answer
            notes = input("Notes (optional): ").strip()
            return winner, notes
        print("Please enter a, b, t, s, or q.")


def main() -> int:
    args = build_parser().parse_args()
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive.")
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive.")
    if args.width <= 0:
        raise SystemExit("--width must be positive.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")

    episodes = discover_rollouts(args.rollouts_dir)
    if len(episodes) < 2:
        raise SystemExit(f"Need at least two rollout episodes under {args.rollouts_dir}.")

    rng = random.Random(args.seed)
    pairs = choose_pairs(episodes, args.policy_a, args.policy_b, args.goal, args.pairs, rng)

    for index, (episode_a, episode_b) in enumerate(pairs, start=1):
        comparison_name = (
            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:03d}_"
            f"{episode_a.policy_name}_vs_{episode_b.policy_name}.mp4"
        )
        safe_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in comparison_name)
        comparison_path = args.comparisons_dir / safe_name

        render_comparison_video(
            episode_a,
            episode_b,
            comparison_path,
            side_width=args.width,
            fps=args.fps,
            max_seconds=args.max_seconds,
        )
        print(f"Rendered {comparison_path}")

        if args.no_label:
            continue

        try:
            preference = prompt_for_preference(comparison_path)
        except KeyboardInterrupt:
            print("\nStopped comparison labeling.")
            break
        if preference is None:
            continue

        winner, notes = preference
        append_preference(args.preferences_path, episode_a, episode_b, comparison_path, winner, notes)
        print(f"Saved preference to {args.preferences_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
