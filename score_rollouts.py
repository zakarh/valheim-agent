from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from compare_rollouts import discover_rollouts
from preference_reward_model import (
    checkpoint_default_goal,
    checkpoint_goal_id,
    checkpoint_goals,
    choose_torch_device,
    load_reward_checkpoint,
    read_video_clip,
    video_frame_count,
)
from valheim_goals import normalize_goal_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score recorded Valheim rollouts with a learned reward model.")
    parser.add_argument("--rollouts-dir", default=Path("rollouts"), type=Path)
    parser.add_argument("--reward-model-path", default=Path("models/reward_model.pt"), type=Path)
    parser.add_argument("--output-path", default=Path("preferences/rollout_scores.csv"), type=Path)
    parser.add_argument("--clips-per-rollout", default=8, type=int)
    parser.add_argument("--goal", default="", help="Override the goal used for scoring every rollout.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def clip_start_frames(total_frames: int, clip_length: int, stride: int, count: int) -> list[int]:
    needed = clip_length * stride
    max_start = max(0, total_frames - needed)
    if count <= 1 or max_start == 0:
        return [0]
    return [round(index * max_start / (count - 1)) for index in range(count)]


def score_rollout(
    model,
    video_path: Path,
    frame_width: int,
    frame_height: int,
    clip_length: int,
    clip_stride: int,
    clips_per_rollout: int,
    goal_id: int,
    device: torch.device,
) -> tuple[float, float, int]:
    total_frames = video_frame_count(video_path)
    starts = clip_start_frames(total_frames, clip_length, clip_stride, clips_per_rollout)
    scores = []

    with torch.no_grad():
        for start_frame in starts:
            clip = read_video_clip(
                video_path=video_path,
                frame_width=frame_width,
                frame_height=frame_height,
                clip_length=clip_length,
                start_frame=start_frame,
                stride=clip_stride,
            )
            goal_ids = torch.tensor([goal_id], dtype=torch.long, device=device)
            score = model(clip.unsqueeze(0).to(device), goal_ids)
            scores.append(float(score.squeeze(0).detach().cpu()))

    mean_score = sum(scores) / max(1, len(scores))
    score_range = max(scores) - min(scores) if scores else 0.0
    return mean_score, score_range, len(scores)


def main() -> int:
    args = build_parser().parse_args()
    if args.clips_per_rollout <= 0:
        raise SystemExit("--clips-per-rollout must be positive.")

    episodes = discover_rollouts(args.rollouts_dir)
    if not episodes:
        raise SystemExit(f"No rollout episodes found under {args.rollouts_dir}.")

    device = choose_torch_device(args.device)
    model, checkpoint = load_reward_checkpoint(args.reward_model_path, device)
    frame_width = int(checkpoint["frame_width"])
    frame_height = int(checkpoint["frame_height"])
    clip_length = int(checkpoint["clip_length"])
    clip_stride = int(checkpoint.get("clip_stride", 1))
    goals = checkpoint_goals(checkpoint)
    default_goal = checkpoint_default_goal(checkpoint)

    rows = []
    for episode in episodes:
        goal_name = normalize_goal_name(args.goal or episode.goal_name)
        goal_id = checkpoint_goal_id(checkpoint, goal_name or default_goal)
        mean_score, score_range, clip_count = score_rollout(
            model=model,
            video_path=episode.video_path,
            frame_width=frame_width,
            frame_height=frame_height,
            clip_length=clip_length,
            clip_stride=clip_stride,
            clips_per_rollout=args.clips_per_rollout,
            goal_id=goal_id,
            device=device,
        )
        rows.append(
            {
                "policy_name": episode.policy_name,
                "goal": goals[goal_id],
                "episode_id": episode.episode_id,
                "rollout": episode.episode_dir.as_posix(),
                "score": mean_score,
                "score_range": score_range,
                "clip_count": clip_count,
            }
        )

    rows.sort(key=lambda row: float(row["score"]), reverse=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("policy_name", "goal", "episode_id", "rollout", "score", "score_range", "clip_count"),
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['score']:+.3f} {row['policy_name']} {row['goal']} {row['episode_id']}")

    print(f"Saved scores to {args.output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
