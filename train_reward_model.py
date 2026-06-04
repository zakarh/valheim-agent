from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from preference_reward_model import (
    PixelRewardNet,
    choose_torch_device,
    read_random_video_clip,
)
from valheim_goals import DEFAULT_GOAL, normalize_goal_name, ordered_unique_goals, parse_goal_list, read_goal_from_meta


@dataclass(frozen=True)
class PreferenceExample:
    video_a: Path
    video_b: Path
    target: float
    weight: float
    winner: str
    goal_name: str


class PreferenceClipDataset(Dataset):
    def __init__(
        self,
        examples: list[PreferenceExample],
        frame_width: int,
        frame_height: int,
        clip_length: int,
        clip_stride: int,
        clips_per_preference: int,
        seed: int,
        goals: tuple[str, ...],
    ) -> None:
        if not examples:
            raise ValueError("No preference examples were provided.")

        self.examples = examples
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clips_per_preference = clips_per_preference
        self.seed = seed
        self.goals = goals
        self.goal_to_id = {goal: index for index, goal in enumerate(goals)}

    def __len__(self) -> int:
        return len(self.examples) * self.clips_per_preference

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index % len(self.examples)]
        rng = random.Random(self.seed + index + random.randint(0, 1_000_000))

        clip_a = read_random_video_clip(
            example.video_a,
            self.frame_width,
            self.frame_height,
            self.clip_length,
            self.clip_stride,
            rng,
        )
        clip_b = read_random_video_clip(
            example.video_b,
            self.frame_width,
            self.frame_height,
            self.clip_length,
            self.clip_stride,
            rng,
        )
        goal_id = torch.tensor(self.goal_to_id[example.goal_name], dtype=torch.long)
        target = torch.tensor(example.target, dtype=torch.float32)
        weight = torch.tensor(example.weight, dtype=torch.float32)
        return clip_a, clip_b, goal_id, target, weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pure-pixel reward model from A/B preferences.")
    parser.add_argument("--preferences-path", default=Path("preferences/preferences.csv"), type=Path)
    parser.add_argument("--model-path", default=Path("models/reward_model.pt"), type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--frame-width", default=160, type=int)
    parser.add_argument("--frame-height", default=96, type=int)
    parser.add_argument("--clip-length", default=16, type=int)
    parser.add_argument("--clip-stride", default=2, type=int)
    parser.add_argument("--clips-per-preference", default=8, type=int)
    parser.add_argument("--tie-weight", default=0.35, type=float)
    parser.add_argument("--val-split", default=0.2, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--goal", default="", help="Only train reward preferences for one goal.")
    parser.add_argument("--goals", default="", help="Comma-separated goals to include/order.")
    parser.add_argument("--default-goal", default=DEFAULT_GOAL, help="Goal used for legacy rollouts.")
    parser.add_argument("--goal-embedding-dim", default=16, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def resolve_rollout_video(raw_path: str, preferences_path: Path) -> Path:
    rollout_dir = Path(raw_path)
    candidates = [rollout_dir / "episode.mp4"]
    if not rollout_dir.is_absolute():
        candidates.append(Path.cwd() / rollout_dir / "episode.mp4")
        candidates.append(preferences_path.parent.parent / rollout_dir / "episode.mp4")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find episode.mp4 for rollout path: {raw_path}")


def resolve_rollout_dir(raw_path: str, preferences_path: Path) -> Path:
    rollout_dir = Path(raw_path)
    candidates = [rollout_dir]
    if not rollout_dir.is_absolute():
        candidates.append(Path.cwd() / rollout_dir)
        candidates.append(preferences_path.parent.parent / rollout_dir)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find rollout path: {raw_path}")


def load_preferences(
    preferences_path: Path,
    tie_weight: float,
    default_goal: str,
    goals: tuple[str, ...] = (),
) -> tuple[list[PreferenceExample], int]:
    if not preferences_path.exists():
        raise FileNotFoundError(preferences_path)

    examples: list[PreferenceExample] = []
    allowed_goals = {normalize_goal_name(goal) for goal in goals}
    skipped_cross_goal = 0
    with preferences_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            winner = (row.get("winner") or "").strip().lower()
            if winner not in {"a", "b", "tie"}:
                continue

            video_a = resolve_rollout_video(row["rollout_a"], preferences_path)
            video_b = resolve_rollout_video(row["rollout_b"], preferences_path)
            rollout_a = resolve_rollout_dir(row["rollout_a"], preferences_path)
            rollout_b = resolve_rollout_dir(row["rollout_b"], preferences_path)
            goal_a = read_goal_from_meta(rollout_a, default_goal=default_goal)
            goal_b = read_goal_from_meta(rollout_b, default_goal=default_goal)
            raw_row_goal = (row.get("goal") or "").strip()
            row_goal = normalize_goal_name(raw_row_goal) if raw_row_goal else ""
            goal_name = row_goal or goal_a
            if goal_a != goal_b and not row_goal:
                skipped_cross_goal += 1
                continue
            if allowed_goals and goal_name not in allowed_goals:
                continue

            if winner == "a":
                target = 1.0
                weight = 1.0
            elif winner == "b":
                target = 0.0
                weight = 1.0
            else:
                target = 0.5
                weight = tie_weight

            examples.append(
                PreferenceExample(
                    video_a=video_a,
                    video_b=video_b,
                    target=target,
                    weight=weight,
                    winner=winner,
                    goal_name=goal_name,
                )
            )

    if not examples:
        raise ValueError(f"No usable preferences found in {preferences_path}.")

    return examples, skipped_cross_goal


def run_epoch(
    model: PixelRewardNet,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_non_ties = 0.0
    total_examples = 0

    for clip_a, clip_b, goal_ids, targets, weights in loader:
        clip_a = clip_a.to(device)
        clip_b = clip_b.to(device)
        goal_ids = goal_ids.to(device)
        targets = targets.to(device)
        weights = weights.to(device)

        score_a = model(clip_a, goal_ids)
        score_b = model(clip_b, goal_ids)
        logits = score_a - score_b

        losses = loss_fn(logits, targets)
        loss = torch.mean(losses * weights)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = clip_a.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size

        non_tie_mask = targets != 0.5
        if torch.any(non_tie_mask):
            predictions = (torch.sigmoid(logits[non_tie_mask]) >= 0.5).float()
            correct = (predictions == targets[non_tie_mask]).float().sum()
            count = float(non_tie_mask.float().sum().detach().cpu())
            total_accuracy += float(correct.detach().cpu())
            total_non_ties += count

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_accuracy / max(1.0, total_non_ties),
        "non_tie_examples": total_non_ties,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise SystemExit("--frame-width and --frame-height must be positive.")
    if args.clip_length <= 0:
        raise SystemExit("--clip-length must be positive.")
    if args.clip_stride <= 0:
        raise SystemExit("--clip-stride must be positive.")
    if args.clips_per_preference <= 0:
        raise SystemExit("--clips-per-preference must be positive.")
    if not 0 <= args.val_split < 1:
        raise SystemExit("--val-split must be between 0 and 1.")
    if args.goal and args.goals:
        raise SystemExit("Use either --goal or --goals, not both.")
    if args.goal_embedding_dim <= 0:
        raise SystemExit("--goal-embedding-dim must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    requested_goals: tuple[str, ...] = ()
    if args.goal:
        requested_goals = parse_goal_list(args.goal, default_goal=args.default_goal)
    elif args.goals:
        requested_goals = parse_goal_list(args.goals, default_goal=args.default_goal)

    examples, skipped_cross_goal = load_preferences(
        args.preferences_path,
        args.tie_weight,
        args.default_goal,
        requested_goals,
    )
    goal_names = requested_goals or ordered_unique_goals(
        [example.goal_name for example in examples],
        default_goal=args.default_goal,
    )
    dataset = PreferenceClipDataset(
        examples=examples,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        clips_per_preference=args.clips_per_preference,
        seed=args.seed,
        goals=goal_names,
    )

    val_count = int(len(dataset) * args.val_split)
    train_count = len(dataset) - val_count
    if val_count:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    device = choose_torch_device(args.device)
    model = PixelRewardNet(
        clip_length=args.clip_length,
        goal_count=len(goal_names),
        goal_embedding_dim=args.goal_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    print(
        f"Training reward model from {len(examples)} preferences "
        f"({train_count} train clips, {val_count} val clips), "
        f"goals={','.join(goal_names)}, device={device}."
    )
    if skipped_cross_goal:
        print(f"Skipped {skipped_cross_goal} cross-goal preferences.")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, loss_fn, optimizer)
        line = (
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_pref_acc={train_metrics['accuracy']:.3f}"
        )
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device, loss_fn)
            line += f" val_loss={val_metrics['loss']:.4f} val_pref_acc={val_metrics['accuracy']:.3f}"
        print(line)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "clip_length": args.clip_length,
            "clip_stride": args.clip_stride,
            "preference_count": len(examples),
            "goals": goal_names,
            "default_goal": goal_names[0],
            "goal_embedding_dim": args.goal_embedding_dim,
        },
        args.model_path,
    )
    print(f"Saved reward model to {args.model_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
