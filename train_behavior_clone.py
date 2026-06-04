from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from behavior_model import ValheimBehaviorNet
from valheim_actions import BUTTONS
from valheim_goals import DEFAULT_GOAL, normalize_goal_name, ordered_unique_goals, parse_goal_list, read_goal_from_meta


@dataclass
class SessionRows:
    session_dir: Path
    rows: list[dict[str, str]]
    goal_name: str
    goal_id: int


class ValheimGameplayDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        frame_width: int,
        frame_height: int,
        frame_stack: int,
        mouse_clip: float,
        default_goal: str,
        goals: tuple[str, ...] = (),
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stack = frame_stack
        self.mouse_clip = mouse_clip
        self.sessions: list[SessionRows] = []
        self.samples: list[tuple[int, int]] = []
        self.default_goal = normalize_goal_name(default_goal)
        requested_goals = tuple(normalize_goal_name(goal) for goal in goals)

        raw_sessions: list[tuple[Path, list[dict[str, str]], str]] = []
        action_paths = sorted(
            {
                *data_dir.glob("session_*/actions.csv"),
                *data_dir.glob("*/session_*/actions.csv"),
            }
        )
        for actions_path in action_paths:
            session_dir = actions_path.parent
            goal_name = read_goal_from_meta(session_dir, default_goal=self.default_goal)
            if requested_goals and goal_name not in requested_goals:
                continue

            with actions_path.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            raw_sessions.append((session_dir, rows, goal_name))

        if requested_goals:
            self.goals = requested_goals
        else:
            self.goals = ordered_unique_goals(
                [goal_name for _session_dir, _rows, goal_name in raw_sessions],
                default_goal=self.default_goal,
            )
        self.goal_to_id = {goal: index for index, goal in enumerate(self.goals)}

        for session_dir, rows, goal_name in raw_sessions:
            goal_id = self.goal_to_id[goal_name]

            session_index = len(self.sessions)
            self.sessions.append(
                SessionRows(
                    session_dir=session_dir,
                    rows=rows,
                    goal_name=self.goals[goal_id],
                    goal_id=goal_id,
                )
            )
            for row_index, row in enumerate(rows):
                if (session_dir / "frames" / row["frame"]).exists():
                    self.samples.append((session_index, row_index))

        if not self.samples:
            raise ValueError(f"No recorded samples found under {data_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        session_index, row_index = self.samples[index]
        session = self.sessions[session_index]
        frames = []

        first_index = max(0, row_index - self.frame_stack + 1)
        needed_padding = self.frame_stack - (row_index - first_index + 1)
        for _ in range(needed_padding):
            frames.append(self._load_frame(session, first_index))
        for current_index in range(first_index, row_index + 1):
            frames.append(self._load_frame(session, current_index))

        row = session.rows[row_index]
        buttons = torch.tensor(
            [float(row.get(button, "0") or 0.0) for button in BUTTONS],
            dtype=torch.float32,
        )
        mouse = torch.tensor(
            [
                self._scaled_mouse_value(row.get("mouse_dx", "0")),
                self._scaled_mouse_value(row.get("mouse_dy", "0")),
            ],
            dtype=torch.float32,
        )
        goal_id = torch.tensor(session.goal_id, dtype=torch.long)
        return torch.stack(frames), goal_id, buttons, mouse

    def _load_frame(self, session: SessionRows, row_index: int) -> torch.Tensor:
        frame_path = session.session_dir / "frames" / session.rows[row_index]["frame"]
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(frame_path)

        resized = cv2.resize(
            gray,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0
        return torch.from_numpy(normalized)

    def _scaled_mouse_value(self, value: str) -> float:
        try:
            raw = float(value or 0.0)
        except ValueError:
            raw = 0.0
        return float(np.clip(raw / self.mouse_clip, -1.0, 1.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pure-pixel behavior cloning model.")
    parser.add_argument("--data-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--model-path", default=Path("models/behavior_clone.pt"), type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--frame-width", default=160, type=int)
    parser.add_argument("--frame-height", default=96, type=int)
    parser.add_argument("--frame-stack", default=4, type=int)
    parser.add_argument("--mouse-clip", default=80.0, type=float)
    parser.add_argument("--mouse-loss-weight", default=2.0, type=float)
    parser.add_argument("--val-split", default=0.1, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--goal", default="", help="Only train one goal/behavior label.")
    parser.add_argument("--goals", default="", help="Comma-separated goals to include/order.")
    parser.add_argument("--default-goal", default=DEFAULT_GOAL, help="Goal used for legacy recordings.")
    parser.add_argument("--goal-embedding-dim", default=16, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(
    model: ValheimBehaviorNet,
    loader: DataLoader,
    device: torch.device,
    button_loss_fn: nn.Module,
    mouse_loss_fn: nn.Module,
    mouse_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_button_loss = 0.0
    total_mouse_loss = 0.0
    total_button_accuracy = 0.0
    total_examples = 0

    for frames, goal_ids, button_targets, mouse_targets in loader:
        frames = frames.to(device)
        goal_ids = goal_ids.to(device)
        button_targets = button_targets.to(device)
        mouse_targets = mouse_targets.to(device)

        button_logits, mouse_pred = model(frames, goal_ids)
        button_loss = button_loss_fn(button_logits, button_targets)
        mouse_loss = mouse_loss_fn(mouse_pred, mouse_targets)
        loss = button_loss + mouse_loss_weight * mouse_loss

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = frames.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_button_loss += float(button_loss.detach().cpu()) * batch_size
        total_mouse_loss += float(mouse_loss.detach().cpu()) * batch_size

        button_probs = torch.sigmoid(button_logits)
        button_accuracy = ((button_probs >= 0.5) == (button_targets >= 0.5)).float().mean()
        total_button_accuracy += float(button_accuracy.detach().cpu()) * batch_size

    return {
        "loss": total_loss / total_examples,
        "button_loss": total_button_loss / total_examples,
        "mouse_loss": total_mouse_loss / total_examples,
        "button_accuracy": total_button_accuracy / total_examples,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise SystemExit("--frame-width and --frame-height must be positive.")
    if args.frame_stack <= 0:
        raise SystemExit("--frame-stack must be positive.")
    if args.mouse_clip <= 0:
        raise SystemExit("--mouse-clip must be positive.")
    if not 0 <= args.val_split < 1:
        raise SystemExit("--val-split must be between 0 and 1.")
    if args.goal and args.goals:
        raise SystemExit("Use either --goal or --goals, not both.")
    if args.goal_embedding_dim <= 0:
        raise SystemExit("--goal-embedding-dim must be positive.")

    device = choose_device(args.device)
    requested_goals: tuple[str, ...] = ()
    if args.goal:
        requested_goals = parse_goal_list(args.goal, default_goal=args.default_goal)
    elif args.goals:
        requested_goals = parse_goal_list(args.goals, default_goal=args.default_goal)

    dataset = ValheimGameplayDataset(
        data_dir=args.data_dir,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        frame_stack=args.frame_stack,
        mouse_clip=args.mouse_clip,
        default_goal=args.default_goal,
        goals=requested_goals,
    )

    val_count = int(len(dataset) * args.val_split)
    train_count = len(dataset) - val_count
    if val_count:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(7),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = ValheimBehaviorNet(
        frame_stack=args.frame_stack,
        button_count=len(BUTTONS),
        goal_count=len(dataset.goals),
        goal_embedding_dim=args.goal_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    button_loss_fn = nn.BCEWithLogitsLoss()
    mouse_loss_fn = nn.MSELoss()

    print(
        f"Training on {train_count} samples, validating on {val_count}, "
        f"goals={','.join(dataset.goals)}, device={device}."
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            button_loss_fn,
            mouse_loss_fn,
            args.mouse_loss_weight,
            optimizer,
        )
        line = (
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"buttons={train_metrics['button_loss']:.4f} "
            f"mouse={train_metrics['mouse_loss']:.4f} "
            f"button_acc={train_metrics['button_accuracy']:.3f}"
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    button_loss_fn,
                    mouse_loss_fn,
                    args.mouse_loss_weight,
                )
            line += (
                f" val_loss={val_metrics['loss']:.4f}"
                f" val_button_acc={val_metrics['button_accuracy']:.3f}"
            )

        print(line)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "buttons": BUTTONS,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "frame_stack": args.frame_stack,
            "mouse_clip": args.mouse_clip,
            "threshold": args.threshold,
            "goals": dataset.goals,
            "default_goal": dataset.goals[0],
            "goal_embedding_dim": args.goal_embedding_dim,
        },
        args.model_path,
    )
    print(f"Saved model to {args.model_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
