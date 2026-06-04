from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from valheim_actions import BUTTONS
from valheim_goals import DEFAULT_GOAL, normalize_goal_name, resolve_goal_id


class ValheimBehaviorNet(nn.Module):
    def __init__(
        self,
        frame_stack: int,
        button_count: int = len(BUTTONS),
        goal_count: int = 1,
        goal_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.goal_count = goal_count
        self.goal_embedding_dim = goal_embedding_dim if goal_count > 1 else 0

        self.encoder = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )
        if goal_count > 1:
            self.goal_embedding = nn.Embedding(goal_count, goal_embedding_dim)
        else:
            self.goal_embedding = None

        head_width = 256 + self.goal_embedding_dim
        self.button_head = nn.Linear(head_width, button_count)
        self.mouse_head = nn.Sequential(
            nn.Linear(head_width, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        frames: torch.Tensor,
        goal_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(frames)
        if self.goal_embedding is not None:
            if goal_ids is None:
                goal_ids = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            goal_features = self.goal_embedding(goal_ids.long())
            features = torch.cat((features, goal_features), dim=1)
        return self.button_head(features), self.mouse_head(features)


def checkpoint_goals(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    goals = checkpoint.get("goals") or checkpoint.get("behaviors") or (DEFAULT_GOAL,)
    return tuple(normalize_goal_name(goal) for goal in goals)


def checkpoint_default_goal(checkpoint: dict[str, Any]) -> str:
    goals = checkpoint_goals(checkpoint)
    return normalize_goal_name(checkpoint.get("default_goal") or goals[0])


def checkpoint_goal_id(checkpoint: dict[str, Any], requested_goal: str | None = "") -> int:
    return resolve_goal_id(checkpoint_goals(checkpoint), requested_goal, checkpoint_default_goal(checkpoint))


def load_behavior_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[ValheimBehaviorNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    buttons = tuple(checkpoint.get("buttons", BUTTONS))
    frame_stack = int(checkpoint["frame_stack"])
    goals = checkpoint_goals(checkpoint)
    goal_embedding_dim = int(checkpoint.get("goal_embedding_dim", 16))

    model = ValheimBehaviorNet(
        frame_stack=frame_stack,
        button_count=len(buttons),
        goal_count=len(goals),
        goal_embedding_dim=goal_embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint
