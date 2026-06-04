from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from valheim_goals import DEFAULT_GOAL, normalize_goal_name, resolve_goal_id


class PixelRewardNet(nn.Module):
    """Scores a short pure-pixel clip with one scalar reward/preference value."""

    def __init__(
        self,
        clip_length: int,
        goal_count: int = 1,
        goal_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.clip_length = clip_length
        self.goal_count = goal_count
        self.goal_embedding_dim = goal_embedding_dim if goal_count > 1 else 0

        encoder_layers: list[nn.Module] = [
            nn.Conv2d(clip_length, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        ]
        if goal_count <= 1:
            encoder_layers.append(nn.Linear(256, 1))
        self.encoder = nn.Sequential(
            *encoder_layers,
        )
        if goal_count > 1:
            self.goal_embedding = nn.Embedding(goal_count, goal_embedding_dim)
            self.reward_head = nn.Linear(256 + goal_embedding_dim, 1)
        else:
            self.goal_embedding = None
            self.reward_head = None

    def forward(
        self,
        clips: torch.Tensor,
        goal_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(clips)
        if self.goal_embedding is not None and self.reward_head is not None:
            if goal_ids is None:
                goal_ids = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            goal_features = self.goal_embedding(goal_ids.long())
            features = torch.cat((features, goal_features), dim=1)
            return self.reward_head(features).squeeze(-1)
        return features.squeeze(-1)


def choose_torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def preprocess_bgr_frame(frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_bgra_frame(frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    return preprocess_bgr_frame(frame[:, :, :3], frame_width, frame_height)


def video_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    finally:
        capture.release()


def read_video_clip(
    video_path: Path,
    frame_width: int,
    frame_height: int,
    clip_length: int,
    start_frame: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        last_frame: np.ndarray | None = None

        while len(frames) < clip_length:
            ok, frame = capture.read()
            if not ok:
                if last_frame is None:
                    raise RuntimeError(f"No readable frames in video: {video_path}")
                frames.append(last_frame)
                continue

            processed = preprocess_bgr_frame(frame, frame_width, frame_height)
            frames.append(processed)
            last_frame = processed

            for _ in range(stride - 1):
                capture.grab()
    finally:
        capture.release()

    return torch.from_numpy(np.stack(frames, axis=0)).float()


def read_random_video_clip(
    video_path: Path,
    frame_width: int,
    frame_height: int,
    clip_length: int,
    stride: int,
    rng: random.Random,
) -> torch.Tensor:
    total_frames = video_frame_count(video_path)
    needed_frames = max(1, clip_length * stride)
    max_start = max(0, total_frames - needed_frames)
    start_frame = rng.randint(0, max_start) if max_start else 0
    return read_video_clip(
        video_path=video_path,
        frame_width=frame_width,
        frame_height=frame_height,
        clip_length=clip_length,
        start_frame=start_frame,
        stride=stride,
    )


def checkpoint_goals(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    goals = checkpoint.get("goals") or checkpoint.get("behaviors") or (DEFAULT_GOAL,)
    return tuple(normalize_goal_name(goal) for goal in goals)


def checkpoint_default_goal(checkpoint: dict[str, Any]) -> str:
    goals = checkpoint_goals(checkpoint)
    return normalize_goal_name(checkpoint.get("default_goal") or goals[0])


def checkpoint_goal_id(checkpoint: dict[str, Any], requested_goal: str | None = "") -> int:
    return resolve_goal_id(checkpoint_goals(checkpoint), requested_goal, checkpoint_default_goal(checkpoint))


def load_reward_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[PixelRewardNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    clip_length = int(checkpoint["clip_length"])
    goals = checkpoint_goals(checkpoint)
    goal_embedding_dim = int(checkpoint.get("goal_embedding_dim", 16))

    model = PixelRewardNet(
        clip_length=clip_length,
        goal_count=len(goals),
        goal_embedding_dim=goal_embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


class LearnedRewardScorer:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        self.model, self.checkpoint = load_reward_checkpoint(model_path, device)
        self.device = device
        self.frame_width = int(self.checkpoint["frame_width"])
        self.frame_height = int(self.checkpoint["frame_height"])
        self.clip_length = int(self.checkpoint["clip_length"])
        self.goals = checkpoint_goals(self.checkpoint)
        self.default_goal = checkpoint_default_goal(self.checkpoint)

    def preprocess_live_frame(self, bgra: np.ndarray) -> np.ndarray:
        return preprocess_bgra_frame(bgra, self.frame_width, self.frame_height)

    def goal_id(self, goal_name: str | None = "") -> int:
        return resolve_goal_id(self.goals, goal_name, self.default_goal)

    def score_clip(
        self,
        frames: list[np.ndarray] | tuple[np.ndarray, ...],
        goal_name: str | None = "",
    ) -> float:
        if len(frames) != self.clip_length:
            raise ValueError(f"Expected {self.clip_length} frames, got {len(frames)}.")

        clip = torch.from_numpy(np.stack(frames, axis=0)).float().unsqueeze(0).to(self.device)
        goal_ids = torch.tensor([self.goal_id(goal_name)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return float(self.model(clip, goal_ids).squeeze(0).detach().cpu())
