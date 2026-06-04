from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from valheim_actions import ACTION_COLUMNS


class RolloutRecorder:
    def __init__(
        self,
        output_dir: Path,
        policy_name: str,
        fps: float,
        frame_width: int,
        jpeg_quality: int = 85,
        save_frames: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if frame_width <= 0:
            raise ValueError("frame_width must be positive.")

        self.output_dir = output_dir
        self.policy_name = policy_name
        self.fps = fps
        self.frame_width = frame_width
        self.jpeg_quality = jpeg_quality
        self.save_frames = save_frames

        self.session_dir = self._make_session_dir(output_dir, policy_name)
        self.frames_dir = self.session_dir / "frames"
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.session_dir / "episode.mp4"
        self.actions_path = self.session_dir / "actions.csv"
        self.meta_path = self.session_dir / "meta.json"
        self.thumbnail_path = self.session_dir / "thumbnail.jpg"

        self.csv_file = None
        self.writer: csv.DictWriter | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.frame_index = 0
        self.started_at: float | None = None

    def __enter__(self) -> "RolloutRecorder":
        self.csv_file = self.actions_path.open("w", newline="", encoding="utf-8")
        fieldnames = (
            "frame_index",
            "timestamp",
            "elapsed",
            "window_left",
            "window_top",
            "window_width",
            "window_height",
        ) + ACTION_COLUMNS
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
        if self.csv_file is not None:
            self.csv_file.close()

    def write_meta(self, metadata: dict[str, Any]) -> None:
        payload = {
            "policy_name": self.policy_name,
            "fps": self.fps,
            "stored_frame_width": self.frame_width,
            "video": self.video_path.name,
            "actions": self.actions_path.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        payload.update(metadata)
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append(
        self,
        bgra: np.ndarray,
        action_row: dict[str, float | int],
        capture_area: dict[str, int],
        timestamp: float,
        elapsed: float,
    ) -> None:
        if self.writer is None:
            raise RuntimeError("RolloutRecorder must be used as a context manager.")

        bgr = self._resize_for_video(bgra)
        if self.video_writer is None:
            self._open_video_writer(bgr)

        assert self.video_writer is not None
        self.video_writer.write(bgr)

        if self.frame_index == 0:
            cv2.imwrite(str(self.thumbnail_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        if self.save_frames:
            frame_path = self.frames_dir / f"{self.frame_index:08d}.jpg"
            cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        row = {
            "frame_index": self.frame_index,
            "timestamp": timestamp,
            "elapsed": elapsed,
            "window_left": capture_area["left"],
            "window_top": capture_area["top"],
            "window_width": capture_area["width"],
            "window_height": capture_area["height"],
        }
        for column in ACTION_COLUMNS:
            row[column] = action_row.get(column, 0)
        self.writer.writerow(row)
        self.frame_index += 1

    def _open_video_writer(self, bgr: np.ndarray) -> None:
        height, width = bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.video_path}.")

    def _resize_for_video(self, bgra: np.ndarray) -> np.ndarray:
        bgr = bgra[:, :, :3]
        if self.frame_width == bgr.shape[1]:
            return bgr

        target_height = max(1, round(bgr.shape[0] * (self.frame_width / bgr.shape[1])))
        return cv2.resize(bgr, (self.frame_width, target_height), interpolation=cv2.INTER_AREA)

    def _make_session_dir(self, output_dir: Path, policy_name: str) -> Path:
        safe_policy = "".join(char if char.isalnum() or char in "-_" else "_" for char in policy_name)
        session_name = datetime.now().strftime("episode_%Y%m%d_%H%M%S")
        session_dir = output_dir / safe_policy / session_name
        session_dir.mkdir(parents=True, exist_ok=False)
        return session_dir
