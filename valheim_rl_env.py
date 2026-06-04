from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from preference_reward_model import LearnedRewardScorer, choose_torch_device
from valheim_actions import BUTTONS
from valheim_capture import (
    DEFAULT_PROCESS_NAME,
    find_window_client_area,
    process_is_running,
    set_dpi_awareness,
    wait_for_process,
    wait_for_window_client_area,
)
from valheim_control import apply_predictions, release_held_inputs
from valheim_goals import DEFAULT_GOAL, normalize_goal_name, parse_goal_list
from valheim_rl_actions import RL_ACTIONS, action_to_predictions


class ValheimPreferenceEnv(gym.Env):
    """Live Valheim environment rewarded by a learned human-preference model."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        reward_model_path: str,
        process_name: str = DEFAULT_PROCESS_NAME,
        obs_width: int = 84,
        obs_height: int = 84,
        obs_stack: int = 4,
        step_seconds: float = 0.15,
        max_episode_steps: int = 500,
        reward_mode: str = "delta",
        reward_scale: float = 1.0,
        reward_clip: float = 2.0,
        goal: str = DEFAULT_GOAL,
        goals: tuple[str, ...] | None = None,
        mouse_clip: float = 80.0,
        mouse_gain: float = 1.0,
        mouse_deadzone: float = 2.0,
        tap_cooldown: float = 0.25,
        process_check_every: float = 2.0,
        window_check_every: float = 0.5,
        send_inputs: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__()

        if obs_width <= 0 or obs_height <= 0:
            raise ValueError("Observation width/height must be positive.")
        if obs_stack <= 0:
            raise ValueError("obs_stack must be positive.")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive.")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive.")
        if reward_mode not in {"delta", "score"}:
            raise ValueError('reward_mode must be "delta" or "score".')

        self.process_name = process_name
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.obs_stack_size = obs_stack
        self.step_seconds = step_seconds
        self.max_episode_steps = max_episode_steps
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.goals = tuple(normalize_goal_name(raw_goal) for raw_goal in goals) if goals else parse_goal_list(goal)
        self.current_goal_index = 0
        self.current_goal = self.goals[0]
        self.mouse_clip = mouse_clip
        self.mouse_gain = mouse_gain
        self.mouse_deadzone = mouse_deadzone
        self.tap_cooldown = tap_cooldown
        self.process_check_every = process_check_every
        self.window_check_every = window_check_every

        self.torch_device = choose_torch_device(device)
        self.reward_scorer = LearnedRewardScorer(
            model_path=Path(reward_model_path),
            device=self.torch_device,
        )
        for goal_name in self.goals:
            self.reward_scorer.goal_id(goal_name)

        goal_channels = len(self.goals) if len(self.goals) > 1 else 0
        self.action_space = spaces.Discrete(len(RL_ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_stack + goal_channels, obs_height, obs_width),
            dtype=np.uint8,
        )

        self.sct = None
        self.capture_area: dict[str, int] | None = None
        self.window_title: str | None = None
        self.obs_frames: deque[np.ndarray] = deque(maxlen=obs_stack)
        self.reward_frames: deque[np.ndarray] = deque(maxlen=self.reward_scorer.clip_length)
        self.last_reward_score = 0.0
        self.last_observation: np.ndarray | None = None
        self.steps = 0
        self.next_process_check_at = 0.0
        self.next_window_check_at = 0.0
        self.last_tap_at: dict[str, float] = {}
        self.held_buttons: set[str] = set()

        self.input_backend = None
        if send_inputs:
            try:
                import pydirectinput
            except ImportError:
                raise ImportError(
                    "Missing dependency: pydirectinput. Install dependencies with "
                    "python -m pip install -r requirements.txt"
                ) from None

            pydirectinput.PAUSE = 0
            self.input_backend = pydirectinput

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if len(self.goals) > 1:
            self.current_goal_index = int(self.np_random.integers(len(self.goals)))
        else:
            self.current_goal_index = 0
        self.current_goal = self.goals[self.current_goal_index]

        set_dpi_awareness()

        if self.sct is None:
            import mss

            self.sct = mss.MSS()

        wait_for_process(self.process_name, poll_interval=1.0)
        self.capture_area, self.window_title = wait_for_window_client_area(
            self.process_name,
            poll_interval=1.0,
        )

        release_held_inputs(self.input_backend, self.held_buttons)
        self.last_tap_at.clear()

        raw_frame = self._capture_raw_frame()
        obs_frame = self._preprocess_observation_frame(raw_frame)
        reward_frame = self.reward_scorer.preprocess_live_frame(raw_frame)

        self.obs_frames.clear()
        self.reward_frames.clear()
        for _ in range(self.obs_stack_size):
            self.obs_frames.append(obs_frame)
        for _ in range(self.reward_scorer.clip_length):
            self.reward_frames.append(reward_frame)

        self.last_reward_score = self.reward_scorer.score_clip(tuple(self.reward_frames), self.current_goal)
        self.last_observation = self._stack_observation()
        self.steps = 0

        now = time.perf_counter()
        self.next_process_check_at = now + self.process_check_every
        self.next_window_check_at = now + self.window_check_every

        return self.last_observation, self._info(reward_score=self.last_reward_score)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.last_observation is None:
            observation, info = self.reset()
            return observation, 0.0, False, False, info

        now = time.perf_counter()
        if now >= self.next_process_check_at:
            if not process_is_running(self.process_name):
                release_held_inputs(self.input_backend, self.held_buttons)
                return self.last_observation, 0.0, True, False, self._info(error="process exited")
            self.next_process_check_at = now + self.process_check_every

        if now >= self.next_window_check_at:
            target = find_window_client_area(self.process_name)
            if target is None:
                release_held_inputs(self.input_backend, self.held_buttons)
                return self.last_observation, 0.0, True, False, self._info(error="window unavailable")
            self.capture_area, self.window_title = target
            self.next_window_check_at = now + self.window_check_every

        probabilities, mouse = action_to_predictions(int(action))
        action_row = apply_predictions(
            self.input_backend,
            BUTTONS,
            probabilities,
            mouse,
            threshold=0.5,
            mouse_clip=self.mouse_clip,
            mouse_gain=self.mouse_gain,
            mouse_deadzone=self.mouse_deadzone,
            tap_cooldown=self.tap_cooldown,
            last_tap_at=self.last_tap_at,
            held_buttons=self.held_buttons,
        )

        time.sleep(self.step_seconds)
        release_held_inputs(self.input_backend, self.held_buttons)

        raw_frame = self._capture_raw_frame()
        self.obs_frames.append(self._preprocess_observation_frame(raw_frame))
        self.reward_frames.append(self.reward_scorer.preprocess_live_frame(raw_frame))
        observation = self._stack_observation()
        self.last_observation = observation

        current_score = self.reward_scorer.score_clip(tuple(self.reward_frames), self.current_goal)
        if self.reward_mode == "delta":
            reward = current_score - self.last_reward_score
        else:
            reward = current_score
        self.last_reward_score = current_score
        reward = float(np.clip(reward * self.reward_scale, -self.reward_clip, self.reward_clip))

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        return observation, reward, terminated, truncated, self._info(
            action=RL_ACTIONS[int(action)].name,
            action_row=action_row,
            reward_score=current_score,
        )

    def render(self) -> np.ndarray | None:
        if self.last_observation is None:
            return None
        latest_gray = self.last_observation[-1]
        return cv2.cvtColor(latest_gray, cv2.COLOR_GRAY2RGB)

    def close(self) -> None:
        release_held_inputs(self.input_backend, self.held_buttons)
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def _capture_raw_frame(self) -> np.ndarray:
        if self.sct is None or self.capture_area is None:
            raise RuntimeError("Environment is not connected. Call reset() first.")
        return np.asarray(self.sct.grab(self.capture_area)).copy()

    def _preprocess_observation_frame(self, bgra: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _stack_observation(self) -> np.ndarray:
        observation = np.stack(tuple(self.obs_frames), axis=0).astype(np.uint8)
        if len(self.goals) <= 1:
            return observation

        goal_planes = np.zeros((len(self.goals), self.obs_height, self.obs_width), dtype=np.uint8)
        goal_planes[self.current_goal_index].fill(255)
        return np.concatenate((observation, goal_planes), axis=0)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "window_title": self.window_title,
            "capture_area": self.capture_area,
            "steps": self.steps,
            "goal": self.current_goal,
        }
        info.update(extra)
        return info
