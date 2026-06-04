from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from valheim_capture import DEFAULT_PROCESS_NAME
from valheim_goals import DEFAULT_GOAL, parse_goal_list
from valheim_rl_env import ValheimPreferenceEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a live Valheim policy with a learned human-preference reward model."
    )
    parser.add_argument("--reward-model-path", default=Path("models/reward_model.pt"), type=Path)
    parser.add_argument("--model-path", default=Path("models/rlhf_dqn.zip"), type=Path)
    parser.add_argument("--checkpoint-dir", default=Path("models/rlhf_checkpoints"), type=Path)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--timesteps", default=10_000, type=int)
    parser.add_argument("--step-seconds", default=0.15, type=float)
    parser.add_argument("--max-episode-steps", default=500, type=int)
    parser.add_argument("--obs-width", default=84, type=int)
    parser.add_argument("--obs-height", default=84, type=int)
    parser.add_argument("--obs-stack", default=4, type=int)
    parser.add_argument("--reward-mode", choices=("delta", "score"), default="delta")
    parser.add_argument("--reward-scale", default=1.0, type=float)
    parser.add_argument("--reward-clip", default=2.0, type=float)
    parser.add_argument("--goal", default="", help="Fixed goal to train. Defaults to general.")
    parser.add_argument("--goals", default="", help="Comma-separated goals for one conditioned DQN.")
    parser.add_argument("--learning-starts", default=1_000, type=int)
    parser.add_argument("--buffer-size", default=50_000, type=int)
    parser.add_argument("--checkpoint-every", default=2_500, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-input", action="store_true", help="Do not send controls; useful for smoke testing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive.")
    if args.goal and args.goals:
        raise SystemExit("Use either --goal or --goals, not both.")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    goals = parse_goal_list(args.goals or args.goal, default_goal=DEFAULT_GOAL)

    env = Monitor(
        ValheimPreferenceEnv(
            reward_model_path=str(args.reward_model_path),
            process_name=args.process_name,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            obs_stack=args.obs_stack,
            step_seconds=args.step_seconds,
            max_episode_steps=args.max_episode_steps,
            reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
            reward_clip=args.reward_clip,
            goal=goals[0],
            goals=goals,
            send_inputs=not args.no_input,
            device=args.device,
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(args.checkpoint_dir),
        name_prefix="rlhf_dqn",
        save_replay_buffer=True,
    )

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    finally:
        model.save(args.model_path)
        metadata_path = args.model_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "policy": "dqn",
                    "model_path": str(args.model_path),
                    "reward_model_path": str(args.reward_model_path),
                    "obs_width": args.obs_width,
                    "obs_height": args.obs_height,
                    "obs_stack": args.obs_stack,
                    "goals": goals,
                    "default_goal": goals[0],
                    "goal_channels": len(goals) if len(goals) > 1 else 0,
                    "reward_mode": args.reward_mode,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env.close()

    print(f"Saved RLHF policy to {args.model_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
