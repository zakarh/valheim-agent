from __future__ import annotations

import json
from pathlib import Path


DEFAULT_GOAL = "general"


def normalize_goal_name(raw_goal: str | None) -> str:
    cleaned = " ".join(str(raw_goal or "").strip().split()).lower()
    if not cleaned:
        return DEFAULT_GOAL
    return cleaned.replace(" ", "_")


def parse_goal_list(raw_goals: str | None, default_goal: str = DEFAULT_GOAL) -> tuple[str, ...]:
    if not raw_goals:
        return (normalize_goal_name(default_goal),)

    goals: list[str] = []
    seen: set[str] = set()
    for raw_goal in raw_goals.split(","):
        goal = normalize_goal_name(raw_goal)
        if goal in seen:
            continue
        goals.append(goal)
        seen.add(goal)

    return tuple(goals or (normalize_goal_name(default_goal),))


def ordered_unique_goals(raw_goals: list[str] | tuple[str, ...], default_goal: str = DEFAULT_GOAL) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_goal in (default_goal, *raw_goals):
        goal = normalize_goal_name(raw_goal)
        if goal in seen:
            continue
        ordered.append(goal)
        seen.add(goal)
    return tuple(ordered)


def resolve_goal_id(
    goals: tuple[str, ...],
    requested_goal: str | None = "",
    default_goal: str | None = "",
) -> int:
    if not goals:
        return 0

    normalized_goals = tuple(normalize_goal_name(goal) for goal in goals)
    if len(normalized_goals) == 1:
        return 0

    goal = normalize_goal_name(requested_goal or default_goal or normalized_goals[0])
    if goal not in normalized_goals:
        available = ", ".join(normalized_goals)
        raise ValueError(f"Unknown goal '{goal}'. Available goals: {available}")
    return normalized_goals.index(goal)


def read_goal_from_meta(path: Path, default_goal: str = DEFAULT_GOAL) -> str:
    meta_path = path / "meta.json" if path.is_dir() else path
    if not meta_path.exists():
        return normalize_goal_name(default_goal)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return normalize_goal_name(default_goal)

    return normalize_goal_name(
        meta.get("goal")
        or meta.get("behavior")
        or meta.get("behavior_name")
        or default_goal
    )


def safe_goal_path(goal: str) -> str:
    normalized = normalize_goal_name(goal)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in normalized)
