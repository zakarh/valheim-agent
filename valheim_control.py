from __future__ import annotations

import time

import numpy as np

from valheim_actions import BUTTONS, HELD_KEYS, MOUSE_BUTTONS, TAP_KEYS, empty_button_state


def release_held_inputs(input_backend, held_buttons: set[str]) -> None:
    if input_backend is None:
        held_buttons.clear()
        return

    for button in tuple(held_buttons):
        input_backend.keyUp(button)
        held_buttons.remove(button)


def apply_predictions(
    input_backend,
    buttons: tuple[str, ...],
    probabilities: np.ndarray,
    mouse: np.ndarray,
    threshold: float,
    mouse_clip: float,
    mouse_gain: float,
    mouse_deadzone: float,
    tap_cooldown: float,
    last_tap_at: dict[str, float],
    held_buttons: set[str],
) -> dict[str, float | int]:
    now = time.perf_counter()
    probability_by_button = dict(zip(buttons, probabilities, strict=True))
    sent = empty_button_state()

    for button in HELD_KEYS:
        should_hold = probability_by_button.get(button, 0.0) >= threshold
        sent[button] = int(should_hold)

        if input_backend is None:
            continue
        if should_hold and button not in held_buttons:
            input_backend.keyDown(button)
            held_buttons.add(button)
        elif not should_hold and button in held_buttons:
            input_backend.keyUp(button)
            held_buttons.remove(button)

    for button in TAP_KEYS:
        if probability_by_button.get(button, 0.0) < threshold:
            continue
        if now - last_tap_at.get(button, 0.0) < tap_cooldown:
            continue

        sent[button] = 1
        if input_backend is not None:
            input_backend.press(button)
        last_tap_at[button] = now

    for button, directinput_name in MOUSE_BUTTONS.items():
        if probability_by_button.get(button, 0.0) < threshold:
            continue
        if now - last_tap_at.get(button, 0.0) < tap_cooldown:
            continue

        sent[button] = 1
        if input_backend is not None:
            input_backend.click(button=directinput_name)
        last_tap_at[button] = now

    dx = int(round(float(mouse[0]) * mouse_clip * mouse_gain))
    dy = int(round(float(mouse[1]) * mouse_clip * mouse_gain))
    if abs(dx) < mouse_deadzone:
        dx = 0
    if abs(dy) < mouse_deadzone:
        dy = 0

    if input_backend is not None and (dx or dy):
        input_backend.moveRel(dx, dy, duration=0)

    sent["mouse_dx"] = dx
    sent["mouse_dy"] = dy
    return sent


def active_button_summary(buttons: tuple[str, ...], probabilities: np.ndarray, limit: int = 5) -> str:
    top_buttons = sorted(
        zip(buttons, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return ", ".join(f"{button}={probability:.2f}" for button, probability in top_buttons)


def all_buttons() -> tuple[str, ...]:
    return BUTTONS
