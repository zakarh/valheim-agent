from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valheim_actions import BUTTONS


@dataclass(frozen=True)
class DiscreteValheimAction:
    name: str
    buttons: tuple[str, ...] = ()
    mouse: tuple[float, float] = (0.0, 0.0)


RL_ACTIONS = (
    DiscreteValheimAction("noop"),
    DiscreteValheimAction("forward", ("w",)),
    DiscreteValheimAction("back", ("s",)),
    DiscreteValheimAction("left", ("a",)),
    DiscreteValheimAction("right", ("d",)),
    DiscreteValheimAction("forward_left", ("w", "a")),
    DiscreteValheimAction("forward_right", ("w", "d")),
    DiscreteValheimAction("sprint_forward", ("w", "shift")),
    DiscreteValheimAction("turn_left", mouse=(-0.45, 0.0)),
    DiscreteValheimAction("turn_right", mouse=(0.45, 0.0)),
    DiscreteValheimAction("look_up", mouse=(0.0, -0.30)),
    DiscreteValheimAction("look_down", mouse=(0.0, 0.30)),
    DiscreteValheimAction("jump", ("space",)),
    DiscreteValheimAction("attack", ("lmb",)),
    DiscreteValheimAction("block", ("rmb",)),
    DiscreteValheimAction("interact", ("e",)),
    DiscreteValheimAction("forward_attack", ("w", "lmb")),
)


def action_to_predictions(action_index: int) -> tuple[np.ndarray, np.ndarray]:
    action = RL_ACTIONS[int(action_index)]
    probabilities = np.zeros(len(BUTTONS), dtype=np.float32)
    button_index = {button: index for index, button in enumerate(BUTTONS)}

    for button in action.buttons:
        probabilities[button_index[button]] = 1.0

    return probabilities, np.asarray(action.mouse, dtype=np.float32)
