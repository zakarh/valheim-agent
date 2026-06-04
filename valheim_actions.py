from __future__ import annotations


BUTTONS = (
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "lmb",
    "rmb",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
)

HELD_KEYS = ("w", "a", "s", "d", "shift", "ctrl")
TAP_KEYS = ("space", "e", "1", "2", "3", "4", "5", "6", "7", "8")
MOUSE_BUTTONS = {"lmb": "left", "rmb": "right"}

ACTION_COLUMNS = BUTTONS + ("mouse_dx", "mouse_dy")


def empty_button_state() -> dict[str, int]:
    return {button: 0 for button in BUTTONS}
