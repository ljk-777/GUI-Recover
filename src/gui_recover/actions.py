"""Parse inert model data; only our compiler can produce executable actions."""

import ast
import json
import math
from dataclasses import asdict, dataclass


class ActionError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    action: str
    value: str | None = None
    position: tuple[float, float] | None = None

    def to_dict(self):
        return asdict(self)


def parse_action(raw: str) -> Action:
    text = raw.strip()
    if len(text) > 8192:
        raise ActionError("Output exceeds 8192 characters")
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise ActionError("Expected exactly one action dictionary") from exc
    if not isinstance(data, dict) or set(data) != {"action", "value", "position"}:
        raise ActionError("Required keys: action, value, position; no extra keys")
    kind, value, pos = data["action"], data["value"], data["position"]
    if kind not in ("CLICK", "INPUT", "HOVER", "ENTER", "SCROLL", "ANSWER"):
        raise ActionError(f"Unsupported action: {kind!r}")
    if kind in ("CLICK", "INPUT", "HOVER"):
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            raise ActionError("Position must be [x, y]")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in pos):
            raise ActionError("Coordinates must be finite numbers in [0, 1]")
        pos = tuple(float(p) for p in pos)
    elif pos is not None:
        raise ActionError(f"{kind} requires position=None")
    if kind in ("INPUT", "ANSWER"):
        if not isinstance(value, str) or len(value) > 2048:
            raise ActionError("Expected a string value of at most 2048 characters")
    elif kind == "SCROLL":
        if value not in ("up", "down", "left", "right"):
            raise ActionError("SCROLL value must be up/down/left/right")
    elif value is not None:
        raise ActionError(f"{kind} requires value=None")
    return Action(kind, value, pos)


def compile_action(action: Action, width: int, height: int) -> str:
    # Validate even when a caller constructs Action directly.
    action = parse_action(json.dumps(action.to_dict()))
    if width < 1 or height < 1:
        raise ActionError("Invalid screenshot dimensions")
    if action.position is not None:
        # BrowserGym maps screenshot pixels back to CSS pixels itself.
        x = min(action.position[0] * width, width - 1)
        y = min(action.position[1] * height, height - 1)
    if action.action == "CLICK":
        return f"mouse_click({x!r}, {y!r})"
    if action.action == "HOVER":
        return f"mouse_move({x!r}, {y!r})"
    if action.action == "INPUT":
        # INPUT appends text to the clicked field; does not silently clear it.
        return f"mouse_click({x!r}, {y!r})\nkeyboard_type({action.value!r})"
    if action.action == "ENTER":
        return "keyboard_press('Enter')"
    if action.action == "SCROLL":
        dx, dy = {"up": (0, -300), "down": (0, 300), "left": (-300, 0), "right": (300, 0)}[action.value]
        return f"scroll_at({width // 2}, {height // 2}, {dx}, {dy})"
    raise ActionError("ANSWER is a local stop request, never an environment success")
