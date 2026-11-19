"""V0.2 alpha: controlled faults and visual-only feedback signals.

The detector sees only screenshot pixels and the proposed action. Injection
labels, reward, task success and DOM are intentionally absent from its API.
"""

from dataclasses import dataclass
import random

import numpy as np

from .actions import Action


class ClickFault:
    def __init__(self, kind="none", seed=0):
        if kind not in ("none", "drop-click", "offset-click"):
            raise ValueError("Unknown fault kind")
        self.kind = kind
        self.seed = seed
        self.random = random.Random(seed)
        self.injected = False

    def apply(self, action):
        if self.kind == "none" or self.injected or action.action != "CLICK":
            return action, None
        self.injected = True
        if self.kind == "drop-click":
            # A missed press with the pointer still moving to the intended target.
            executed = Action("HOVER", position=action.position)
        else:
            dx = self.random.choice((-0.15, 0.15))
            dy = self.random.choice((-0.15, 0.15))
            executed = Action("CLICK", position=tuple(
                min(1.0, max(0.0, p + d)) for p, d in zip(action.position, (dx, dy))))
        return executed, {"kind": self.kind, "seed": self.seed,
                          "proposed": action.to_dict(), "executed": executed.to_dict()}


@dataclass(frozen=True)
class VisualSignal:
    unchanged_after_click: bool
    mean_pixel_difference: float
    execution_error: str


def detect(before, after, proposed, threshold=0.002):
    if before.screenshot.size != after.screenshot.size:
        difference = 1.0
    else:
        a = np.asarray(before.screenshot, dtype=np.float32)
        b = np.asarray(after.screenshot, dtype=np.float32)
        difference = float(np.abs(a - b).mean() / 255)
    return VisualSignal(proposed.action == "CLICK" and difference <= threshold,
                        difference, after.last_action_error)


class FeedbackRecovery:
    def __init__(self, enabled=False, limit=2):
        self.enabled = enabled
        self.limit = limit
        self.used = 0

    def feedback(self, signal):
        if not self.enabled or self.used >= self.limit:
            return None
        if signal.execution_error:
            message = "The last action raised an execution error. Reobserve and choose a valid next action."
        elif signal.unchanged_after_click:
            message = ("The previous click produced little or no visible change. "
                       "Reobserve the screenshot and check the intended target. "
                       "Choose the next action; do not assume the previous click completed the task.")
        else:
            return None
        self.used += 1
        return message
