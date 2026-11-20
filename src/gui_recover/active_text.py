"""Identify a selected final character from two native screenshots."""

import numpy as np
from PIL import Image

from .recovery_env import Decision, FRAME_SIZE, Phase, ProbeDecision, transition_view
from .visual_text import glyph_decision


def inspect_tail_selection(original, selected, region):
    """Locate a compact selection by pixel change, independent of its color."""
    if original.size != FRAME_SIZE or selected.size != FRAME_SIZE:
        return {"status": "unknown", "reason": "unexpected_screenshot_size"}
    left, top, right, bottom = region
    left, top, right, bottom = left + 3, top + 3, right - 3, bottom - 3
    if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
        return {"status": "unknown", "reason": "invalid_field_region"}
    bounds = np.s_[top:bottom, left:right]
    first = np.asarray(original.convert("RGB"))[bounds]
    second = np.asarray(selected.convert("RGB"))[bounds]
    changed = np.max(np.abs(first.astype(np.int16) - second), axis=2) >= 20
    ys, xs = np.nonzero(changed)
    if not len(xs):
        return {"status": "unknown", "reason": "no_selection_change"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    density = len(xs) / (width * height)
    if not (2 <= width <= 30 and 8 <= height <= 30 and density >= .45):
        return {"status": "unknown", "reason": "unsupported_selection_geometry"}
    ink = int(np.count_nonzero(np.asarray(original.convert("L"))[top+y0:top+y1, left+x0:left+x1] < 160))
    # Chromium can leave one antialiased pixel from the previous glyph at
    # the left edge of a selected space.
    status = "blank_tail" if ink <= 1 else "visible_tail" if ink >= 4 else "unknown"
    return {"status": status, "reason": "ink_in_selection_bounds", "ink_pixels": ink,
            "selection_region": [left+x0, top+y0, left+x1, top+y1]}


class ActiveTextPolicy:
    """Ask for one reversible probe of a visually matched short input."""

    def __init__(self, verifier, choose_when_matched=None):
        self.verifier = verifier
        self.choose_when_matched = choose_when_matched
        self.evidence = []

    def __call__(self, observation):
        if observation["phase"] != Phase.REVIEW:
            return Decision.CONTINUE
        view = transition_view(observation)
        glyph = self.verifier.verify(view)
        selection = None
        if observation["probed"]:
            if observation["probe_restored"] and glyph["status"] == "matched":
                selection = inspect_tail_selection(view.after, Image.fromarray(observation["selected"]), glyph["region"])
            else:
                selection = {"status": "unknown", "reason": "probe_did_not_restore_visual_state"}
            action = (Decision.RETRY_FIXED_ZOOM if selection["status"] == "blank_tail" else Decision.CONTINUE)
        elif glyph["status"] == "matched" and observation["action_mask"][ProbeDecision.PROBE_LAST_CHAR]:
            action = ProbeDecision.PROBE_LAST_CHAR
            if self.choose_when_matched is not None:
                action = ProbeDecision(self.choose_when_matched(observation))
                if action not in (ProbeDecision.CONTINUE, ProbeDecision.PROBE_LAST_CHAR):
                    raise ValueError("A visually matched field allows continuation or probing")
        else:
            action = glyph_decision(observation, glyph["status"])
        self.evidence.append({"glyph": glyph, "selection": selection, "decision": action.name})
        return action
