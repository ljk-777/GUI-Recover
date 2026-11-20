"""Read and recover checkbox state from images in the MiniWoB Chromium fixture."""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import log
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .actions import Action
from .verification import TransitionView


_SIZE = (498, 321)
_TARGET = re.compile(r'The checkbox labeled "([A-Za-z0-9]{2,8})"')
_GOAL = re.compile(r'Select ([A-Za-z0-9]{2,8}(?:, [A-Za-z0-9]{2,8})*) and click Submit\.')


@dataclass(frozen=True)
class Checkbox:
    top: int
    bottom: int
    checked: bool
    ink: np.ndarray

    @property
    def point(self):
        return (24.0/_SIZE[0], (self.top+self.bottom)/2/_SIZE[1])


def _trim(ink):
    y, x = np.nonzero(ink > 0.16)
    if not len(y):
        return None
    return ink[y.min():y.max()+1, x.min():x.max()+1]


def _checkboxes(rgb):
    gray = rgb.mean(axis=2)
    dark = (gray[78:254, 12:38] < 220).sum(axis=1) >= 3
    active = np.flatnonzero(dark) + 78
    runs = [r for r in np.split(active, np.flatnonzero(np.diff(active) > 1)+1) if 17 <= len(r) <= 22]
    if not 2 <= len(runs) <= 6 or abs(int(runs[0][0])-83) > 1:
        return None
    if any(abs(int(r[0])-83-28.5*i) > 2 for i, r in enumerate(runs)):
        return None
    rows = []
    for run in runs:
        top, bottom = int(run[0]), int(run[-1])+1
        patch = rgb[top:bottom, 14:34]
        red, green, blue = patch.transpose(2, 0, 1)
        blue_fill = ((blue-red > 60) & (green-red > 40)).mean()
        white_center = (gray[top+5:bottom-5, 20:28] > 242).mean()
        if blue_fill > .35 and .10 <= white_center <= .80:
            checked = True
        elif blue_fill == 0 and white_center >= .95:
            checked = False
        else:
            return None
        mask = _trim(1-gray[top+4:bottom+6, 40:220]/255)
        if mask is None or not 6 <= mask.shape[0] <= 18:
            return None
        rows.append(Checkbox(top, bottom, checked, mask))
    return rows


def _similarity(template, ink):
    height, width = template.shape
    ih, iw = ink.shape
    resized = np.asarray(Image.fromarray((ink*255).astype(np.uint8))
                         .resize((width, height), Image.Resampling.BILINEAR), dtype=float)/255
    norm = np.linalg.norm(template)*np.linalg.norm(resized)
    cosine = float(np.vdot(template, resized)/norm) if norm else 0.0
    return cosine-.25*(abs(log(iw/width))+abs(log(ih/height)))


def _unknown(reason, **details):
    return {"status": "unknown", "reason": reason, **details}


class CheckboxStateVerifier:
    """Profile for one viewport, font and native checkbox style. Other UIs abstain."""

    def __init__(self, font_path, *, minimum_score=.70, minimum_margin=.18):
        self.font = ImageFont.truetype(str(font_path), 15)
        if not (0 < minimum_score <= 1 and 0 < minimum_margin <= 1):
            raise ValueError("Invalid evidence thresholds")
        self.minimum_score, self.minimum_margin = minimum_score, minimum_margin

    @lru_cache(maxsize=128)
    def template(self, label):
        image = Image.new("L", (150, 30))
        ImageDraw.Draw(image).text((1, 0), label, fill=255, font=self.font)
        return _trim(np.asarray(image, dtype=float)/255)

    def match_label(self, label, rows):
        template = self.template(label)
        if template is None:
            return None, _unknown("target_font_unreadable")
        scores = [_similarity(template, row.ink) for row in rows]
        best, runner_up = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
        margin = scores[best]-scores[runner_up]
        evidence = {"target_score": scores[best], "target_margin": margin}
        if scores[best] < self.minimum_score or margin < self.minimum_margin:
            return None, _unknown("ambiguous_target_label", **evidence)
        return best, evidence

    def verify(self, view: TransitionView):
        target, goal = _TARGET.fullmatch(view.target), _GOAL.fullmatch(view.goal)
        if view.proposed.action != "CLICK" or not target or not goal or target[1] not in goal[1].split(", "):
            return _unknown("no_supported_target_state")
        if view.before.size != _SIZE or view.after.size != _SIZE:
            return _unknown("unsupported_viewport")
        before, after = (np.asarray(p.convert("RGB"), dtype=np.int16) for p in (view.before, view.after))
        old, new = _checkboxes(before), _checkboxes(after)
        if old is None or new is None or len(old) != len(new):
            return _unknown("checkbox_grid_unreadable")
        if not all(abs(a.top-b.top) <= 1 and abs(a.bottom-b.bottom) <= 1
                   and np.array_equal(a.ink, b.ink) for a, b in zip(old, new)):
            return _unknown("checkbox_grid_changed")
        changed = np.any(before != after, axis=2)
        for row in old:
            changed[row.top-2:row.bottom+2, 12:38] = False
        if changed.any():
            return _unknown("changes_outside_checkbox_controls")

        best, evidence = self.match_label(target[1], old)
        if best is None:
            return evidence

        other_changes = [i for i, (a, b) in enumerate(zip(old, new))
                         if i != best and a.checked != b.checked]
        evidence.update(target_row=best, checked_before=old[best].checked, checked_after=new[best].checked,
                        other_changed_rows=other_changes)
        if new[best].checked and not other_changes:
            return {"status": "achieved", "reason": "target_checked", **evidence}
        if new[best].checked or old[best].checked or len(other_changes) > 1:
            return _unknown("unexpected_state_delta", **evidence)
        if other_changes:
            wrong = other_changes[0]
            x, y = (a*d for a, d in zip(view.proposed.position, _SIZE))
            row = old[wrong]
            if row.checked or not (12 <= x <= 220 and row.top-2 <= y < row.bottom+8):
                return _unknown("unattributed_checkbox_change", **evidence)
            return {"status": "not_achieved", "reason": "different_checkbox_checked",
                    "repair": {"kind": "undo_wrong_check", "point": row.point}, **evidence}
        return {"status": "not_achieved", "reason": "target_unchecked",
                "repair": {"kind": "check_target", "point": new[best].point}, **evidence}

    def recover(self, view, execute):
        """Reinspect after each repair. Execute at most one undo and one target click."""
        trace = []
        for kind in ("undo_wrong_check", "check_target"):
            verdict = self.verify(view)
            proposal = verdict.get("repair")
            if not proposal or proposal["kind"] != kind:
                continue
            action = Action("CLICK", position=proposal["point"])
            after = execute(action)
            trace.append({"kind": kind, "action": action.to_dict(), "evidence": verdict})
            if after is None:
                return {"actions": trace, "final": _unknown("task_already_ended")}
            view = replace(view, after=after)
        return {"actions": trace, "final": self.verify(view)}
