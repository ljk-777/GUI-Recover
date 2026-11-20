"""Locate answer evidence in a screenshot and read a native-pixel crop."""

import json
import re
from decimal import Decimal

from .actions import Action, ActionError
from .qwen_grounding import read_reply
from .verification import pixel_hash


_AMOUNT = re.compile(r"(?<![\w.])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?![\w.])")


def evidence_box(raw, size):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"bbox_2d"}:
        raise ActionError("Expected one evidence box")
    box = data["bbox_2d"]
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        raise ActionError("Evidence box must contain four integer pixels")
    left, top, right, bottom = box
    width, height = size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ActionError("Evidence box lies outside the prepared screenshot")
    return tuple(box)


def native_crop(screen, size, box):
    """Map prepared-image coordinates to the original pixels with context."""
    left, top, right, bottom = box
    width, height = size
    pad_x = max(16, (right - left) // 8)
    pad_y = max(16, (bottom - top) // 8)
    bounds = (
        max(0, (left - pad_x) * screen.width // width),
        max(0, (top - pad_y) * screen.height // height),
        min(screen.width, ((right + pad_x) * screen.width + width - 1) // width),
        min(screen.height, ((bottom + pad_y) * screen.height + height - 1) // height),
    )
    return screen.crop(bounds), bounds


def evidence_answer(raw, goal):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "evidence"}:
        raise ActionError("Expected an answer and its visual evidence")
    answer, facts = data["answer"], data["evidence"]
    if (not isinstance(facts, list) or len(facts) > 6
            or any(not isinstance(fact, str) or not 0 < len(fact.strip()) <= 200 for fact in facts)):
        raise ActionError("Expected a bounded list of visual evidence descriptions")
    if answer is None:
        return None, "evidence_abstained"
    if not isinstance(answer, str) or not 0 < len(answer.strip()) <= 512:
        raise ActionError("Expected a nonempty short answer or null")
    if not facts:
        return None, "evidence_incomplete"
    if "price range" in goal.casefold():
        # A range answer needs both visible endpoints. Ignore formatting
        # differences such as thousands separators when matching amounts.
        bounds = [Decimal(value.replace(",", "")) for value in _AMOUNT.findall(answer)]
        observed = {Decimal(value.replace(",", "")) for value in _AMOUNT.findall(" ".join(facts))}
        if len(bounds) != 2 or not set(bounds) <= observed:
            return None, "evidence_incomplete"
    return answer.strip(), "evidence_answer"


class EvidenceAnswerPolicy:
    """Use one model invocation per prediction, including crop localization."""

    def __init__(self, actor):
        self.actor = actor
        self.checked_frames = set()
        self.pending_crop = None

    @property
    def last_metrics(self):
        return self.actor.last_metrics

    def reset(self, seed):
        self.actor.reset(seed)
        self.checked_frames.clear()
        self.pending_crop = None

    def set_feedback(self, message):
        self.actor.set_feedback(message)

    def locate(self, observation, frame):
        canvas = self.actor.prepare_image(observation.screenshot)
        width, height = canvas.size
        prompt = (
            f"Find the visual evidence needed to answer this question: {observation.goal}\n"
            "Point to the region containing the requested item and the relevant details. "
            "Include labels or row context that identify the target. For a price range include both ends. "
            "Select a useful region if it can be inspected more closely. "
            f"Use integer pixel coordinates in this {width} by {height} image with top-left origin. "
            'Return one JSON object like {"bbox_2d":[left,top,right,bottom]}. '
            'If the relevant region is not visible, return {"bbox_2d":null}. Do not answer the question.'
        )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="locate", evidence_frame=frame, provider_format="evidence_invalid")
        try:
            box = evidence_box(raw, canvas.size)
        except (ValueError, RecursionError):
            return None
        self.last_metrics["provider_format"] = "evidence_region" if box else "evidence_abstained"
        if box:
            crop, bounds = native_crop(observation.screenshot, canvas.size, box)
            self.pending_crop = frame, crop, bounds
            self.last_metrics["evidence_bounds"] = list(bounds)
        return None

    def read(self, observation, frame, crop, bounds):
        canvas = self.actor.prepare_image(crop)
        prompt = (
            f"Question: {observation.goal}\n"
            "Inspect only the visible evidence in this crop. Identify the requested subject and list "
            "the relevant observations in evidence before writing the short answer. "
            "For a count, inspect and enumerate the visible items. For a range, read and give both endpoints. "
            "If the requested subject, scope, or any required detail cannot be verified, use null for answer. "
            'Return one JSON object with exactly two keys: "evidence" (a list of short strings), '
            'then "answer" (short string or null).'
        )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="read", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_crop_sha256=pixel_hash(crop), provider_format="evidence_invalid")
        try:
            answer, status = evidence_answer(raw, observation.goal)
        except (ValueError, RecursionError):
            return None
        self.last_metrics["provider_format"] = status
        if answer is not None:
            return json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)
        return None

    def predict(self, observation, history):
        frame = pixel_hash(observation.screenshot)
        if self.pending_crop is not None:
            pending_frame, crop, bounds = self.pending_crop
            self.pending_crop = None
            if frame == pending_frame:
                return self.read(observation, frame, crop, bounds)
        if frame not in self.checked_frames:
            self.checked_frames.add(frame)
            return self.locate(observation, frame)
        return self.actor.predict(observation, history, allow_answer=False)
