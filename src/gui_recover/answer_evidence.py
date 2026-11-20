"""Locate answer evidence in a screenshot and read a native-pixel crop."""

import json
import re
from decimal import Decimal

from PIL import Image, ImageDraw

from .actions import Action, ActionError
from .qwen_grounding import native_action_after_note, normalize_qwen_action, parse_visible_answer, read_reply
from .verification import pixel_hash


_AMOUNT = re.compile(r"(?<![\w.])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?![\w.])")


def evidence_box(raw, size):
    data = read_reply(raw)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict) or set(data) not in ({"bbox_2d"}, {"bbox_2d", "label"}):
        raise ActionError("Expected one evidence box")
    if "label" in data and (not isinstance(data["label"], str) or len(data["label"]) > 200):
        raise ActionError("Expected a short evidence label")
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


def evidence_clipped(goal, box, size):
    """A range or ordinal row needs the visible extent of that row."""
    requires_row = re.search(r"\brows?\b", goal.casefold())
    return bool(requires_row and box[3] >= size[1] - 2)


def evidence_navigation(raw, size):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        data = native_action_after_note(raw)
    if not isinstance(data, dict) or "action" not in data:
        return None
    action = normalize_qwen_action(data, size, native_navigation=True)
    if action.action == "ANSWER":
        raise ActionError("An evidence search cannot submit an answer")
    return action


def evidence_answer(raw, goal):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "evidence"}:
        raise ActionError("Expected an answer and its visual evidence")
    answer, facts = data["answer"], data["evidence"]
    if (not isinstance(facts, list) or len(facts) > 6
            or any(not isinstance(fact, str) or not 0 < len(fact.strip()) <= 200 for fact in facts)):
        raise ActionError("Expected a bounded list of visual evidence descriptions")
    if answer is None or (isinstance(answer, str) and answer.strip().casefold() == "null"):
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


def confirmation_image(screen, crop, bounds):
    overview = screen.convert("RGB")
    overview.thumbnail((448, 336), Image.Resampling.LANCZOS)
    ow, oh = overview.size
    scaled = (bounds[0] * ow // screen.width, bounds[1] * oh // screen.height,
              min(ow - 1, bounds[2] * ow // screen.width),
              min(oh - 1, bounds[3] * oh // screen.height))
    ImageDraw.Draw(overview).rectangle(scaled, outline="#176fde", width=2)
    panel = Image.new("RGB", (ow + 12 + crop.width, max(oh, crop.height)), "white")
    panel.paste(overview)
    panel.paste(crop, (ow + 12, 0))
    return panel


def answer_signature(answer):
    value = " ".join(answer.casefold().rstrip(".").split())
    if re.fullmatch(r"\$?[0-9][0-9,. ]*(?:[–—-]\s*\$?[0-9][0-9,. ]*)?", value):
        amounts = tuple(Decimal(amount.replace(",", "")) for amount in _AMOUNT.findall(value))
        if amounts:
            return amounts
    return value


class EvidenceAnswerPolicy:
    """Use one model invocation per prediction, including crop localization."""

    def __init__(self, actor, *, confirm=False):
        self.actor = actor
        self.confirm = confirm
        self.checked_frames = set()
        self.pending_crop = None
        self.pending_answer = None

    @property
    def last_metrics(self):
        return self.actor.last_metrics

    def reset(self, seed):
        self.actor.reset(seed)
        self.checked_frames.clear()
        self.pending_crop = None
        self.pending_answer = None

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
        if self.confirm:
            prompt = (
                f"Task: {observation.goal}\n"
                f"Use the visible state of this {width} by {height} screenshot. "
                'If a specific visible region could support all requested details, return {"bbox_2d":[left,top,right,bottom]}. '
                "Choose a tight box with the relevant text or photo and enough context. "
                "If the user first needs a search, category, sort, or item opened, do one navigation action to reach it. "
                'Return {"action":"CLICK","value":null,"position":[x,y]} for a visible control, '
                '{"action":"INPUT","value":"query","position":[x,y]} to type into a visible search field, '
                '{"action":"ENTER","value":null,"position":null} to submit a filled search field, or '
                '{"action":"SCROLL","value":"down","position":null} to reveal lower content. '
                'Coordinates are integer pixels. Return {"bbox_2d":null} if neither option is clear. '
                "Return one JSON response and never answer the task directly."
            )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="locate", evidence_frame=frame, provider_format="evidence_invalid")
        try:
            navigation = evidence_navigation(raw, canvas.size) if self.confirm else None
            if navigation is not None:
                self.last_metrics["provider_format"] = "evidence_navigation"
                return json.dumps(navigation.to_dict(), ensure_ascii=False)
            box = evidence_box(raw, canvas.size)
        except (ValueError, RecursionError):
            return None
        self.last_metrics["provider_format"] = "evidence_region" if box else "evidence_abstained"
        if box:
            if evidence_clipped(observation.goal, box, canvas.size):
                self.last_metrics["provider_format"] = "evidence_clipped"
                return None
            crop, bounds = native_crop(observation.screenshot, canvas.size, box)
            self.pending_crop = frame, crop, bounds
            self.last_metrics["evidence_bounds"] = list(bounds)
        return None

    def read(self, observation, frame, crop, bounds):
        canvas = self.actor.prepare_image(crop)
        prompt = (
            f"Question: {observation.goal}\n"
            "Inspect this crop. Identify the requested subject and list at most three brief, relevant facts. "
            "Omit unrelated text. For a count, count each visible group separately before adding. "
            "For a price range, give both endpoints. For a total price, read rates and pack sizes before calculating. "
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
            if self.confirm:
                self.pending_answer = frame, crop, bounds, answer
                return None
            return json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)
        return None

    def verify(self, observation, frame, crop, bounds, candidate):
        panel = confirmation_image(observation.screenshot, crop, bounds)
        canvas = self.actor.prepare_image(panel)
        prompt = (
            f"Question: {observation.goal}\n"
            "The left panel is the whole viewport and the blue outline marks the area enlarged on the right. "
            "Check the requested item's identity and the full scope in the overview, then read the enlargement. "
            "For a page, row, or search question, verify that the required results are visible. "
            "Give every requested component. A single price cannot answer a price range. "
            "If anything required is missing, return {\"answer\":null}. "
            "Otherwise return one JSON object with one key answer and a short string value."
        )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="confirm", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_panel_sha256=pixel_hash(panel), provider_format="evidence_invalid")
        try:
            answer = parse_visible_answer(raw, allow_numeric=True)
        except (ValueError, RecursionError):
            return None
        if answer is None:
            self.last_metrics["provider_format"] = "evidence_verify_abstained"
            return None
        if answer_signature(candidate) != answer_signature(answer):
            self.last_metrics["provider_format"] = "evidence_verify_disagreed"
            return None
        self.last_metrics["provider_format"] = "evidence_verify_agreed"
        return json.dumps(Action("ANSWER", candidate).to_dict(), ensure_ascii=False)

    def predict(self, observation, history):
        frame = pixel_hash(observation.screenshot)
        if self.pending_answer is not None:
            pending_frame, crop, bounds, candidate = self.pending_answer
            self.pending_answer = None
            if frame == pending_frame:
                return self.verify(observation, frame, crop, bounds, candidate)
        if self.pending_crop is not None:
            pending_frame, crop, bounds = self.pending_crop
            self.pending_crop = None
            if frame == pending_frame:
                return self.read(observation, frame, crop, bounds)
        if frame not in self.checked_frames:
            self.checked_frames.add(frame)
            return self.locate(observation, frame)
        return self.actor.predict(observation, history, allow_answer=False)
