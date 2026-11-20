"""Locate answer evidence in a screenshot and read a native-pixel crop."""

import json
import re
from decimal import Decimal

from PIL import Image, ImageDraw

from .actions import Action, ActionError
from .qwen_grounding import native_action_after_note, normalize_qwen_action, parse_visible_answer, read_reply
from .screen_text import exact_initial_search, from_readable_control, listing_detail
from .verification import pixel_hash
from .visible_links import agreed_listing_answer, link_answer_count, listing_selection, preview_destinations


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


def evidence_navigation(raw, size, *, compact_navigation=True):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        data = native_action_after_note(raw)
    if not isinstance(data, dict) or "action" not in data:
        return None
    action = normalize_qwen_action(data, size, native_navigation=True, compact_navigation=compact_navigation)
    if action.action == "ANSWER":
        raise ActionError("An evidence search cannot submit an answer")
    return action


def evidence_answer(raw, goal, *, allow_string_abstention=False):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "evidence"}:
        raise ActionError("Expected an answer and its visual evidence")
    answer, facts = data["answer"], data["evidence"]
    if (not isinstance(facts, list) or len(facts) > 6
            or any(not isinstance(fact, str) or not 0 < len(fact.strip()) <= 200 for fact in facts)):
        raise ActionError("Expected a bounded list of visual evidence descriptions")
    if answer is None or (allow_string_abstention and isinstance(answer, str) and answer.strip().casefold() == "null"):
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


def localized_candidate(raw, goal, size):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != {"bbox_2d", "evidence", "answer"}:
        return None
    box = evidence_box(json.dumps({"bbox_2d": data["bbox_2d"]}), size)
    if box is None:
        raise ActionError("A candidate needs a selectable region")
    reply = json.dumps({"answer": data["answer"], "evidence": data["evidence"]})
    answer, _ = evidence_answer(reply, goal, allow_string_abstention=True)
    return box, answer


def has_transcribed_answer(answer, box, rows):
    text = " ".join(answer.casefold().split())
    pattern = re.compile(r"(?<![\w.])" + re.escape(text) + r"(?![\w.])")
    left, top, right, bottom = box
    for row in rows:
        x0, y0, x1, y1 = row["bounds"]
        if left <= (x0 + x1) / 2 < right and top <= (y0 + y1) / 2 < bottom:
            if pattern.search(row["text"].casefold()):
                return True
    return False


def parse_evidence_location(raw, goal, size, *, confirm, compact_navigation=True, rows=None, guarded_ocr=False,
                            navigation_only=False):
    """Return a browser action or a validated evidence region and optional candidate."""
    candidate = None
    localized = None
    if rows is not None:
        control = from_readable_control(raw, rows, size, guarded=guarded_ocr)
        if control is not None:
            return control, None, None, "evidence_ocr_navigation"
        localized = localized_candidate(raw, goal, size)
    if localized is not None:
        box, candidate = localized
        if guarded_ocr and candidate is not None and not has_transcribed_answer(candidate, box, rows):
            candidate = None
    else:
        navigation = evidence_navigation(raw, size, compact_navigation=compact_navigation) if confirm else None
        if navigation is not None:
            return navigation, None, None, "evidence_navigation"
        box = evidence_box(raw, size)
    if box is None:
        return None, None, None, "evidence_abstained"
    if navigation_only:
        raise ActionError("A listing link must come from a verified title preview")
    if evidence_clipped(goal, box, size):
        return None, None, None, "evidence_clipped"
    return None, box, candidate, "evidence_locate_read" if candidate is not None else "evidence_region"


def confirmation_image(screen, crop, bounds, navigation_context=None):
    overview = screen.convert("RGB")
    overview.thumbnail((448, 336), Image.Resampling.LANCZOS)
    ow, oh = overview.size
    scaled = (bounds[0] * ow // screen.width, bounds[1] * oh // screen.height,
              min(ow - 1, bounds[2] * ow // screen.width),
              min(oh - 1, bounds[3] * oh // screen.height))
    ImageDraw.Draw(overview).rectangle(scaled, outline="#176fde", width=2)
    previous = None
    if navigation_context is not None:
        before, (x, y) = navigation_context
        previous = before.convert("RGB")
        previous.thumbnail((448, 336), Image.Resampling.LANCZOS)
        px = min(previous.width - 1, int(x * previous.width))
        py = min(previous.height - 1, int(y * previous.height))
        ImageDraw.Draw(previous).ellipse((px - 7, py - 7, px + 7, py + 7), outline="#d85319", width=3)
    left_height = oh + (12 + previous.height if previous is not None else 0)
    panel = Image.new("RGB", (ow + 12 + crop.width, max(left_height, crop.height)), "white")
    panel.paste(overview)
    if previous is not None:
        panel.paste(previous, (0, oh + 12))
    panel.paste(crop, (ow + 12, 0))
    return panel


def answer_signature(answer):
    value = " ".join(answer.casefold().rstrip(".").split())
    if re.fullmatch(r"\$?[0-9][0-9,. ]*(?:[–—-]\s*\$?[0-9][0-9,. ]*)?", value):
        amounts = tuple(Decimal(amount.replace(",", "")) for amount in _AMOUNT.findall(value))
        if amounts:
            return amounts
    return value


def indexed_text(rows):
    return "\n".join(f"{index}: {json.dumps(row['text'], ensure_ascii=False)} at {row['bounds']}"
                     for index, row in enumerate(rows))


class EvidenceAnswerPolicy:
    """Use one model invocation per prediction, including crop localization."""

    def __init__(self, actor, *, confirm=False, use_navigation_context=False, screen_reader=None,
                 listing_preview=None, listing_site=None):
        if screen_reader is not None and not confirm:
            raise ValueError("The OCR policy needs independent confirmation")
        if listing_preview is not None and (screen_reader is None or listing_site is None):
            raise ValueError("A listing preview needs visible text and a registered origin")
        self.actor = actor
        self.confirm = confirm
        self.use_navigation_context = use_navigation_context
        self.screen_reader = screen_reader
        self.listing_preview = listing_preview
        self.listing_site = listing_site
        self.checked_frames = set()
        self.pending_crop = None
        self.pending_answer = None
        self.pending_links = None
        self.navigation_context = None

    @property
    def last_metrics(self):
        return self.actor.last_metrics

    def reset(self, seed):
        self.actor.reset(seed)
        self.checked_frames.clear()
        self.pending_crop = None
        self.pending_answer = None
        self.pending_links = None
        self.navigation_context = None

    def set_feedback(self, message):
        self.actor.set_feedback(message)

    def observe_transition(self, before, after, action):
        if (self.use_navigation_context and action.action == "CLICK"
                and pixel_hash(before.screenshot) != pixel_hash(after.screenshot)):
            self.navigation_context = before.screenshot.copy(), action.position

    def locate(self, observation, frame, history):
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
        rows = None
        navigation_rows = None
        link_count = link_answer_count(observation.goal) if self.listing_preview is not None else None
        if self.screen_reader is not None:
            rows = self.screen_reader(observation.screenshot, canvas.size)
            if not history:
                try:
                    shortcut = exact_initial_search(observation.goal, rows, canvas.size)
                except ActionError:
                    shortcut = None
                if shortcut is not None:
                    self.actor.last_metrics = {
                        "model_invocations": 0, "provider_output": None, "provider_image_size": list(canvas.size),
                        "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
                        "provider_format": "evidence_exact_search", "peak_cuda_allocated_bytes": 0,
                    }
                    return json.dumps(shortcut.to_dict(), ensure_ascii=False)
            navigation_rows = None if link_count is None and listing_detail(rows) else rows
        if navigation_rows is not None:
            controls = indexed_text(navigation_rows)
            recent = [{"action": item["action"]["action"],
                       "value": (item["action"]["value"] or "")[:80]}
                      for item in history[-4:]]
            prompt += (
                "\nUse the indexed visible text as screenshot evidence. "
                'For a button or field with an identified line, use {"action":"CLICK","value":null,"control":index} '
                'or {"action":"INPUT","value":"query","control":index}. Only use INPUT on a visible field or placeholder. '
                'On a page with an empty Keyword input showing an e.g. placeholder, '
                'use {"action":"SEARCH","value":"query","control":placeholder_index} to type and submit in one action. '
                "If the current page shows a search form and the target is elsewhere, search by the requested keyword "
                "or category instead of scrolling unrelated items. If the field already contains the query, submit it. "
                "After a SEARCH, inspect its results instead of typing the same query again. "
                "Use the visible row order and item photos to identify a requested listing. If publisher or item "
                "details are absent from the gallery, open a matching item. Do not scroll just to find details in a gallery. "
                'When the answer is already legible, you may return {"bbox_2d":[x0,y0,x1,y1],"evidence":["visible fact"],'
                '"answer":"candidate"}. Use a candidate only for literal text visible inside the region. For counts, colors '
                "and other visual details, return the box only. A separate read will confirm the answer. Do not guess. "
                f"Previous browser actions: {json.dumps(recent, ensure_ascii=False)}.\nVisible text:\n{controls}"
            )
        if link_count is not None:
            prompt = (
                f"Task: {observation.goal}\n"
                "Find all requested listings using the screenshot, their positions, and the indexed text. "
                f"The answer needs {link_count[0]} to {link_count[1]} distinct listing links. "
                "When the full requested group is visible, select the text index for each matching listing TITLE, "
                'such as {"listings":[10,12]}. Exclude navigation, prices, labels, and unrelated cards. '
                "Check any required ordering, category, price, and result scope before selecting. "
                "If the right results are not shown, issue one navigation command. "
                'Use {"action":"CLICK","value":null,"control":index} for a visible button or filter, '
                '{"action":"SEARCH","value":"short query","control":index} on the empty Keyword placeholder to search, '
                '{"action":"ENTER","value":null,"position":null} to submit a filled search field, or '
                '{"action":"SCROLL","value":"down","position":null} to see lower listings. '
                'Return {"listings":null} only if you cannot select or navigate. Never type or invent a URL. '
                f"Return one JSON object.\nPrevious actions: {json.dumps(recent, ensure_ascii=False)}.\n"
                f"Visible text:\n{controls}"
            )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="locate", evidence_frame=frame, provider_format="evidence_invalid")
        if rows is not None:
            self.last_metrics["screen_text"] = rows
        try:
            selection = listing_selection(raw, rows, canvas.size, link_count) if link_count is not None else None
            if selection is None:
                navigation, box, candidate, status = parse_evidence_location(
                    raw, observation.goal, canvas.size, confirm=self.confirm, rows=navigation_rows, guarded_ocr=True,
                    navigation_only=link_count is not None)
        except (ValueError, RecursionError):
            return None
        if selection is not None:
            indices, points = selection
            self.last_metrics["provider_format"] = "evidence_link_abstained"
            if indices:
                receipt = self.listing_preview(points)
                destinations = preview_destinations(receipt, frame, points, self.listing_site)
                self.last_metrics["listing_preview"] = receipt
                self.last_metrics["provider_format"] = "evidence_link_preview"
                if destinations is not None:
                    self.pending_links = frame, rows, indices, destinations, link_count
            return None
        self.last_metrics["provider_format"] = status
        if navigation is not None:
            return json.dumps(navigation.to_dict(), ensure_ascii=False)
        if box:
            crop, bounds = native_crop(observation.screenshot, canvas.size, box)
            self.last_metrics["evidence_bounds"] = list(bounds)
            if candidate is not None:
                self.pending_answer = frame, crop, bounds, candidate
            else:
                self.pending_crop = frame, crop, bounds
        return None

    def read(self, observation, frame, crop, bounds):
        canvas = self.actor.prepare_image(crop)
        visibility_check = ("If the requested details cannot be read, use null for answer. "
                            "A separate visual check will confirm that this is the correct item. "
                            if self.navigation_context is not None else
                            "If the requested subject, scope, or any required detail cannot be verified, use null for answer. ")
        prompt = (
            f"Question: {observation.goal}\n"
            "Inspect this crop. Identify the requested subject and list at most three brief, relevant facts. "
            "Omit unrelated text. For a count, count each visible group separately before adding. "
            "For a price range, give both endpoints. For a total price, read rates and pack sizes before calculating. "
            f"{visibility_check}"
            'Return one JSON object with exactly two keys: "evidence" (a list of short strings), '
            'then "answer" (short string or null).'
        )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="read", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_crop_sha256=pixel_hash(crop), provider_format="evidence_invalid")
        try:
            answer, status = evidence_answer(raw, observation.goal, allow_string_abstention=self.confirm)
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
        panel = confirmation_image(observation.screenshot, crop, bounds, self.navigation_context)
        canvas = self.actor.prepare_image(panel)
        context_description = (
            "The bottom-left panel shows the previous viewport. Its orange circle marks the click that changed the view. "
            "Use it when the question depends on how this item was selected. "
            if self.navigation_context is not None else ""
        )
        prompt = (
            f"Question: {observation.goal}\n"
            "The upper-left panel is the current viewport and the blue outline marks the area enlarged on the right. "
            f"{context_description}"
            "Check the requested item's identity and the full scope in the overview, then read the enlargement. "
            "For a page, row, or search question, verify that the required results are visible. "
            "Give every requested component. A single price cannot answer a price range. "
            "If anything required is missing, return {\"answer\":null}. "
            "Otherwise return one JSON object with one key answer and a short string value."
        )
        raw = self.actor.ask(canvas, prompt)
        self.last_metrics.update(evidence_stage="confirm", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_panel_sha256=pixel_hash(panel), provider_format="evidence_invalid")
        if self.navigation_context is not None:
            previous, position = self.navigation_context
            self.last_metrics.update(evidence_context_sha256=pixel_hash(previous), evidence_context_position=list(position))
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

    def verify_links(self, observation, frame, rows, selected, destinations, count):
        canvas = self.actor.prepare_image(observation.screenshot)
        raw = self.actor.ask(canvas, (
            f"Visible text from the screenshot:\n{indexed_text(rows)}\n\n"
            f"Question: {observation.goal}\n"
            "Identify the complete set of matching listing titles using the screenshot. "
            "Check the scope, ordering, and visible card details. Exclude headings and controls. "
            'Return one JSON object keyed "listings". Its value must be an array of the actual text indices '
            f"from the listing titles, with {count[0]} to {count[1]} distinct entries. "
            "Use null if any required listing or condition is unverifiable."
        ))
        self.last_metrics.update(evidence_stage="link_confirm", evidence_frame=frame,
                                 provider_format="evidence_link_disagreed")
        try:
            selection = listing_selection(raw, rows, canvas.size, count)
            answer = agreed_listing_answer(selected, destinations, selection[0] if selection is not None else None)
        except (ValueError, RecursionError):
            answer = None
        if answer is None:
            return None
        self.last_metrics["provider_format"] = "evidence_link_agreed"
        return json.dumps(answer.to_dict(), ensure_ascii=False)

    def predict(self, observation, history):
        frame = pixel_hash(observation.screenshot)
        if self.pending_links is not None:
            pending_frame, rows, indices, destinations, count = self.pending_links
            self.pending_links = None
            if frame == pending_frame:
                return self.verify_links(observation, frame, rows, indices, destinations, count)
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
            return self.locate(observation, frame, history)
        return self.actor.predict(observation, history, allow_answer=False)
