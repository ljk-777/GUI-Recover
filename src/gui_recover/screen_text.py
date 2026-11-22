"""Optional OCR for grounding actions in rendered text."""

import json
import math
import re
from threading import Lock

from .actions import Action, ActionError, parse_action
from .qwen_grounding import normalize_qwen_action, read_reply


def visible_rows(result, source_size, canvas_size):
    """Map OCR quadrilaterals to bounded rectangles in the model canvas."""
    if result.boxes is None:
        return []
    source_width, source_height = source_size
    width, height = canvas_size
    rows = []
    for points, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
        if not math.isfinite(score) or score < 0.65 or points.shape != (4, 2):
            continue
        if not all(math.isfinite(value) for point in points for value in point):
            continue
        xs, ys = points.T
        left = max(0, math.floor(min(xs) * width / source_width))
        top = max(0, math.floor(min(ys) * height / source_height))
        right = min(width, math.ceil(max(xs) * width / source_width))
        bottom = min(height, math.ceil(max(ys) * height / source_height))
        label = " ".join(text.split())[:100]
        if label and left < right and top < bottom:
            rows.append({"text": label, "bounds": [left, top, right, bottom]})
    rows.sort(key=lambda row: (row["bounds"][1], row["bounds"][0]))
    return rows[:56]


class RapidScreenReader:
    """Read pixels locally on the CPU. Imports the optional backend lazily."""

    def __init__(self):
        from importlib.metadata import version

        from rapidocr import RapidOCR

        self.engine = RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": 2,
                                       "EngineConfig.onnxruntime.inter_op_num_threads": 1})
        self._default_orientation = self.engine.use_cls
        self._lock = Lock()
        self.config = {"rapidocr": version("rapidocr"), "onnxruntime": version("onnxruntime")}

    def __call__(self, screen, canvas_size):
        return self._read(screen, canvas_size, use_cls=self._default_orientation)

    def read_upright(self, screen, canvas_size):
        return self._read(screen, canvas_size, use_cls=False)

    def _read(self, screen, canvas_size, *, use_cls):
        import numpy as np

        image = np.ascontiguousarray(np.asarray(screen.convert("RGB"))[:, :, ::-1])
        # RapidOCR retains the last orientation setting on the shared engine.
        with self._lock:
            result = self.engine(image, use_cls=use_cls)
        return visible_rows(result, screen.size, canvas_size)


def from_readable_control(raw, rows, size, *, guarded=False):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "control" not in data:
        return None
    if set(data) != {"action", "value", "control"} or data["action"] not in ("CLICK", "INPUT", "SEARCH"):
        raise ActionError("Expected one click, input, or search on an OCR line")
    index = data["control"]
    if type(index) is not int or not 0 <= index < len(rows):
        raise ActionError("OCR control index is out of range")
    row = rows[index]
    left, top, right, bottom = row["bounds"]
    width, height = size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ActionError("OCR control lies outside the model canvas")
    point = [(left + right) // 2, (top + bottom) // 2]
    label = row["text"].casefold()
    empty_keyword = label.startswith(("e.g.", "eg,")) and any(
        item["text"].casefold() == "keyword" for item in rows)
    if data["action"] == "SEARCH":
        if not empty_keyword:
            raise ActionError("Search needs a recognized empty keyword field")
        typed = normalize_qwen_action({"action": "INPUT", "value": data["value"], "position": point}, size)
        return parse_action(json.dumps(Action("SEARCH", typed.value, typed.position).to_dict()))
    if guarded and data["action"] == "INPUT" and not empty_keyword:
        raise ActionError("Text input needs a recognized field rather than an arbitrary label")
    return normalize_qwen_action({"action": data["action"], "value": data["value"], "position": point}, size)


def listing_detail(rows):
    labels = [row["text"].casefold() for row in rows]
    return (any(text.startswith("contact publisher") for text in labels)
            or (any(text.startswith("published date:") for text in labels)
                and any(text.startswith("location:") for text in labels)))


def empty_keyword_field(rows):
    fields = [index for index, row in enumerate(rows) if row["text"].casefold().startswith(("e.g.", "eg,"))]
    labels = {row["text"].casefold() for row in rows}
    return fields[0] if len(fields) == 1 and {"keyword", "search"} <= labels else None


def keyword_search(query, rows, size):
    field = empty_keyword_field(rows)
    if field is None:
        return None
    raw = json.dumps({"action": "SEARCH", "value": query, "control": field})
    return from_readable_control(raw, rows, size, guarded=True)


def narrow_keyword_search(query, rows, size):
    """Replace a visible filled query with its first distinctive term."""
    words = query.split()
    if len(words) < 2 or re.fullmatch(r"[^\W_]{4,30}", words[0]) is None:
        return None
    return replace_keyword_search(query, words[0], rows, size)


def replace_keyword_search(query, replacement, rows, size):
    """Replace a verified filled keyword field in the sidebar."""
    labels = {row["text"].casefold() for row in rows}
    fields = [row["bounds"] for row in rows if row["text"].casefold() == query.casefold()]
    if len(fields) != 1 or not {"your search", "apply"} <= labels:
        return None
    left, top, right, bottom = fields[0]
    width, height = size
    return Action("SEARCH_REPLACE", replacement, ((left + right) / (2 * width), (top + bottom) / (2 * height)))


def exact_initial_search(goal, rows, size):
    match = re.match(r'^\s*search for\s+"([^"\n]{1,80})"', goal, re.IGNORECASE)
    return keyword_search(match[1], rows, size) if match is not None else None
