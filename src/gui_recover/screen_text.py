"""Optional OCR for grounding actions in rendered text."""

import json
import math
import re

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
        self.config = {"rapidocr": version("rapidocr"), "onnxruntime": version("onnxruntime")}

    def __call__(self, screen, canvas_size):
        import numpy as np

        image = np.ascontiguousarray(np.asarray(screen.convert("RGB"))[:, :, ::-1])
        result = self.engine(image)
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


def exact_initial_search(goal, rows, size):
    match = re.match(r'^\s*search for\s+"([^"\n]{1,80})"', goal, re.IGNORECASE)
    if match is None or not any(row["text"].casefold() == "search" for row in rows):
        return None
    fields = [index for index, row in enumerate(rows) if row["text"].casefold().startswith(("e.g.", "eg,"))]
    if len(fields) != 1 or not any(row["text"].casefold() == "keyword" for row in rows):
        return None
    raw = json.dumps({"action": "SEARCH", "value": match[1], "control": fields[0]})
    return from_readable_control(raw, rows, size, guarded=True)
