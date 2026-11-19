"""Frozen-actor repair inputs. No environment, teacher or outcome labels here."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from .actions import ActionError
from .grounding import parse_point
from .sft import file_sha256
from .verification import TransitionView, pixel_hash


REPAIR_MODES = ("plain", "feedback", "zoom", "feedback_zoom")
FEEDBACK = ("A recovery check was requested after the previous proposed interaction: {action}.\n"
            "The target may not have been reached. Reinspect the current image. "
            "Repeating the previous location is appropriate only if it still locates the target.\n"
            "Locate the CURRENT target: {target}")


@dataclass(frozen=True)
class RepairInput:
    picture: Image.Image
    query: str
    crop: tuple[int, int, int, int]
    original_size: tuple[int, int]
    mode: str

    def map_point(self, point):
        if point is None:
            return None
        point = parse_point(json.dumps(point))
        left, top, right, bottom = self.crop
        width, height = self.original_size
        return ((left + point[0] * (right-left))/width,
                (top + point[1] * (bottom-top))/height)


def prepare_repair(view: TransitionView, mode: str, fraction=0.6):
    if mode not in REPAIR_MODES or not 0 < fraction <= 1:
        raise ValueError("Unknown repair mode or invalid crop fraction")
    if view.before.size != view.after.size:
        raise ValueError("Repair requires unchanged screenshot dimensions")
    picture = view.after.convert("RGB")
    width, height = picture.size
    crop = (0, 0, width, height)
    if mode in ("zoom", "feedback_zoom"):
        cw, ch = max(1, round(width*fraction)), max(1, round(height*fraction))
        x, y = view.proposed.position or (0.5, 0.5)
        left = min(max(round(x*width-cw/2), 0), width-cw)
        top = min(max(round(y*height-ch/2), 0), height-ch)
        crop = (left, top, left+cw, top+ch)
        # This is a declared model-input transform, never a replacement for
        # the native environment evidence. Keep the canvas/pixel budget fixed.
        picture = picture.crop(crop).resize((width, height), Image.Resampling.BICUBIC)
    query = view.target
    if mode in ("feedback", "feedback_zoom"):
        # Coordinates are deliberately omitted after the earlier anchoring
        # regressions. Only the agent's own proposed action/value is admitted.
        action = {k: v for k, v in view.proposed.to_dict().items() if k != "position"}
        query = FEEDBACK.format(action=json.dumps(action, separators=(",", ":")), target=view.target)
    return RepairInput(picture, query, crop, (width, height), mode)


class RepairGrounder:
    """Observable-input cache; logical requests and actual generations differ."""
    def __init__(self, actor, directory):
        self.actor = actor
        self.directory = Path(directory)
        (self.directory/"model_inputs").mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.calls = self.requests = 0
        self.actual_input_tokens = self.actual_generated_tokens = 0

    @staticmethod
    def key(picture, query):
        return hashlib.sha256(json.dumps([pixel_hash(picture), query], ensure_ascii=False).encode()).hexdigest()

    def prime(self, picture, query, raw, provenance):
        key = self.key(picture, query)
        if key in self.cache and self.cache[key]["raw"] != raw:
            raise ValueError("Conflicting original predictions for identical observable inputs")
        if key in self.cache:
            return
        try:
            point = parse_point(raw)
        except ActionError:
            point = None
        self.cache[key] = {"raw": raw, "point": point, "metrics": None,
                           "source_prediction_provenance": provenance}

    def ground(self, picture, query):
        self.requests += 1
        key = self.key(picture, query)
        cached = key in self.cache
        if not cached:
            started = time.perf_counter()
            raw = self.actor.ground(picture, query)
            try:
                point = parse_point(raw)
            except ActionError:
                point = None
            metrics = deepcopy(self.actor.last_metrics)
            self.actual_input_tokens += metrics.get("input_tokens", 0)
            self.actual_generated_tokens += metrics.get("generated_tokens", 0)
            self.cache[key] = {"raw": raw, "point": point, "metrics": metrics,
                               "generation_seconds": time.perf_counter()-started}
            self.calls += 1
        result = {**deepcopy(self.cache[key]), "cache_key": key, "cache_hit": cached}
        result.update(input_pixel_hash=pixel_hash(picture), input_size=list(picture.size), query=query)
        with (self.directory/"requests.jsonl").open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False)+"\n")
        return result

    def repair(self, view, mode, fraction=0.6, input_path=None):
        prepared = prepare_repair(view, mode, fraction)
        result = self.ground(prepared.picture, prepared.query)
        if input_path is None:
            input_path = self.directory/"model_inputs"/(result["input_pixel_hash"]+".png")
        input_path = Path(input_path)
        prepared.picture.save(input_path)
        result.update(local_point=result["point"], point=prepared.map_point(result["point"]),
                      crop=list(prepared.crop), original_size=list(prepared.original_size),
                      repair_mode=mode, input_image=input_path.name,
                      input_image_sha256=file_sha256(input_path))
        return result

    def counters(self):
        return {"logical_requests": self.requests, "actual_generations": self.calls,
                "actual_input_tokens": self.actual_input_tokens,
                "actual_generated_tokens": self.actual_generated_tokens,
                "scope": "Actual token totals exclude cached/reused original predictions with unavailable token counts"}


class ContextualRepairPolicy:
    def __init__(self, grounder, mode, fraction=0.6, input_path=None):
        if mode not in REPAIR_MODES:
            raise ValueError("Unknown repair policy")
        self.grounder, self.mode, self.fraction, self.input_path = grounder, mode, fraction, input_path
        self.config = {"mode": mode, "crop_fraction": fraction,
                       "feedback_sha256": hashlib.sha256(FEEDBACK.encode()).hexdigest(),
                       "uses_teacher": False, "coordinates_in_feedback": False}

    def predict(self, view):
        return self.grounder.repair(view, self.mode, self.fraction, self.input_path)
