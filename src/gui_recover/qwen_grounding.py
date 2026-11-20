"""Ground controls with a separate Qwen2.5-VL checkpoint."""

import json
import re
from math import isfinite
from pathlib import Path

from .actions import ActionError, _unique_dict, parse_action
from .policies import ShowUIPolicy


QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_7B_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_7B_MIRROR_REVISION = "c03ef5ea2cb88c1aaf8c1231f58dcd8b4a80374f"
_LINE_SCROLL = re.compile(r"Action: SCROLL\nValue: ([+-]?[0-9]{1,6})(?:\nPosition: \[[0-9]{1,5}, [0-9]{1,5}\])?")
VISIBLE_ANSWER_PROMPT = (
    "Check whether this screenshot contains enough evidence to answer the question. "
    "For questions about a particular item or photo, check that it is the requested one. "
    "If any required detail is missing, illegible, or requires another view, return {\"answer\":null}. "
    "Otherwise return one JSON object with the short answer as the string value of the key answer. "
    "Include every requested part. Read the screenshot directly and avoid guessing. "
    "Return one JSON object only.\n"
)


def read_reply(raw):
    if len(raw) > 4096:
        raise ActionError("Qwen response is too long")
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    return json.loads(raw, object_pairs_hook=_unique_dict)


def parse_pixel_point(raw, width, height):
    try:
        data = read_reply(raw)
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict) or set(data) != {"point_2d"}:
            raise ValueError("Expected a point_2d object")
        point = data["point_2d"]
        if not isinstance(point, list) or len(point) != 2 or any(type(x) is not int for x in point):
            raise ValueError("Expected two integer image pixels")
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Point lies outside the prepared image")
    except (ValueError, RecursionError) as error:
        raise ActionError("Invalid Qwen pixel coordinates") from error
    return ((x + 0.5) / width, (y + 0.5) / height)


def parse_visible_answer(raw, *, allow_numeric=False):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer"}:
        raise ActionError("Expected one answer field")
    answer = data["answer"]
    if answer is None:
        return None
    if allow_numeric and (type(answer) is int or (type(answer) is float and isfinite(answer))):
        answer = str(answer)
    if not isinstance(answer, str) or not 0 < len(answer.strip()) <= 512:
        raise ActionError("Expected a nonempty short answer or null")
    return answer.strip()


def native_input_pair(raw):
    """Recognize one coordinate click followed by one text input."""
    text = raw.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_dict)
    first, offset = decoder.raw_decode(text)
    remainder = text[offset:].lstrip()
    second, offset = decoder.raw_decode(remainder)
    if (remainder[offset:].strip() or not isinstance(first, dict) or not isinstance(second, dict)
            or set(first) != {"action", "coordinate"} or first["action"] != "click"
            or set(second) not in ({"action", "text"}, {"action", "value"}) or second["action"] != "type"):
        raise ActionError("Unexpected native action sequence")
    return {"action": "INPUT", "position": first["coordinate"], "value": second.get("text", second.get("value"))}


def normalize_qwen_action(data, size, *, native_navigation=False):
    if not isinstance(data, dict) or set(data) != {"action", "value", "position"}:
        raise ActionError("Expected one action dictionary")
    if not isinstance(data["action"], str):
        raise ActionError("Expected one named action")
    kind, value, position = data["action"].upper(), data["value"], data["position"]
    if kind == "LEFT_CLICK":
        kind = "CLICK"
    if kind == "CLICK" and isinstance(value, str):
        # A visible label may accompany the predicted point, but the executor
        # consumes only the verified position.
        value = None
    if kind in ("CLICK", "INPUT"):
        position = parse_pixel_point(json.dumps({"point_2d": position}), *size)
    elif kind == "SCROLL":
        finite_number = type(value) is int or (type(value) is float and isfinite(value))
        if finite_number and value:
            value = "up" if value < 0 else "down"
        position = None
    action = parse_action(json.dumps({"action": kind, "value": value, "position": position}))
    if action.action not in ("CLICK", "INPUT", "SCROLL", "ANSWER") and not (
            native_navigation and action.action == "ENTER"):
        raise ActionError("Unsupported Qwen navigation action")
    return action


class QwenGroundingPolicy:
    """A bounded point backend with the same ground/image/query contract as ShowUI."""

    _generate = ShowUIPolicy._generate

    def __init__(self, model, *, revision=QWEN_REVISION, max_new_tokens=48, max_pixels=448 * 448,
                 device="cuda", load_in_4bit=False, native_navigation=False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device not in ("cuda", "cpu"):
            raise ValueError("Expected Qwen device cuda or cpu")
        if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("Qwen CUDA grounding requires a BF16 GPU")
        if load_in_4bit and device != "cuda":
            raise ValueError("The 4-bit profile requires CUDA")
        model_dir = Path(model).resolve(strict=True)
        manifest = None
        weights = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            manifest = json.loads((model_dir / "download_manifest.json").read_text())
            if (manifest["repository"] != QWEN_7B_MODEL
                    or manifest["weight_revision"] != QWEN_7B_MIRROR_REVISION):
                raise ValueError("Expected the pinned Qwen 7B snapshot")
            weights = {"device_map": {"": 0},
                       "quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)}
        model = str(model_dir)
        self.device = "cuda:0" if device == "cuda" else "cpu"
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.native_navigation = native_navigation
        self.min_pixels = 56 * 56
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels, trust_remote_code=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            trust_remote_code=False, **weights)
        if not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.last_metrics = {}
        self.recovery_feedback = None
        self.config = {"model": QWEN_MODEL, "revision": revision, "dtype": "bfloat16", "device": device,
            "min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "max_new_tokens": self.max_new_tokens,
            "do_sample": False, "coordinate_format": "integer pixels in the resized image"}
        if manifest is not None:
            self.config.update(model=QWEN_7B_MODEL, revision=QWEN_7B_MIRROR_REVISION,
                               quantization="bitsandbytes NF4 double quantization", asset_provenance=manifest)
        if native_navigation:
            self.config["native_navigation"] = True

    def reset(self, seed):
        self.torch.manual_seed(seed)
        self.recovery_feedback = None
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def set_feedback(self, message):
        self.recovery_feedback = message

    def prepare_image(self, screen):
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        height, width = smart_resize(screen.height, screen.width, factor=28,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return screen.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)

    def ask(self, canvas, request):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": canvas}, {"type": "text", "text": request}]}]
        raw = self._generate(messages)
        self.last_metrics.update(provider_output=raw, provider_image_size=list(canvas.size))
        return raw

    def ground(self, screen, query):
        canvas = self.prepare_image(screen)
        width, height = canvas.size
        request = (f"Locate the center of this clickable element: {query}\n"
                   f"The image is {width} pixels wide and {height} pixels high. "
                   "Use absolute pixels with the top-left as origin. "
                   'Return exactly one JSON object of the form {"point_2d":[x,y]} with integer coordinates.')
        raw = self.ask(canvas, request)
        try:
            return json.dumps(parse_pixel_point(raw, width, height))
        except ActionError:
            return ""

    def check_answer(self, observation):
        canvas = self.prepare_image(observation.screenshot)
        raw = self.ask(canvas, f"{VISIBLE_ANSWER_PROMPT}Question: {observation.goal}")
        self.last_metrics["provider_format"] = "answer_invalid"
        try:
            answer = parse_visible_answer(raw)
        except (ValueError, RecursionError):
            return None
        self.last_metrics["provider_format"] = "answer_abstained" if answer is None else "answer_json"
        return answer

    def predict(self, observation, history, *, allow_answer=True):
        canvas = self.prepare_image(observation.screenshot)
        width, height = canvas.size
        native_navigation = getattr(self, "native_navigation", False)
        history = history[-4:]
        if not allow_answer:
            pixel_history = []
            for entry in history:
                action = dict(entry["action"])
                if action["position"] is not None:
                    x, y = action["position"]
                    action["position"] = [min(width - 1, int(x * width)), min(height - 1, int(y * height))]
                pixel_history.append({**entry, "action": action})
            history = pixel_history
        allowed_actions = ("ANSWER has the exact short answer as value and null position. Use only these four actions."
                           if allow_answer else "Use only CLICK, INPUT or SCROLL to find the missing visual evidence. "
                           'For example, {"action":"SCROLL","value":"down","position":null}. '
                           "Click a relevant photo or listing to inspect it, or scroll to reveal more of the page.")
        if native_navigation:
            allowed_actions = allowed_actions.replace(" Use only these four actions.", "")
            allowed_actions += ' When a search box is focused, use {"action":"ENTER","value":null,"position":null} to submit it.'
        request = (
            f"Task: {observation.goal}\n"
            f"Previous actions: {json.dumps(history, ensure_ascii=False)}\n"
            f"Latest execution error: {observation.last_action_error}\n"
            f"{self.recovery_feedback or ''}\n"
            "Return exactly one JSON object with keys action, value, position. "
            f"Use integer pixel positions in this {width} by {height} image. "
            "CLICK has null value and [x,y] position. INPUT has string value and [x,y]. "
            "SCROLL has value up or down and null position. "
            f"{allowed_actions} Inspect the image and avoid repeating a failed action."
        )
        raw = self.ask(canvas, request)
        response_format = "json"
        self.last_metrics["provider_format"] = "invalid"
        try:
            try:
                data = read_reply(raw)
            except json.JSONDecodeError:
                # Some single scroll replies use labelled lines. The scroll
                # executor always uses the viewport center and a fixed step.
                match = _LINE_SCROLL.fullmatch(raw.strip())
                if match is None:
                    if not native_navigation:
                        raise
                    data = native_input_pair(raw)
                    response_format = "native_navigation"
                else:
                    data = {"action": "SCROLL", "value": int(match[1]), "position": None}
                    response_format = "line_scroll"
            if native_navigation and isinstance(data, dict):
                if set(data) == {"action", "coordinate"} and data["action"] == "click":
                    data = {"action": "CLICK", "value": None, "position": data["coordinate"]}
                    response_format = "native_navigation"
                elif data == {"action": "key", "key": "Enter"} or data == {"action": "key", "key": "Return"}:
                    data = {"action": "ENTER", "value": None, "position": None}
                    response_format = "native_navigation"
            action = normalize_qwen_action(data, canvas.size, native_navigation=native_navigation)
            if action.action == "ANSWER" and not allow_answer:
                self.last_metrics["provider_format"] = "answer_blocked"
                return None
            self.last_metrics["provider_format"] = response_format
            return json.dumps(action.to_dict())
        except (ValueError, RecursionError):
            return ""
