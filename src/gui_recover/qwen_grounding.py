"""Ground controls with a separate Qwen2.5-VL checkpoint."""

import json
from pathlib import Path

from .actions import ActionError, _unique_dict
from .policies import ShowUIPolicy


QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_pixel_point(raw, width, height):
    if len(raw) > 4096:
        raise ActionError("Qwen grounding response is too long")
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    try:
        data = json.loads(raw, object_pairs_hook=_unique_dict)
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


class QwenGroundingPolicy:
    """A bounded point backend with the same ground/image/query contract as ShowUI."""

    _generate = ShowUIPolicy._generate

    def __init__(self, model, *, revision=QWEN_REVISION, max_new_tokens=48, max_pixels=448 * 448, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device not in ("cuda", "cpu"):
            raise ValueError("Expected Qwen device cuda or cpu")
        if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("Qwen CUDA grounding requires a BF16 GPU")
        model = str(Path(model).resolve(strict=True))
        self.device = "cuda:0" if device == "cuda" else "cpu"
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.min_pixels = 56 * 56
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels, trust_remote_code=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            trust_remote_code=False).to(self.device).eval()
        self.last_metrics = {}
        self.config = {"model": QWEN_MODEL, "revision": revision, "dtype": "bfloat16", "device": device,
            "min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "max_new_tokens": self.max_new_tokens,
            "do_sample": False, "coordinate_format": "integer pixels in the resized image"}

    def ground(self, screen, query):
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        height, width = smart_resize(screen.height, screen.width, factor=28,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        canvas = screen.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        request = (f"Locate the center of this clickable element: {query}\n"
                   f"The image is {width} pixels wide and {height} pixels high. "
                   "Use absolute pixels with the top-left as origin. "
                   'Return exactly one JSON object of the form {"point_2d":[x,y]} with integer coordinates.')
        messages = [{"role": "user", "content": [
            {"type": "image", "image": canvas}, {"type": "text", "text": request}]}]
        raw = self._generate(messages)
        self.last_metrics.update(provider_output=raw, provider_image_size=[width, height])
        try:
            return json.dumps(parse_pixel_point(raw, width, height))
        except ActionError:
            return ""
