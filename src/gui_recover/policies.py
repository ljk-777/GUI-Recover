"""Policy implementations receive screenshots, task text and action history."""

import json
from pathlib import Path
from typing import Protocol

from .environment import VisualObservation
from .prompts import GROUNDING, OFFICIAL_MINIWOB, OFFICIAL_WEB

MODEL_ID = "showlab/ShowUI-2B"
MODEL_REVISION = "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"


def sft_adapter_metadata(directory, revision):
    from .sft import file_sha256
    path = Path(directory)
    training = json.loads((path / "training.json").read_text())
    if training.get("status") != "complete" or training.get("base_revision") != revision:
        raise ValueError("SFT adapter has not passed reload acceptance for this base revision")
    sha = file_sha256(path / "adapter_model.safetensors")
    if sha != training["adapter_sha256"]:
        raise ValueError("SFT adapter checksum differs")
    return {"path": str(path), "sha256": sha, "base_revision": revision,
            "base_weights_sha256": training["base_weights_sha256"],
            "training_code": training["runtime"]["git_commit"], "updates": training["completed_updates"]}
NAV_PROMPT = """Navigate the web screen using the task, screenshot and previous actions.
Return exactly ONE dictionary with keys action, value, position. No prose or action list.
Position is [x,y] in normalized screenshot coordinates, each between 0 and 1.
Actions:
CLICK: position required, value=None.
INPUT: click position then type value (string); appends to the focused field.
HOVER: position required, value=None.
ENTER: value=None, position=None.
SCROLL: value is up/down/left/right, position=None.
ANSWER: value is your final message, position=None; requests stopping.
Observe the new screenshot after each action before choosing the next action.
"""


class Policy(Protocol):
    def predict(self, obs: VisualObservation, history: list[dict]) -> str: ...


class OracleSmokePolicy:
    """Privileged diagnostic baseline, NOT a model score."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, history):
        return json.dumps({"action": "CLICK", "value": None,
                           "position": self.env.diagnostic_button_position()})


class ShowUIPolicy:
    def __init__(self, model=MODEL_ID, revision=MODEL_REVISION, max_pixels=784 * 784,
                 max_new_tokens=128, prompt_profile="baseline", adapter=None):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        adapter_config = sft_adapter_metadata(adapter, revision) if adapter else None
        if adapter_config:
            from .sft import file_sha256
            if file_sha256(Path(model) / "pytorch_model.bin") != adapter_config["base_weights_sha256"]:
                raise ValueError("Adapter requires the exact local training base weights")

        if not torch.cuda.is_available():
            raise RuntimeError("This inference profile requires CUDA; run oracle-smoke on the Mac")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This inference profile requires a BF16-capable GPU")
        self.torch = torch
        self.device = "cuda:0"
        self.prompt_profile = prompt_profile
        self.prompt = {"baseline": NAV_PROMPT, "official-web": OFFICIAL_WEB,
                       "official-miniwob": OFFICIAL_MINIWOB}[prompt_profile]
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model, revision=revision,
                min_pixels=256 * 28 * 28, max_pixels=max_pixels, trust_remote_code=False)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model, revision=revision,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                trust_remote_code=False).to("cuda:0").eval()
        if adapter_config:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False).eval()
        self.config = {"model": model, "revision": revision, "dtype": "bfloat16",
                       "attention": "sdpa", "min_pixels": 256 * 28 * 28,
                       "max_pixels": max_pixels, "max_new_tokens": max_new_tokens,
                       "do_sample": False, "history_limit": 8, "prompt": self.prompt,
                       "prompt_profile": prompt_profile}
        if adapter_config:
            self.config["adapter"] = adapter_config
        provenance = Path(model) / "download_manifest.json"
        if provenance.is_file():
            self.config["asset_provenance"] = json.loads(provenance.read_text())
        self.last_metrics = {}
        self.recovery_feedback = None

    def set_feedback(self, message):
        self.recovery_feedback = message

    def reset(self, seed):
        self.recovery_feedback = None
        self.torch.manual_seed(seed)
        self.torch.cuda.reset_peak_memory_stats()

    def predict(self, obs, history):
        context = f"Task: {obs.goal}"
        if self.prompt_profile == "baseline" or history:
            context += f"\nHistory: {json.dumps(history[-8:])}"
        if self.prompt_profile == "baseline" or obs.last_action_error:
            context += f"\nExecution error: {obs.last_action_error}"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": self.prompt},
            {"type": "text", "text": context},
            {"type": "image", "image": obs.screenshot},
        ]}]
        if self.prompt_profile == "official-miniwob":
            # Matches the source template's interleaved text history (tttt).
            messages[0]["content"] = [
                {"type": "text", "text": self.prompt},
                {"type": "text", "text": f"Task: {obs.goal}"},
                *[{"type": "text", "text": str(item["action"])} for item in history[-8:]],
                {"type": "image", "image": obs.screenshot},
            ]
        if self.recovery_feedback:
            messages[0]["content"].insert(-1, {"type": "text", "text": self.recovery_feedback})
        return self._generate(messages)

    def ground(self, screenshot, query):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": GROUNDING},
            {"type": "image", "image": screenshot},
            {"type": "text", "text": query},
        ]}]
        return self._generate(messages)

    def _generate(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        inputs = self.processor(text=[text], images=images, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        result = self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        cuda = self.device.startswith("cuda")
        self.last_metrics = {"input_tokens": int(inputs.input_ids.shape[1]),
                "model_invocations": 1,
                "generated_tokens": int(ids.shape[1] - inputs.input_ids.shape[1]),
                "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved()}
        return result
