"""Shared-weight, paired grounding evaluation; labels never enter generation."""

import contextlib
import json
from pathlib import Path
import time

from .actions import ActionError
from .grounding import parse_point
from .policies import MODEL_REVISION, sft_adapter_metadata
from .sft import file_sha256, grounding_messages, grounding_query


HEADS = {
    "base_plain": ("base", False), "base_history": ("base", True),
    "visual_plain": ("visual", False), "visual_history": ("visual", True),
    "history_plain": ("history", False), "history_history": ("history", True),
}
DEFAULT_HEADS = ("base_plain", "visual_plain", "history_history")


def verify_ablation_inputs(data, visual, history):
    manifests = [json.loads((Path(p)/"training.json").read_text()) for p in (visual,history)]
    sha = file_sha256(Path(data)/"manifest.json")
    if [m.get("context_mode","none") for m in manifests] != ["none","previous-action"]:
        raise ValueError("Expected no-history and action-history adapters")
    for key in ("base_revision","base_weights_sha256","dataset_manifest_sha256","train_records",
                "validation_records","completed_updates"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Ablation differs in {key}")
    if manifests[0]["config"] != manifests[1]["config"]:
        raise ValueError("Ablation requires the identical config, data order and update budget")
    if manifests[0]["dataset_manifest_sha256"] != sha:
        raise ValueError("Dataset manifest differs from training")
    if manifests[0]["adapter_sha256"] == manifests[1]["adapter_sha256"]:
        raise ValueError("Conditioning intervention produced identical adapter bytes")
    if manifests[0]["runtime"]["git_commit"] != manifests[1]["runtime"]["git_commit"]:
        raise ValueError("Adapters were trained from different code revisions")
    for path in (visual,history):
        sft_adapter_metadata(path,MODEL_REVISION)
    return manifests


class AblationGrounder:
    def __init__(self, model_path, data, visual_adapter, history_adapter):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.training = verify_ablation_inputs(data,visual_adapter,history_adapter)
        if file_sha256(Path(model_path)/"pytorch_model.bin") != self.training[0]["base_weights_sha256"]:
            raise ValueError("Evaluation base differs")
        if file_sha256(Path(visual_adapter)/"tokenizer.json") != file_sha256(Path(history_adapter)/"tokenizer.json"):
            raise ValueError("Adapter tokenizers differ")
        self.profile = self.training[0]["config"]["training"]
        self.seed = self.training[0]["config"]["seed"]
        self.processor = AutoProcessor.from_pretrained(visual_adapter,
            min_pixels=self.profile["min_pixels"],max_pixels=self.profile["max_pixels"],trust_remote_code=False)
        base = Qwen2VLForConditionalGeneration.from_pretrained(model_path,revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,attn_implementation="sdpa",trust_remote_code=False).to("cuda:0")
        self.model = PeftModel.from_pretrained(base,visual_adapter,adapter_name="visual",is_trainable=False)
        self.model.load_adapter(history_adapter,adapter_name="history",is_trainable=False)
        self.model.eval()
        self.calls = 0
        torch.cuda.reset_peak_memory_stats()

    def predict(self, picture, query, previous, head):
        weights,with_history = HEADS[head]
        action = previous if with_history else None
        messages = grounding_messages(picture,query,previous_action=action)
        prompt = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs = self.processor(text=[prompt],images=[picture],return_tensors="pt").to("cuda:0")
        if weights != "base":
            self.model.set_adapter(weights)
        context = self.model.disable_adapter() if weights == "base" else contextlib.nullcontext()
        self.torch.manual_seed(self.seed)
        before = time.perf_counter()
        with context,self.torch.inference_mode():
            ids = self.model.generate(**inputs,max_new_tokens=32,do_sample=False)
        raw = self.processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        try:
            point = parse_point(raw)
        except ActionError:
            point = None
        self.calls += 1
        return {"raw":raw,"point":point,"input_query":grounding_query(query,action),
                "inference_seconds":time.perf_counter()-before}
