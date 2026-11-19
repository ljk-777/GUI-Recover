"""Independent BF16 inference: same base, processor and budget for both arms."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from gui_recover.actions import ActionError
from gui_recover.cli import fingerprint
from gui_recover.grounding import parse_point, point_hits_boxes
from gui_recover.policies import MODEL_REVISION
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256, grounding_messages, load_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    training = json.loads((Path(args.adapter)/"training.json").read_text())
    if training["status"] != "complete" or training["base_revision"] != MODEL_REVISION:
        raise ValueError("Adapter has not passed training/reload acceptance")
    if training["base_weights_sha256"] != file_sha256(Path(args.model)/"pytorch_model.bin"):
        raise ValueError("Evaluation base differs from training base")
    dataset_sha = hashlib.sha256((Path(args.data)/"manifest.json").read_bytes()).hexdigest()
    if dataset_sha != training["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed after training")
    if training["adapter_sha256"] != file_sha256(Path(args.adapter)/"adapter_model.safetensors"):
        raise ValueError("Adapter checksum differs")
    rows = load_records(args.data, "test")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    profile = training["config"]["training"]
    processor = AutoProcessor.from_pretrained(args.adapter, min_pixels=profile["min_pixels"],
                                              max_pixels=profile["max_pixels"], trust_remote_code=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False).to("cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    torch.cuda.reset_peak_memory_stats()
    manifest = {"runtime": fingerprint(), "dataset_manifest_sha256": dataset_sha,
        "adapter_sha256": training["adapter_sha256"], "training_code": training["runtime"]["git_commit"],
        "evaluation_dtype": "bfloat16 base in both arms; adapter disabled for baseline",
        "min_pixels": profile["min_pixels"], "max_pixels": profile["max_pixels"],
        "max_new_tokens": 32, "do_sample": False,
        "metric": "visible teacher-box hit per grounding query, not episode success",
        "teacher_labels": "DOM target boxes used for evaluation only"}
    write_json(directory/"manifest.json", manifest)
    predictions = []
    for row in rows:
        picture = Image.open(Path(args.data)/row["image"]).convert("RGB")
        prompt = processor.apply_chat_template(grounding_messages(picture, row["query"]),
                                               tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[picture], return_tensors="pt").to("cuda:0")
        paired = {"id": row["id"], "task": row["task"], "seed": row["seed"], "kind": row["kind"],
                  "image_sha256": row["image_sha256"], "query": row["query"]}
        for arm in ("baseline", "sft"):
            torch.manual_seed(training["config"]["seed"])
            before = time.perf_counter()
            context = model.disable_adapter() if arm == "baseline" else contextlib.nullcontext()
            with context, torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            raw = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
            try:
                point = parse_point(raw)
            except (ActionError, ValueError):
                point = None
            paired[arm] = {"raw": raw, "point": point, "hit": point_hits_boxes(point, row["target_boxes"]),
                           "inference_seconds": time.perf_counter() - before}
        predictions.append(paired)
        with (directory/"predictions.jsonl").open("a") as file:
            file.write(json.dumps(paired) + "\n")
        print(row["id"], paired["baseline"]["hit"], paired["sft"]["hit"], flush=True)
    def summarize(items):
        return {"queries": len(items), **{arm: {"hits": sum(r[arm]["hit"] for r in items),
            "invalid": sum(r[arm]["point"] is None for r in items)} for arm in ("baseline", "sft")},
            "newly_hit": sum(not r["baseline"]["hit"] and r["sft"]["hit"] for r in items),
            "degraded": sum(r["baseline"]["hit"] and not r["sft"]["hit"] for r in items)}
    summary = {"all": summarize(predictions), "by_task": {
        task: summarize([r for r in predictions if r["task"] == task]) for task in sorted({r["task"] for r in predictions})},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(directory/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
