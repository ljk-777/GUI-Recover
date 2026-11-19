"""Single-GPU QLoRA smoke: fixed steps, assistant-only loss, adapter reload gate."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from gui_recover.cli import fingerprint
from gui_recover.policies import MODEL_ID, MODEL_REVISION
from gui_recover.runner import write_json
from gui_recover.sft import encode_record, file_sha256, load_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["base_revision"] != MODEL_REVISION:
        raise ValueError("Base revision differs from frozen ShowUI")
    base_sha = file_sha256(Path(args.model)/"pytorch_model.bin")
    if base_sha != "68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3":
        raise ValueError("Base weight hash differs from frozen ShowUI")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze a clean code revision before training")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires an allocated BF16 CUDA GPU")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    profile = config["training"]
    train = load_records(args.data, "train")
    validation = load_records(args.data, "validation")
    # Test labels/outputs are never read by the trainer or used for model selection.
    manifest = {"status": "started", "runtime": runtime, "config": config,
        "base_model": args.model, "base_model_id": MODEL_ID, "base_revision": MODEL_REVISION,
        "base_weights_sha256": base_sha,
        "dataset_manifest_sha256": hashlib.sha256((Path(args.data)/"manifest.json").read_bytes()).hexdigest(),
        "train_records": len(train), "validation_records": len(validation),
        "packages": {p: importlib.metadata.version(p) for p in ("peft", "accelerate", "bitsandbytes", "psutil")},
        "precision": "NF4 double quantized base; BF16 compute; decoder q/v LoRA only",
        "loss": "assistant answer and end token only; all image/user tokens masked"}
    write_json(directory / "training.json", manifest)
    processor = AutoProcessor.from_pretrained(args.model, revision=MODEL_REVISION,
        min_pixels=profile["min_pixels"], max_pixels=profile["max_pixels"], trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    def base_model(checkpointing):
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, revision=MODEL_REVISION,
            quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", trust_remote_code=False)
        model.config.use_cache = False
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})

    model = get_peft_model(base_model(True), LoraConfig(r=profile["lora_rank"],
        lora_alpha=profile["lora_alpha"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not parameters or any("lora_" not in n or ".model.layers." not in n for n in parameters):
        raise ValueError("Trainable parameters escaped the decoder-only LoRA boundary")
    initial = {n: p.detach().float().cpu().clone() for n, p in parameters.items()}
    manifest["trainable_parameters"] = sum(p.numel() for p in parameters.values())
    manifest["trainable_names"] = list(parameters)
    write_json(directory / "training.json", manifest)
    torch.cuda.reset_peak_memory_stats()

    def batch(row):
        return encode_record(processor, args.data, row).to("cuda:0")

    def mean_loss(current, rows):
        current.eval()
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                loss = float(current(**batch(row)).loss)
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite evaluation loss")
                values.append(loss)
        return sum(values) / len(values)

    manifest["validation_loss_before"] = mean_loss(model, validation)
    optimizer = torch.optim.AdamW(parameters.values(), lr=profile["learning_rate"])
    order = list(range(len(train)))
    random.shuffle(order)
    records = []
    started = time.perf_counter()
    for step in range(profile["steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(profile["gradient_accumulation"]):
            index = (step * profile["gradient_accumulation"] + micro) % len(order)
            encoded = batch(train[order[index]])
            supervised = int((encoded["labels"] != -100).sum())
            if not 1 <= supervised < encoded["input_ids"].numel():
                raise ValueError("Assistant-only label mask is invalid")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**encoded).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            losses.append(float(loss.detach()))
            (loss / profile["gradient_accumulation"]).backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        entry = {"step": step + 1, "loss": sum(losses) / len(losses), "gradient_norm": norm,
                 "elapsed_seconds": time.perf_counter() - started}
        records.append(entry)
        with (directory/"steps.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    optimizer.zero_grad(set_to_none=True)
    manifest["validation_loss_after"] = mean_loss(model, validation)
    manifest["adapter_delta_l2"] = math.sqrt(sum(float(((p.detach().float().cpu() - initial[n]) ** 2).sum())
                                                  for n, p in parameters.items()))
    if manifest["adapter_delta_l2"] <= 0 or not any(r["gradient_norm"] > 0 for r in records):
        raise ValueError("Adapter did not learn any parameter update")
    manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    model.eval()
    probe_loss = mean_loss(model, train[:1])
    model.save_pretrained(directory, safe_serialization=True)
    processor.save_pretrained(directory)
    # Release training state before checking an independently loaded quantized model.
    del encoded, loss, optimizer, parameters, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(base_model(False), directory, is_trainable=False)
    reloaded_loss = mean_loss(reloaded, train[:1])
    manifest["reload_probe_loss_before"] = probe_loss
    manifest["reload_probe_loss_after"] = reloaded_loss
    manifest["reload_probe_loss_abs_difference"] = abs(probe_loss - reloaded_loss)
    if abs(probe_loss - reloaded_loss) > 0.002:
        raise ValueError("Reloaded adapter did not reproduce the saved probe loss")
    manifest["adapter_sha256"] = hashlib.sha256((directory/"adapter_model.safetensors").read_bytes()).hexdigest()
    manifest["status"] = "complete"
    manifest["completed_updates"] = len(records)
    manifest["elapsed_training_seconds"] = records[-1]["elapsed_seconds"]
    write_json(directory / "training.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "completed_updates", "trainable_parameters",
        "validation_loss_before", "validation_loss_after", "adapter_delta_l2", "reload_probe_loss_abs_difference",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}), flush=True)


if __name__ == "__main__":
    main()
