"""Three predeclared SFT seeds; validation gates and shared fresh-holdout acceptance."""

import argparse
import json
from pathlib import Path

from PIL import Image

from gui_recover.cli import fingerprint
from gui_recover.grounding import point_hits_boxes
from gui_recover.runner import write_json
from gui_recover.sft import audit_dataset, file_sha256, load_records
from gui_recover.sft_balanced import BalancedStudyGrounder, balanced_heads, balanced_query, validation_gate
from gui_recover.sft_execution import execute_case
from gui_recover.sft_history import study_summary
from gui_recover.sft_recovery import audit_corrections


def main():
    parser = argparse.ArgumentParser()
    for arg in ("stage", "config", "reference-data", "balanced-data", "data", "model", "visual-adapter", "history-adapter", "out"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--balanced-adapters", nargs=3, required=True)
    parser.add_argument("--validation-eval")
    args = parser.parse_args()
    if args.stage not in ("validation", "test"):
        raise ValueError("Expected validation or test stage")
    config = json.loads(Path(args.config).read_text())
    seeds = config["balanced_training"]["seeds"]
    if len(set(seeds)) != 3 or config["balanced_training"]["primary_seed"] != seeds[0]:
        raise ValueError("Expected three distinct seeds and fixed primary")
    data, reference, out = Path(args.data), Path(args.reference_data), Path(args.out)
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean training/evaluation code")
    if args.stage == "validation":
        if data.resolve() != reference.resolve():
            raise ValueError("Use original validation only")
    else:
        if not args.validation_eval:
            raise ValueError("Test requires frozen validation gate")
        validation_dir = Path(args.validation_eval)
        gate = json.loads((validation_dir / "validation-gate.json").read_text())
        validation_predictions = [json.loads(x) for x in (validation_dir / "predictions.jsonl").read_text().splitlines()]
        if any(r["split"] != "validation" for r in validation_predictions):
            raise ValueError("Gate consumed a non-validation row")
        expected = validation_gate(study_summary(validation_predictions, balanced_heads(seeds)), seeds)
        if (any(gate[k] != v for k, v in expected.items()) or gate["config_sha256"] != file_sha256(Path(args.config))
                or gate["predictions_sha256"] != file_sha256(validation_dir / "predictions.jsonl")
                or gate["source_code"] != runtime["git_commit"]):
            raise ValueError("Validation evidence changed")
        source = json.loads((data / "manifest.json").read_text())
        if source["config"]["splits"] != config["splits"]:
            raise ValueError("Fresh scene protocol differs")
        audit_dataset(data)
        previous = {(r["task"], r["seed"]) for s in ("train", "validation", "test") for r in load_records(reference, s)}
        if previous & {(r["task"], r["seed"]) for r in load_records(data, "test")}:
            raise ValueError("Previous scene reused in fresh test")
    rows = load_records(data, args.stage)
    audit_corrections(data, rows)
    out.mkdir(parents=True, exist_ok=False)
    model = BalancedStudyGrounder(args.model, reference, args.visual_adapter, args.history_adapter,
        args.balanced_data, args.balanced_adapters, config)
    hashes = [m["adapter_sha256"] for m in model.replicates]
    if args.stage == "test" and hashes != gate["adapter_sha256"]:
        raise ValueError("Checkpoint bytes differ from validation")
    heads = balanced_heads(seeds)
    write_json(out / "manifest.json", {"runtime": runtime, "stage": args.stage, "heads": heads,
        "config_sha256": file_sha256(Path(args.config)), "adapter_sha256": hashes,
        "data_manifest_sha256": file_sha256(data / "manifest.json"),
        "reference_manifest_sha256": file_sha256(reference / "manifest.json"),
        "validation_gate_sha256": file_sha256(Path(args.validation_eval) / "validation-gate.json") if args.validation_eval else None,
        "replicate_training_code": [m["runtime"]["git_commit"] for m in model.replicates]})
    predictions, index = [], {}
    for row in rows:
        picture = Image.open(data / row["image"]).convert("RGB")
        result = {k: row[k] for k in ("id", "split", "task", "seed", "kind", "query", "image_sha256", "target_boxes")}
        result.update({"previous_action": row.get("previous_action"), "recovery_mode": row.get("recovery_mode"), "heads": {}})
        cache = {}
        for head in heads:
            weights = "_".join(head.split("_")[:2]) if head.startswith("balanced_") else head.split("_")[0]
            key = (weights, balanced_query(row["query"], row.get("previous_action"), head))
            if key not in cache:
                cache[key] = model.predict(picture, row["query"], row.get("previous_action"), head)
            result["heads"][head] = {**cache[key], "hit": point_hits_boxes(cache[key]["point"], row["target_boxes"])}
        predictions.append(result)
        index[row["id"]] = result
        with (out / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print("grounded", args.stage, row["id"], flush=True)
    offline = study_summary(predictions, heads)
    summary = {"offline": offline}
    if args.stage == "validation":
        gate = {**validation_gate(offline, seeds), "config_sha256": file_sha256(Path(args.config)),
            "predictions_sha256": file_sha256(out / "predictions.jsonl"), "adapter_sha256": hashes,
            "source_code": runtime["git_commit"]}
        write_json(out / "validation-gate.json", gate)
        summary["validation_gate"] = gate
    else:
        selected, normal_seeds = set(source["online_case_ids"]), set(config["online_normal_seeds"])
        episodes = []
        online_heads = config["balanced_training"]["online_heads"]
        for row in rows:
            conditional = row["id"] in selected
            if not conditional and not (row["kind"] == "normal_entry" and row["seed"] in normal_seeds):
                continue
            for head in online_heads:
                result = execute_case(model, head, row, index[row["id"]]["heads"][head], data, out, conditional)
                episodes.append(result)
                write_json(out / "online-results.json", episodes)
                print("executed", row["id"], head, result["reason"], flush=True)
        def online_summary(items):
            return {h: {"episodes": sum(r["head"] == h for r in items),
                "successes": sum(r["head"] == h and r["success"] for r in items),
                "prefix_mismatches": sum(r["head"] == h and not r["prefix_matched"] for r in items),
                "runtime_errors": sum(r["head"] == h and r["reason"] == "runtime_error" for r in items),
                "resets": sum(r["resets_after_failure"] for r in items if r["head"] == h)} for h in online_heads}
        summary.update({"validation_gate": gate,
            "online_normal": online_summary([r for r in episodes if not r["conditional"]]),
            "online_conditional": online_summary([r for r in episodes if r["conditional"]]),
            "online_by_mode": {m: online_summary([r for r in episodes if r.get("recovery_mode") == m])
                for m in ("retry_in_place", "restart_after_terminal")}})
    summary.update({"actual_generation_calls_including_submit": model.calls,
        "peak_cuda_allocated_bytes": model.torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": model.torch.cuda.max_memory_reserved()})
    write_json(out / "summary.json", summary)
    if args.stage == "test" and any(r["reason"] == "runtime_error" for r in episodes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
