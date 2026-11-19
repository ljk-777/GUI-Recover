"""Select history prompts on validation, then accept once on a fresh holdout."""

import argparse
import json
from pathlib import Path

from PIL import Image

from gui_recover.cli import fingerprint
from gui_recover.grounding import point_hits_boxes
from gui_recover.runner import write_json
from gui_recover.sft import audit_dataset, file_sha256, load_records
from gui_recover.sft_execution import execute_case
from gui_recover.sft_history import (
    CANDIDATES, STUDY_HEADS, HistoryStudyGrounder, select_history_prompt, study_query, study_summary,
)
from gui_recover.sft_recovery import audit_corrections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual-adapter", required=True)
    parser.add_argument("--history-adapter", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["history_study"]["candidates"] != list(CANDIDATES):
        raise ValueError("Candidate list differs from frozen implementation")
    data, reference, out = Path(args.data), Path(args.reference_data), Path(args.out)
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean prompt-study code")
    selection = None
    if args.stage == "validation":
        if data.resolve() != reference.resolve() or args.selection:
            raise ValueError("Select only on the reference validation split")
        heads = list(STUDY_HEADS)
    else:
        if not args.selection:
            raise ValueError("Test requires frozen validation selection")
        selection_path = Path(args.selection)
        selection = json.loads(selection_path.read_text())
        validation_file = selection_path.parent / "predictions.jsonl"
        development = [json.loads(x) for x in validation_file.read_text().splitlines()]
        expected = select_history_prompt(development)
        if any(selection[k] != v for k, v in expected.items()):
            raise ValueError("Selection differs from validation-only rule")
        if (selection["predictions_sha256"] != file_sha256(validation_file)
                or selection["config_sha256"] != file_sha256(Path(args.config))
                or selection["runtime_code"] != runtime["git_commit"]
                or selection["validation_split_sha256"] != file_sha256(reference / "validation.jsonl")):
            raise ValueError("Frozen selection evidence differs")
        heads = list(dict.fromkeys(("base_plain", "history_plain", "history_legacy", selection["selected_head"])))
        # Do not load this test split until the validation selection has passed.
        source = json.loads((data / "manifest.json").read_text())
        if source["config"] != config or source["reused_development"]["source_manifest_sha256"] != file_sha256(reference / "manifest.json"):
            raise ValueError("Fresh cohort differs from preregistered protocol")
        audit_dataset(data)
        old_scenes = {(r["task"], r["seed"]) for r in json.loads((reference / "manifest.json").read_text())["source_cohort"]}
        test_rows = load_records(data, "test")
        if old_scenes & {(r["task"], r["seed"]) for r in test_rows}:
            raise ValueError("Fresh test reuses a previous scene")
        old_images = {r["image_sha256"] for s in ("train", "validation", "test") for r in load_records(reference, s)}
        if old_images & {r["image_sha256"] for r in test_rows}:
            raise ValueError("Fresh test duplicates an old screenshot")
    rows = load_records(data, args.stage)
    audit_corrections(data, rows)
    out.mkdir(parents=True, exist_ok=False)
    model = HistoryStudyGrounder(args.model, reference, args.visual_adapter, args.history_adapter)
    if selection and selection["history_adapter_sha256"] != model.training[1]["adapter_sha256"]:
        raise ValueError("Test weights differ from validation selection")
    write_json(out / "manifest.json", {"runtime": runtime, "stage": args.stage,
        "config_sha256": file_sha256(Path(args.config)), "heads": {h: STUDY_HEADS[h] for h in heads},
        "reference_manifest_sha256": file_sha256(reference / "manifest.json"),
        "data_manifest_sha256": file_sha256(data / "manifest.json"),
        "history_adapter_sha256": model.training[1]["adapter_sha256"],
        "selection_sha256": file_sha256(Path(args.selection)) if args.selection else None,
        "limitation": "Frozen weights; evaluator-selected failures, scripted controller, not autonomous detection"})
    predictions, index = [], {}
    for row in rows:
        picture = Image.open(data / row["image"]).convert("RGB")
        result = {k: row[k] for k in ("id", "split", "task", "seed", "kind", "query", "image_sha256", "target_boxes")}
        result.update({"previous_action": row.get("previous_action"),
                       "recovery_mode": row.get("recovery_mode"), "heads": {}})
        cache = {}
        for head in heads:
            weights = STUDY_HEADS[head][0].split("_")[0]
            key = (weights, study_query(row["query"], row.get("previous_action"), head))
            if key not in cache:
                cache[key] = model.predict(picture, row["query"], row.get("previous_action"), head)
            result["heads"][head] = {**cache[key], "hit": point_hits_boxes(cache[key]["point"], row["target_boxes"])}
        predictions.append(result)
        index[row["id"]] = result
        with (out / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print("grounded", args.stage, row["id"], flush=True)
    summary = {"offline": study_summary(predictions, heads)}
    if args.stage == "validation":
        selection = {**select_history_prompt(predictions),
            "predictions_sha256": file_sha256(out / "predictions.jsonl"),
            "config_sha256": file_sha256(Path(args.config)), "runtime_code": runtime["git_commit"],
            "validation_split_sha256": file_sha256(reference / "validation.jsonl"),
            "history_adapter_sha256": model.training[1]["adapter_sha256"]}
        write_json(out / "selection.json", selection)
        summary["selection"] = selection
    else:
        selected = set(source["online_case_ids"])
        normal_seeds = set(config["online_normal_seeds"])
        episodes = []
        for row in rows:
            conditional = row["id"] in selected
            if not conditional and not (row["kind"] == "normal_entry" and row["seed"] in normal_seeds):
                continue
            for head in heads:
                result = execute_case(model, head, row, index[row["id"]]["heads"][head], data, out, conditional)
                episodes.append(result)
                write_json(out / "online-results.json", episodes)
                print("executed", row["id"], head, result["reason"], flush=True)
        def online_summary(items):
            return {h: {"episodes": sum(r["head"] == h for r in items),
                "successes": sum(r["head"] == h and r["success"] for r in items),
                "prefix_mismatches": sum(r["head"] == h and not r["prefix_matched"] for r in items),
                "runtime_errors": sum(r["head"] == h and r["reason"] == "runtime_error" for r in items),
                "resets": sum(r["resets_after_failure"] for r in items if r["head"] == h)} for h in heads}
        summary.update({"selection": selection,
            "online_normal": online_summary([r for r in episodes if not r["conditional"]]),
            "online_conditional": online_summary([r for r in episodes if r["conditional"]]),
            "online_by_mode": {m: online_summary([r for r in episodes if r.get("recovery_mode") == m])
                for m in ("retry_in_place", "restart_after_terminal")}})
    summary.update({"actual_generation_calls_including_submit": model.calls,
        "peak_cuda_allocated_bytes": model.torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": model.torch.cuda.max_memory_reserved()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if args.stage == "test" and any(r["reason"] == "runtime_error" for r in episodes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
