"""Compare a reloaded recovery adapter with the frozen native retry baseline."""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from gui_recover.contextual_repair import RepairGrounder
from gui_recover.multistep_recovery import Recovery, paired_rollouts, summarize, validate_plan
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text_audit import checked_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "model", "adapter", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--layouts-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.layouts_per_task < 1:
        parser.error("Expected at least one held-out layout per task")
    source = args.run
    receipt = json.loads((source / "before-rollouts.json").read_text())
    plan = json.loads(checked_file(source / "preflight.json", receipt["plan_sha256"]).read_text())
    validate_plan(plan)
    frozen = [json.loads(line) for line in (source / "evaluation" / "episodes.jsonl").read_text().splitlines()]
    scenes, counts = [], Counter()
    for scene in plan["selected"]:
        task = scene["task"]
        if scene["split"] in ("test", "transfer") and counts[task] < args.layouts_per_task:
            counts[task] += 1
            scenes.append(scene)
    conditions = ("healthy", "miss", "natural")
    scenes_by_id = {(s["task"], s["seed"]): s for s in scenes}
    baseline = [r for r in frozen if r["method"] == "fixed_retry" and r["regime"] in conditions
                and (r["task"], r["seed"]) in scenes_by_id]
    args.out.mkdir(parents=True, exist_ok=False)
    loaded = ShowUIPolicy(str(args.model), revision=MODEL_REVISION, max_new_tokens=32, adapter=args.adapter)

    class CacheOnly:
        def ground(self, *_):
            raise RuntimeError("Missing frozen ShowUI first-action prediction")

    prior = RepairGrounder(CacheOnly(), args.out / "first-actions")
    for row in baseline:
        if row["regime"] != "natural":
            continue
        branch = checked_file(source / "evaluation" / row["trace"], row["trace_sha256"]).parent
        record = row["frames"]["initial"]
        with Image.open(checked_file(branch / record["image"], record["sha256"])) as image:
            query = scenes_by_id[row["task"], row["seed"]]["fixture"]["target"]
            prior.prime(image, query, row["source_prediction"]["raw"], "frozen_heldout_first_action")
    repairer = RepairGrounder(loaded, args.out / "adapted-inference")
    results = paired_rollouts(scenes, prior, Recovery(prior, None, None, repair_actor=repairer),
        args.out / "evaluation", methods=("fixed_retry",), allowed_regimes=conditions)
    pairs = baseline + [{**r, "method": "sft_retry"} for r in results]
    metrics = summarize(pairs, ("fixed_retry", "sft_retry"))
    write_json(args.out / "summary.json", {"episodes_per_method": len(results), "conditions": conditions,
        "baseline_index_sha256": file_sha256(source / "evaluation" / "episodes.jsonl"),
        "adapter_sha256": file_sha256(args.adapter / "adapter_model.safetensors"),
        "first_action_generations": prior.calls, "adapted_inference": repairer.counters(),
        "metrics": metrics})
    print(json.dumps(metrics["all"], indent=2))


if __name__ == "__main__":
    main()
