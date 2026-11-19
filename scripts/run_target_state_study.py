"""Execute target-state verification and guarded recovery on fresh MiniWoB layouts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.multistep_study import register_multistep, run_multistep
from gui_recover.policies import ShowUIPolicy
from gui_recover.repair_study import StudyEncoder
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.target_state import CheckboxStateVerifier
from gui_recover.verification_training import LearnedVerifier


def recovery_summary(folder, plan):
    rows = [json.loads(p.read_text()) for p in Path(folder).glob("*/record.json")]
    if not rows:
        raise ValueError("The study did not record any transitions")
    counts = {}
    for scene in plan["accepted"]:
        task, seed = scene["task"], scene["seed"]
        subset = [r for r in rows if (r["task"], r["seed"]) == (task, seed)]
        expected = {"correct", "blank", "natural"}
        if scene["fixture"]["wrong_control"] is not None:
            expected.add("wrong_control")
        if len(subset) != len(expected) or {r["role"] for r in subset} != expected:
            raise ValueError(f"Missing or duplicate branches for {task} {seed}")
        for r in subset:
            group = counts.setdefault(task, {}).setdefault(r["role"], Counter())
            group["traces"] += 1
            if r["status"] != "ok":
                group["invalid"] += 1
                continue
            initial, final = r["achieved"], r["achieved_after_recovery"]
            actions = r.get("recovery", {}).get("actions", [])
            group.update({"initial_correct": initial, "final_correct": final,
                          "recovered": final and not initial, "regressed": initial and not final,
                          "completed_with_oracle_continuation": r.get("completed_with_oracle_continuation", False),
                          "repair_attempts": bool(actions), "repair_clicks": len(actions),
                          "undone_wrong_clicks": sum(a["kind"] == "undo_wrong_check" for a in actions)})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "prior-plan", "model", "frozen-detector", "font", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--profile", choices=("test", "smoke"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the committed protocol and registered MiniWoB revision")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0")
    for path, name in ((args.prior_plan, "prior_plan_sha256"),
                       (args.model/"pytorch_model.bin", "weights_sha256"),
                       (args.font, "font_sha256"),
                       (args.frozen_detector/"selection.json", "frozen_selection_sha256")):
        if file_sha256(path) != config[name]:
            raise ValueError(f"{name} does not match the protocol")
    old = json.loads(args.prior_plan.read_text())
    profile = config[args.profile]
    seeds = list(range(profile["first_seed"], profile["first_seed"]+profile["reserve"]))
    groups = {"multistep_tasks": [{"task": task, "count": count, "candidate_seeds": seeds}
                                 for task, count in profile["tasks"].items()]}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/"protocol.json", {"config": config, "profile": args.profile,
                                         "runtime": runtime, "config_sha256": file_sha256(args.config)})
    plan = register_multistep(groups, known_layouts=[r["initial_sha256"] for r in old["accepted"]])
    write_json(args.out/"plan.json", plan)
    print("registered", len(plan["accepted"]), "rejected", len(plan["rejected"]), flush=True)

    usage = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], text=True).strip()
    if int(usage) != 0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out/"inference")
    encoder = StudyEncoder(actor)
    learned = {"frozen": LearnedVerifier(args.frozen_detector, encoder)}
    verifier = CheckboxStateVerifier(args.font, minimum_score=config["minimum_label_score"],
                                     minimum_margin=config["minimum_label_margin"])
    detection = run_multistep(plan, grounder, learned, args.out/"episodes", recovery=verifier)
    write_json(args.out/"summary.json", {
        "profile": args.profile, "plan_sha256": file_sha256(args.out/"plan.json"),
        "detection": detection, "recovery": recovery_summary(args.out/"episodes", plan),
        "inference": grounder.counters(), "features": encoder.frozen.counters(),
        "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
