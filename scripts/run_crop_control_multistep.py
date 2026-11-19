"""Run the fixed crop controls and multistep first-click test with frozen assets."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.crop_control_study import crop_comparison
from gui_recover.multistep_study import register_multistep, run_multistep
from gui_recover.policies import ShowUIPolicy
from gui_recover.repair_study import StudyEncoder
from gui_recover.repair_study_data import collect_test
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification_training import LearnedVerifier


def groups(tasks, profile):
    seeds = list(range(profile["start"], profile["start"] + profile["reserve"]))
    return [{"task": task, "count": profile["count"], "candidate_seeds": seeds} for task in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "inherited-detectors", "prior-development", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--profile", choices=("formal", "smoke"), required=True)
    args = parser.parse_args()
    registered = json.loads(args.config.read_text())
    profile = registered[args.profile]
    config = {**registered, "profile": args.profile,
              "test_tasks": groups(registered["crop_tasks"], profile["crop"]),
              "multistep_tasks": groups(registered["multistep_tasks"], profile["multistep"])}
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Run from clean committed source with the registered MiniWoB revision")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("This run assigns the visible GPU 0")
    selection_hash = file_sha256(args.prior_development/"selection.json")
    if selection_hash != config["prior_comparison_selection_sha256"]:
        raise ValueError("Previous repair selection does not match")
    learned_dirs = {"frozen": args.inherited_detectors/"frozen", "pixels": args.inherited_detectors/"pixels",
                    "action_phase": args.prior_development/"action-phase"}
    for name, path in learned_dirs.items():
        if file_sha256(path/"selection.json") != config["detector_selection_sha256"][name]:
            raise ValueError(f"Frozen {name} detector metadata differs")
    if file_sha256(args.model/"pytorch_model.bin") != config["base_weights_sha256"]:
        raise ValueError("Frozen actor weights differ")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/"protocol.json", {"config": config, "registered_config_sha256": file_sha256(args.config),
                                          "runtime": runtime, "prior_selection_sha256": selection_hash})
    plan = register_multistep(config)
    write_json(args.out/"multistep-plan.json", plan)
    print("registered multistep", len(plan["accepted"]), "rejected", len(plan["rejected"]), flush=True)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) != 0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out/"inference")
    encoder = StudyEncoder(actor)
    learned = {b: LearnedVerifier(path, encoder) for b, path in learned_dirs.items()}
    collect_test(args.reference, args.out/"crop-data", config, runtime, grounder, selection_hash)
    crop = crop_comparison(args.out/"crop-data", grounder, args.out/"crop-execution", config)
    multistep = run_multistep(plan, grounder, learned, args.out/"multistep")
    write_json(args.out/"summary.json", {
        "profile": args.profile, "protocol_sha256": file_sha256(args.out/"protocol.json"),
        "multistep_plan_sha256": file_sha256(args.out/"multistep-plan.json"),
        "crop_manifest_sha256": file_sha256(args.out/"crop-data/manifest.json"),
        "crop": crop, "multistep": multistep, "inference": grounder.counters(),
        "features": encoder.frozen.counters(),
        "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
