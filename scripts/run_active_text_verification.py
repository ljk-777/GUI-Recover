"""Compare a reversible visual probe with the glyph rule and immediate replacement."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from gui_recover.active_text_study import TASK, evaluate_registered
from gui_recover.cli import fingerprint
from gui_recover.policies import ShowUIPolicy
from gui_recover.repair_study_data import png_sha, preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification_env import TransitionExecution
from gui_recover.visual_text_study import prepare_references


def register(args, config, profile):
    earlier = []
    for path in args.prior_plan:
        data = json.loads(path.read_text())
        earlier.extend(data["selected"] if isinstance(data, dict) else data)
    if args.exclude_run:
        plan_path = args.exclude_run / "preflight.json"
        summary = json.loads((args.exclude_run / "summary.json").read_text())
        if file_sha256(plan_path) != summary["plan_sha256"]:
            raise ValueError("Engineering smoke registry checksum changed")
        earlier.extend(json.loads(plan_path.read_text())["selected"])
    development = []
    for seed in config["development_seeds"]:
        sample = TransitionExecution(TASK, seed, None)
        try:
            development.append({"task": TASK, "seed": seed, "initial_sha256": png_sha(sample.obs.screenshot)})
        finally:
            sample.close()
    write_json(args.out / "development-layouts.json", development)
    seeds = range(profile["start"], profile["start"] + profile["reserve"])
    return preflight(args.reference, args.out,
                     [{"task": TASK, "count": profile["layouts"], "candidate_seeds": list(seeds)}],
                     prior_scenes=earlier + development, include_goal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--prior-plan", action="append", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--exclude-run", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    profile = config[args.profile]
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the registered clean source and pinned MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen ShowUI inference")
    if [file_sha256(path) for path in args.prior_plan] != config["prior_plan_sha256"]:
        raise ValueError("Registered earlier layouts changed")
    if file_sha256(args.model / "pytorch_model.bin") != config["weights_sha256"]:
        raise ValueError("Frozen ShowUI weights changed")
    if args.profile == "formal" and args.exclude_run is None:
        raise ValueError("Provide the completed engineering run to exclude its layouts")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", {"profile": args.profile, "config": config, "runtime": runtime,
                                           "config_sha256": file_sha256(args.config)})
    plan = register(args, config, profile)
    bank_sha = prepare_references(args.out)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used):
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    evaluate_registered(args.out, plan, args.profile, profile["regimes"], actor, bank_sha)


if __name__ == "__main__":
    main()
