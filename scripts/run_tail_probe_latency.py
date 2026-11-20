"""Compare two-step and batched tail probes with the same visual policy."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import median
import subprocess

from gui_recover.active_text_study import collect
from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_study import prepare_references

from run_active_text_verification import register


POLICIES = ("active_tail", "batched_tail")
OUTCOME_FIELDS = ("success", "outcome", "waits", "repairs", "probes", "actor_requests", "selection_statuses")


def compare(rows, threshold):
    pairs = defaultdict(dict)
    for row in rows:
        methods = pairs[row["seed"], row["regime"]]
        if row["policy"] in methods:
            raise ValueError("Duplicate timing branch")
        methods[row["policy"]] = row
    if any(set(pair) != set(POLICIES) for pair in pairs.values()):
        raise ValueError("Incomplete timing pair")
    matching = sum(all(pair["active_tail"][key] == pair["batched_tail"][key] for key in OUTCOME_FIELDS)
                   for pair in pairs.values())
    probes = [pair for pair in pairs.values() if pair["active_tail"]["probes"] == pair["batched_tail"]["probes"] == 1]
    if not probes:
        raise ValueError("At least one executed probe pair is needed for timing")
    gains = [1 - pair["batched_tail"]["probe_seconds"] / pair["active_tail"]["probe_seconds"] for pair in probes]
    return {
        "pairs": len(pairs), "matching_outcome_pairs": matching, "executed_probe_pairs": len(probes),
        "median_paired_latency_reduction": median(gains),
        "by_policy": {policy: {
            "successes": sum(pair[policy]["success"] for pair in pairs.values()),
            "median_probe_seconds": median(pair[policy]["probe_seconds"] for pair in probes),
            **{field: sum(pair[policy][field] for pair in pairs.values()) for field in ("probes", "browser_operations", "actor_requests")},
        } for policy in POLICIES},
        "gate": {"identical_outcomes": matching == len(pairs), "median_paired_reduction": median(gains) >= threshold},
    }


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
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the registered clean source and pinned MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen ShowUI inference")
    if [file_sha256(path) for path in args.prior_plan] != config["prior_plan_sha256"]:
        raise ValueError("Registered comparison layouts changed")
    if file_sha256(args.model / "pytorch_model.bin") != config["weights_sha256"]:
        raise ValueError("Frozen ShowUI weights changed")
    if args.profile == "formal" and args.exclude_run is None:
        raise ValueError("Provide the engineering run to exclude its layouts")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", {"config": config, "runtime": runtime, "profile": args.profile,
                                           "config_sha256": file_sha256(args.config), "policies": POLICIES})
    spec = config[args.profile]
    plan = register(args, config, spec)
    bank_sha = prepare_references(args.out)
    bank_file = args.out / "text-references.json"
    bank = json.loads(bank_file.read_text())
    verifier = VisualTextVerifier(bank)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   text=True).strip()
    if int(used):
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out / "inference")
    rows = collect(args.out, plan, spec["regimes"], grounder, verifier, bank["browser_version"],
                   policies=POLICIES, alternate_order=True)
    if file_sha256(bank_file) != bank_sha:
        raise RuntimeError("Frozen text references changed")
    result = compare(rows, config["minimum_median_paired_latency_reduction"])
    write_json(args.out / "summary.json", {"profile": args.profile, "episodes": len(rows),
        "bank_sha256": bank_sha, "plan_sha256": file_sha256(args.out / "preflight.json"),
        "index_sha256": file_sha256(args.out / "episodes.jsonl"), "comparison": result,
        "inference": grounder.counters(), "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved()})
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
