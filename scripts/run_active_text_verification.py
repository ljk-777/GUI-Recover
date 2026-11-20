"""Compare a reversible visual probe with the glyph rule and immediate replacement."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import median
import subprocess

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.recovery_env import RecoveryDecisionEnv
from gui_recover.recovery_rollouts import record_rollout
from gui_recover.repair_study_data import png_sha, preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification_env import TransitionExecution
from gui_recover.visual_text import ContentRecoveryPolicy, VisualTextVerifier
from gui_recover.visual_text_study import prepare_references


TASK = "enter-text"
POLICIES = ("glyph_check", "fixed_zoom", "active_tail")


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


def summarize(rows, regimes):
    conditions = {}
    for regime in regimes:
        by_seed = defaultdict(dict)
        for row in rows:
            if row["regime"] == regime:
                pair = by_seed[row["seed"]]
                if row["policy"] in pair:
                    raise ValueError("Duplicate paired rollout")
                pair[row["policy"]] = row
        if not by_seed or any(set(pair) != set(POLICIES) for pair in by_seed.values()):
            raise ValueError("Missing paired rollout")
        metrics = {}
        for policy in POLICIES:
            group = [pair[policy] for pair in by_seed.values()]
            metrics[policy] = {
                "episodes": len(group), "successes": sum(row["success"] for row in group),
                **{key: sum(row[key] for row in group) for key in
                   ("decisions", "waits", "repairs", "probes", "actor_requests", "browser_operations")},
            }
        for baseline in POLICIES[:2]:
            metrics[f"active_vs_{baseline}"] = {
                "gains": sum(pair["active_tail"]["success"] and not pair[baseline]["success"] for pair in by_seed.values()),
                "losses": sum(pair[baseline]["success"] and not pair["active_tail"]["success"] for pair in by_seed.values()),
            }
        conditions[regime] = metrics
    return conditions


def collect(out, plan, regimes, grounder, verifier, browser_version):
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as index:
        for layout in plan["selected"]:
            scene = {**layout, "split": "test"}
            for regime in regimes:
                for policy in POLICIES:
                    choose = None
                    if policy == "glyph_check":
                        choose = ContentRecoveryPolicy(verifier)
                    elif policy == "active_tail":
                        choose = ActiveTextPolicy(verifier)
                    branch = out / "rollouts" / f"{TASK}-{layout['seed']}-{regime}-{policy}"
                    env = RecoveryDecisionEnv(TASK, grounder, regime=regime, input_repair="replace", enable_tail_probe=True)
                    try:
                        row = record_rollout(env, scene, policy, branch, choose=choose)
                        if env.execution.env.metadata()["browser_version"] != browser_version:
                            raise ValueError("Task and reference browsers differ")
                        row["browser_operations"] = len(env.execution.events)
                    finally:
                        env.close()
                    evidence = choose.evidence if choose else []
                    evidence_path = branch / "visual-verification.json"
                    write_json(evidence_path, evidence)
                    row.update(trace=(branch / "transitions.jsonl").relative_to(out).as_posix(),
                               visual_verification_sha256=file_sha256(evidence_path),
                               selection_statuses=[item["selection"]["status"] for item in evidence if item.get("selection")])
                    rows.append(row)
                    index.write(json.dumps(row) + "\n")
                    index.flush()
                    print(branch.name, row["outcome"], flush=True)
    return rows


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
    bank_path = args.out / "text-references.json"
    bank = json.loads(bank_path.read_text())
    verifier = VisualTextVerifier(bank)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used):
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out / "inference")
    rows = collect(args.out, plan, profile["regimes"], grounder, verifier, bank["browser_version"])
    if file_sha256(bank_path) != bank_sha:
        raise RuntimeError("Frozen text references changed")
    conditions = summarize(rows, profile["regimes"])
    probe_rows = [row for row in rows if row["probes"]]
    write_json(args.out / "summary.json", {
        "profile": args.profile, "layouts": len(plan["selected"]), "episodes": len(rows), "conditions": conditions,
        "plan_sha256": file_sha256(args.out / "preflight.json"), "bank_sha256": bank_sha,
        "index_sha256": file_sha256(args.out / "episodes.jsonl"),
        "selection_statuses": dict(Counter(status for row in rows for status in row["selection_statuses"])),
        "probe_time_seconds": {"total": sum(row["probe_seconds"] for row in probe_rows),
                               "median": median(row["probe_seconds"] for row in probe_rows) if probe_rows else None},
        "gate": {
            "preserved_glyph_successes": not any(m["active_vs_glyph_check"]["losses"] for m in conditions.values()),
            "no_clean_repairs": conditions["none"]["active_tail"]["repairs"] == 0,
            "added_trailing_space_success": conditions["trailing-space-input"]["active_vs_glyph_check"]["gains"] > 0,
        },
        "inference": grounder.counters(), "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
