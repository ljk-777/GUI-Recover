"""Learn when to probe visually matched input, then compare fixed rules on fresh layouts."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from random import Random
from statistics import mean
import subprocess

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.probe_learning import COSTS, FrozenProbeChoice, ProbeLearner, adjusted_return, skip_after_wait
from gui_recover.recovery_env import RecoveryDecisionEnv
from gui_recover.recovery_rollouts import record_rollout
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification_env import DEFERRED_TAIL_REGIME
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_study import prepare_references


TASK = "enter-text"
POLICIES = ("always_probe", "skip_after_wait", "learned_probe")
METRICS = ("waits", "probes", "repairs", "actor_requests", "browser_operations")


def registered_layouts(root, expected_profile):
    report = json.loads((root / "summary.json").read_text())
    path = root / "preflight.json"
    if report["profile"] != expected_profile or file_sha256(path) != report["plan_sha256"]:
        raise ValueError("Prior active-probe experiment does not match its registry")
    return report, json.loads(path.read_text())["selected"]


def register(out, profile, earlier, references):
    groups, membership = [], {}
    for split in ("train", "test"):
        spec = profile[split]
        seeds = range(spec["start"], spec["start"] + spec["reserve"])
        if set(seeds) & membership.keys():
            raise ValueError("Train and test reserves overlap")
        membership.update((seed, split) for seed in seeds)
        groups.append({"task": TASK, "candidate_seeds": list(seeds), "count": spec["count"]})
    plan = preflight(references, out, groups, prior_scenes=earlier, include_goal=True)
    scenes = [{**row, "split": membership[row["seed"]]} for row in plan["selected"]]
    write_json(out / "splits.json", scenes)
    return scenes


def collect(out, split, queue, grounder, verifier, browser_version, selectors, learner=None):
    index = out / f"{split}.jsonl"
    rows = []
    with index.open("x", encoding="utf-8") as stream:
        for sequence, (scene, regime, policy) in enumerate(queue):
            rule = ActiveTextPolicy(verifier, selectors[policy])
            branch = out / "rollouts" / split / f"{sequence:03d}-{scene['seed']}-{regime}-{policy}"
            env = RecoveryDecisionEnv(TASK, grounder, regime=regime, input_repair="replace", enable_tail_probe=True)
            try:
                row = record_rollout(env, scene, policy, branch, choose=rule)
                if env.execution.env.metadata()["browser_version"] != browser_version:
                    raise ValueError("Task and reference browsers differ")
                row["browser_operations"] = len(env.execution.events)
            finally:
                env.close()
            trace = branch / "transitions.jsonl"
            detail = branch / "visual-verification.json"
            write_json(detail, rule.evidence)
            row.update(trace=trace.relative_to(out).as_posix(), visual_verification_sha256=file_sha256(detail),
                       adjusted_return=adjusted_return(row),
                       selection_statuses=[item["selection"]["status"] for item in rule.evidence if item["selection"]])
            if learner is not None:
                learner.finish_episode(row["adjusted_return"])
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(split, branch.name, row["outcome"], flush=True)
    return rows, file_sha256(index)


def summarize(test, regimes):
    conditions = {}
    for regime in regimes:
        pairs = defaultdict(dict)
        for row in test:
            if row["regime"] == regime:
                member = pairs[row["seed"]]
                if row["policy"] in member:
                    raise ValueError("Duplicate held-out paired branch")
                member[row["policy"]] = row
        if not pairs or any(set(pair) != set(POLICIES) for pair in pairs.values()):
            raise ValueError("Invalid paired test branches")
        report = {}
        for policy in POLICIES:
            group = [pair[policy] for pair in pairs.values()]
            report[policy] = {
                "episodes": len(group), "successes": sum(r["success"] for r in group),
                "mean_adjusted_return": mean(r["adjusted_return"] for r in group),
                **{metric: sum(r[metric] for r in group) for metric in METRICS},
            }
        report["learned_vs_always"] = {
            "gains": sum(pair["learned_probe"]["success"] and not pair["always_probe"]["success"] for pair in pairs.values()),
            "losses": sum(pair["always_probe"]["success"] and not pair["learned_probe"]["success"] for pair in pairs.values()),
        }
        conditions[regime] = report
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "out", "accepted-rule", "rule-smoke"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--prior-plan", action="append", type=Path, required=True)
    parser.add_argument("--exclude-run", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    profile = config[args.profile]
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the registered clean source and pinned MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen ShowUI inference")
    if config["costs"] != COSTS or file_sha256(args.model / "pytorch_model.bin") != config["weights_sha256"]:
        raise ValueError("Frozen weights or costs changed")
    accepted, earlier = registered_layouts(args.accepted_rule, "formal")
    _, engineering = registered_layouts(args.rule_smoke, "smoke")
    earlier.extend(engineering)
    earlier.extend(json.loads((args.accepted_rule / "development-layouts.json").read_text()))
    prior_receipts = []
    for path in args.prior_plan:
        data = json.loads(path.read_text())
        earlier.extend(data["selected"] if isinstance(data, dict) else data)
        prior_receipts.append(file_sha256(path))
    if args.profile == "formal":
        if args.exclude_run is None:
            raise ValueError("Provide the learner smoke to exclude its layouts")
        _, learner_smoke = registered_layouts(args.exclude_run, "smoke")
        earlier.extend(learner_smoke)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", {"config": config, "profile": args.profile, "runtime": runtime,
        "config_sha256": file_sha256(args.config), "reference_rule_plan_sha256": accepted["plan_sha256"],
        "reference_rule_gate": accepted["gate"],
        "prior_plan_sha256": prior_receipts})
    scenes = register(args.out, profile, earlier, args.reference)
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
    learner = ProbeLearner(seed=config["seed"], epsilon=config["epsilon"], min_visits=config["min_visits"])
    rng = Random(config["seed"])
    cycle = [(s, g, "explore_probe") for s in scenes if s["split"] == "train" for g in config["train_regimes"]]
    train_queue = []
    for _ in range(profile["passes"]):
        rng.shuffle(cycle)
        train_queue.extend(cycle)
    train, train_sha = collect(args.out, "train", train_queue, grounder, verifier, bank["browser_version"],
                               {"explore_probe": learner.choose}, learner=learner)
    q_file = args.out / "probe-q.json"
    write_json(q_file, learner.checkpoint())
    checkpoint_sha = file_sha256(q_file)
    write_json(args.out / "before-test.json",
               {"q_sha256": checkpoint_sha, "train_index_sha256": train_sha, "test_started": False})
    frozen = FrozenProbeChoice(json.loads(q_file.read_text()))
    print("frozen probe Q", json.dumps(frozen.q), flush=True)
    test_queue = [(s, r, p) for s in scenes if s["split"] == "test"
                  for r in profile["test_regimes"] for p in POLICIES]
    test, test_sha = collect(args.out, "test", test_queue, grounder, verifier, bank["browser_version"],
                             dict(always_probe=None, skip_after_wait=skip_after_wait, learned_probe=frozen))
    if file_sha256(q_file) != checkpoint_sha or file_sha256(bank_path) != bank_sha:
        raise RuntimeError("Frozen checkpoint or reference changed during test")
    scores = summarize(test, profile["test_regimes"])
    primary = [scores[g] for g in profile["test_regimes"] if g != DEFERRED_TAIL_REGIME]
    write_json(args.out / "summary.json", {
        "profile": args.profile, "episodes_by_split": {"train": len(train), "test": len(test)},
        "index_sha256": {"train": train_sha, "test": test_sha}, "q_sha256": checkpoint_sha,
        "plan_sha256": file_sha256(args.out / "preflight.json"), "bank_sha256": bank_sha,
        "conditions": scores, "learned_routing": dict(frozen.counts),
        "promotion_gate": {
            "no_clean_repairs": scores["none"]["learned_probe"]["repairs"] == 0,
            "preserved_primary_successes": not any(s["learned_vs_always"]["losses"] for s in primary),
            "fewer_primary_probes": sum(s["learned_probe"]["probes"] for s in primary)
            < sum(s["always_probe"]["probes"] for s in primary),
            "preserved_delayed_tail_challenge": scores[DEFERRED_TAIL_REGIME]["learned_vs_always"]["losses"] == 0,
        },
        "inference": grounder.counters(), "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
