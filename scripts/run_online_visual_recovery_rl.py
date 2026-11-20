"""Train a recovery Q policy from live MiniWoB interactions and freeze it for test."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from random import Random
from statistics import mean
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.online_recovery_q import FrozenOnlinePolicy, OnlineRecoveryQ
from gui_recover.policies import ShowUIPolicy
from gui_recover.recovery_env import Decision, RecoveryDecisionEnv
from gui_recover.recovery_q import DEFAULT_COSTS
from gui_recover.recovery_rollouts import record_rollout
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text import ContentRecoveryPolicy, VisualTextVerifier
from gui_recover.visual_text_study import prepare_references, presence_wait


TASK = "enter-text"
POLICIES = ("presence_wait", "glyph_check", "online_q")


def summarize(rows, regimes):
    conditions = {}
    for regime in regimes:
        pairs = defaultdict(dict)
        for row in rows:
            if row["regime"] == regime:
                pair = pairs[row["seed"]]
                if row["policy"] in pair:
                    raise ValueError("Duplicate test policy branch")
                pair[row["policy"]] = row
        if not pairs or any(set(pair) != set(POLICIES) for pair in pairs.values()):
            raise ValueError("Missing paired test policy branch")
        metrics = {}
        for policy in POLICIES:
            sample = [pair[policy] for pair in pairs.values()]
            metrics[policy] = {
                "episodes": len(sample),
                "successes": sum(row["success"] for row in sample),
                "waits": sum(row["waits"] for row in sample),
                "repairs": sum(row["repairs"] for row in sample),
                "actor_requests": sum(row["actor_requests"] for row in sample),
                "mean_adjusted_return": mean(row["adjusted_return"] for row in sample),
            }
        for baseline in POLICIES[:2]:
            metrics[f"paired_vs_{baseline}"] = {
                "gains": sum(pair["online_q"]["success"] and not pair[baseline]["success"] for pair in pairs.values()),
                "losses": sum(pair[baseline]["success"] and not pair["online_q"]["success"] for pair in pairs.values()),
            }
        conditions[regime] = metrics
    return conditions


def collect(out, split, queue, grounder, selectors, verifier, browser_version, learner=None):
    index = out / f"{split}.jsonl"
    rows = []
    with index.open("x", encoding="utf-8") as stream:
        for order, (scene, regime, policy) in enumerate(queue):
            choose = ContentRecoveryPolicy(verifier) if policy == "glyph_check" else selectors[policy]
            branch = out / "rollouts" / split / f"{order:03d}-{scene['seed']}-{regime}-{policy}"
            env = RecoveryDecisionEnv(TASK, grounder, regime=regime, input_repair="replace")
            try:
                row = record_rollout(env, scene, policy, branch, choose=choose,
                                     on_transition=learner.observe if learner else None)
                if env.execution.env.metadata()["browser_version"] != browser_version:
                    raise ValueError("Task and reference browsers differ")
            finally:
                env.close()
            trace = branch / "transitions.jsonl"
            row["trace"] = trace.relative_to(out).as_posix()
            row["adjusted_return"] = (row["return"] - DEFAULT_COSTS[Decision.WAIT] * row["waits"]
                                      - DEFAULT_COSTS[Decision.RETRY_FIXED_ZOOM] * row["repairs"])
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(split, branch.name, row["outcome"], flush=True)
    return rows, file_sha256(index)


def register(args, profile):
    previous = []
    for path in args.prior_plan:
        data = json.loads(path.read_text())
        previous.extend(data["selected"] if isinstance(data, dict) else data)
    for path in args.exclude_run:
        report = json.loads((path / "summary.json").read_text())
        plan = path / "preflight.json"
        if file_sha256(plan) != report["plan_sha256"]:
            raise ValueError("Engineering registry checksum changed")
        previous.extend(json.loads(plan.read_text())["selected"])
    groups, membership = [], {}
    for split in ("train", "test"):
        spec = profile[split]
        seeds = range(spec["start"], spec["start"] + spec["reserve"])
        if set(seeds) & membership.keys():
            raise ValueError("Split reserves overlap")
        membership.update((seed, split) for seed in seeds)
        groups.append({"task": TASK, "candidate_seeds": list(seeds), "count": spec["count"]})
    plan = preflight(args.reference, args.out, groups, prior_scenes=previous, include_goal=True)
    scenes = [{**row, "split": membership[row["seed"]]} for row in plan["selected"]]
    write_json(args.out / "splits.json", scenes)
    return scenes, file_sha256(args.out / "preflight.json"), prepare_references(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--prior-plan", action="append", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--exclude-run", action="append", type=Path, default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = config[args.profile]
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the registered clean source and pinned MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen ShowUI inference")
    if [file_sha256(p) for p in args.prior_plan] != config["prior_plan_sha256"]:
        raise ValueError("Registered earlier layouts changed")
    if file_sha256(args.model / "pytorch_model.bin") != config["weights_sha256"]:
        raise ValueError("Frozen ShowUI weights changed")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", {"profile": args.profile, "config": config, "runtime": runtime,
                                           "config_sha256": file_sha256(args.config)})
    scenes, plan_sha, bank_sha = register(args, selected)
    bank = json.loads((args.out / "text-references.json").read_text())
    verifier = VisualTextVerifier(bank)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used):
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out / "inference")
    learner = OnlineRecoveryQ(verifier, seed=config["seed"], epsilon=config["epsilon"])
    regimes = selected["regimes"]
    train_cycle = [(s, g, "explore") for s in scenes if s["split"] == "train" for g in regimes]
    train_queue = []
    rng = Random(config["seed"])
    for _ in range(selected["passes"]):
        rng.shuffle(train_cycle)
        train_queue.extend(train_cycle)
    train, train_sha = collect(args.out, "train", train_queue, grounder, {"explore": learner.choose},
                               verifier, bank["browser_version"], learner=learner)
    checkpoint = args.out / "online-q.json"
    write_json(checkpoint, learner.checkpoint())
    digest = file_sha256(checkpoint)
    write_json(args.out / "before-test.json", {
        "q_sha256": digest, "train_index_sha256": train_sha, "test_started": False,
    })
    frozen = FrozenOnlinePolicy(verifier, json.loads(checkpoint.read_text()))
    print("frozen Q", json.dumps(frozen.q), flush=True)
    test_queue = [(s, g, p) for s in scenes if s["split"] == "test" for g in regimes for p in POLICIES]
    test, test_sha = collect(args.out, "test", test_queue, grounder,
                             {"presence_wait": presence_wait, "online_q": frozen}, verifier, bank["browser_version"])
    if file_sha256(checkpoint) != digest or file_sha256(args.out / "text-references.json") != bank_sha:
        raise RuntimeError("Frozen reference or checkpoint changed")
    scores = summarize(test, regimes)
    gates = {
        "preserved_glyph_successes": not any(s["paired_vs_glyph_check"]["losses"] for s in scores.values()),
        "no_clean_repairs": scores["none"]["online_q"]["repairs"] == 0,
        "higher_adjusted_return_than_glyph": mean(r["adjusted_return"] for r in test if r["policy"] == "online_q")
        > mean(r["adjusted_return"] for r in test if r["policy"] == "glyph_check"),
    }
    write_json(args.out / "summary.json", {
        "profile": args.profile, "index_sha256": {"train": train_sha, "test": test_sha},
        "plan_sha256": plan_sha, "bank_sha256": bank_sha, "q_sha256": digest,
        "episodes_by_split": {"train": len(train), "test": len(test)},
        "learned_test_routing": dict(frozen.counts), "conditions": scores, "promotion_gate": gates,
        "inference": grounder.counters(), "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
