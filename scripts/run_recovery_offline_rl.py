"""Collect real branching trajectories, freeze fitted Q, and evaluate fresh rollouts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.recovery_env import Decision, Phase, REGIMES, RecoveryDecisionEnv
from gui_recover.recovery_q import FittedRecoveryPolicy, feature_state, fit_recovery_q
from gui_recover.recovery_rollouts import baseline_decision, iter_learning_episodes, record_rollout
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256


TASK = "enter-text"
TRAIN_POLICIES = {"try_continue": (False, Decision.CONTINUE), "try_plain": (False, Decision.RETRY_PLAIN),
    "try_fixed": (False, Decision.RETRY_FIXED_ZOOM), "wait_continue": (True, Decision.CONTINUE),
    "wait_plain": (True, Decision.RETRY_PLAIN), "wait_fixed": (True, Decision.RETRY_FIXED_ZOOM)}
TEST_POLICIES = ("continue", "visual_fixed", "wait_visual_fixed", "learned")


def choose_exploratory(observation, name):
    if observation["phase"] != Phase.REVIEW:
        return Decision.CONTINUE
    wait, final = TRAIN_POLICIES[name]
    return Decision.WAIT if wait and not observation["waited"] else final


def choose_wait_fixed(observation):
    if (observation["phase"] == Phase.REVIEW and observation["action_mask"][Decision.WAIT]
            and feature_state(observation) != "immediate:achieved"):
        return Decision.WAIT
    return baseline_decision(observation, "visual_fixed")


def summarize(rows, policies):
    result = {}
    for regime in REGIMES:
        by_layout = {}
        for row in rows:
            if row["regime"] == regime:
                pair = by_layout.setdefault(row["seed"], {})
                if row["policy"] in pair:
                    raise ValueError("Duplicate evaluation rollout")
                pair[row["policy"]] = row
        if not by_layout or any(set(p) != set(policies) for p in by_layout.values()):
            raise ValueError("Evaluation is missing a paired rollout")
        result[regime] = {}
        for policy in policies:
            selected = [p[policy] for p in by_layout.values()]
            totals = Counter()
            for r in selected:
                totals.update(r["decision_counts"])
            result[regime][policy] = {"episodes": len(selected), "successes": sum(r["success"] for r in selected),
                "waits": sum(r["waits"] for r in selected), "repairs": sum(r["repairs"] for r in selected),
                "actor_requests": sum(r["actor_requests"] for r in selected), "actions": dict(totals),
                "cost_adjusted_return": sum(r["cost_adjusted_return"] for r in selected)/len(selected)}
            if "visual_fixed" in policies:
                result[regime][policy].update(
                    gains_vs_visual_fixed=sum(p[policy]["success"] and not p["visual_fixed"]["success"]
                                              for p in by_layout.values()),
                    losses_vs_visual_fixed=sum(not p[policy]["success"] and p["visual_fixed"]["success"]
                                                for p in by_layout.values()))
    return result


def collect(split, scenes, selectors, grounder, out, costs):
    rows = []
    for scene in scenes:
        if scene["split"] != split:
            continue
        for regime in REGIMES:
            for policy, choose in selectors.items():
                name = f"{TASK}-{scene['seed']}-{regime}-{policy}"
                env = RecoveryDecisionEnv(TASK, grounder, regime=regime)
                branch = out/"rollouts"/split/name
                try:
                    item = record_rollout(env, scene, policy, branch, choose=choose)
                finally:
                    env.close()
                trace = branch/"transitions.jsonl"
                counts = Counter(Decision(r["action"]) for r in map(json.loads, trace.read_text().splitlines()))
                rows.append({**item, "trace": str(trace.relative_to(out)),
                    "cost_adjusted_return": item["return"]-sum(costs[a]*count for a, count in counts.items()),
                    "decision_counts": {a.name:n for a,n in counts.items()}})
                print("rollout", split, name, item["outcome"], flush=True)
    index = out/f"{split}.jsonl"
    with index.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    return rows, file_sha256(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "previous-splits", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--profile", required=True, choices=("smoke", "test"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the registered clean source and pinned MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen inference")
    if file_sha256(args.model/"pytorch_model.bin") != config["weights_sha256"]:
        raise ValueError("Frozen weight checksum changed")
    if file_sha256(args.previous_splits) != config["previous_splits_sha256"]:
        raise ValueError("The previous layout registry changed")
    if any(not p.is_dir() or not any(p.glob("*.jsonl")) for p in args.reference):
        raise ValueError("Missing previous source corpus")
    groups, split_by_seed = [], {}
    for split, profile in config[args.profile].items():
        seeds = range(profile["start"], profile["start"]+profile["reserve"])
        if set(seeds)&split_by_seed.keys():
            raise ValueError("Split reserves overlap")
        split_by_seed.update({s:split for s in seeds})
        groups.append({"task":TASK,"count":profile["count"],"candidate_seeds":list(seeds)})
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"config":config,"config_sha256":file_sha256(args.config),
                                     "profile":args.profile,"runtime":runtime})
    previous = json.loads(args.previous_splits.read_text())
    plan = preflight(args.reference, out, groups, prior_scenes=previous)
    scenes = [{**r,"split":split_by_seed[r["seed"]]} for r in plan["selected"]]
    write_json(out/"splits.json", scenes)
    print("registered",len(scenes),"rejected",len(plan["rejected_layouts"]),flush=True)

    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) != 0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, out/"inference")
    train_selectors = {name: (lambda obs, n=name: choose_exploratory(obs, n)) for name in TRAIN_POLICIES}
    train, train_sha = collect("train", scenes, train_selectors, grounder, out, config["costs"])
    indices = {"train": train_sha}
    write_json(out/"summary.json", {"stage":"fit", "index_sha256":indices})
    fitted = fit_recovery_q(iter_learning_episodes(out,"train"), config["costs"])
    checkpoint = out/"recovery-q.json"
    write_json(checkpoint, fitted)
    digest = file_sha256(checkpoint)
    write_json(out/"before-evaluation.json", {"train_index_sha256":train_sha,"q_sha256":digest,
        "splits_sha256":file_sha256(out/"splits.json"),"evaluation_started":False})
    frozen = FittedRecoveryPolicy(json.loads(checkpoint.read_text()))
    print("frozen Q",fitted["q"],flush=True)
    baselines = {"continue":None,"visual_fixed":None,"wait_visual_fixed":choose_wait_fixed,"learned":frozen}
    validation, indices["validation"] = collect("validation",scenes,
        {k:baselines[k] for k in ("wait_visual_fixed","learned")},grounder,out,config["costs"])
    frozen = FittedRecoveryPolicy(json.loads(checkpoint.read_text()))
    baselines["learned"] = frozen
    test, indices["test"] = collect("test",scenes,baselines,grounder,out,config["costs"])
    if file_sha256(checkpoint) != digest:
        raise RuntimeError("Checkpoint changed during evaluation")
    score = summarize(test, TEST_POLICIES)
    write_json(out/"summary.json", {"profile":args.profile,"index_sha256":indices,"q_sha256":digest,
        "splits_sha256":file_sha256(out/"splits.json"),"fit":fitted["fit"],
        "rollouts_by_split":{"train":len(train),"validation":len(validation),"test":len(test)},
        "validation":summarize(validation,("wait_visual_fixed","learned")),"test":score,
        "learned_test_routing":dict(frozen.counts),"inference":grounder.counters(),
        "gate":{"clean_preserved":score["none"]["learned"]["losses_vs_visual_fixed"]==0,
                "drop_preserved":score["drop-input"]["learned"]["losses_vs_visual_fixed"]==0,
                "deferred_gain":score["deferred-input"]["learned"]["gains_vs_visual_fixed"]>0,
                "deferred_preserved":score["deferred-input"]["learned"]["losses_vs_visual_fixed"]==0},
        "peak_cuda_reserved_bytes":actor.torch.cuda.max_memory_reserved()})
    print("completed", out.name,flush=True)


if __name__ == "__main__":
    main()
