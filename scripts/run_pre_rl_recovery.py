"""Register disjoint visual layouts and record frozen recovery policy rollouts."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.recovery_env import REWARD_PROFILE, TASKS, RecoveryDecisionEnv
from gui_recover.recovery_rollouts import POLICIES, record_rollout, summarize_rollouts
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("config", "model", "out"):
        parser.add_argument("--"+opt, type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--profile", choices=("smoke", "test"), required=True)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != protocol["miniwob_commit"]:
        raise ValueError("Registered clean source and MiniWoB revision required")
    if protocol["reward_profile"] != REWARD_PROFILE or os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the registered reward and assigned GPU 0")
    if file_sha256(args.model/"pytorch_model.bin") != protocol["weights_sha256"]:
        raise ValueError("Frozen actor weights differ")
    if any(not p.is_dir() or not any(p.glob("*.jsonl")) for p in args.reference):
        raise ValueError("Registered reference source is missing")
    groups, split_by_seed = [], {}
    tasks = ("click-button", "enter-text") if args.profile == "smoke" else TASKS
    for split, profile in protocol[args.profile].items():
        seeds = range(profile["start"], profile["start"]+profile["reserve"])
        if set(seeds)&split_by_seed.keys():
            raise ValueError("Split seed reserves overlap")
        split_by_seed.update({s:split for s in seeds})
        groups.extend({"task": task, "count": profile["count"], "candidate_seeds": list(seeds)} for task in tasks)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/"protocol.json", {"config": protocol, "profile": args.profile,
        "config_sha256": file_sha256(args.config), "runtime": runtime})
    plan = preflight(args.reference, args.out, groups)
    scenes = [{**r, "split": split_by_seed[r["seed"]]} for r in plan["selected"]]
    write_json(args.out/"splits.json", scenes)
    print("registered", len(scenes), "rejected", len(plan["rejected_layouts"]), flush=True)

    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(gpu) != 0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=protocol["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor, args.out/"inference")
    episodes = []
    for scene in scenes:
        split = scene["split"]
        samples = ("none", "drop-input") if scene["task"] == "enter-text" else ("none",)
        for regime in samples:
            policies = POLICIES if split == "test" else ("continue", "visual_fixed")
            for policy in policies:
                name = "-".join((scene["task"], str(scene["seed"]), regime, policy))
                env = RecoveryDecisionEnv(scene["task"], grounder, regime=regime,
                                          max_decisions=protocol["max_decisions"])
                item = args.out/"rollouts"/split/name
                try:
                    result = record_rollout(env, scene, policy, item)
                finally:
                    env.close()
                episodes.append({**result, "trace": str((item/"transitions.jsonl").relative_to(args.out))})
                print("rollout", split, name, result["outcome"], flush=True)
    indices = {}
    for split in protocol[args.profile]:
        path = args.out/f"{split}.jsonl"
        with path.open("x") as index:
            for item in episodes:
                if item["split"] == split:
                    index.write(json.dumps(item)+"\n")
        indices[split] = file_sha256(path)
    write_json(args.out/"summary.json", {**summarize_rollouts(episodes), "profile": args.profile,
        "splits_sha256": file_sha256(args.out/"splits.json"), "index_sha256": indices,
        "inference": grounder.counters(),
        "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved()})
    print("completed", args.out.name, flush=True)


if __name__ == "__main__":
    main()
