"""Compare input replacement and waiting on identical fresh browser prefixes."""

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
from gui_recover.recovery_q import FittedRecoveryPolicy
from gui_recover.recovery_rollouts import baseline_decision, record_rollout
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256


PRIMARY = ("legacy_visual", "learned_wait", "replace_visual")
STRESS = ("legacy_forced", "replace_forced")
ADDITIONAL = ("partial-input", "wrong-input")


def choose_visual(observation):
    return baseline_decision(observation, "visual_fixed")


def choose_forced(observation):
    return Decision.RETRY_FIXED_ZOOM if observation["phase"] == Phase.REVIEW else Decision.CONTINUE


def summarize(rows, costs):
    scenarios = {}
    for regime, policies in [(r,PRIMARY) for r in REGIMES] + [(r,STRESS) for r in ADDITIONAL]:
        by_seed = {}
        for row in rows:
            if row["regime"] == regime:
                variants = by_seed.setdefault(row["seed"], {})
                if row["policy"] in variants:
                    raise ValueError("Duplicate evaluation branch")
                variants[row["policy"]] = row
        if not by_seed or any(set(p) != set(policies) for p in by_seed.values()):
            raise ValueError("Paired trial is incomplete")
        group = {}
        for name in policies:
            subset = [p[name] for p in by_seed.values()]
            browser_events = Counter()
            for r in subset:
                browser_events.update(r["browser_events"])
            n = len(subset)
            group[name] = {"episodes": n, "successes": sum(r["success"] for r in subset),
                "waits": sum(r["waits"] for r in subset), "repairs": sum(r["repairs"] for r in subset),
                "actor_requests": sum(r["actor_requests"] for r in subset),
                "browser_events": dict(browser_events),
                "cost_adjusted_return": sum(r["return"]-costs[Decision.WAIT]*r["waits"]
                                     -costs[Decision.RETRY_FIXED_ZOOM]*r["repairs"] for r in subset)/n}
            for other in policies:
                if name != other:
                    group[name]["versus_"+other] = {
                        "gains": sum(p[name]["success"] and not p[other]["success"] for p in by_seed.values()),
                        "losses": sum(not p[name]["success"] and p[other]["success"] for p in by_seed.values())}
        scenarios[regime] = group
    none, drop, delayed = (scenarios[k]["replace_visual"] for k in REGIMES)
    return {"scenarios": scenarios,
        "gate": {"normal_preserved": none["versus_legacy_visual"]["losses"] == 0 and
                                     none["versus_learned_wait"]["losses"] == 0,
                 "drop_preserved": drop["versus_legacy_visual"]["losses"] == 0 and
                                   drop["versus_learned_wait"]["losses"] == 0,
                 "deferred_gain": delayed["versus_legacy_visual"]["gains"] > 0,
                 "deferred_preserved": delayed["versus_learned_wait"]["losses"] == 0,
                 "deferred_without_wait": delayed["waits"] == 0,
                 "partial_and_wrong_preserved": all(scenarios[r]["replace_forced"]["versus_legacy_forced"]["losses"] == 0
                                                       for r in ADDITIONAL)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "q-policy", "prior-pre-rl", "prior-rl", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--profile", required=True, choices=("smoke", "test"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the committed protocol and registered MiniWoB")
    if os.getenv("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the assigned GPU 0 for frozen inference")
    for file, name in ((args.q_policy,"q_sha256"),(args.model/"pytorch_model.bin","weights_sha256"),
                       (args.prior_pre_rl,"prior_pre_rl_sha256"),(args.prior_rl,"prior_rl_sha256")):
        if file_sha256(file) != config[name]:
            raise ValueError(f"Registered input changed: {name}")
    if any(not r.is_dir() or not any(r.glob("*.jsonl")) for r in args.reference):
        raise ValueError("A previous layout registry is missing")
    previous = sum((json.loads(p.read_text()) for p in (args.prior_pre_rl,args.prior_rl)), [])
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    profile = config[args.profile]
    protocol = {"runtime":runtime,"config":config,"config_sha256":file_sha256(args.config),"profile":args.profile}
    write_json(out/"protocol.json", protocol)
    plan = preflight(args.reference, out, [{"task":"enter-text","count":profile["layouts"],
        "candidate_seeds":list(range(profile["start"],profile["start"]+profile["reserve"]))}], prior_scenes=previous)
    print("registered", len(plan["selected"]), "rejected", len(plan["rejected_layouts"]), flush=True)
    used = subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],text=True).strip()
    if int(used) != 0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor = ShowUIPolicy(str(args.model), revision=config["model_revision"], max_new_tokens=32)
    grounder = RepairGrounder(actor,out/"inference")
    checkpoint = json.loads(args.q_policy.read_text())
    learned = FittedRecoveryPolicy(checkpoint)
    selectors = {"legacy_visual":("legacy",choose_visual),"learned_wait":("legacy",learned),
                 "replace_visual":("replace",choose_visual),"legacy_forced":("legacy",choose_forced),
                 "replace_forced":("replace",choose_forced)}
    rows = []
    with (out/"episodes.jsonl").open("x") as index:
        for source in plan["selected"]:
            scene = {**source,"split":"test"}
            for regime in (*REGIMES,*ADDITIONAL):
                for policy in (PRIMARY if regime in REGIMES else STRESS):
                    mode, decide = selectors[policy]
                    name = f"{scene['task']}-{scene['seed']}-{regime}-{policy}"
                    branch = out/"rollouts"/name
                    env = RecoveryDecisionEnv(scene["task"],grounder,regime=regime,input_repair=mode)
                    try:
                        row = record_rollout(env, scene, policy, branch, choose=decide)
                    finally:
                        env.close()
                    trace = branch/"transitions.jsonl"
                    events = Counter(event["event"] for step in map(json.loads,trace.read_text().splitlines())
                                     for event in step["evidence"]["browser_events"])
                    row.update(input_repair=mode,trace=str(trace.relative_to(out)),browser_events=dict(events))
                    index.write(json.dumps(row)+"\n")
                    index.flush()
                    rows.append(row)
                    print(name, row["outcome"], flush=True)
    if file_sha256(args.q_policy) != config["q_sha256"]:
        raise RuntimeError("The frozen Q checkpoint changed")
    summary = summarize(rows,checkpoint["costs"])
    write_json(out/"summary.json", {**summary,"profile":args.profile,
        "episodes":len(rows),"index_sha256":file_sha256(out/"episodes.jsonl"),
        "plan_sha256":file_sha256(out/"preflight.json"),"q_sha256":config["q_sha256"],
        "learned_routing":dict(learned.counts),"inference":grounder.counters(),
        "peak_cuda_reserved_bytes":actor.torch.cuda.max_memory_reserved()})
    print("completed",out.name,flush=True)


if __name__=="__main__":
    main()
