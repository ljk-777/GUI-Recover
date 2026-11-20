"""Evaluate visual text verification and presence-only routing on paired tasks."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.recovery_env import Decision, Phase, RecoveryDecisionEnv, SUPPORTED_REGIMES, transition_view
from gui_recover.recovery_rollouts import baseline_decision, record_rollout
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification import appearance_verdict
from gui_recover.visual_text import ContentRecoveryPolicy, VisualTextVerifier


POLICIES = ("presence_wait", "glyph_check")
VISIBLE_ERRORS = ("partial-input", "wrong-input", "case-input")


def presence_wait(observation):
    if (observation["phase"] == Phase.REVIEW and observation["action_mask"][Decision.WAIT]
            and appearance_verdict(transition_view(observation))["status"] != "achieved"):
        return Decision.WAIT
    return baseline_decision(observation,"visual_fixed")


def summarize(rows):
    scores = {}
    for condition in SUPPORTED_REGIMES:
        paired = {}
        for r in rows:
            if r["regime"] == condition:
                group = paired.setdefault(r["seed"],{})
                if r["policy"] in group:
                    raise ValueError("Duplicate policy branch")
                group[r["policy"]] = r
        if not paired or any(set(p)!=set(POLICIES) for p in paired.values()):
            raise ValueError("Missing paired branch")
        totals = {}
        for policy in POLICIES:
            branch = [p[policy] for p in paired.values()]
            totals[policy] = {"episodes":len(branch),"successes":sum(r["success"] for r in branch),
                **{k:sum(r[k] for r in branch) for k in ("waits","repairs","actor_requests")},
                "outcomes":dict(Counter(r["outcome"] for r in branch))}
        totals["paired_glyph_vs_presence"] = {
            "gains":sum(p["glyph_check"]["success"] and not p["presence_wait"]["success"] for p in paired.values()),
            "losses":sum(not p["glyph_check"]["success"] and p["presence_wait"]["success"] for p in paired.values())}
        totals["glyph_statuses"] = dict(Counter(s for p in paired.values() for s in p["glyph_check"]["visual_statuses"]))
        scores[condition] = totals
    preserved = ("none","drop-input","deferred-input")
    gate = {"preserved_normal_and_missing":all(scores[r]["paired_glyph_vs_presence"]["losses"]==0 for r in preserved),
            "no_normal_repair":scores["none"]["glyph_check"]["repairs"]==0,
            "gained_visible_errors":all(scores[r]["paired_glyph_vs_presence"]["gains"]>0 for r in VISIBLE_ERRORS)}
    return {"conditions":scores,"gate":gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config","model","prior-pre-rl","prior-rl","prior-idempotent","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reference",action="append",type=Path,required=True)
    parser.add_argument("--profile",choices=("smoke","test"),required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    runtime=fingerprint()
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"]!=config["miniwob_commit"]:
        raise ValueError("Use the committed protocol and pinned MiniWoB source")
    if os.getenv("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Use the assigned GPU 0 for frozen inference")
    for path,key in ((args.model/"pytorch_model.bin","weights_sha256"),(args.prior_pre_rl,"prior_pre_rl_sha256"),
                   (args.prior_rl,"prior_rl_sha256"),(args.prior_idempotent,"prior_idempotent_sha256")):
        if file_sha256(path)!=config[key]:
            raise ValueError(f"Registered artifact changed: {key}")
    if any(not r.is_dir() or not any(r.glob("*.jsonl")) for r in args.reference):
        raise ValueError("Missing prior layout corpus")
    previous=[]
    for p in (args.prior_pre_rl,args.prior_rl):
        previous.extend(json.loads(p.read_text()))
    previous.extend(json.loads(args.prior_idempotent.read_text())["selected"])
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    profile=config[args.profile]
    write_json(out/"protocol.json",{"config":config,"profile":args.profile,"runtime":runtime,
        "config_sha256":file_sha256(args.config)})
    plan=preflight(args.reference,out,[{"task":"enter-text","count":profile["layouts"],
         "candidate_seeds":list(range(profile["start"],profile["start"]+profile["reserve"]))}],
         prior_scenes=previous,include_goal=True)
    print("registered",len(plan["selected"]),"rejected",len(plan["rejected_layouts"]),flush=True)
    bank_path=out/"text-references.json"
    # Rendering runs in an isolated browser process with visible goal text
    # as its only input. It never opens the task or consumes rollout outcomes.
    subprocess.run([sys.executable,"-m","gui_recover.visual_text","--plan",str(out/"preflight.json"),
                    "--out",str(bank_path)],check=True)
    bank_sha=file_sha256(bank_path)
    bank=json.loads(bank_path.read_text())
    verifier=VisualTextVerifier(bank)
    write_json(out/"before-rollouts.json",{"plan_sha256":file_sha256(out/"preflight.json"),
        "bank_sha256":bank_sha,"actor_calls":0,"outcomes_observed":False})
    used=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],text=True).strip()
    if int(used)!=0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor=ShowUIPolicy(str(args.model),revision=config["model_revision"],max_new_tokens=32)
    grounder=RepairGrounder(actor,out/"inference")
    rows=[]
    with (out/"episodes.jsonl").open("x") as index:
        for source in plan["selected"]:
            scene={**source,"split":"test"}
            for condition in SUPPORTED_REGIMES:
                for policy in POLICIES:
                    content=ContentRecoveryPolicy(verifier) if policy=="glyph_check" else None
                    name=f"{scene['task']}-{scene['seed']}-{condition}-{policy}"
                    branch=out/"rollouts"/name
                    env=RecoveryDecisionEnv(scene["task"],grounder,regime=condition,input_repair="replace")
                    try:
                        result=record_rollout(env,scene,policy,branch,choose=content or presence_wait)
                        if env.execution.env.metadata()["browser_version"]!=bank["browser_version"]:
                            raise ValueError("Task and reference Chromium builds differ")
                    finally:
                        env.close()
                    diagnostics=content.evidence if content else []
                    write_json(branch/"visual-verification.json",diagnostics)
                    trace=branch/"transitions.jsonl"
                    result.update(trace=str(trace.relative_to(out)),
                        visual_verification_sha256=file_sha256(branch/"visual-verification.json"),
                        visual_statuses=[d["status"] for d in diagnostics])
                    index.write(json.dumps(result)+"\n")
                    index.flush()
                    rows.append(result)
                    print(name,result["outcome"],flush=True)
    if file_sha256(bank_path)!=bank_sha:
        raise ValueError("Reference templates changed during evaluation")
    write_json(out/"summary.json",{**summarize(rows),"profile":args.profile,"episodes":len(rows),
        "index_sha256":file_sha256(out/"episodes.jsonl"),"plan_sha256":file_sha256(out/"preflight.json"),
        "bank_sha256":bank_sha,"reference_texts":len(bank["templates"]),
        "inference":grounder.counters(),"peak_cuda_reserved_bytes":actor.torch.cuda.max_memory_reserved()})
    print("completed",out.name,flush=True)


if __name__=="__main__":
    main()
