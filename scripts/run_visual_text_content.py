"""Evaluate visual text verification and presence-only routing on paired tasks."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.policies import ShowUIPolicy
from gui_recover.repair_study_data import preflight
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text_study import prepare_references, record_text_study


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
    bank_sha = prepare_references(out)
    used=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],text=True).strip()
    if int(used)!=0:
        raise RuntimeError("Assigned GPU 0 is busy")
    actor=ShowUIPolicy(str(args.model),revision=config["model_revision"],max_new_tokens=32)
    record_text_study(out, plan, args.profile, actor, bank_sha)


if __name__ == "__main__":
    main()
