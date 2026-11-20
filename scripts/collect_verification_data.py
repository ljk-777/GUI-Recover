"""Frozen actor, paired normal/injected transitions, plus queued-input controls."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from gui_recover.actions import Action, ActionError
from gui_recover.cli import fingerprint
from gui_recover.grounding import parse_point
from gui_recover.policies import ShowUIPolicy
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification_backbone import FrozenFeatures
from gui_recover.verification_data import audit_transition_splits, observation_rows, view_from_record
from gui_recover.verification_env import TransitionExecution, task_request, teacher_evidence


def main():
    parser = argparse.ArgumentParser()
    for name in ("config","model","out"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--only-split", choices=("train","validation","test"),
                        help="Collect one split for an explicitly audited dataset assembly")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean data protocol")
    if file_sha256(Path(args.model)/"pytorch_model.bin") != config["base_weights_sha256"]:
        raise ValueError("Frozen actor weights differ")
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"images").mkdir()
    (out/"traces").mkdir()
    shutil.copyfile("vendor/miniwob-plusplus/LICENSE",out/"MINIWOB_LICENSE.txt")
    actor = ShowUIPolicy(args.model,max_new_tokens=32)
    encoder = FrozenFeatures(actor)
    manifest = {"runtime":runtime,"config":config,"actor":actor.config,"source_scenes":[],
        "scope":"Teacher labels offline only; queued input is an explicit synthetic delivery fixture",
        "render_fixture":"CSS caret hidden, solid button borders, double-RAF native CDP capture; raw PNGs unedited; not stock benchmark rendering"}
    def frame(obs,name):
        path=out/"images"/f"{name}.png"
        obs.screenshot.save(path)
        return {"image":f"images/{path.name}","sha256":file_sha256(path)}
    def state(execution,name):
        evidence=teacher_evidence(execution)
        return {**frame(execution.obs,name),"reward":execution.reward,"terminated":execution.done,
            "truncated":execution.truncated,"effect_achieved":bool(evidence["effect_achieved"]),"teacher_evidence":evidence,
            "task_info":deepcopy(execution.info.get("task_info",{}))}
    for split,groups in config["splits"].items():
        if args.only_split is not None and split != args.only_split:
            continue
        records=[]
        for group in groups:
            task=group["task"]
            for index,seed in enumerate(group["seeds"]):
                scene=f"{task}-{seed}"
                first=TransitionExecution(task,seed,None)
                first_closed=False
                try:
                    goal=first.obs.goal
                    target,kind,value=task_request(task,goal)
                    initial=frame(first.obs,scene+"-before")
                    actor.torch.manual_seed(seed)
                    raw=actor.ground(first.obs.screenshot,target)
                    try:
                        proposed=Action(kind,value,parse_point(raw))
                    except ActionError:
                        proposed=None
                    fault=(config["input_faults"] if kind=="INPUT" else config["click_faults"])[index % (3 if kind=="INPUT" else 2)]
                    regimes=["none",fault]
                    if fault=="drop-input":
                        regimes.append("deferred-input")
                    reference=False
                    for regime in regimes:
                        execution=first if regime=="none" else TransitionExecution(task,seed,proposed,regime)
                        execution.proposed=proposed
                        row={"id":scene+"-"+regime,"scene_id":scene,"split":split,"task":task,"seed":seed,
                            "goal":goal,"target":target,"regime":regime,"deferred":regime=="deferred-input",
                            "proposed_action":proposed.to_dict() if proposed else None,"raw":raw,
                            "before":initial,"immediate":None,"after_wait":None,"status":"invalid_source",
                            "reference_achieved":False}
                        try:
                            if proposed is not None:
                                # Counterfactual resets must reproduce the exact actor input.
                                current=frame(execution.obs,row["id"]+"-reset-check")
                                row["reset_check"]=current
                                if current["sha256"] != initial["sha256"] or execution.obs.goal!=goal:
                                    row["status"]="prefix_mismatch"
                                else:
                                    execution.first()
                                    row["immediate"]=state(execution,row["id"]+"-immediate")
                                    if regime=="none":
                                        reference=row["immediate"]["effect_achieved"]
                                    if not (execution.done or execution.truncated):
                                        execution.wait_once()
                                        row["after_wait"]=state(execution,row["id"]+"-after-wait")
                                    row["status"]="ok"
                            row["reference_achieved"]=bool(reference)
                        except Exception as exc:
                            row.update({"status":"runtime_error","error":str(exc)})
                        finally:
                            trace=out/"traces"/(row["id"]+".json")
                            write_json(trace,{"events":execution.events,"record":row})
                            row["trace"]=str(trace.relative_to(out))
                            row["trace_sha256"]=file_sha256(trace)
                            execution.close()
                            if execution is first:
                                first_closed=True
                        records.append(row)
                    manifest["source_scenes"].append({"scene_id":scene,"split":split,"task":task,"seed":seed,
                        "source_status":records[-len(regimes)]["status"],"regimes":regimes})
                    (out/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
                    print("collected",split,scene,flush=True)
                finally:
                    if not first_closed:
                        first.close()
        views,collisions=observation_rows(out,records)
        write_json(out/f"{split}-views.json",views)
        write_json(out/f"{split}-collisions.json",collisions)
        by_id={r["id"]:r for r in records}
        for backend in ("pixels","frozen"):
            features=np.asarray([encoder.encode(view_from_record(out,by_id[r["transition_id"]],r["phase"]),backend)
                                 for r in views],dtype=np.float32)
            np.savez_compressed(out/f"{split}-{backend}.npz",features=features,
                ids=np.asarray([r["transition_id"]+":"+r["phase"] for r in views]))
        write_json(out/"progress.json",{"splits_finished":split,"scenes":len(manifest["source_scenes"]),
            "records":len(records),"views":len(views),"statuses":dict(Counter(r["status"] for r in records))})
    collected=[args.only_split] if args.only_split else list(config["splits"])
    manifest["partial_split"]=args.only_split
    manifest["audit"]=None if args.only_split else audit_transition_splits(out)
    manifest["feature_counters"]=encoder.counters()
    manifest["source_actor_generations"]=len(manifest["source_scenes"])
    manifest["split_sha256"]={s:file_sha256(out/f"{s}.jsonl") for s in collected}
    manifest["features_sha256"]={s:{b:file_sha256(out/f"{s}-{b}.npz") for b in ("pixels","frozen")} for s in collected}
    manifest["views_sha256"]={s:file_sha256(out/f"{s}-views.json") for s in collected}
    manifest["cuda_peak_allocated_bytes"]=actor.torch.cuda.max_memory_allocated()
    manifest["cuda_peak_reserved_bytes"]=actor.torch.cuda.max_memory_reserved()
    write_json(out/"manifest.json",manifest)


if __name__=="__main__":
    main()
