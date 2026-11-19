"""Fresh test discrimination plus actual selective-recovery execution."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from gui_recover.cli import fingerprint
from gui_recover.policies import ShowUIPolicy
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification import STATUSES,appearance_verdict,probabilities
from gui_recover.verification_backbone import FrozenFeatures
from gui_recover.verification_data import audit_transition_splits,load_transitions,view_from_record
from gui_recover.verification_runner import GroundingCache,run_verification_case
from gui_recover.verification_training import LearnedVerifier,probability_verdicts,routing_metrics


def main():
    parser=argparse.ArgumentParser()
    for name in ("config","data","classifiers","model","out"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    data,out=Path(args.data),Path(args.out)
    runtime=fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean acceptance code")
    manifests={b:json.loads((Path(args.classifiers)/b/"selection.json").read_text()) for b in ("pixels","frozen")}
    ledger=json.loads((Path(args.classifiers)/"training-ledger.json").read_text())
    source=json.loads((data/"manifest.json").read_text())
    if source["config"]!=config or ledger["dataset_manifest_sha256"]!=file_sha256(data/"manifest.json"):
        raise ValueError("Frozen collection/training ledger differs")
    for b,m in manifests.items():
        if (m["config_sha256"]!=file_sha256(Path(args.config)) or m["dataset_manifest_sha256"]!=file_sha256(data/"manifest.json")
                or m["runtime"]["git_commit"]!=runtime["git_commit"]
                or ledger["selection_sha256"][b]!=file_sha256(Path(args.classifiers)/b/"selection.json")):
            raise ValueError("Validation selection differs from frozen acceptance")
    # Test records are not read until both selections have been checked.
    audit=audit_transition_splits(data)
    records=load_transitions(data,"test")
    views=json.loads((data/"test-views.json").read_text())
    if file_sha256(data/"test-views.json")!=source["views_sha256"]["test"]:
        raise ValueError("Test view definitions changed")
    by_id={r["id"]:r for r in records}
    out.mkdir(parents=True,exist_ok=False)
    all_verdicts={}
    for backend in ("pixels","frozen"):
        directory=Path(args.classifiers)/backend
        metadata=manifests[backend]
        if metadata["classifier_sha256"]!=file_sha256(directory/"classifier.npz"):
            raise ValueError("Classifier bytes differ")
        with np.load(directory/"classifier.npz",allow_pickle=False) as archive:
            model={k:archive[k] for k in archive.files}
        with np.load(data/f"test-{backend}.npz",allow_pickle=False) as archive:
            if file_sha256(data/f"test-{backend}.npz")!=source["features_sha256"]["test"][backend]:
                raise ValueError("Test features changed")
            if archive["ids"].tolist()!=[r["transition_id"]+":"+r["phase"] for r in views]:
                raise ValueError("Test feature IDs differ")
            values=probabilities(model,archive["features"].astype(np.float64))
        all_verdicts[backend]=probability_verdicts(views,values,metadata["threshold"])
    all_verdicts["appearance"]={(v["transition_id"],v["phase"]):appearance_verdict(
        view_from_record(data,by_id[v["transition_id"]],v["phase"])) for v in views}
    predictions=[]
    for view in views:
        key=(view["transition_id"],view["phase"])
        predictions.append({**view,"predictions":{b:all_verdicts[b][key] for b in all_verdicts}})
    write_json(out/"predictions.json",predictions)
    offline={}
    for backend,verdicts in all_verdicts.items():
        confusion={label:{pred:0 for pred in STATUSES} for label in STATUSES}
        for view in views:
            confusion[view["label"]][verdicts[(view["transition_id"],view["phase"])]["status"]]+=1
        offline[backend]={"confusion":confusion,"routing":routing_metrics(records,verdicts)}
    write_json(out/"offline-summary.json",offline)
    if file_sha256(Path(args.model)/"pytorch_model.bin")!=config["base_weights_sha256"]:
        raise ValueError("Actor weights changed")
    actor=ShowUIPolicy(args.model,max_new_tokens=32)
    encoder=FrozenFeatures(actor)
    verifier=LearnedVerifier(Path(args.classifiers)/config["classifier"]["primary_backend"],encoder)
    grounder=GroundingCache(actor)
    selected=[r for r in records if r["seed"] in config["online"]["seeds"]]
    write_json(out/"manifest.json",{"runtime":runtime,"config":config,"data_audit":audit,
        "data_manifest_sha256":file_sha256(data/"manifest.json"),
        "classifier_sha256":{b:m["classifier_sha256"] for b,m in manifests.items()},
        "selections":manifests,"online_case_ids":[r["id"] for r in selected],
        "scope":"Same fixed actor/controller, one repair and one wait; oracle explicitly privileged; no restart"})
    episodes=[]
    for row in selected:
        for strategy in config["online"]["strategies"]:
            result=run_verification_case(row,strategy,grounder,verifier,out)
            episodes.append(result)
            write_json(out/"online-results.json",episodes)
            print("executed",row["id"],strategy,result["reason"],flush=True)
    def aggregate(rows):
        return {s:{"episodes":sum(r["strategy"]==s for r in rows),
            "successes":sum(r["strategy"]==s and r["success"] for r in rows),
            "repairs":sum(r["repairs"] for r in rows if r["strategy"]==s),
            "waits":sum(r["waits"] for r in rows if r["strategy"]==s),
            "logical_actor_calls":sum(r["logical_actor_calls"] for r in rows if r["strategy"]==s),
            "runtime_errors":sum(r["reason"]=="runtime_error" for r in rows if r["strategy"]==s),
            "prefix_mismatches":sum(r["reason"]=="prefix_mismatch" for r in rows if r["strategy"]==s)} for s in config["online"]["strategies"]}
    summary={"offline":offline,"online_all":aggregate(episodes),
        "online_by_regime":{regime:aggregate([r for r in episodes if r["regime"]==regime]) for regime in sorted({r["regime"] for r in episodes})},
        "online_real_errors":aggregate([r for r in episodes if r["regime"]=="none" and by_id[r["id"]].get("immediate") and not by_id[r["id"]]["immediate"]["effect_achieved"]]),
        "online_healthy_controls":aggregate([r for r in episodes if r["regime"]=="none" and by_id[r["id"]].get("immediate") and by_id[r["id"]]["immediate"]["effect_achieved"]]),
        "actor_new_generations":grounder.calls,"actor_requests":grounder.requests,"feature_counters":encoder.counters(),
        "runtime_errors":[r for r in episodes if r["reason"]=="runtime_error"],
        "peak_cuda_allocated_bytes":actor.torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes":actor.torch.cuda.max_memory_reserved()}
    write_json(out/"summary.json",summary)
    if summary["runtime_errors"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
