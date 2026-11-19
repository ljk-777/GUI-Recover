"""CPU fitting/calibration reads train + validation only, never test records."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from gui_recover.cli import fingerprint
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.verification import fit_classifier,probabilities
from gui_recover.verification_data import load_transitions
from gui_recover.verification_training import choose_threshold


def main():
    parser=argparse.ArgumentParser()
    for name in ("config","data","out"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    profile=config["classifier"]
    data,out=Path(args.data),Path(args.out)
    runtime=fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean training code")
    source=json.loads((data/"manifest.json").read_text())
    if source["config"]!=config:
        raise ValueError("Collection config differs from classifier config")
    out.mkdir(parents=True,exist_ok=False)
    views={s:json.loads((data/f"{s}-views.json").read_text()) for s in ("train","validation")}
    records=load_transitions(data,"validation")
    fit=[i for i,r in enumerate(views["train"]) if r["fit_eligible"]]
    if not fit:
        raise ValueError("No injection-transfer training examples")
    group_counts=Counter(views["train"][i]["scene_id"] for i in fit)
    weights=[1/group_counts[views["train"][i]["scene_id"]] for i in fit]
    labels=[views["train"][i]["label"] for i in fit]
    for backend in ("pixels","frozen"):
        directory=out/backend
        directory.mkdir()
        arrays={}
        for split in ("train","validation"):
            if (file_sha256(data/f"{split}-{backend}.npz")!=source["features_sha256"][split][backend]
                    or file_sha256(data/f"{split}-views.json")!=source["views_sha256"][split]):
                raise ValueError("Development features or view labels changed")
            with np.load(data/f"{split}-{backend}.npz",allow_pickle=False) as archive:
                ids=[r["transition_id"]+":"+r["phase"] for r in views[split]]
                if ids!=archive["ids"].tolist():
                    raise ValueError("Feature IDs differ from development views")
                arrays[split]=archive["features"].astype(np.float64)
        trials=[]
        best=None
        for l2 in profile["l2"]:
            model=fit_classifier(arrays["train"][fit],labels,weights,l2=l2,rank=profile["rank"],steps=profile["steps"])
            values=probabilities(model,arrays["validation"])
            chosen,thresholds=choose_threshold(records,views["validation"],values,profile)
            m=chosen["routing"]
            score=(m["injected_failure_recall"]["rate"] or 0,-m["healthy_false_trigger"]["rate"],
                   -(m["pending_premature"]["rate"] or 0),l2,chosen["threshold"])
            trials.append({"l2":l2,"chosen":chosen,"thresholds":thresholds})
            if best is None or score>best[0]:
                best=(score,model,l2,chosen)
        _,model,l2,chosen=best
        np.savez_compressed(directory/"classifier.npz",**model)
        selection={"runtime":runtime,"backend":backend,"l2":l2,"threshold":chosen["threshold"],
            "classifier_sha256":file_sha256(directory/"classifier.npz"),
            "dataset_manifest_sha256":file_sha256(data/"manifest.json"),
            "config_sha256":file_sha256(Path(args.config)),"fit_views":len(fit),
            "fit_scene_groups":len(group_counts),"fit_labels":dict(Counter(labels)),
            "excluded_train_views":len(views["train"])-len(fit),"validation_routing":chosen["routing"],
            "rule":"Injection/healthy validation only; normal false trigger <=5%, pending premature <=0; maximize injected live failure recall; fixed tie-breaks",
            "train_features_sha256":file_sha256(data/f"train-{backend}.npz"),
            "validation_features_sha256":file_sha256(data/f"validation-{backend}.npz")}
        write_json(directory/"selection.json",selection)
        write_json(directory/"calibration-trials.json",trials)
        print("selected",backend,l2,chosen["threshold"],flush=True)
    write_json(out/"training-ledger.json",{"runtime":runtime,"config_sha256":file_sha256(Path(args.config)),
        "dataset_manifest_sha256":file_sha256(data/"manifest.json"),
        "selection_sha256":{b:file_sha256(out/b/"selection.json") for b in ("pixels","frozen")},
        "status":"both classifiers frozen before test evaluation"})


if __name__=="__main__":
    main()
