"""Validation-only selective-routing calibration; natural errors stay held out."""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .sft import file_sha256
from .verification import STATUSES, confidence_verdict, probabilities


def routed_decision(record, verdicts):
    first=verdicts[(record["id"],"immediate")]
    wait=first["status"]=="unknown" and record.get("after_wait") is not None
    final=verdicts[(record["id"],"after_wait")] if wait else first
    return {"trigger":final["status"]=="not_achieved", "wait":wait,
            "premature":record["deferred"] and first["status"]=="not_achieved"}


def routing_metrics(records,verdicts,*,development_only=False):
    rows=[r for r in records if r["status"]=="ok" and
          (not development_only or r["reference_achieved"])]
    results=[]
    for row in rows:
        d=routed_decision(row,verdicts)
        live=not (row["immediate"]["terminated"] or row["immediate"]["truncated"])
        achieved=row["immediate"]["effect_achieved"]
        results.append({**d,"id":row["id"],"live":live,
            "healthy":live and achieved and not row["deferred"],
            "injected_failure":live and row["reference_achieved"] and row["regime"] not in ("none","deferred-input") and not achieved,
            "natural_failure":live and row["regime"]=="none" and not achieved,
            "pending":live and row["deferred"]})
    def outcome(flag,key="trigger"):
        subset=[r for r in results if r[flag]]
        return {"count":len(subset),"triggers":sum(r[key] for r in subset),
                "rate":sum(r[key] for r in subset)/len(subset) if subset else None}
    return {"healthy_false_trigger":outcome("healthy"),"injected_failure_recall":outcome("injected_failure"),
        "natural_failure_recall":outcome("natural_failure"),"pending_premature":outcome("pending","premature"),
        "records":len(rows),"waits":sum(r["wait"] for r in results),"decisions":results}


def probability_verdicts(views,values,threshold):
    return {(r["transition_id"],r["phase"]):confidence_verdict(p,threshold) for r,p in zip(views,values)}


def choose_threshold(records,views,values,profile):
    candidates=[]
    for threshold in profile["thresholds"]:
        m=routing_metrics(records,probability_verdicts(views,values,threshold),development_only=True)
        false=m["healthy_false_trigger"]["rate"]
        pending=m["pending_premature"]["rate"]
        if false is None:
            raise ValueError("Validation has no eligible healthy live controls")
        eligible=(false<=profile["normal_false_trigger_budget"] and
                  (pending is None or pending<=profile["premature_pending_trigger_budget"]))
        candidates.append({"threshold":threshold,"eligible":eligible,"routing":m})
    good=[c for c in candidates if c["eligible"]]
    if not good:
        raise ValueError("No conservative fallback threshold passed")
    def score(c):
        m=c["routing"]
        return (m["injected_failure_recall"]["rate"] or 0,
                -m["healthy_false_trigger"]["rate"],-(m["pending_premature"]["rate"] or 0),c["threshold"])
    return max(good,key=score),candidates


class LearnedVerifier:
    def __init__(self,directory,encoder):
        self.directory=Path(directory)
        self.metadata=json.loads((self.directory/"selection.json").read_text())
        if file_sha256(self.directory/"classifier.npz") != self.metadata["classifier_sha256"]:
            raise ValueError("Classifier checksum differs")
        with np.load(self.directory/"classifier.npz",allow_pickle=False) as archive:
            self.model={key:archive[key] for key in archive.files}
        self.encoder=encoder
        self.calls=0
        self.seconds=0.0

    def verify(self,view):
        started=time.perf_counter()
        self.calls+=1
        if view.before.size!=view.after.size:
            return {"status":"unknown","reason":"size_changed"}
        feature=self.encoder.encode(view,self.metadata["backend"])
        result=confidence_verdict(probabilities(self.model,feature[None,:])[0],self.metadata["threshold"])
        self.seconds+=time.perf_counter()-started
        return result
