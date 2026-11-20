"""Offline schemas and audits; these helpers are not visual verifier inputs."""

from collections import defaultdict
import json
from pathlib import Path

from PIL import Image

from .actions import parse_action
from .sft import file_sha256
from .verification import TransitionView, observation_key


def load_transitions(root, split):
    root = Path(root).resolve()
    records = [json.loads(x) for x in (root/f"{split}.jsonl").read_text().splitlines()]
    ids = set()
    for row in records:
        if row["split"] != split or row["id"] in ids:
            raise ValueError("Duplicate ID or incorrect split")
        ids.add(row["id"])
        trace=(root/row["trace"]).resolve()
        if not trace.is_relative_to(root) or file_sha256(trace) != row["trace_sha256"]:
            raise ValueError("Trace path or checksum differs")
        if row["status"] == "ok":
            parse_action(json.dumps(row["proposed_action"]))
        for phase in ("before", "reset_check", "immediate", "after_wait"):
            if row.get(phase) is None:
                continue
            frame = row[phase]
            path = (root/frame["image"]).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != frame["sha256"]:
                raise ValueError("Transition image path or checksum differs")
    return records


def view_from_record(root, row, phase):
    root = Path(root)
    return TransitionView(Image.open(root/row["before"]["image"]).convert("RGB"),
        Image.open(root/row[phase]["image"]).convert("RGB"),
        parse_action(json.dumps(row["proposed_action"])), row["goal"], row["target"], phase)


def observation_rows(root, records):
    result = []
    for row in records:
        if row["status"] != "ok":
            continue
        for phase in ("immediate", "after_wait"):
            if row.get(phase) is None:
                continue
            label = "unknown" if phase == "immediate" and row["deferred"] else (
                "achieved" if row[phase]["effect_achieved"] else "not_achieved")
            result.append({"transition_id": row["id"], "scene_id": row["scene_id"], "phase": phase,
                "split": row["split"], "label": label, "raw_label": label,
                "observation_key": observation_key(view_from_record(root, row, phase)),
                "fit_eligible": row["reference_achieved"], "regime": row["regime"],
                "effect_achieved": row[phase]["effect_achieved"],
                "live": not (row[phase]["terminated"] or row[phase]["truncated"]),
                "deferred": row["deferred"] and phase == "immediate"})
    groups = defaultdict(list)
    for item in result:
        groups[item["observation_key"]].append(item)
    collisions = []
    for key, group in groups.items():
        labels = {r["raw_label"] for r in group}
        if len(labels) > 1:
            collisions.append({"observation_key": key, "labels": sorted(labels),
                               "views": [r["transition_id"]+":"+r["phase"] for r in group]})
            for item in group:
                item["label"] = "unknown"
    return result, collisions


def audit_transition_splits(root):
    rows = {s: load_transitions(root,s) for s in ("train","validation","test")}
    groups = {s: {r["scene_id"] for r in records} for s, records in rows.items()}
    pictures = {s: {r["before"]["sha256"] for r in records} for s, records in rows.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if groups[a]&groups[b] or pictures[a]&pictures[b]:
            raise ValueError("Scene or initial screenshot crosses splits")
    return {"scenes": {s:len(g) for s,g in groups.items()},
            "transitions": {s:len(r) for s,r in rows.items()}, "cross_split_scene_and_initial_image_overlap":0}
