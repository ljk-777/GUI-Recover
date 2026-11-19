"""Pre-fit development regrouping; preserves all scenes and the entire test split."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from .runner import write_json
from .sft import file_sha256
from .verification_data import audit_transition_splits, load_transitions, observation_rows


def scene_plan(config):
    plan = {}
    for split, groups in config["splits"].items():
        for group in groups:
            faults = config["input_faults"] if group["task"] == "enter-text" else config["click_faults"]
            for index, seed in enumerate(group["seeds"]):
                scene = f"{group['task']}-{seed}"
                if scene in plan:
                    raise ValueError("Duplicate planned scene")
                plan[scene] = (split, faults[index % len(faults)])
    return plan


def validate_regroup(original, revised):
    a, b = deepcopy(original), deepcopy(revised)
    a.pop("splits"); b.pop("splits")
    if a != b or original["splits"]["test"] != revised["splits"]["test"]:
        raise ValueError("Only development split membership may change; test is immutable")
    old, new = scene_plan(original), scene_plan(revised)
    if old.keys() != new.keys() or any(old[s][1] != new[s][1] for s in old):
        raise ValueError("All scenes and assigned faults must be preserved")
    for split in original["splits"]:
        counts = lambda cfg: {g["task"]: len(g["seeds"]) for g in cfg["splits"][split]}
        if counts(original) != counts(revised):
            raise ValueError("Per-task split sizes must be preserved")
    return old, new


def regroup_development_data(source, out, original, revised, collection_commit, runtime):
    """Copy immutable raw evidence, reindex cached vectors, regenerate split views.

    Historical trace.record.split remains the original collection split. Current
    JSONL membership is explicit, with both mappings and hashes in the manifest.
    No model fitting, inference or outcome-dependent sample selection occurs here.
    """
    old, new = validate_regroup(original, revised)
    source, out = Path(source), Path(out)
    rows = {s: load_transitions(source, s) for s in original["splits"]}
    by_scene = {}
    for split, records in rows.items():
        for row in records:
            by_scene.setdefault(row["scene_id"], []).append(row)
    if by_scene.keys() != old.keys():
        raise ValueError("Source does not contain exactly the planned scenes")
    for scene, records in by_scene.items():
        fault = old[scene][1]
        expected = {"none", fault} | ({"deferred-input"} if fault == "drop-input" else set())
        if {r["regime"] for r in records} != expected or len(records) != len(expected):
            raise ValueError("Source variants differ from the collection plan")
    vectors = {b: {} for b in ("pixels", "frozen")}
    for split in rows:
        source_views = json.loads((source/f"{split}-views.json").read_text())
        for backend in vectors:
            with np.load(source/f"{split}-{backend}.npz", allow_pickle=False) as a:
                keys = [v["transition_id"]+":"+v["phase"] for v in source_views]
                if a["ids"].tolist() != keys:
                    raise ValueError("Source feature IDs differ")
                vectors[backend].update(zip(keys, a["features"]))
    source_files = {str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    shutil.copytree(source, out)  # Must be new; original assets and traces stay byte-identical.
    scenes = []
    for split, groups in revised["splits"].items():
        records = []
        for group in groups:
            for seed in group["seeds"]:
                scene = f"{group['task']}-{seed}"
                copied = deepcopy(by_scene[scene])
                for row in copied:
                    row["split"] = split
                records.extend(copied)
                scenes.append({"scene_id": scene, "split": split, "task": group["task"], "seed": seed,
                               "source_status": copied[0]["status"], "regimes": [r["regime"] for r in copied]})
        if split == "test":
            if records != rows[split]:
                raise ValueError("Test records changed")
            continue  # Preserve test JSON, features and labels byte-for-byte.
        (out/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        views, collisions = observation_rows(out, records)
        write_json(out/f"{split}-views.json", views)
        write_json(out/f"{split}-collisions.json", collisions)
        for backend in vectors:
            ids = [v["transition_id"]+":"+v["phase"] for v in views]
            np.savez_compressed(out/f"{split}-{backend}.npz", ids=np.asarray(ids),
                                features=np.asarray([vectors[backend][key] for key in ids]))
    manifest = {"runtime": runtime, "config": revised, "source_scenes": scenes,
        "source_actor_generations": len(scenes), "new_actor_generations_during_regroup": 0,
        "collection_source_commit": collection_commit,
        "scope": "Pre-fit development regrouping; original traces preserve historical split; test unchanged",
        "render_fixture": "CSS caret hidden, solid button borders, double-RAF native CDP capture; raw PNGs unedited; not stock benchmark rendering",
        "regroup": {"reason": "Identical train/validation initial images detected before any classifier fitting",
            "original_config": original, "source_files_sha256": source_files,
            "moves": [{"scene_id": s, "from": old[s][0], "to": new[s][0]} for s in old if old[s][0] != new[s][0]],
            "selection": "Explicit development membership in revised config; no rows removed, no fault changed, no test membership changed"},
        "audit": audit_transition_splits(out),
        "split_sha256": {s: file_sha256(out/f"{s}.jsonl") for s in rows},
        "views_sha256": {s: file_sha256(out/f"{s}-views.json") for s in rows},
        "features_sha256": {s: {b: file_sha256(out/f"{s}-{b}.npz") for b in vectors} for s in rows}}
    for name, digest in source_files.items():
        if name.startswith(("images/", "traces/", "test")) and file_sha256(out/name) != digest:
            raise ValueError("Immutable evidence changed")
    write_json(out/"manifest.json", manifest)
    return manifest
