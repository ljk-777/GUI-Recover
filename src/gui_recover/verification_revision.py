"""Audited focus-event label correction plus a genuinely fresh test split."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from .runner import is_success, write_json
from .sft import file_sha256
from .verification_data import audit_transition_splits, load_transitions, observation_rows


def revise_focus_record(record):
    row = deepcopy(record)
    if row["task"] != "focus-text":
        return row
    for phase in ("immediate", "after_wait"):
        state = row.get(phase)
        if state is None:
            continue
        previous = state["teacher_evidence"]
        if previous["kind"] == "focus_event_success":
            continue
        if previous["kind"] != "input_focus":
            raise ValueError("Unrecognized focus teacher evidence")
        achieved = is_success(state["reward"], state["terminated"], state["truncated"], state["task_info"])
        state["teacher_evidence"] = {"kind": "focus_event_success",
            "focused_after_step": previous["effect_achieved"], "effect_achieved": achieved,
            "original_teacher_evidence": previous}
        state["effect_achieved"] = achieved
    return row


def assemble_revised_data(development, fresh_test, out, config, runtime):
    development, fresh_test, out = map(Path, (development, fresh_test, out))
    old = json.loads((development/"manifest.json").read_text())
    fresh = json.loads((fresh_test/"manifest.json").read_text())
    if (old["config"]["splits"]["train"] != config["splits"]["train"] or
            old["config"]["splits"]["validation"] != config["splits"]["validation"] or
            fresh["config"] != config or fresh["partial_split"] != "test" or
            fresh["runtime"]["git_commit"] != runtime["git_commit"]):
        raise ValueError("Development membership or fresh collection protocol differs")
    old_test = {r["seed"] for r in load_transitions(development, "test")}
    if old_test & {seed for g in config["splits"]["test"] for seed in g["seeds"]}:
        raise ValueError("Previously exposed test seeds cannot be reused")
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir(); (out/"traces").mkdir()
    shutil.copyfile(development/"MINIWOB_LICENSE.txt", out/"MINIWOB_LICENSE.txt")
    scenes, lineage, changes = [], {}, []
    for split in config["splits"]:
        source = fresh_test if split == "test" else development
        original = load_transitions(source, split)
        rows = original if split == "test" else [revise_focus_record(r) for r in original]
        by_scene = {}
        for row in rows:
            by_scene.setdefault(row["scene_id"], []).append(row)
            for phase in ("before", "reset_check", "immediate", "after_wait"):
                if row.get(phase):
                    path = row[phase]["image"]
                    shutil.copyfile(source/path, out/path)
            shutil.copyfile(source/row["trace"], out/row["trace"])
        for scene, records in by_scene.items():
            reference = next(r for r in records if r["regime"] == "none")
            achieved = reference["status"] == "ok" and reference["immediate"]["effect_achieved"]
            for row in records:
                if split != "test":
                    row["reference_achieved"] = bool(achieved)
            scenes.append({"scene_id": scene, "split": split, "task": reference["task"],
                "seed": reference["seed"], "source_status": reference["status"], "regimes": [r["regime"] for r in records]})
        for before, after in zip(original, rows):
            if before != after:
                changes.append(after["id"])
        if split == "test":
            for name in (f"{split}.jsonl", f"{split}-views.json", f"{split}-collisions.json"):
                shutil.copyfile(source/name, out/name)
        else:
            (out/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            views, collisions = observation_rows(out, rows)
            write_json(out/f"{split}-views.json", views)
            write_json(out/f"{split}-collisions.json", collisions)
        views = json.loads((out/f"{split}-views.json").read_text())
        for backend in ("pixels", "frozen"):
            name = f"{split}-{backend}.npz"
            with np.load(source/name, allow_pickle=False) as a:
                if a["ids"].tolist() != [v["transition_id"]+":"+v["phase"] for v in views]:
                    raise ValueError("Label revision changed feature alignment")
            shutil.copyfile(source/name, out/name)
        lineage[split] = {"source_manifest_sha256": file_sha256(source/"manifest.json"),
            "source_split_sha256": file_sha256(source/f"{split}.jsonl"),
            "source_features_sha256": {b: file_sha256(source/f"{split}-{b}.npz") for b in ("pixels", "frozen")}}
    m = {"runtime": runtime, "config": config, "source_scenes": scenes,
        "source_actor_generations": len(scenes), "collection_generations_including_superseded_tests": 390,
        "collection_source_commit": old["collection_source_commit"],
        "fresh_test_source_commit": fresh["runtime"]["git_commit"],
        "render_fixture": fresh["render_fixture"], "regroup": old["regroup"],
        "label_revision": {"reason": "focus-text deliberately blurs inside its successful focus event handler",
            "changed_development_records": changes, "lineage": lineage,
            "old_test_excluded": True, "raw_images_traces_and_features_unchanged": True,
            "historical_trace_record": "Retains original label evidence and original split; corrected current labels are in JSONL"},
        "audit": audit_transition_splits(out),
        "split_sha256": {s: file_sha256(out/f"{s}.jsonl") for s in config["splits"]},
        "views_sha256": {s: file_sha256(out/f"{s}-views.json") for s in config["splits"]},
        "features_sha256": {s: {b: file_sha256(out/f"{s}-{b}.npz") for b in ("pixels", "frozen")} for s in config["splits"]}}
    write_json(out/"manifest.json", m)
    return m
