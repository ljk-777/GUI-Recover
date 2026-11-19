"""Teacher/evaluation preparation for the repair study, outside inference."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil

from .actions import Action, ActionError, parse_action
from .runner import write_json
from .sft import file_sha256
from .sft_recovery import visible_target_boxes
from .verification_data import load_transitions
from .verification_env import TransitionExecution, task_request, teacher_evidence


def png_sha(picture):
    content = io.BytesIO()
    picture.save(content, format="PNG")
    return hashlib.sha256(content.getvalue()).hexdigest()


def balanced(rows, limit):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["seed"], r["id"])):
        groups[row["task"]].append(row)
    selected = []
    while groups and len(selected) < limit:
        for task in sorted(list(groups)):
            selected.append(groups[task].pop(0))
            if not groups[task]:
                del groups[task]
            if len(selected) == limit:
                break
    return selected


def state(execution, root, name):
    path = root/"images"/(name+".png")
    execution.obs.screenshot.save(path)
    evidence = teacher_evidence(execution)
    return {"image": str(path.relative_to(root)), "sha256": file_sha256(path),
        "reward": execution.reward, "terminated": execution.done, "truncated": execution.truncated,
        "effect_achieved": bool(evidence["effect_achieved"]), "teacher_evidence": evidence,
        "task_info": deepcopy(execution.info.get("task_info", {}))}


def boxes(execution, kind="entry"):
    return visible_target_boxes(execution.env, execution.obs, kind)[1]


def make_probes(rows, split, profile):
    valid = [r for r in rows if r["status"] == "ok"]
    healthy = [r for r in valid if r["immediate"]["effect_achieved"]]
    natural = [r for r in valid if not r["immediate"]["effect_achieved"]]
    healthy_limit = profile["development_healthy_probes"] if split == "train" else len(healthy)
    submit = [r for r in healthy if r["task"] == "enter-text"]
    submit_limit = profile["development_submit_probes"] if split == "train" else len(submit)
    result = []
    for kind, chosen in (("natural_error", natural), ("healthy_recheck", balanced(healthy, healthy_limit)),
                         ("normal_submit", balanced(submit, submit_limit))):
        for row in chosen:
            result.append({"id": row["scene_id"]+"-"+kind, "source_id": row["id"],
                "scene_id": row["scene_id"], "task": row["task"], "seed": row["seed"],
                "split": split, "kind": kind, "phase": "immediate",
                "target": "Submit button" if kind == "normal_submit" else row["target"],
                "target_boxes": row["teacher_boxes"]["submit" if kind == "normal_submit" else "immediate"],
                "source_live": not (row["immediate"]["terminated"] or row["immediate"]["truncated"])})
    return result


def prepare_development(reference, out, config, runtime):
    reference, out = Path(reference), Path(out)
    source_manifest = json.loads((reference/"manifest.json").read_text())
    if file_sha256(reference/"manifest.json") != config["reference_manifest_sha256"]:
        raise ValueError("Development source manifest differs from frozen protocol")
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir(); (out/"traces").mkdir()
    shutil.copyfile(reference/"MINIWOB_LICENSE.txt", out/"MINIWOB_LICENSE.txt")
    manifest = {"runtime": runtime, "config": config, "source_manifest_sha256": file_sha256(reference/"manifest.json"),
        "source_collection_code": source_manifest["collection_source_commit"], "splits": {},
        "scope": "Old train/validation only; no reference test rows used for repair selection"}
    for split in ("train", "validation"):
        originals = [r for r in load_transitions(reference, split) if r["regime"] == "none"]
        if config.get("smoke"):
            # Engineering rows are fixed by task/seed, never by candidate predictions.
            keys = {(g["task"], seed) for g in config["smoke_development"][split] for seed in g["seeds"]}
            originals = [r for r in originals if (r["task"], r["seed"]) in keys]
        rows = []
        for original in originals:
            row = deepcopy(original)
            for phase in ("before", "reset_check", "immediate", "after_wait"):
                if row.get(phase):
                    shutil.copyfile(reference/row[phase]["image"], out/row[phase]["image"])
            shutil.copyfile(reference/row["trace"], out/row["trace"])
            execution = TransitionExecution(row["task"], row["seed"], parse_action(json.dumps(row["proposed_action"])))
            try:
                if png_sha(execution.obs.screenshot) != row["before"]["sha256"]:
                    raise ValueError("Development initial screenshot does not replay exactly: "+row["id"])
                teacher_boxes = {"before": boxes(execution)}
                execution.first()
                if png_sha(execution.obs.screenshot) != row["immediate"]["sha256"]:
                    raise ValueError("Development action prefix does not replay exactly: "+row["id"])
                if teacher_evidence(execution)["effect_achieved"] != row["immediate"]["effect_achieved"]:
                    raise ValueError("Replayed source teacher disagrees")
                teacher_boxes["immediate"] = boxes(execution)
                if row["task"] == "enter-text":
                    teacher_boxes["submit"] = boxes(execution, "submit")
                if row.get("after_wait"):
                    execution.wait_once()
                    if png_sha(execution.obs.screenshot) != row["after_wait"]["sha256"]:
                        raise ValueError("Development wait prefix differs")
                    teacher_boxes["after_wait"] = boxes(execution)
                row["teacher_boxes"] = teacher_boxes
                row["original_record_sha256"] = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
                rows.append(row)
                (out/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                print("annotated", split, row["scene_id"], flush=True)
            finally:
                execution.close()
        probes = make_probes(rows, split, config)
        write_json(out/f"{split}-probes.json", probes)
        manifest["splits"][split] = {"sources": len(rows), "natural_errors": sum(not r["immediate"]["effect_achieved"] for r in rows),
            "healthy": sum(r["immediate"]["effect_achieved"] for r in rows), "probes": len(probes),
            "records_sha256": file_sha256(out/f"{split}.jsonl"), "probes_sha256": file_sha256(out/f"{split}-probes.json")}
    write_json(out/"manifest.json", manifest)
    return manifest


def preflight(reference_roots, out, groups, *, prior_scenes=()):
    """Choose the first noncolliding reserved layouts, before actor inference."""
    prior_scenes = list(prior_scenes)
    known = {row["initial_sha256"] for row in prior_scenes}
    scenes = {(row["task"], row["seed"]) for row in prior_scenes}
    registry = []
    for directory in map(Path, reference_roots):
        for split in ("train", "validation", "test"):
            if not (directory/f"{split}.jsonl").exists():
                continue
            for row in load_transitions(directory, split):
                if row["regime"] == "none":
                    known.add(row["before"]["sha256"]); scenes.add((row["task"], row["seed"]))
        registry.append({"directory_name": directory.name,
                         "split_sha256": {s: file_sha256(directory/f"{s}.jsonl") for s in ("train", "validation", "test") if (directory/f"{s}.jsonl").exists()}})
    selected, rejected = [], []
    for group in groups:
        accepted = 0
        for seed in group["candidate_seeds"]:
            if accepted >= group["count"]:
                break
            execution = TransitionExecution(group["task"], seed, None)
            try:
                sha = png_sha(execution.obs.screenshot)
                if sha in known or (group["task"], seed) in scenes:
                    rejected.append({"task": group["task"], "seed": seed, "sha256": sha})
                else:
                    selected.append({"task": group["task"], "seed": seed, "initial_sha256": sha})
                    known.add(sha); scenes.add((group["task"], seed)); accepted += 1
            finally:
                execution.close()
        if accepted != group["count"]:
            raise ValueError("Predeclared reserve exhausted before obtaining unique test layouts")
        print("preflight", group["task"], accepted, flush=True)
    report = {"selected": selected, "rejected_layouts": rejected, "prior_registry": registry,
              "outcomes_observed": False, "actor_calls": 0}
    write_json(Path(out)/"preflight.json", report)
    return report


def collect_test(reference_roots, out, config, runtime, grounder, selection_sha):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir(); (out/"traces").mkdir()
    shutil.copyfile(Path(reference_roots[0])/"MINIWOB_LICENSE.txt", out/"MINIWOB_LICENSE.txt")
    plan = preflight(reference_roots, out, config["test_tasks"])
    rows = []
    for scene in plan["selected"]:
        task, seed = scene["task"], scene["seed"]
        name = f"{task}-{seed}"
        execution = TransitionExecution(task, seed, None)
        row = {"id": name+"-none", "scene_id": name, "split": "test", "task": task, "seed": seed,
            "regime": "none", "deferred": False, "status": "invalid_source", "immediate": None,
            "after_wait": None, "reference_achieved": False, "proposed_action": None}
        try:
            if png_sha(execution.obs.screenshot) != scene["initial_sha256"]:
                raise ValueError("Preflight input changed before source prediction")
            before_path = out/"images"/(name+"-before.png")
            execution.obs.screenshot.save(before_path)
            target, kind, value = task_request(task, execution.obs.goal)
            row.update(goal=execution.obs.goal, target=target,
                before={"image": str(before_path.relative_to(out)), "sha256": file_sha256(before_path)})
            row["teacher_boxes"] = {"before": boxes(execution)}
            prediction = grounder.ground(execution.obs.screenshot, target)
            row["raw"] = prediction["raw"]
            row["source_prediction"] = prediction
            if prediction["point"] is not None:
                execution.proposed = Action(kind, value, tuple(prediction["point"]))
                row["proposed_action"] = execution.proposed.to_dict()
                execution.first()
                row["immediate"] = state(execution, out, name+"-immediate")
                row["reference_achieved"] = row["immediate"]["effect_achieved"]
                row["teacher_boxes"]["immediate"] = boxes(execution)
                if task == "enter-text":
                    row["teacher_boxes"]["submit"] = boxes(execution, "submit")
                if not (execution.done or execution.truncated):
                    execution.wait_once()
                    row["after_wait"] = state(execution, out, name+"-after-wait")
                    row["teacher_boxes"]["after_wait"] = boxes(execution)
                row["status"] = "ok"
        except Exception as exc:
            row.update(status="runtime_error", error=str(exc))
        finally:
            trace = out/"traces"/(row["id"]+".json")
            write_json(trace, {"events": execution.events, "record": row})
            row.update(trace=str(trace.relative_to(out)), trace_sha256=file_sha256(trace))
            execution.close()
        rows.append(row)
        (out/"test.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        print("collected fresh", name, row["status"], flush=True)
    probes = make_probes(rows, "test", config)
    write_json(out/"test-probes.json", probes)
    manifest = {"runtime": runtime, "config": config, "selection_sha256_before_collection": selection_sha,
        "preflight_sha256": file_sha256(out/"preflight.json"), "source_scenes": len(rows),
        "records_sha256": file_sha256(out/"test.jsonl"), "probes_sha256": file_sha256(out/"test-probes.json"),
        "scope": "Fresh natural source actions only; no artificial fault; original rendering fixture"}
    write_json(out/"manifest.json", manifest)
    if any(r["status"] == "runtime_error" for r in rows):
        raise ValueError("Collection runtime errors retained; abort model comparison")
    return manifest
