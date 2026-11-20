"""Export training and validation supervision from verified browser repairs."""

import argparse
import json
from pathlib import Path

from PIL import Image

from gui_recover.actions import parse_action
from gui_recover.contextual_repair import prepare_repair
from gui_recover.grounding import point_hits_boxes
from gui_recover.multistep_recovery import BUTTONS, validate_plan
from gui_recover.runner import write_json
from gui_recover.sft import audit_dataset, file_sha256
from gui_recover.verification import TransitionView, pixel_hash
from gui_recover.visual_text_audit import checked_file


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_examples(run, destination):
    index = run / "development" / "episodes.jsonl"
    rows = read_rows(index)
    found = json.loads((run / "preflight.json").read_text())
    validate_plan(found)
    scenes = {(r["task"], r["seed"]): r for r in found["selected"]}
    seen = {}
    partitions = {s: [] for s in ("train", "validation", "test")}
    exclusions = []
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()

    def keep(scene, split, role, picture, query, point, bounds, provenance):
        identity = f"{scene['task']}-{scene['seed']}-{role}"
        if not point_hits_boxes(point, [bounds]):
            exclusions.append({"id": identity, "reason": "executed_point_outside_teacher_bounds"})
            return
        fingerprint = pixel_hash(picture)
        if seen.setdefault(fingerprint, split) != split:
            raise ValueError("Identical supervision screenshot crossed a split")
        name = f"images/{identity}.png"
        picture.save(destination / name)
        partitions[split].append({"id": identity, "split": split, "task": scene["task"], "seed": scene["seed"],
            "kind": role, "query": query, "image": name, "image_sha256": file_sha256(destination / name),
            "target_boxes": [bounds], "target_point": point, "answer": json.dumps(point, separators=(",", ":")),
            "provenance": provenance})

    for row in rows:
        if row["split"] not in ("train", "validation"):
            raise ValueError("Development trace includes evaluation data")
        scene = scenes[row["task"], row["seed"]]
        fixture = scene["fixture"]
        path = checked_file(run / "development" / row["trace"], row["trace_sha256"])
        if row["method"] != "visual" or not row["success"] or not row["recovered_stage"]:
            continue
        if row["regime"] == "healthy" and row["stage_correct"]:
            role, source, point = "healthy_anchor", "initial", list(fixture["correct"])
        elif not row["stage_correct"] and row["repairs"] == 1:
            role = row["regime"]
            repair, = row["repair_evidence"]
            source, point = repair["observation"], repair["position"]
        else:
            continue
        frame = row["frames"][source]
        label = {"trace_sha256": row["trace_sha256"], "frame_sha256": frame["sha256"],
                 "first_stage_correct": row["stage_correct"], "exact_native_success": row["success"]}
        picture = Image.open(checked_file(path.parent / frame["image"], frame["sha256"])).convert("RGB")
        box = list(fixture["target_box"])
        query = fixture["target"]
        if role != "healthy_anchor":
            proposed = parse_action(json.dumps(row["proposed"]))
            transitioned = TransitionView(picture, picture, proposed, scene["goal"], query)
            crop = prepare_repair(transitioned, "feedback_zoom",
                                  fixed_crop=None if row["task"] == BUTTONS else (0, 0, .60, .60))
            cw, ch = crop.original_size
            x1, y1, x2, y2 = crop.crop
            x, y = point
            point = [(x*cw-x1)/(x2-x1), (y*ch-y1)/(y2-y1)]
            box = [max(0, (box[0]*cw-x1)/(x2-x1)), max(0, (box[1]*ch-y1)/(y2-y1)),
                   min(1, (box[2]*cw-x1)/(x2-x1)), min(1, (box[3]*ch-y1)/(y2-y1))]
            label.update(crop=list(crop.crop), originating_frame=source)
            if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                exclusions.append({"id": f"{scene['task']}-{scene['seed']}-{role}", "reason": "target_outside_crop"})
                continue
            picture, query = crop.picture, crop.query
        keep(scene, row["split"], role, picture, query, point, box, label)

    # These untouched prefixes are for offline evaluation only. The trainer reads only train and validation.
    for row in read_rows(run / "evaluation" / "episodes.jsonl"):
        if row["regime"] != "healthy" or row["method"] != "continue":
            continue
        scene = scenes[row["task"], row["seed"]]
        fixture = scene["fixture"]
        frame = row["frames"]["initial"]
        branch = (run / "evaluation" / row["trace"]).parent
        picture = Image.open(checked_file(branch / frame["image"], frame["sha256"])).convert("RGB")
        keep(scene, "test", "untouched_initial", picture, fixture["target"], list(fixture["correct"]),
             list(fixture["target_box"]), {"frame_sha256": frame["sha256"], "source_split": row["split"]})
    for split, subset in partitions.items():
        with (destination / f"{split}.jsonl").open("x") as stream:
            for record in subset:
                stream.write(json.dumps(record) + "\n")
    checked = audit_dataset(destination)
    correction_counts = {s: sum(r["kind"] != "healthy_anchor" for r in partitions[s])
                         for s in ("train", "validation")}
    if not all(correction_counts.values()):
        raise ValueError("Both development splits require a verified correction")
    write_json(destination / "manifest.json", {"source_plan_sha256": file_sha256(run / "preflight.json"),
        "development_index_sha256": file_sha256(index), "corrective_samples": correction_counts,
        "target": "executed successful repair coordinates or healthy anchors",
        "data_integrity": checked, "excluded": exclusions})
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_examples(args.run, args.out), indent=2))


if __name__ == "__main__":
    main()
