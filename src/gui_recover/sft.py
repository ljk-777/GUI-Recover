"""Grounding supervision utilities. Teacher metadata never enters the prompt."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from .grounding import parse_point, point_hits_boxes
from .actions import parse_action
from .prompts import GROUNDING


REQUERY = "Locate the center of the empty editable text input box."
HISTORY_PROFILES = ("legacy", "target-last", "current-target", "action-only")


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def grounding_query(query, previous_action=None, *, history_profile="legacy"):
    if history_profile not in HISTORY_PROFILES:
        raise ValueError("Unknown grounding history profile")
    if previous_action is None:
        return query
    # The previous PROPOSED action is the only admitted prefix. Rewards,
    # teacher boxes, fault labels and executed-code strings are not accepted.
    action = parse_action(json.dumps(previous_action)).to_dict()
    previous = "Previous proposed action: " + json.dumps(action, separators=(",", ":"))
    recheck = "Recheck the current screenshot and return the target location."
    if history_profile == "legacy":
        return query + "\n" + previous + "\n" + recheck
    if history_profile == "target-last":
        # Same text/information as legacy; only the three lines are reordered.
        return previous + "\n" + recheck + "\n" + query
    if history_profile == "action-only":
        # Explicit information-removal diagnostic, not an equal-information ablation.
        action = {key: action[key] for key in ("action", "value")}
    return ("Historical action context: " + json.dumps(action, separators=(",", ":"))
            + "\nLocate the CURRENT target in the current screenshot: " + query)


def grounding_messages(picture, query, answer=None, previous_action=None, *, history_profile="legacy"):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": GROUNDING}, {"type": "image", "image": picture},
        {"type": "text", "text": grounding_query(query, previous_action, history_profile=history_profile)}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def completion_labels(full_ids, prompt_ids):
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("Assistant token boundary is not an exact nonempty prompt prefix")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]


def load_records(directory, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown dataset split")
    root = Path(directory).resolve()
    rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines() if line]
    if not rows:
        raise ValueError("Empty split")
    ids = set()
    for row in rows:
        if row["split"] != split or row["id"] in ids or not row["query"].strip():
            raise ValueError("Invalid or duplicate supervision record")
        ids.add(row["id"])
        if row.get("previous_action") is not None:
            parse_action(json.dumps(row["previous_action"]))
        image = (root / row["image"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError("Image must exist inside the dataset")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("Image checksum differs")
        boxes = row["target_boxes"]
        if not boxes or any(len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1) for b in boxes):
            raise ValueError("Invalid visible target boxes")
        point = parse_point(row["answer"])
        if list(point) != row["target_point"] or not point_hits_boxes(point, boxes):
            raise ValueError("Supervised answer is outside the teacher target")
    return rows


def audit_dataset(directory):
    rows = {s: load_records(directory, s) for s in ("train", "validation", "test")}
    scenes, pictures, tasks = {}, {}, {}
    for split, records in rows.items():
        scenes[split] = {(r["task"], r["seed"]) for r in records}
        pictures[split] = {r["image_sha256"] for r in records}
        tasks[split] = {r["task"] for r in records}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if scenes[a] & scenes[b] or pictures[a] & pictures[b]:
            raise ValueError("Scene or identical screenshot leaks across splits")
    return {"records": {s: len(r) for s, r in rows.items()},
            "scenes": {s: len(v) for s, v in scenes.items()},
            "tasks": {s: sorted(v) for s, v in tasks.items()},
            "unseen_test_tasks": sorted(tasks["test"] - tasks["train"]),
            "cross_split_scene_and_image_overlap": 0}


def encode_record(processor, directory, row, context_mode="none"):
    # Only query/image/answer are consumed. DOM boxes, task_info and labels
    # remain evaluator/teacher metadata even if a record contains more fields.
    picture = Image.open(Path(directory) / row["image"]).convert("RGB")
    if context_mode not in ("none", "previous-action"):
        raise ValueError("Unknown grounding context mode")
    previous = row.get("previous_action") if context_mode == "previous-action" else None
    prompt = processor.apply_chat_template(grounding_messages(picture, row["query"], previous_action=previous),
                                           tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(grounding_messages(picture, row["query"], row["answer"], previous),
                                         tokenize=False, add_generation_prompt=False)
    prefix = processor(text=[prompt], images=[picture], return_tensors="pt")
    encoded = processor(text=[full], images=[picture], return_tensors="pt")
    labels = completion_labels(encoded["input_ids"][0].tolist(), prefix["input_ids"][0].tolist())
    if len(labels) > 768:
        raise ValueError("Sequence exceeds frozen smoke budget; never truncate image tokens")
    encoded["labels"] = encoded["input_ids"].new_tensor([labels])
    return encoded
