"""Reconstruct a live format study from its saved trajectories on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.runner import INVALID_ACTION_FEEDBACK, aggregate, write_json
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import checked_file, read_json, require
from run_visualwebarena_live import TASK_SETS, exact_vwa_success


MODES = ("none", "retry", "feedback")


def audit_episode(directory, budget, mode):
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    row = read_json(directory / "episode.json")
    require(events[0]["event"] == "reset" and events[-1] == {"event": "episode_end", **row},
            f"Incomplete trajectory: {directory.name}")
    require(row["reason"] != "runtime_error" and row["recovery"] == row["fault_kind"] == "none",
            f"Unexpected runtime or intervention: {directory.name}")
    require(row["invalid_action_retry"] == mode, f"Incorrect study mode: {directory.name}")
    by_event = {}
    for event in events:
        by_event.setdefault(event["event"], []).append(event)
    predictions = by_event["prediction"]
    transitions = by_event.get("transition", [])
    require(0 < len(predictions) <= budget and [e["step"] for e in predictions] == list(range(len(predictions))),
            f"Invalid prediction budget: {directory.name}")
    invalid_steps = []
    valid_steps = []
    formats = Counter()
    for prediction in predictions:
        formats[prediction["metrics"]["provider_format"]] += 1
        try:
            parse_action(prediction["raw"])
        except ActionError:
            invalid_steps.append(prediction["step"])
        else:
            valid_steps.append(prediction["step"])
    require([e["step"] for e in transitions] == valid_steps, "Unparsed action reached the browser")
    require([e["step"] for e in by_event.get("error", [])] == invalid_steps, "Invalid prediction log differs")
    for event in transitions:
        action = parse_action(predictions[event["step"]]["raw"]).to_dict()
        require(json.loads(json.dumps(action)) == event["action"] == event["proposed_action"],
                "Executed action differs from prediction")

    retries = by_event.get("invalid_action_retry", [])
    repaired = by_event.get("invalid_action_repaired", [])
    first_invalid = invalid_steps[0] if invalid_steps else None
    should_retry = mode != "none" and first_invalid is not None and first_invalid + 1 < budget
    require(len(retries) == int(should_retry), "Missing or extra invalid-action retry")
    if retries:
        retry = retries[0]
        next_step = retry["next_prediction"]
        expected_message = INVALID_ACTION_FEEDBACK if mode == "feedback" else None
        require(retry["step"] == first_invalid and next_step == first_invalid + 1 < len(predictions)
                and retry["mode"] == mode and retry["message"] == expected_message, "Invalid retry evidence")
        require([event["step"] for event in repaired] == ([next_step] if next_step in valid_steps else []),
                "Repair count differs from the immediate retry outcome")
    else:
        require(not repaired, "Repair without a retry")
    counts = {"predictions": len(predictions), "model_calls": sum(e["metrics"]["model_invocations"] for e in predictions),
              "steps": len(transitions), "invalid_predictions": len(invalid_steps),
              "invalid_action_retries": len(retries), "repaired_predictions": len(repaired)}
    require(all(row[key] == value for key, value in counts.items()), "Episode counters differ from trace")

    terminal = transitions[-1] if transitions else {"reward": 0, "terminated": False, "truncated": False}
    require(all(row[key] == terminal[key] for key in ("reward", "terminated", "truncated")),
            "Terminal reward differs from the final transition")
    require(row["success"] == exact_vwa_success(terminal["reward"], terminal["terminated"], terminal["truncated"], {}),
            "Success differs from the native reward")
    frames = []
    for index, event in enumerate([events[0], *transitions]):
        name = f"{index:04d}.png"
        require(event["screenshot"] == name, "Unexpected screenshot name")
        with Image.open(checked_file(directory / name, event["sha256"])) as picture:
            require(list(picture.size) == event["size"], "Recorded screenshot dimensions differ")
            frames.append(pixel_hash(picture))
    before_retry = None
    if first_invalid is not None:
        before_retry = (first_invalid, frames[first_invalid],
                        predictions[first_invalid]["metrics"]["provider_output"])
    return {"row": row, "counts": counts, "formats": formats, "screenshots": len(frames),
            "reset": (events[0]["goal"], events[0]["seed"]), "reset_pixels": frames[0],
            "first_invalid": before_retry,
            "peak_cuda_allocated_bytes": max(e["metrics"]["peak_cuda_allocated_bytes"] for e in predictions)}


def audit(directory):
    summary = read_json(directory / "summary.json")
    task_set = summary["task_set"]
    task_ids = TASK_SETS[task_set]
    budget = summary["max_steps"]
    require(summary["study"] == "format" and summary["budget_unit"] == "predictions"
            and 1 <= budget <= 12 and list(task_ids) == summary["task_ids"], "Unexpected study manifest")
    require(set(summary["strategies"]) == set(MODES), "Unexpected format study strategies")
    studies = {mode: [] for mode in MODES}
    by_task = {}
    screenshots = 0
    for task_id in task_ids:
        trio = {}
        for mode in MODES:
            name = f"visualwebarena.{task_id}"
            record = audit_episode(directory / f"{name}-{mode}", budget, mode)
            require(record["row"]["task"] == name and record["row"]["seed"] == record["reset"][1] == 0,
                    "Task or seed mismatch")
            studies[mode].append(record)
            trio[mode] = record
            screenshots += record["screenshots"]
        require(len({record["reset"] for record in trio.values()}) == 1, "Paired task goals or seeds differ")
        invalid = {record["first_invalid"] for record in trio.values()}
        by_task[task_id] = {
            "identical_reset_pixels": len({record["reset_pixels"] for record in trio.values()}) == 1,
            "identical_first_invalid": len(invalid) == 1 and None not in invalid,
            "strategies": {mode: {**record["counts"], "success": record["row"]["success"],
                                  "reason": record["row"]["reason"]}
                           for mode, record in trio.items()}}
    formats = {}
    for mode, records in studies.items():
        counts = {key: sum(record["counts"][key] for record in records) for key in records[0]["counts"]}
        actual = {**aggregate([record["row"] for record in records]), **counts}
        require(summary["strategies"][mode] == actual, f"Summary disagrees with {mode} trajectories")
        formats[mode] = dict(sum((record["formats"] for record in records), Counter()))
    return {"task_set": task_set, "prediction_budget": budget,
            "screenshots": screenshots, "strategies": summary["strategies"], "provider_formats": formats,
            "peak_cuda_allocated_bytes": max(record["peak_cuda_allocated_bytes"]
                                             for records in studies.values() for record in records),
            "by_task": by_task}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.run))
