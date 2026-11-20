"""Reconstruct a live VisualWebArena study from its saved trajectories on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.answer_evidence import evidence_answer, evidence_box, evidence_clipped, native_crop
from gui_recover.qwen_grounding import parse_visible_answer, read_reply
from gui_recover.runner import INVALID_ACTION_FEEDBACK, aggregate, write_json
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import checked_file, read_json, require
from run_visualwebarena_live import STUDY_MODES, TASK_SETS, exact_vwa_success


def audit_evidence(directory, predictions, frame_before_prediction, frames, goal):
    visited = set()
    answers = set()
    stages = Counter()
    pending = None
    for prediction in predictions:
        step = prediction["step"]
        index = frame_before_prediction[step]
        frame = frames[index]
        metrics = prediction["metrics"]
        raw = prediction["raw"]
        reply = metrics["provider_output"]
        if pending is not None and frame == pending[0]:
            stage = "read"
        elif frame not in visited:
            stage = "locate"
        else:
            stage = None
        require(metrics.get("evidence_stage") == stage, "Evidence check schedule disagrees with the observed frame")
        if stage is None:
            if raw is None:
                require(metrics["provider_format"] == "answer_blocked"
                        and read_reply(reply)["action"].upper() == "ANSWER", "Unexpected non-action navigation decision")
            elif raw:
                require(parse_action(raw).action != "ANSWER", "Navigation bypassed the evidence check")
            continue
        require(metrics["evidence_frame"] == frame, "Evidence was attributed to the wrong screenshot")
        stages[stage] += 1
        status = "evidence_invalid"
        if stage == "locate":
            visited.add(frame)
            pending = None
            require(raw is None, "Evidence localization generated a browser action")
            size = metrics["provider_image_size"]
            try:
                box = evidence_box(reply, size)
            except (ValueError, RecursionError):
                pass
            else:
                if box is None:
                    status = "evidence_abstained"
                elif evidence_clipped(goal, box, size):
                    status = "evidence_clipped"
                else:
                    status = "evidence_region"
                    with Image.open(directory / f"{index:04d}.png") as screen:
                        crop, bounds = native_crop(screen, size, box)
                    require(metrics["evidence_bounds"] == list(bounds), "Evidence crop bounds differ from localization")
                    pending = frame, list(bounds), pixel_hash(crop)
        else:
            _, bounds, crop_digest = pending
            pending = None
            require(metrics["evidence_bounds"] == bounds and metrics["evidence_crop_sha256"] == crop_digest,
                    "Evidence-read pixels differ from the recorded screenshot crop")
            candidate = None
            try:
                candidate, status = evidence_answer(reply, goal)
            except (ValueError, RecursionError):
                pass
            if candidate is None:
                require(raw is None, "Unsupported answer reached the browser")
            else:
                action = parse_action(raw)
                require(action.action == "ANSWER" and action.value == candidate, "Answer differs from the crop read")
                answers.add(step)
        require(metrics["provider_format"] == status, "Evidence status differs from the provider reply")
    return stages, answers


def audit_episode(directory, budget, recovery_mode, retry_mode, *, answer_check=False, answer_study=False,
                  evidence_check=False):
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    row = read_json(directory / "episode.json")
    require(events[0]["event"] == "reset" and events[-1] == {"event": "episode_end", **row},
            f"Incomplete trajectory: {directory.name}")
    require(row["reason"] != "runtime_error" and row["recovery"] == recovery_mode and row["fault_kind"] == "none",
            f"Unexpected runtime or intervention: {directory.name}")
    require(row["invalid_action_retry"] == retry_mode, f"Incorrect retry mode: {directory.name}")
    require(row.get("answer_check", False) == answer_check, f"Incorrect answer check mode: {directory.name}")
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
    transition_by_step = {event["step"]: event for event in transitions}
    frame_before_prediction = {}
    checked_frames = set()
    checked_answers = set()
    observation = events[0]
    frame_index = 0
    for prediction in predictions:
        step = prediction["step"]
        metrics = prediction["metrics"]
        provider_format = metrics["provider_format"]
        formats[provider_format] += 1
        require(metrics["model_invocations"] == 1, "Expected one model invocation per prediction")
        digest = observation["sha256"]
        frame_before_prediction[step] = frame_index
        if step in transition_by_step:
            observation = transition_by_step[step]
            frame_index += 1
        checking = prediction.get("decision") == "answer_check"
        require(checking == (answer_check and digest not in checked_frames),
                "Answer check schedule disagrees with the observed frame")
        if checking:
            require(prediction["observation_sha256"] == digest, "Answer check used an unexpected screenshot")
            checked_frames.add(digest)
            try:
                candidate = parse_visible_answer(metrics["provider_output"])
            except (ValueError, RecursionError):
                candidate = None
                expected_format = "answer_invalid"
            else:
                expected_format = "answer_abstained" if candidate is None else "answer_json"
            require(provider_format == expected_format, "Provider answer status differs from its raw reply")
            if candidate is None:
                require(prediction["raw"] is None, "Missing answer generated a browser action")
                continue
            action = parse_action(prediction["raw"])
            require(action.action == "ANSWER" and action.value == candidate, "Answer differs from its raw reply")
            checked_answers.add(step)
        if evidence_check and prediction["raw"] is None:
            continue
        try:
            parse_action(prediction["raw"])
        except ActionError:
            invalid_steps.append(step)
        else:
            valid_steps.append(step)
    require([e["step"] for e in transitions] == valid_steps, "Unparsed action reached the browser")
    require([e["step"] for e in by_event.get("error", [])] == invalid_steps, "Invalid prediction log differs")
    for event in transitions:
        action = parse_action(predictions[event["step"]]["raw"]).to_dict()
        require(json.loads(json.dumps(action)) == event["action"] == event["proposed_action"],
                "Executed action differs from prediction")

    retries = by_event.get("invalid_action_retry", [])
    repaired = by_event.get("invalid_action_repaired", [])
    first_invalid = invalid_steps[0] if invalid_steps else None
    should_retry = retry_mode != "none" and first_invalid is not None and first_invalid + 1 < budget
    require(len(retries) == int(should_retry), "Missing or extra invalid-action retry")
    if retries:
        retry = retries[0]
        next_step = retry["next_prediction"]
        expected_message = INVALID_ACTION_FEEDBACK if retry_mode == "feedback" else None
        require(retry["step"] == first_invalid and next_step == first_invalid + 1 < len(predictions)
                and retry["mode"] == retry_mode and retry["message"] == expected_message, "Invalid retry evidence")
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
    evidence_stages = {}
    if evidence_check:
        evidence_stages, checked_answers = audit_evidence(
            directory, predictions, frame_before_prediction, frames, events[0]["goal"])
    answer_outcome = None
    if answer_study:
        submissions = [event for event in transitions if event["action"]["action"] == "ANSWER"]
        counts.update(answer_checks=len(checked_frames) + evidence_stages.get("locate", 0),
                      answers_submitted=len(submissions), answers_from_checks=len(checked_answers))
        require(all(row[key] == counts[key] for key in ("answer_checks", "answers_submitted", "answers_from_checks")),
                "Answer counters differ from trace")
        if submissions:
            last = submissions[-1]
            source = "navigation"
            if last["step"] in checked_answers:
                source = "evidence-check" if evidence_check else "answer-check"
            answer_outcome = {"source": source, "prediction": last["step"] + 1, "native_reward": last["reward"]}
    before_retry = None
    if first_invalid is not None:
        before_retry = (first_invalid, frames[frame_before_prediction[first_invalid]],
                        predictions[first_invalid]["metrics"]["provider_output"])
    first_unchanged_scroll = None
    first_scroll_followup = None
    if recovery_mode != "none":
        scrolls = {event["step"]: frames[index + 1] for index, event in enumerate(transitions)
                   if event["action"]["action"] == "SCROLL" and frames[index] == frames[index + 1]}
        feedback = by_event.get("recovery_feedback", [])
        scroll_feedback = [event["step"] for event in feedback if event["scroll"]]
        require(len(feedback) == row["recovery_feedbacks"] <= 2
                and all(step in scrolls and step + 1 < len(predictions) for step in scroll_feedback)
                and (recovery_mode == "scroll-feedback" or not scroll_feedback), "Invalid scroll feedback")
        counts.update(scroll_detection_signals=len(scrolls), scroll_feedbacks=len(scroll_feedback))
        require(all(row[key] == counts[key] for key in ("scroll_detection_signals", "scroll_feedbacks")),
                "Scroll counters differ from trace")
        if scrolls:
            step = next(iter(scrolls))
            first_unchanged_scroll = (step, scrolls[step], predictions[step]["metrics"]["provider_output"])
            for index, event in enumerate(transitions):
                if event["step"] == step + 1:
                    first_scroll_followup = {"action": event["action"]["action"],
                                             "viewport_changed": frames[index] != frames[index + 1]}
                    break
    return {"row": row, "counts": counts, "formats": formats, "screenshots": len(frames),
            "reset": (events[0]["goal"], events[0]["seed"]), "reset_pixels": frames[0],
            "first_invalid": before_retry, "first_unchanged_scroll": first_unchanged_scroll,
            "first_scroll_followup": first_scroll_followup, "answer_outcome": answer_outcome,
            "evidence_stages": dict(evidence_stages),
            "peak_cuda_allocated_bytes": max(e["metrics"]["peak_cuda_allocated_bytes"] for e in predictions)}


def audit(directory):
    summary = read_json(directory / "summary.json")
    task_set = summary["task_set"]
    task_ids = TASK_SETS[task_set]
    budget = summary["max_steps"]
    study = summary["study"]
    require(study in ("format", "scroll", "answer", "evidence") and summary["budget_unit"] == "predictions"
            and 1 <= budget <= 12 and list(task_ids) == summary["task_ids"], "Unexpected study manifest")
    modes = STUDY_MODES[study]
    require(set(summary["strategies"]) == set(modes), "Unexpected study strategies")
    studies = {mode: [] for mode in modes}
    by_task = {}
    screenshots = 0
    for task_id in task_ids:
        paired = {}
        for mode in modes:
            name = f"visualwebarena.{task_id}"
            if study in ("answer", "evidence"):
                recovery_mode, retry_mode = "scroll-feedback", "feedback"
            else:
                recovery_mode, retry_mode = ("none", mode) if study == "format" else (mode, "feedback")
            record = audit_episode(directory / f"{name}-{mode}", budget, recovery_mode, retry_mode,
                                   answer_check=mode == "answer-check", answer_study=study in ("answer", "evidence"),
                                   evidence_check=mode == "evidence-check")
            require(record["row"]["task"] == name and record["row"]["seed"] == record["reset"][1] == 0,
                    "Task or seed mismatch")
            studies[mode].append(record)
            paired[mode] = record
            screenshots += record["screenshots"]
        require(len({record["reset"] for record in paired.values()}) == 1, "Paired task goals or seeds differ")
        invalid = {record["first_invalid"] for record in paired.values()}
        scrolls = {record["first_unchanged_scroll"] for record in paired.values()}
        by_task[task_id] = {
            "identical_reset_pixels": len({record["reset_pixels"] for record in paired.values()}) == 1,
            "identical_first_invalid": len(invalid) == 1 and None not in invalid,
            "strategies": {mode: {**record["counts"], "success": record["row"]["success"],
                                  "reason": record["row"]["reason"]}
                           for mode, record in paired.items()}}
        if study == "scroll":
            by_task[task_id]["identical_first_unchanged_scroll"] = len(scrolls) == 1 and None not in scrolls
            by_task[task_id]["first_scroll_followup"] = {
                mode: record["first_scroll_followup"] for mode, record in paired.items()}
        if study in ("answer", "evidence"):
            by_task[task_id]["answer_outcome"] = {mode: record["answer_outcome"] for mode, record in paired.items()}
        if study == "evidence":
            by_task[task_id]["evidence_calls"] = paired["evidence-check"]["evidence_stages"]
    formats = {}
    for mode, records in studies.items():
        counts = {key: sum(record["counts"][key] for record in records) for key in records[0]["counts"]}
        actual = {**aggregate([record["row"] for record in records]), **counts}
        require(summary["strategies"][mode] == actual, f"Summary disagrees with {mode} trajectories")
        formats[mode] = dict(sum((record["formats"] for record in records), Counter()))
    report = {"study": study, "task_set": task_set, "prediction_budget": budget,
              "screenshots": screenshots, "strategies": summary["strategies"], "provider_formats": formats,
              "peak_cuda_allocated_bytes": max(record["peak_cuda_allocated_bytes"]
                                               for records in studies.values() for record in records),
              "by_task": by_task}
    if study == "evidence":
        wrong = {mode: sum(outcome is not None and outcome["native_reward"] != 1.0 for outcome in (
            task["answer_outcome"][mode] for task in by_task.values())) for mode in modes}
        successes = {mode: summary["strategies"][mode]["successes"] for mode in modes}
        report["wrong_submissions"] = wrong
        if task_set == "evidence-transfer":
            report["target_met"] = (len(task_ids) == 8 and budget == 8
                                    and all(task["identical_reset_pixels"] for task in by_task.values())
                                    and successes["evidence-check"] >= 2
                                    and successes["evidence-check"] > successes["answer-check"]
                                    and wrong["evidence-check"] < wrong["answer-check"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.run))
