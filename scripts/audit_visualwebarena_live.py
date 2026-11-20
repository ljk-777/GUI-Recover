"""Reconstruct a live VisualWebArena study from its saved trajectories on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.answer_evidence import (
    answer_signature, confirmation_image, evidence_answer, native_crop, parse_evidence_location,
)
from gui_recover.qwen_grounding import native_action_after_note, parse_visible_answer, read_reply
from gui_recover.runner import INVALID_ACTION_FEEDBACK, aggregate, write_json
from gui_recover.screen_text import exact_initial_search, listing_detail
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import checked_file, read_json, require
from run_visualwebarena_live import ANSWER_STUDIES, EVIDENCE_STUDIES, STUDY_MODES, TASK_SETS, exact_vwa_success


def audit_evidence(directory, predictions, frame_before_prediction, frames, goal, *,
                   confirm=False, click_origins=None, compact_navigation=False, use_screen_text=False,
                   guarded_ocr=False, direct_navigation=False):
    visited = set()
    answers = set()
    stages = Counter()
    pending = None
    pending_confirmation = None
    for prediction in predictions:
        step = prediction["step"]
        index = frame_before_prediction[step]
        frame = frames[index]
        metrics = prediction["metrics"]
        raw = prediction["raw"]
        reply = metrics["provider_output"]
        if pending_confirmation is not None and frame == pending_confirmation[0]:
            stage = "confirm"
        elif pending is not None and frame == pending[0]:
            stage = "read"
        elif frame not in visited:
            stage = "locate"
        else:
            stage = None
        require(metrics.get("evidence_stage") == stage, "Evidence check schedule disagrees with the observed frame")
        if stage is None:
            if raw is None:
                try:
                    blocked = read_reply(reply)
                except json.JSONDecodeError:
                    blocked = native_action_after_note(reply)
                require(metrics["provider_format"] == "answer_blocked" and blocked["action"].upper() == "ANSWER",
                        "Unexpected non-action navigation decision")
            elif raw:
                require(parse_action(raw).action != "ANSWER", "Navigation bypassed the evidence check")
            continue
        require(metrics["evidence_frame"] == frame, "Evidence was attributed to the wrong screenshot")
        stages[stage] += 1
        status = "evidence_invalid"
        if stage == "locate":
            visited.add(frame)
            pending = None
            size = metrics["provider_image_size"]
            require(("screen_text" in metrics) == use_screen_text, "Unexpected OCR mode")
            rows = metrics.get("screen_text")
            if rows is not None:
                require(isinstance(rows, list) and len(rows) <= 56, "Unexpected OCR row count")
                for line in rows:
                    coords = line["bounds"]
                    require(isinstance(line["text"], str) and 0 < len(line["text"]) <= 100
                            and len(coords) == 4 and all(type(value) is int for value in coords)
                            and 0 <= coords[0] < coords[2] <= size[0] and 0 <= coords[1] < coords[3] <= size[1],
                            "Invalid OCR bounds")
            shortcut = None
            if direct_navigation and rows is not None and step == 0:
                try:
                    shortcut = exact_initial_search(goal, rows, size)
                except ActionError:
                    pass
            if shortcut is not None:
                require(metrics["provider_format"] == "evidence_exact_search" and reply is None
                        and parse_action(raw) == shortcut, "Search action differs from the visible field and task text")
                stages["grounded_search"] += 1
                continue
            if direct_navigation and rows is not None and listing_detail(rows):
                rows = None
            try:
                navigation, box, candidate, status = parse_evidence_location(
                    reply, goal, size, confirm=confirm, compact_navigation=compact_navigation, rows=rows,
                    guarded_ocr=guarded_ocr)
            except (ValueError, RecursionError):
                require(raw is None, "Unusable evidence localization generated a browser action")
            else:
                require((raw is None) == (navigation is None), "Evidence localization disagrees with the browser action")
                if navigation is not None:
                    require(parse_action(raw) == navigation, "Evidence search executed an unexpected action")
                elif box is not None:
                    with Image.open(directory / f"{index:04d}.png") as screen:
                        crop, bounds = native_crop(screen, size, box)
                    require(metrics["evidence_bounds"] == list(bounds), "Evidence crop bounds differ from localization")
                    if candidate is not None:
                        pending_confirmation = frame, list(bounds), candidate
                    else:
                        pending = frame, list(bounds), pixel_hash(crop)
        elif stage == "read":
            _, bounds, crop_digest = pending
            pending = None
            require(metrics["evidence_bounds"] == bounds and metrics["evidence_crop_sha256"] == crop_digest,
                    "Evidence-read pixels differ from the recorded screenshot crop")
            candidate = None
            try:
                candidate, status = evidence_answer(reply, goal, allow_string_abstention=confirm)
            except (ValueError, RecursionError):
                pass
            if candidate is None:
                require(raw is None, "Unsupported answer reached the browser")
            elif confirm:
                require(raw is None, "Candidate answer bypassed the independent check")
                pending_confirmation = frame, bounds, candidate
            else:
                action = parse_action(raw)
                require(action.action == "ANSWER" and action.value == candidate, "Answer differs from the crop read")
                answers.add(step)
        else:
            _, bounds, candidate = pending_confirmation
            pending_confirmation = None
            context = None
            origin = click_origins.get(index) if click_origins is not None else None
            if origin is not None:
                origin_index, position = origin
                with Image.open(directory / f"{origin_index:04d}.png") as previous:
                    context = previous.copy(), position
                require(metrics["evidence_context_sha256"] == frames[origin_index]
                        and metrics["evidence_context_position"] == position, "Context differs from the recorded click")
                stages["navigation_context"] += 1
            else:
                require("evidence_context_sha256" not in metrics, "Untracked navigation context reached confirmation")
            with Image.open(directory / f"{index:04d}.png") as screen:
                panel = confirmation_image(screen, screen.crop(bounds), bounds, context)
            require(metrics["evidence_bounds"] == bounds and metrics["evidence_panel_sha256"] == pixel_hash(panel),
                    "Confirmation pixels differ from the recorded viewport and crop")
            try:
                independent = parse_visible_answer(reply, allow_numeric=True)
            except (ValueError, RecursionError):
                pass
            else:
                if independent is None:
                    status = "evidence_verify_abstained"
                elif answer_signature(candidate) != answer_signature(independent):
                    status = "evidence_verify_disagreed"
                else:
                    status = "evidence_verify_agreed"
            if status == "evidence_verify_agreed":
                action = parse_action(raw)
                require(action.action == "ANSWER" and action.value == candidate, "Submission differs from agreed answer")
                answers.add(step)
            else:
                require(raw is None, "Answer reached the browser without agreement")
        require(metrics["provider_format"] == status, "Evidence status differs from the provider reply")
    return stages, answers


def audit_episode(directory, budget, recovery_mode, retry_mode, *, answer_check=False, answer_study=False,
                  evidence_check=False, confirm=False, use_navigation_context=False, compact_navigation=False,
                  use_screen_text=False, guarded_ocr=False, direct_navigation=False):
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
        is_shortcut = direct_navigation and use_screen_text and provider_format == "evidence_exact_search" and step == 0
        require(metrics["model_invocations"] == (0 if is_shortcut else 1),
                "Unexpected model invocation count for the decision")
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
        click_origins = None
        if use_navigation_context:
            click_origins = {}
            origin = None
            for index, event in enumerate(transitions):
                if event["action"]["action"] == "CLICK" and frames[index] != frames[index + 1]:
                    origin = index, event["action"]["position"]
                click_origins[index + 1] = origin
        evidence_stages, checked_answers = audit_evidence(
            directory, predictions, frame_before_prediction, frames, events[0]["goal"],
            confirm=confirm, click_origins=click_origins, compact_navigation=compact_navigation,
            use_screen_text=use_screen_text, guarded_ocr=guarded_ocr, direct_navigation=direct_navigation)
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
                source = "evidence-confirm" if confirm else ("evidence-check" if evidence_check else "answer-check")
                if use_navigation_context:
                    source = "evidence-ocr" if use_screen_text else "evidence-context"
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
    compact_navigation = summary["model"].get("compact_navigation_replies", task_set == "evidence-navigation-diagnostic")
    require(study in ("format", "scroll", *ANSWER_STUDIES) and summary["budget_unit"] == "predictions"
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
            if study in ANSWER_STUDIES:
                recovery_mode, retry_mode = "scroll-feedback", "feedback"
            else:
                recovery_mode, retry_mode = ("none", mode) if study == "format" else (mode, "feedback")
            record = audit_episode(directory / f"{name}-{mode}", budget, recovery_mode, retry_mode,
                                   answer_check=mode == "answer-check", answer_study=study in ANSWER_STUDIES,
                                   evidence_check=study in EVIDENCE_STUDIES and mode != "answer-check",
                                   confirm=mode in ("evidence-confirm", "evidence-context", "evidence-ocr"),
                                   use_navigation_context=mode in ("evidence-context", "evidence-ocr"),
                                   compact_navigation=compact_navigation, use_screen_text=mode == "evidence-ocr",
                                   guarded_ocr=summary.get("screen_reader", {}).get("guarded_ocr", False),
                                   direct_navigation=summary.get("screen_reader", {}).get("direct_navigation", False))
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
        if study in ANSWER_STUDIES:
            by_task[task_id]["answer_outcome"] = {mode: record["answer_outcome"] for mode, record in paired.items()}
        if study in EVIDENCE_STUDIES:
            by_task[task_id]["evidence_calls"] = paired[modes[-1]]["evidence_stages"]
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
    if study in EVIDENCE_STUDIES:
        wrong = {mode: sum(outcome is not None and outcome["native_reward"] != 1.0 for outcome in (
            task["answer_outcome"][mode] for task in by_task.values())) for mode in modes}
        successes = {mode: summary["strategies"][mode]["successes"] for mode in modes}
        report["wrong_submissions"] = wrong
        if task_set in ("evidence-transfer", "evidence-confirm-transfer", "evidence-context-transfer"):
            candidate = modes[-1]
            report["target_met"] = (len(task_ids) == 8 and budget == 8
                                    and all(task["identical_reset_pixels"] for task in by_task.values())
                                    and successes[candidate] >= 2
                                    and successes[candidate] > successes["answer-check"]
                                    and wrong[candidate] < wrong["answer-check"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.run))
