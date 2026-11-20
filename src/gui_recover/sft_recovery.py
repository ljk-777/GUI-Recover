"""Explicit teacher/evaluator helpers for actual executed grounding failures.

These functions are NOT imported by visual policy inference.
"""

from collections import defaultdict
import json
from pathlib import Path
import re

from .actions import Action, parse_action
from .grounding import point_hits_boxes
from .sft import file_sha256


def target_description(env, obs, kind="entry"):
    page, task = env.env.unwrapped.page, env.task
    if task == "enter-text":
        value = re.fullmatch(r'Enter "(.*)" into the text field and press Submit\.', obs.goal).group(1)
        if kind == "submit":
            return "Submit button", page.locator("#subbtn"), "CLICK", None
        return "text field", page.locator("#tt"), "INPUT", value
    if task == "click-button":
        text = re.fullmatch(r'Click on the "(.*)" button\.', obs.goal).group(1)
        return f'The "{text}" button', page.get_by_role("button", name=text, exact=True), "CLICK", None
    if task == "click-test":
        return "Click Me! button", page.locator("#subbtn"), "CLICK", None
    if task == "focus-text":
        return "text field", page.locator("#tt"), "CLICK", None
    if task == "click-dialog":
        return 'The "x" close button of the dialog', page.locator("button.ui-dialog-titlebar-close"), "CLICK", None
    if task == "click-tab":
        number = re.fullmatch(r'Click on Tab #(\d)\.', obs.goal).group(1)
        return f"Tab #{number}", page.locator(f'#area a[href="#tabs-{number}"]'), "CLICK", None
    raise ValueError("Unsupported teacher task")


def visible_target_boxes(env, obs, kind="entry"):
    query, locator, action, value = target_description(env, obs, kind)
    boxes = []
    scale = env.env.unwrapped.page._bgym_scale_factor
    for element in locator.all():
        b = element.bounding_box()
        if b is None:
            continue
        box = [max(0, b["x"] * scale / obs.screenshot.width),
               max(0, b["y"] * scale / obs.screenshot.height),
               min(1, (b["x"] + b["width"]) * scale / obs.screenshot.width),
               min(1, (b["y"] + b["height"]) * scale / obs.screenshot.height)]
        if (box[2]-box[0])*obs.screenshot.width >= 4 and (box[3]-box[1])*obs.screenshot.height >= 4:
            boxes.append(box)
    return query, boxes, action, value


def center(box):
    return [round((box[0]+box[2])/2, 3), round((box[1]+box[3])/2, 3)]


def select_online_cases(rows, limit):
    """Round robin by task/recovery mode, using no candidate model results."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["recovery_mode"])].append(row["id"])
    for group in groups.values():
        group.sort()
    selected = []
    while groups and len(selected) < limit:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == limit:
                break
    return selected


def audit_corrections(directory, rows):
    root = Path(directory)
    counts = defaultdict(int)
    for row in rows:
        if row["kind"] != "correction":
            continue
        trace = json.loads((root/row["source_trace"]).read_text())
        action = parse_action(json.dumps(row["previous_action"]))
        if action.to_dict() != parse_action(json.dumps(trace["proposed_action"])).to_dict():
            raise ValueError("Correction action prefix differs from executed source proposal")
        if action.to_dict() != parse_action(json.dumps(trace["executed_action"])).to_dict():
            raise ValueError("Source proposal was not actually executed unchanged")
        if not trace["executed"] or trace["first_action_completed"]:
            raise ValueError("Correction lacks a genuinely executed failed action")
        if point_hits_boxes(action.position, trace["initial_target_boxes"]):
            raise ValueError("Correction source was not a grounding miss")
        for field in ("initial", "after_action", "current"):
            frame = trace[field]
            if file_sha256(root/frame["image"]) != frame["sha256"]:
                raise ValueError("Source trace screenshot checksum differs")
        if row["image_sha256"] != trace["current"]["sha256"]:
            raise ValueError("Correction input differs from source state")
        if row["recovery_mode"] == "restart_after_terminal":
            if not (trace["terminated"] or trace["truncated"]) or trace["initial"]["sha256"] != trace["current"]["sha256"]:
                raise ValueError("Restart must reproduce the same initial scene")
        elif row["recovery_mode"] == "retry_in_place":
            if trace["terminated"] or trace["truncated"] or trace["after_action"] != trace["current"]:
                raise ValueError("In-place correction must use the actual nonterminal state")
        else:
            raise ValueError("Unknown recovery mode")
        if row.get("source_trace_sha256") and file_sha256(root/row["source_trace"]) != row["source_trace_sha256"]:
            raise ValueError("Source trace checksum differs")
        counts[row["recovery_mode"]] += 1
    return dict(counts)
