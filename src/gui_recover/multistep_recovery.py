"""Paired, bounded recovery after the first action in native MiniWoB tasks."""

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .actions import Action
from .active_text import inspect_tail_selection
from .multistep_study import fixture as click_fixture, record_frame, task_state
from .runner import is_success, write_json
from .sft import file_sha256
from .verification import TransitionView, pixel_hash
from .verification_capture import capture_after_repaint
from .verification_env import DEFERRED_TAIL_REGIME, TransitionExecution, task_request


BUTTONS = "click-button-sequence"
CHECKBOXES = "click-checkboxes"
FORM = "enter-text"
TASKS = (BUTTONS, CHECKBOXES, FORM)
METHODS = ("continue", "fixed_retry", "visual")
EXTRA_BUDGET = 3
FOCUS_COLOR = (239, 9, 177)
FOCUS_STYLE = "button:focus { outline: 3px solid rgb(239,9,177) !important; }"


def layout_fixture(run):
    """Use privileged geometry only to construct controlled trials and their shared suffix."""
    if run.env.task != FORM:
        return click_fixture(run)
    page = run.env.env.unwrapped.page
    points = page.evaluate("""() => ['tt', 'subbtn'].map(id => {
        const r = document.getElementById(id).getBoundingClientRect();
        const x = r.left + r.width / 2, y = r.top + r.height / 2;
        return document.elementFromPoint(x, y)?.id === id ? [x / innerWidth, y / innerHeight] : null;
    })""")
    if not all(points):
        return None, "form_control_unreachable"
    return {"target": "text field", "correct": points[0], "blank": [0.9, 0.9],
            "wrong_control": None, "continuation": points[1:]}, None


def register(groups, known_hashes=()):
    plan = {"selected": [], "rejected": [], "actor_calls": 0, "outcomes_observed": False}
    used = set(known_hashes)
    for group in groups:
        task, split, count = group["task"], group["split"], group["count"]
        if task not in TASKS or split not in ("train", "validation", "test", "transfer") or count < 1:
            raise ValueError("Unsupported task, split or layout count")
        accepted = 0
        for seed in group["seeds"]:
            if accepted == count:
                break
            run = TransitionExecution(task, seed, None, batched_probe=task == FORM)
            try:
                image = pixel_hash(run.obs.screenshot)
                case = {"task": task, "seed": seed, "split": split, "goal": run.obs.goal,
                        "initial_pixel_hash": image}
                data, reason = layout_fixture(run)
                if data and task == CHECKBOXES and data["wrong_control"] is None:
                    data, reason = None, "no_distractor_checkbox"
                if data and data["correct"] == data.get("wrong_control"):
                    data, reason = None, "overlapping_controls"
                if not data or image in used:
                    plan["rejected"].append({**case, "reason": reason or "duplicate_initial_pixels"})
                    continue
                plan["selected"].append({**case, "fixture": data})
                used.add(image)
                accepted += 1
            finally:
                run.close()
        if accepted != count:
            raise ValueError(f"Insufficient distinct layouts for {split}/{task}")
    validate_plan(plan)
    return plan


def validate_plan(plan):
    scenes = plan["selected"]
    keys = {(r["task"], r["seed"]) for r in scenes}
    hashes = {r["initial_pixel_hash"] for r in scenes}
    if not scenes or len(keys) != len(scenes) or len(hashes) != len(scenes):
        raise ValueError("Duplicate or missing layouts in the frozen plan")
    transfer = {r["task"] for r in scenes if r["split"] == "transfer"}
    if transfer & {r["task"] for r in scenes if r["split"] != "transfer"}:
        raise ValueError("Transfer task families cannot appear in other partitions")
    if plan["actor_calls"] != 0 or plan["outcomes_observed"] is not False:
        raise ValueError("Scene registry must be frozen before acting")


def regimes(task):
    return ("healthy", "miss", "wrong", "natural", "deferred", "trailing", "deferred_tail") if task == FORM else (
        "healthy", "miss", "wrong", "natural")


def proposal(scene, regime, screenshot, actor):
    data, task = scene["fixture"], scene["task"]
    kind, value = task_request(task, scene["goal"])[1:] if task == FORM else ("CLICK", None)
    prediction = actor.ground(screenshot, data["target"]) if regime == "natural" else None
    if prediction is not None:
        point = prediction["point"]
    elif regime == "wrong" and task != FORM:
        point = data["wrong_control"]
    else:
        point = data["correct"]
    return (Action(kind, value, tuple(point)) if point is not None else None), prediction


def first_action(run, scene, regime):
    if regime == "miss" and scene["task"] != FORM:
        # The controller still sees the intended action. Executed coordinates are evaluation evidence.
        run._step(Action("CLICK", position=tuple(scene["fixture"]["blank"])), "injected_blank")
    else:
        run.first()


def correct_stage(run, scene):
    """Privileged label computed for reporting. Controllers never receive it."""
    if scene["task"] == FORM:
        expected = task_request(FORM, run.obs.goal)[2]
        return run.env.env.unwrapped.page.locator("#tt").input_value() == expected
    expected = ["subbtn"] if scene["task"] == BUTTONS else scene["fixture"]["targets"][:1]
    return task_state(run) == expected


def focus_probe(run):
    """Expose native button focus in a temporary screenshot without clicking again."""
    page = run.env.env.unwrapped.page
    original = run.obs.screenshot
    style = page.add_style_tag(content=FOCUS_STYLE)
    try:
        shown = capture_after_repaint(page)
    finally:
        style.evaluate("element => element.remove()")
    restored = capture_after_repaint(page)
    pixels = np.asarray(shown.convert("RGB"))
    prior = np.asarray(original.convert("RGB"))
    marked = np.all(pixels == FOCUS_COLOR, axis=2) & ~np.all(prior == FOCUS_COLOR, axis=2)
    count = int(marked.sum())
    stable = pixel_hash(original) == pixel_hash(restored)
    status = "unknown" if not stable or 0 < count < 20 else "focused" if count else "unfocused"
    return {"status": status, "restored": stable, "marker_pixels": count}, shown, restored


class Recovery:
    """Only screenshots, goal, proposed action and physical operation results enter decisions."""

    def __init__(self, actor, text, checkbox, *, choose_probe=None, repair_actor=None):
        self.actor = actor
        self.text = text
        self.checkbox = checkbox
        self.choose_probe = choose_probe
        self.repair_actor = repair_actor or actor

    def run(self, method, run, target, evidence, save):
        start = len(run.events)
        evidence.update(waits=0, probes=0, repairs=0, relocation_requests=0, decisions=[])
        if method == "continue" or run.done or run.truncated:
            return
        source = run.before
        point = run.proposed.position

        def view(phase="immediate"):
            return TransitionView(source.screenshot, run.obs.screenshot, run.proposed, source.goal, target, phase)

        def relocate(current, *, execute=True):
            evidence["relocation_requests"] += 1
            result = self.repair_actor.repair(current, "feedback_zoom", fixed_crop=(0, 0, .60, .60))
            evidence["relocation"] = result
            if execute and result["point"] is not None:
                repair(result["point"])
            return result["point"]

        def repair(position):
            if len(run.events) - start + evidence["focus_probes"] >= EXTRA_BUDGET:
                raise RuntimeError("Recovery exhausted its fixed browser operation budget")
            evidence["repairs"] += 1
            run.repair(position, input_profile="replace" if target == "text field" else "legacy")
            save(run.obs, f"repair-{evidence['repairs']}")

        if method == "fixed_retry":
            relocate(view())
            return
        if method != "visual":
            raise ValueError(f"Unknown recovery method {method}")
        if target == 'The "ONE" button':
            verdict, shown, restored = focus_probe(run)
            evidence["focus_probes"] += 1
            evidence["probes"] += 1
            save(replace(run.obs, screenshot=shown), "focus")
            save(replace(run.obs, screenshot=restored), "focus-restored")
            evidence["decisions"].append(verdict)
            if verdict["status"] == "unfocused":
                relocate(view())
            return
        if target.startswith('The checkbox labeled "'):
            if self.checkbox is None:
                raise ValueError("Checkbox recovery requires a font")
            current = view()
            for kind in ("undo_wrong_check", "check_target"):
                verdict = self.checkbox.verify(current)
                evidence["decisions"].append(verdict)
                action = verdict.get("repair")
                if action and action["kind"] == kind:
                    repair(action["point"])
                    if run.done or run.truncated:
                        break
                    current = view()
            return
        if target != "text field":
            raise ValueError("Unknown checkpoint target")
        glyph = self.text.verify(view())
        evidence["decisions"].append(glyph)
        if glyph["status"] in ("empty", "unknown"):
            evidence["waits"] += 1
            run.wait_once()
            save(run.obs, "wait")
            if run.done or run.truncated:
                return
            glyph = self.text.verify(view("after_wait"))
            evidence["decisions"].append(glyph)
        if glyph["status"] == "unknown":
            found = relocate(view("after_wait"), execute=False)
            if found is not None:
                point = found
                grounded = replace(view("after_wait"), proposed=replace(run.proposed, position=tuple(point)))
                glyph = self.text.verify(grounded)
                evidence["decisions"].append(glyph)
        if glyph["status"] in ("empty", "different"):
            repair(point)
        elif glyph["status"] == "matched":
            waited = bool(evidence["waits"])
            if self.choose_probe is not None and not self.choose_probe(waited):
                evidence["decisions"].append({"status": "skipped_probe"})
                return
            evidence["probes"] += 1
            selected, restored = run.probe_text_tail(batched=True)
            save(replace(run.obs, screenshot=selected), "selected")
            save(run.obs, "restored")
            verdict = inspect_tail_selection(run.obs.screenshot, selected, glyph["region"]) if restored else {
                "status": "unknown", "reason": "probe_changed_original_frame"}
            evidence["decisions"].append(verdict)
            if verdict["status"] == "blank_tail" and not (run.done or run.truncated):
                repair(point)


def run_trial(scene, regime, method, actor, recovery, out):
    task, data = scene["task"], scene["fixture"]
    mutation = ({"miss": "drop-input", "wrong": "wrong-input", "deferred": "deferred-input", "trailing": "trailing-space-input",
                 "deferred_tail": DEFERRED_TAIL_REGIME}.get(regime, "none") if task == FORM else "none")
    run = TransitionExecution(task, scene["seed"], None, regime=mutation, batched_probe=task == FORM)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    row = {key: scene[key] for key in ("task", "seed", "split", "goal")}
    row.update(regime=regime, method=method, trace="record.json", frames={}, focus_probes=0,
               waits=0, probes=0, repairs=0, relocation_requests=0, decisions=[])

    def save(obs, name):
        if name in row["frames"]:
            raise ValueError("Duplicate browser frame name")
        row["frames"][name] = record_frame(obs, out, name)

    try:
        save(run.obs, "initial")
        if row["frames"]["initial"]["pixel_hash"] != scene["initial_pixel_hash"] or run.obs.goal != scene["goal"]:
            raise RuntimeError("Reset differs from the registered layout")
        run.proposed, prediction = proposal(scene, regime, run.obs.screenshot, actor)
        row["proposed"] = run.proposed.to_dict() if run.proposed is not None else None
        if prediction is not None:
            row["source_prediction"] = {key: prediction[key] for key in ("raw", "point", "cache_key")}
        if run.proposed is None:
            row.update(stage_correct=False, live=False, success=False, outcome="invalid_prediction",
                       extra_operations=0, events=[], reward=0.0, terminated=False, truncated=False)
            return row
        first_action(run, scene, regime)
        save(run.obs, "first")
        row.update(stage_correct=correct_stage(run, scene), live=not (run.done or run.truncated))
        start = len(run.events)
        recovery.run(method, run, data["target"], row, save)
        row["extra_operations"] = len(run.events) - start + row["focus_probes"]
        if row["extra_operations"] > EXTRA_BUDGET:
            raise RuntimeError("Recovery used more than the registered budget")
        row["recovered_stage"] = correct_stage(run, scene)
        if not (run.done or run.truncated):
            run.flush_pending()
        for step, point in enumerate(data["continuation"], 1):
            if run.done or run.truncated:
                break
            run._step(Action("CLICK", position=tuple(point)), "scripted_continuation")
            save(run.obs, f"suffix-{step}")
        success = is_success(run.reward, run.done, run.truncated, run.info.get("task_info", {}))
        errors = any(e["execution_error"] or e["task_info"].get("error") for e in run.events)
        row.update(success=bool(success), outcome="execution_error" if errors else "success" if success else "failed",
                   events=run.events, reward=run.reward, terminated=run.done, truncated=run.truncated)
        return row
    finally:
        write_json(out / "record.json", row)
        run.close()


def paired_rollouts(scenes, actor, recovery, out, *, methods=METHODS, allowed_regimes=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for i, scene in enumerate(scenes):
            conditions = regimes(scene["task"]) if allowed_regimes is None else allowed_regimes
            for j, regime in enumerate(conditions):
                for method in methods if (i + j) % 2 == 0 else methods[::-1]:
                    branch = out / "rollouts" / f"{scene['task']}-{scene['seed']}-{regime}-{method}"
                    row = run_trial(scene, regime, method, actor, recovery, branch)
                    row["trace"] = (branch / "record.json").relative_to(out).as_posix()
                    row["trace_sha256"] = file_sha256(branch / "record.json")
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(scene["split"], branch.name, row["outcome"], flush=True)
    return rows


def summarize(rows, methods=METHODS):
    pairs = defaultdict(dict)
    for row in rows:
        pair = pairs[row["task"], row["seed"], row["regime"]]
        if row["method"] in pair:
            raise ValueError("Repeated rollout in a matched group")
        pair[row["method"]] = row
    for pair in pairs.values():
        if set(pair) != set(methods):
            raise ValueError("Missing method in a matched group")
        prefix = lambda r: (r["frames"]["initial"]["pixel_hash"], r["frames"].get("first", {}).get("pixel_hash"),
                            r["proposed"], r["stage_correct"], r["live"])
        if len({json.dumps(prefix(r)) for r in pair.values()}) != 1:
            raise ValueError("Matched rollouts disagree before recovery")
    results = {}
    for group in ("all", *TASKS):
        result = results[group] = {}
        for method in methods:
            subset = [r for r in rows if r["method"] == method and (group == "all" or r["task"] == group)]
            healthy = [r for r in subset if r["live"] and r["stage_correct"]]
            natural_failed = [r for r in subset if r["regime"] == "natural" and not r["stage_correct"]]
            result[method] = {"episodes": len(subset), "exact_successes": sum(r["success"] for r in subset),
                "healthy_live": len(healthy), "healthy_repairs": sum(r["repairs"] > 0 for r in healthy),
                "healthy_checks": sum(bool(r["extra_operations"]) for r in healthy),
                "natural_failures": len(natural_failed), "natural_recovered": sum(r["success"] for r in natural_failed),
                "nonrecoverable_natural": sum(not r["live"] for r in natural_failed),
                "outcomes": dict(Counter(r["outcome"] for r in subset)),
                **{name: sum(r[name] for r in subset) for name in (
                    "extra_operations", "waits", "probes", "repairs", "relocation_requests")}}
    return results
