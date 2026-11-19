"""A held-out first-click diagnostic on two actual MiniWoB multistep tasks."""

from collections import Counter
import json
from pathlib import Path

from .actions import Action
from .repair_study_data import png_sha
from .runner import is_success, write_json
from .verification import TransitionView, appearance_verdict, pixel_hash
from .verification_env import TransitionExecution
from .verification_training import routed_decision
from .sft import file_sha256


TASKS = ("click-button-sequence", "click-checkboxes")


def fixture(execution):
    """Read the goal and setup geometry for the controlled actions only."""
    page = execution.env.env.unwrapped.page
    raw = page.evaluate("""() => {
        const point = el => {
            const r = el.getBoundingClientRect();
            for (const y of [0.5, 0.2, 0.8]) {
                for (const x of [0.5, 0.2, 0.8]) {
                    const p = [r.left + r.width*x, r.top + r.height*y];
                    if (el.contains(document.elementFromPoint(...p))) return p;
                }
            }
            return null;
        };
        const boxes = [...document.querySelectorAll('#boxes label')];
        const item = el => ({name: el.textContent.trim(), point: point(el)});
        if (boxes.length) return {controls: boxes.map(el => ({
            name: el.textContent.trim(), point: point(el.querySelector('input'))
        })), submit: point(document.getElementById('subbtn')), width: innerWidth, height: innerHeight};
        return {controls: [...document.querySelectorAll('#area button')].map(item),
                width: innerWidth, height: innerHeight};
    }""")
    controls = raw["controls"]
    by_name = {r["name"]: r["point"] for r in controls}
    if len(by_name) != len(controls):
        return None, "duplicate_visible_labels"
    if execution.env.task == "click-button-sequence":
        targets = ["ONE", "TWO"]
        wrong = "TWO"
        target = 'The "ONE" button'
    else:
        goal = execution.obs.goal
        prefix, suffix = "Select ", " and click Submit."
        if not (goal.startswith(prefix) and goal.endswith(suffix)):
            raise ValueError("Unexpected checkbox goal format")
        requested = goal[len(prefix):-len(suffix)]
        if requested == "nothing":
            return None, "goal_requires_no_checkbox"
        targets = requested.split(", ")
        wrong = next((c["name"] for c in controls if c["name"] not in targets and c["point"]), None)
        target = f'The checkbox labeled "{targets[0]}"'
        if raw["submit"] is None:
            return None, "submit_unreachable"
    if any(by_name.get(name) is None for name in targets):
        return None, "goal_control_unreachable"
    def normalized(p):
        return None if p is None else (p[0]/raw["width"], p[1]/raw["height"])
    return {"target": target, "targets": targets, "correct": normalized(by_name[targets[0]]),
            "blank": (0.9, 0.9), "wrong_control": normalized(by_name[wrong]) if wrong else None,
            "wrong_state": ["subbtn2" if execution.env.task == "click-button-sequence" else wrong] if wrong else None,
            "continuation": [normalized(by_name[n]) for n in targets[1:]]
                            + ([normalized(raw["submit"])] if "submit" in raw else [])}, None


def task_state(execution):
    page = execution.env.env.unwrapped.page
    if execution.env.task == "click-button-sequence":
        return page.evaluate("() => [...buttonsPushed]")
    return page.evaluate("""() => [...document.querySelectorAll('#boxes label')]
        .filter(el => el.querySelector('input').checked).map(el => el.textContent.trim())""")


def record_frame(obs, out, name):
    path = out/f"{name}.png"
    obs.screenshot.save(path)
    return {"image": path.name, "sha256": file_sha256(path), "pixel_hash": pixel_hash(obs.screenshot)}


def register_multistep(config, *, known_layouts=()):
    plan = {"accepted": [], "rejected": [], "actor_calls": 0, "outcomes_observed": False}
    seen = set(known_layouts)
    for group in config["multistep_tasks"]:
        accepted = 0
        for seed in group["candidate_seeds"]:
            if accepted == group["count"]:
                break
            execution = TransitionExecution(group["task"], seed, None)
            try:
                layout = png_sha(execution.obs.screenshot)
                inputs, reason = fixture(execution)
                row = {"task": group["task"], "seed": seed, "initial_sha256": layout, "goal": execution.obs.goal}
                if layout in seen or not inputs:
                    plan["rejected"].append({**row, "reason": reason or "duplicate_initial_image"})
                else:
                    plan["accepted"].append({**row, "fixture": inputs})
                    seen.add(layout)
                    accepted += 1
            finally:
                execution.close()
        if accepted != group["count"]:
            raise ValueError("Predeclared multistep layout reserve exhausted")
    return plan


def first_click(scene, role, grounder, verifiers, out, *, recovery=None):
    execution = TransitionExecution(scene["task"], scene["seed"], None)
    inputs = scene["fixture"]
    out.mkdir(parents=True, exist_ok=False)
    evidence = {"task": scene["task"], "seed": scene["seed"], "role": role,
                "target": inputs["target"], "goal": scene["goal"], "decisions": {}}
    try:
        before = execution.obs
        evidence["before"] = record_frame(before, out, "before")
        if evidence["before"]["sha256"] != scene["initial_sha256"] or before.goal != scene["goal"]:
            raise RuntimeError("Multistep reset did not replay")
        evidence["before_state"] = task_state(execution)
        if evidence["before_state"]:
            raise RuntimeError("Unexpected initial checkbox or sequence state")
        if role == "natural":
            prediction = grounder.ground(before.screenshot, inputs["target"])
            evidence["source_prediction"] = prediction
            point = prediction["point"]
        else:
            point = inputs[role]
        if point is None:
            evidence["status"] = "invalid_prediction"
            return evidence
        execution.proposed = Action("CLICK", position=tuple(point))
        execution.first()
        if execution.obs.last_action_error or execution.info.get("task_info", {}).get("error"):
            raise RuntimeError(f"First click execution failed: {execution.events[-1]}")
        immediate = execution.obs
        state = task_state(execution)
        expected = ["subbtn"] if scene["task"] == "click-button-sequence" else inputs["targets"][:1]
        live = not (execution.done or execution.truncated)
        achieved = state == expected
        evidence.update(status="ok", proposed=execution.proposed.to_dict(), achieved=achieved, live=live,
                        expected_state=expected, immediate_state=state,
                        immediate=record_frame(immediate, out, "immediate"))
        controlled = {"correct": expected, "blank": [], "wrong_control": inputs["wrong_state"]}
        if role in controlled and (not live or state != controlled[role]):
            raise RuntimeError(f"Controlled {role} action disagrees with the first-step goal")
        if live:
            execution.wait_once()
            evidence["after_wait"] = record_frame(execution.obs, out, "after-wait")
            evidence["after_wait_state"] = task_state(execution)
            if execution.done or execution.truncated or evidence["after_wait_state"] != state:
                raise RuntimeError("The shared hover wait changed the multistep task outcome")
            views = {
                "immediate": TransitionView(before.screenshot, immediate.screenshot, execution.proposed,
                                            before.goal, inputs["target"], "immediate"),
                "after_wait": TransitionView(before.screenshot, execution.obs.screenshot, execution.proposed,
                                             before.goal, inputs["target"], "after_wait"),
            }
            for backend, verify in verifiers.items():
                verdict = {("click", phase): verify(view) for phase, view in views.items()}
                decision = routed_decision({"id": "click", "after_wait": True, "deferred": False}, verdict)
                evidence["decisions"][backend] = {**decision, **{p: verdict["click", p] for p in views}}
        continue_task = role == "correct"
        if recovery is not None:
            if live:
                decision = evidence["decisions"]["target_state"]
                phase = "after_wait" if decision["wait"] else "immediate"

                def execute_repair(action):
                    execution._step(action, "target_state_recovery")
                    n = sum(e["event"] == "target_state_recovery" for e in execution.events)
                    evidence[f"repair_{n}"] = record_frame(execution.obs, out, f"repair-{n}")
                    if execution.obs.last_action_error or execution.info.get("task_info", {}).get("error"):
                        raise RuntimeError(f"Repair execution failed: {execution.events[-1]}")
                    if not (execution.done or execution.truncated):
                        return execution.obs.screenshot

                evidence["recovery"] = recovery.recover(views[phase], execute_repair) if decision["trigger"] else {
                    "actions": [], "final": decision[phase]}
            evidence["state_after_recovery"] = task_state(execution)
            evidence["achieved_after_recovery"] = evidence["state_after_recovery"] == expected
            continue_task = live and evidence["achieved_after_recovery"] and not (execution.done or execution.truncated)
        if continue_task:
            for n, next_point in enumerate(inputs["continuation"], 1):
                execution._step(Action("CLICK", position=tuple(next_point)), "oracle_continuation")
                evidence[f"continuation_{n}"] = record_frame(execution.obs, out, f"continuation-{n}")
            evidence["completed_with_oracle_continuation"] = is_success(execution.reward, execution.done,
                execution.truncated, execution.info.get("task_info", {}))
            end = task_state(execution)
            expected_end = {"subbtn", "subbtn2"} if scene["task"] == "click-button-sequence" else set(inputs["targets"])
            evidence["continuation_state"] = end
            if not evidence["completed_with_oracle_continuation"] or set(end) != expected_end:
                raise RuntimeError("Healthy control did not complete the full task with its continuation")
        return evidence
    finally:
        evidence["events"] = execution.events
        write_json(out/"record.json", evidence)
        execution.close()


def summarize_multistep(rows, backends):
    result = {}
    for task in (*TASKS, "all"):
        subset = [r for r in rows if task == "all" or task == r["task"]]
        counts = Counter(r["role"] for r in subset)
        live = [r for r in subset if r.get("live")]
        groups = {
            "healthy_control": [r for r in live if r["role"] == "correct"],
            "blank_control": [r for r in live if r["role"] == "blank"],
            "wrong_control": [r for r in live if r["role"] == "wrong_control"],
            "healthy_natural": [r for r in live if r["role"] == "natural" and r["achieved"]],
            "failed_natural": [r for r in live if r["role"] == "natural" and not r["achieved"]],
        }
        result[task] = {"counts": dict(counts), "terminal": sum(r.get("live") is False for r in subset),
                        "invalid": sum(r["status"] != "ok" for r in subset),
                        "same_pixels_correct": sum(r["before"]["pixel_hash"] == r["immediate"]["pixel_hash"]
                                                   for r in groups["healthy_control"]),
                        "groups": {name: {"count": len(group), "detectors": {b: {
                            "triggers": sum(r["decisions"][b]["trigger"] for r in group),
                            "waits": sum(r["decisions"][b]["wait"] for r in group),
                            "unresolved": sum(
                                r["decisions"][b]["after_wait" if r["decisions"][b]["wait"] else "immediate"]["status"] == "unknown"
                                for r in group)
                        } for b in backends}} for name, group in groups.items()}}
    return result


def run_multistep(plan, grounder, learned, out, *, recovery=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    verifiers = {name: model.verify for name, model in learned.items()}
    verifiers["appearance"] = appearance_verdict
    if recovery is not None:
        verifiers["target_state"] = recovery.verify
    rows = []
    for scene in plan["accepted"]:
        for role in ("correct", "blank", "wrong_control", "natural"):
            if role == "wrong_control" and scene["fixture"][role] is None:
                continue
            directory = out/f"{scene['task']}-{scene['seed']}-{role}"
            row = first_click(scene, role, grounder, verifiers, directory, recovery=recovery)
            rows.append(row)
            print("multistep", scene["task"], scene["seed"], role, row.get("achieved"), flush=True)
    summary = summarize_multistep(rows, tuple(verifiers))
    write_json(out/"summary.json", summary)
    return summary
