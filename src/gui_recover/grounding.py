"""Explicitly scripted task decomposition; no DOM/target coordinates are used."""

import ast
from dataclasses import dataclass
import json
import re

from .actions import Action, ActionError, parse_action
from .prompts import GROUNDING


@dataclass(frozen=True)
class PlannedStep:
    action: str
    query: str
    value: str | None = None


def plan_from_goal(goal):
    """Narrow MiniWoB language grammar, NOT an autonomous learned planner."""
    if goal == "Click the button.":
        return [PlannedStep("CLICK", "Click Me! button")]
    match = re.fullmatch(r'Click on the "([^"\n]+)" button\.', goal)
    if match:
        return [PlannedStep("CLICK", f'The "{match[1]}" button')]
    match = re.fullmatch(r'Enter "([^"\n]+)" into the text field and press Submit\.', goal)
    if match:
        return [PlannedStep("INPUT", "text field", match[1]),
                PlannedStep("CLICK", "Submit button")]
    raise ValueError("Unsupported goal for scripted planner")


def parse_point(raw):
    text = raw.strip()
    if len(text) > 8192:
        raise ActionError("Grounding output too long")
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        point = ast.literal_eval(text)
        # Reuse the action boundary's checks for type, length, finiteness and range.
        return parse_action(json.dumps({"action": "CLICK", "value": None, "position": point})).position
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ActionError("Expected exactly one normalized point [x,y]") from exc


def point_hits_boxes(point, boxes):
    return bool(point is not None and any(x0 <= point[0] < x1 and y0 <= point[1] < y1
                                         for x0, y0, x1, y1 in boxes))


class ScriptedGroundingPolicy:
    """Text-derived two-step plan; ShowUI supplies every coordinate."""

    def __init__(self, backend):
        self.backend = backend
        self.last_metrics = {}
        self.config = {**backend.config, "planner": "scripted_miniwob_goal_patterns_v1",
                       "grounding_prompt": GROUNDING, "uses_dom": False,
                       "advances_plan": "after each executed logical action; no visual completion verifier"}

    def reset(self, seed):
        self.last_metrics = {}
        self.backend.reset(seed)

    def predict(self, obs, history):
        plan = plan_from_goal(obs.goal)
        if len(history) >= len(plan):
            peaks = {key: value for key, value in self.last_metrics.items() if key.startswith("peak_cuda_")}
            self.last_metrics = {**peaks, "model_invocations": 0, "input_tokens": 0, "generated_tokens": 0,
                                 "grounding_raw": None, "subgoal": None,
                                 "scripted_plan_exhausted": True}
            return json.dumps(Action("ANSWER", "Scripted plan exhausted").to_dict())
        step = plan[len(history)]
        raw = self.backend.ground(obs.screenshot, step.query)
        self.last_metrics = {**self.backend.last_metrics, "grounding_raw": raw,
                             "model_invocations": 1,
                             "subgoal": step.query, "scripted_step": len(history)}
        try:
            point = parse_point(raw)
        except ActionError:
            # Preserve raw grounding evidence in metrics; runner counts invalid_action.
            return json.dumps({"invalid_grounding_output": raw})
        return json.dumps(Action(step.action, step.value, point).to_dict())
