"""Explicitly scripted task decomposition; no DOM/target coordinates are used."""

import ast
from dataclasses import dataclass
import json
import re

from .actions import Action, ActionError, parse_action
from .prompts import GROUNDING
from .input_guard import inspect_input


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

    def __init__(self, backend, input_guard=False, blind_input_retry=False, input_guard_version="v2"):
        if input_guard and blind_input_retry:
            raise ValueError("Choose visual guard or blind retry, not both")
        self.backend = backend
        self.input_guard = input_guard
        if input_guard_version not in ("v1", "v2", "v3"):
            raise ValueError("Unknown input guard version")
        self.input_guard_version = input_guard_version
        self.blind_input_retry = blind_input_retry
        self.last_metrics = {}
        self.config = {**backend.config, "planner": "scripted_miniwob_goal_patterns_v1",
                       "grounding_prompt": GROUNDING, "uses_dom": False,
                       "advances_plan": "history order by default; visual input check when input_guard enabled",
                       "input_guard": {"enabled": input_guard, "retry_limit": 1,
                                       "blind_retry": blind_input_retry,
                                       "signal": ("enclosed_uniform_input_contrast_v3" if input_guard_version == "v3"
                                                  else f"enclosed_white_input_ink_{input_guard_version}")}}
        self._clear_state()

    def _clear_state(self):
        self.last_metrics = {}
        self.step_index = 0
        self.input_attempts = 0
        self.block_reason = None
        self.guard_metrics = {"checks": 0, "empty": 0, "text_visible": 0, "unknown": 0,
                              "retries": 0, "blocked": 0, "blind_retries": 0}

    def reset(self, seed):
        self._clear_state()
        self.backend.reset(seed)

    def observe_transition(self, before, after, proposed):
        if not self.input_guard and not self.blind_input_retry:
            return None
        if proposed.action != "INPUT":
            self.step_index += 1
            return None
        if self.blind_input_retry:
            self.input_attempts += 1
            if self.input_attempts == 1:
                self.guard_metrics["blind_retries"] += 1
                decision = "unconditional_retry"
            else:
                self.step_index += 1
                decision = "advance"
            return {"kind": "blind_input_retry", "decision": decision, "input_attempts": self.input_attempts}
        signal = inspect_input(before.screenshot, after.screenshot, proposed.position,
                               version=self.input_guard_version)
        status = signal["status"]
        self.input_attempts += 1
        self.guard_metrics["checks"] += 1
        self.guard_metrics[status] += 1
        if status == "text_visible":
            self.step_index += 1
            decision = "advance"
        elif status == "empty" and self.input_attempts < 2:
            self.guard_metrics["retries"] += 1
            decision = "reground_and_retry"
        else:
            self.block_reason = "Input completion could not be verified; submission blocked"
            self.guard_metrics["blocked"] += 1
            decision = "block_submission"
        return {"kind": "input_guard", "signal": signal, "decision": decision,
                "input_attempts": self.input_attempts}

    def predict(self, obs, history):
        plan = plan_from_goal(obs.goal)
        index = self.step_index if self.input_guard or self.blind_input_retry else len(history)
        if self.block_reason or index >= len(plan):
            peaks = {key: value for key, value in self.last_metrics.items() if key.startswith("peak_cuda_")}
            self.last_metrics = {**peaks, "model_invocations": 0, "input_tokens": 0, "generated_tokens": 0,
                                 "grounding_raw": None, "subgoal": None,
                                 "scripted_plan_exhausted": True}
            return json.dumps(Action("ANSWER", self.block_reason or "Scripted plan exhausted").to_dict())
        step = plan[index]
        raw = self.backend.ground(obs.screenshot, step.query)
        self.last_metrics = {**self.backend.last_metrics, "grounding_raw": raw,
                             "model_invocations": 1,
                             "subgoal": step.query, "scripted_step": index}
        try:
            point = parse_point(raw)
        except ActionError:
            # Preserve raw grounding evidence in metrics; runner counts invalid_action.
            return json.dumps({"invalid_grounding_output": raw})
        return json.dumps(Action(step.action, step.value, point).to_dict())
