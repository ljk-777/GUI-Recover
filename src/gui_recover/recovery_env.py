"""Gymnasium interface for learning bounded visual recovery decisions."""

from enum import IntEnum
import json
from string import printable
from time import perf_counter

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from .actions import Action, parse_action
from .recovery import INPUT_FAULTS
from .text_goals import NATIVE_TEXT_TASKS
from .verification import TransitionView
from .verification_env import DEFERRED_TAIL_REGIME, INPUT_REPAIR_PROFILES, TransitionExecution, task_request


TASKS = ("click-button", "click-dialog", "click-tab", *NATIVE_TEXT_TASKS)
REGIMES = ("none", "drop-input", "deferred-input")
SUPPORTED_REGIMES = ("none", "deferred-input", *INPUT_FAULTS)
FRAME_SIZE = (498, 321)
FIXED_CROP = (0., 0., .6, .6)
REWARD_PROFILE = "exact_terminal_v1"


class Decision(IntEnum):
    CONTINUE = 0
    WAIT = 1
    RETRY_PLAIN = 2
    RETRY_FIXED_ZOOM = 3


class ProbeDecision(IntEnum):
    CONTINUE = Decision.CONTINUE
    WAIT = Decision.WAIT
    RETRY_PLAIN = Decision.RETRY_PLAIN
    RETRY_FIXED_ZOOM = Decision.RETRY_FIXED_ZOOM
    PROBE_LAST_CHAR = 4


class Phase(IntEnum):
    SOURCE = 0
    REVIEW = 1
    CLOSED = 2


def exact_success(terminated, truncated, reward, task_info):
    """The supported tasks finish with raw +1 on exact completion."""
    return bool(terminated and not truncated and reward == 1.0
                and task_info.get("DONE_GLOBAL") is True
                and task_info.get("RAW_REWARD_GLOBAL") == 1.0
                and not task_info.get("error"))


def transition_view(observation):
    """Reconstruct the verifier's sealed input using policy-visible data only."""
    if observation["phase"] != Phase.REVIEW:
        raise ValueError("A visual transition is available only in review")
    return TransitionView(Image.fromarray(observation["before"]), Image.fromarray(observation["current"]),
                          parse_action(observation["proposed_action"]), observation["goal"],
                          observation["target"], "after_wait" if observation["waited"] else "immediate")


class RecoveryDecisionEnv(gym.Env):
    """One frozen source action, at most one wait/retry, and a fixed continuation.

    The grounder implements ground(image, target) and repair(view, mode, **options).
    Its outputs enter the existing coordinate-only compiler. The policy receives
    only the returned observation. The info dictionary is for the evaluator.
    """
    metadata = {"render_modes": []}

    def __init__(self, task, grounder, *, regime="none", max_decisions=None, input_repair="legacy",
                 enable_tail_probe=False, batched_probe=False):
        super().__init__()
        known_regime = regime in SUPPORTED_REGIMES or (enable_tail_probe and regime == DEFERRED_TAIL_REGIME)
        if task not in TASKS or not known_regime or (regime != "none" and task not in NATIVE_TEXT_TASKS):
            raise ValueError("Unsupported task or fault for the recovery environment")
        if input_repair not in INPUT_REPAIR_PROFILES or (input_repair == "replace" and task not in NATIVE_TEXT_TASKS):
            raise ValueError("Unsupported input repair profile for the task")
        if enable_tail_probe and task not in NATIVE_TEXT_TASKS:
            raise ValueError("Tail selection requires a native text task")
        if batched_probe and not enable_tail_probe:
            raise ValueError("A batched probe requires the tail-selection action")
        max_budget = 4 if enable_tail_probe else 3
        if max_decisions is None:
            max_decisions = max_budget
        if type(max_decisions) is not int or not 1 <= max_decisions <= max_budget:
            raise ValueError(f"The decision budget must be between 1 and {max_budget}")
        self.task, self.grounder = task, grounder
        self.regime, self.max_decisions = regime, max_decisions
        self.input_repair = input_repair
        self.enable_tail_probe = enable_tail_probe
        self.batched_probe = batched_probe
        self.decision_type = ProbeDecision if enable_tail_probe else Decision
        self.action_space = spaces.Discrete(len(self.decision_type))
        image_shape = (FRAME_SIZE[1], FRAME_SIZE[0], 3)
        self.observation_space = spaces.Dict({
            "before": spaces.Box(0, 255, image_shape, dtype=np.uint8),
            "current": spaces.Box(0, 255, image_shape, dtype=np.uint8),
            "goal": spaces.Text(2048, min_length=1, charset=printable),
            "target": spaces.Text(2048, min_length=1, charset=printable),
            "proposed_action": spaces.Text(8192, min_length=0, charset=printable),
            "phase": spaces.Discrete(len(Phase)),
            "waited": spaces.Discrete(2),
            "remaining_decisions": spaces.Discrete(max_budget + 1),
            "action_mask": spaces.MultiBinary(len(self.decision_type)),
        })
        if enable_tail_probe:
            self.observation_space["selected"] = spaces.Box(0, 255, image_shape, dtype=np.uint8)
            self.observation_space["probed"] = spaces.Discrete(2)
            self.observation_space["probe_restored"] = spaces.Discrete(2)
        self.execution = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        seed = int(self.np_random.integers(0, 2**31)) if seed is None else int(seed)
        options = {"batched_probe": True} if self.batched_probe else {}
        self.execution = TransitionExecution(self.task, seed, None, self.regime, **options)
        try:
            self.target, self.action_kind, self.value = task_request(self.task, self.execution.obs.goal)
            self.phase = Phase.SOURCE
            self.decisions = self.waits = self.repairs = self.actor_requests = 0
            self.selected = None
            self.probe_restored = False
            self.probe_seconds = 0.
            self.terminated = self.truncated = self.success = False
            obs = self._observation()
        except Exception:
            self.close()
            raise
        return obs, {}

    def _mask(self):
        mask = np.zeros(len(self.decision_type), dtype=np.int8)
        if self.phase == Phase.CLOSED:
            return mask
        mask[Decision.CONTINUE] = 1
        if self.phase == Phase.SOURCE:
            return mask
        mask[Decision.WAIT] = not self.waits and self.selected is None
        mask[Decision.RETRY_PLAIN] = mask[Decision.RETRY_FIXED_ZOOM] = 1
        if self.enable_tail_probe:
            mask[ProbeDecision.PROBE_LAST_CHAR] = self.selected is None
        return mask

    def _observation(self):
        x = self.execution
        if x.obs.screenshot.size != FRAME_SIZE or x.before.screenshot.size != FRAME_SIZE:
            raise RuntimeError("Browser screenshot shape differs from the declared observation space")
        obs = {"before": np.array(x.before.screenshot, dtype=np.uint8),
                "current": np.array(x.obs.screenshot, dtype=np.uint8),
                "goal": x.obs.goal, "target": self.target,
                "proposed_action": json.dumps(x.proposed.to_dict()) if x.proposed else "",
                "phase": int(self.phase), "waited": int(bool(self.waits)),
                "remaining_decisions": self.max_decisions-self.decisions, "action_mask": self._mask()}
        if self.enable_tail_probe:
            if self.selected is not None and self.selected.size != FRAME_SIZE:
                raise RuntimeError("Probe screenshot shape differs from the declared observation space")
            obs.update(selected=(np.array(self.selected, dtype=np.uint8) if self.selected is not None
                                 else np.zeros_like(obs["current"])),
                       probed=int(self.selected is not None), probe_restored=int(self.probe_restored))
        return obs

    def _finish(self, reason, *, truncated=False, success=False):
        self.phase = Phase.CLOSED
        self.terminated, self.truncated = not truncated, truncated
        self.success = success
        return reason

    def _browser_outcome(self):
        x = self.execution
        task_info = x.info.get("task_info", {})
        if x.obs.last_action_error or task_info.get("error"):
            return self._finish("browser_error")
        if x.truncated:
            return self._finish("browser_timeout", truncated=True)
        if x.done:
            success = exact_success(x.done, x.truncated, x.reward, task_info)
            return self._finish("success" if success else "browser_failure", success=success)
        return None

    def _ground(self, target):
        self.actor_requests += 1
        return self.grounder.ground(self.execution.obs.screenshot, target)

    def _continue_task(self, predictions):
        if self.task in NATIVE_TEXT_TASKS:
            # Submitting before observing queued input runs against the
            # current field. Only a wait or the repair fixture delivers it.
            p = predictions["continuation"] = self._ground("Submit button")
            if p["point"] is None:
                return self._finish("invalid_continuation_prediction")
            self.execution._step(Action("CLICK", position=tuple(p["point"])), "submit")
            reason = self._browser_outcome()
            if reason:
                return reason
        return self._finish("accepted_incomplete")

    def step(self, action):
        if self.execution is None or self.phase == Phase.CLOSED:
            raise gym.error.ResetNeeded("Reset before taking another action")
        if not self.action_space.contains(action):
            raise ValueError("Action outside the discrete recovery space")
        decision = self.decision_type(int(action))
        predictions = {}
        x = self.execution
        n_events = len(x.events)
        self.decisions += 1
        if not self._mask()[decision]:
            reason = self._finish("masked_action")
        elif self.phase == Phase.SOURCE:
            p = predictions["source"] = self._ground(self.target)
            if p["point"] is None:
                reason = self._finish("invalid_source_prediction")
            else:
                x.proposed = Action(self.action_kind, self.value, tuple(p["point"]))
                x.first()
                reason = self._browser_outcome()
                if reason is None:
                    self.phase = Phase.REVIEW
        elif decision == Decision.WAIT:
            self.waits += 1
            x.wait_once()
            reason = self._browser_outcome()
        elif decision == ProbeDecision.PROBE_LAST_CHAR:
            started = perf_counter()
            self.selected, self.probe_restored = x.probe_text_tail(batched=self.batched_probe)
            self.probe_seconds = perf_counter() - started
            reason = self._browser_outcome()
        elif decision == Decision.CONTINUE:
            reason = self._continue_task(predictions)
        else:
            self.repairs += 1
            self.actor_requests += 1
            view = transition_view(self._observation())
            fixed = decision == Decision.RETRY_FIXED_ZOOM
            mode = "feedback_zoom" if fixed else "plain"
            p = predictions["repair"] = self.grounder.repair(view, mode,
                **({"fixed_crop": FIXED_CROP, "fraction": FIXED_CROP[2]} if fixed else {}))
            if p["point"] is None:
                reason = self._finish("invalid_repair_prediction")
            else:
                x.repair(p["point"], input_profile=self.input_repair)
                reason = self._browser_outcome() or self._continue_task(predictions)
        if reason is None and self.decisions >= self.max_decisions:
            reason = self._finish("decision_budget", truncated=True)
        info = {"outcome": reason, "success": self.success, "browser_events": x.events[n_events:],
                "predictions": predictions, "actor_requests": self.actor_requests,
                "decisions": self.decisions, "repairs": self.repairs, "waits": self.waits}
        if self.enable_tail_probe:
            info.update(probes=int(self.selected is not None), probe_seconds=self.probe_seconds)
        return self._observation(), float(self.success), self.terminated, self.truncated, info

    def close(self):
        if self.execution is not None:
            execution, self.execution = self.execution, None
            execution.close()
