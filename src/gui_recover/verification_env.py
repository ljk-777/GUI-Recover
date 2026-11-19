"""Controlled execution fixtures and teacher-only labels, outside visual inference."""

import re
from copy import deepcopy

from .actions import Action
from .environment import MiniwobEnvironment
from .recovery import ActionFault
from .runner import is_success


def task_request(task, goal):
    if task == "enter-text":
        value = re.fullmatch(r'Enter "(.*)" into the text field and press Submit\.',goal).group(1)
        return "text field", "INPUT", value
    if task == "click-button":
        value = re.fullmatch(r'Click on the "(.*)" button\.',goal).group(1)
        return f'The "{value}" button', "CLICK", None
    if task == "focus-text":
        return "text field", "CLICK", None
    if task == "click-test":
        return "Click Me! button", "CLICK", None
    if task == "click-dialog":
        return 'The "x" close button of the dialog', "CLICK", None
    if task == "click-tab":
        number = re.fullmatch(r'Click on Tab #(\d)\.',goal).group(1)
        return f"Tab #{number}", "CLICK", None
    raise ValueError("Unsupported fixed-controller task")


def teacher_effect(execution):
    """Offline labels / explicitly privileged oracle baseline only."""
    return teacher_evidence(execution)["effect_achieved"]


def teacher_evidence(execution):
    """Raw offline evidence retained for label auditing, never for a visual model."""
    env = execution.env
    if env.task == "enter-text":
        _, _, expected = task_request(env.task, execution.obs.goal)
        observed=env.env.unwrapped.page.locator("#tt").input_value()
        return {"kind":"exact_input_value","expected":expected,"observed":observed,"effect_achieved":observed==expected}
    if env.task == "focus-text":
        focused=env.env.unwrapped.page.locator("#tt").evaluate("el => el === document.activeElement")
        return {"kind":"input_focus","effect_achieved":bool(focused)}
    return {"kind":"environment_success","effect_achieved":is_success(execution.reward,execution.done,
        execution.truncated,execution.info.get("task_info",{}))}


class TransitionExecution:
    """Fixture knows interventions; the verifier receives only a TransitionView.

    deferred-input is a queued-delivery simulator, not a claim of real network
    latency. Pending delivery is flushed after a controller repair, or on wait,
    before normal continuation. A premature retry can therefore duplicate input.
    """
    def __init__(self, task, seed, proposed, regime="none"):
        self.env = MiniwobEnvironment(task)
        self.seed, self.proposed, self.regime = seed, proposed, regime
        self.pending = None
        self.events = []
        self.obs, self.info = self.env.reset(seed)
        self.before = self.obs
        self.reward, self.done, self.truncated = 0.0, False, False

    def first(self):
        if self.regime == "deferred-input":
            if self.proposed.action != "INPUT":
                raise ValueError("Deferred fixture supports INPUT only")
            self.pending = self.proposed
            actual = Action("CLICK",position=self.proposed.position)
        else:
            actual, _ = ActionFault(self.regime,self.seed).apply(self.proposed)
        self._step(actual,"first")
        return self.obs

    def _step(self, actual, event):
        self.obs,self.reward,self.done,self.truncated,self.info,code = self.env.step(actual)
        self.events.append({"event":event,"executed_action":actual.to_dict(),"executed_code":code,
            "reward":self.reward,"terminated":self.done,"truncated":self.truncated,
            "task_info":deepcopy(self.info.get("task_info",{})),"execution_error":self.obs.last_action_error})

    def wait_once(self):
        if self.done or self.truncated:
            return self.obs
        self.env.env.unwrapped.page.wait_for_timeout(150)
        if self.pending is not None:
            actual,self.pending = self.pending,None
            self._step(actual,"pending_delivered_on_wait")
        else:
            self._step(Action("HOVER",position=self.proposed.position or (0.0,0.0)),"wait_hover")
        return self.obs

    def flush_pending(self):
        if self.pending is not None and not (self.done or self.truncated):
            actual,self.pending = self.pending,None
            self._step(actual,"pending_delivered_before_continuation")

    def repair(self, point):
        if self.proposed.action == "INPUT":
            _,_,value = task_request(self.env.task,self.obs.goal)
            self.obs,self.reward,self.done,self.truncated,self.info,code = self.env.replace_input(point,value)
            self.events.append({"event":"replace_input","position":list(point),"value":value,
                "executed_code":code,"reward":self.reward,"terminated":self.done,"truncated":self.truncated,
                "task_info":deepcopy(self.info.get("task_info",{})),"execution_error":self.obs.last_action_error})
        else:
            self._step(Action("CLICK",position=tuple(point)),"retry_click")

    def close(self):
        self.env.close()
