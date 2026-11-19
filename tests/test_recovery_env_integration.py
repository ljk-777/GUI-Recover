import os

import pytest

from gui_recover.recovery_env import Decision, RecoveryDecisionEnv
from gui_recover.sft_recovery import center, visible_target_boxes
from gui_recover.verification_env import TransitionExecution


pytestmark = [pytest.mark.integration, pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")]


def test_sparse_reward_requires_real_completion_after_a_dropped_input():
    seed = 4001
    source = TransitionExecution("enter-text", seed, None)
    try:
        field = center(visible_target_boxes(source.env, source.obs)[1][0])
        submit = center(visible_target_boxes(source.env, source.obs, "submit")[1][0])
    finally:
        source.close()

    class KnownPoints:
        # Real browser integration with a test-only, privileged point source.
        def ground(self, picture, target):
            return {"point": submit if target == "Submit button" else field}

        def repair(self, view, mode, *, fixed_crop, fraction):
            assert mode == "feedback_zoom" and fixed_crop == (0.,0.,.6,.6) and fraction == .6
            return {"point": field}

    env=RecoveryDecisionEnv("enter-text", KnownPoints(), regime="drop-input")
    try:
        env.reset(seed=seed)
        obs, r, done, truncated, _ = env.step(Decision.CONTINUE)
        assert env.observation_space.contains(obs) and (r,done,truncated)==(0.,False,False)
        _, r, done, truncated, proof = env.step(Decision.CONTINUE)
        assert (r,done,truncated,proof["outcome"])==(0.,True,False,"browser_failure")
        env.reset(seed=seed)
        env.step(Decision.CONTINUE)
        _, r, done, truncated, proof = env.step(Decision.RETRY_FIXED_ZOOM)
        assert (r,done,truncated,proof["outcome"])==(1.,True,False,"success")
        assert proof["actor_requests"]==3 and proof["browser_events"][-1]["task_info"]["RAW_REWARD_GLOBAL"]==1
    finally:
        env.close()


def test_wait_distinguishes_delayed_input_from_a_retry_that_duplicates_it():
    seed = 4001
    source = TransitionExecution("enter-text", seed, None)
    try:
        field = center(visible_target_boxes(source.env, source.obs)[1][0])
        submit = center(visible_target_boxes(source.env, source.obs, "submit")[1][0])
    finally:
        source.close()

    class KnownPoints:
        def ground(self, picture, target):
            return {"point": submit if target == "Submit button" else field}

        def repair(self, view, mode):
            return {"point": field}

    env = RecoveryDecisionEnv("enter-text", KnownPoints(), regime="deferred-input")
    try:
        env.reset(seed=seed)
        env.step(Decision.CONTINUE)
        _, reward, terminated, truncated, info = env.step(Decision.RETRY_PLAIN)
        assert (reward, terminated, truncated, info["outcome"]) == (0., True, False, "browser_failure")
        assert [e["event"] for e in info["browser_events"]] == [
            "pending_delivered_before_continuation", "append_input_repair", "submit"]
        env.reset(seed=seed)
        env.step(Decision.CONTINUE)
        _, reward, terminated, truncated, info = env.step(Decision.WAIT)
        assert (reward, terminated, truncated) == (0., False, False)
        assert info["browser_events"][0]["event"] == "pending_delivered_on_wait"
        _, reward, terminated, truncated, info = env.step(Decision.CONTINUE)
        assert (reward, terminated, truncated, info["outcome"]) == (1., True, False, "success")
    finally:
        env.close()


def test_replacement_after_delayed_delivery_is_idempotent():
    from gui_recover.actions import Action
    from gui_recover.verification_env import task_request, teacher_evidence

    execution = TransitionExecution("enter-text", 4001, None, "deferred-input")
    try:
        field = center(visible_target_boxes(execution.env, execution.obs)[1][0])
        submit = center(visible_target_boxes(execution.env, execution.obs, "submit")[1][0])
        _, _, value = task_request("enter-text", execution.obs.goal)
        execution.proposed = Action("INPUT", value, tuple(field))
        execution.first()
        assert teacher_evidence(execution)["observed"] == ""
        execution.repair(field, input_profile="replace")
        assert teacher_evidence(execution)["observed"] == value
        execution.repair(field, input_profile="replace")
        assert teacher_evidence(execution)["observed"] == value
        assert [event["event"] for event in execution.events] == [
            "first", "pending_delivered_before_continuation", "idempotent_input_repair", "idempotent_input_repair"]
        execution._step(Action("CLICK", position=tuple(submit)), "submit")
        assert execution.done and execution.reward == 1.0 and execution.info["task_info"]["RAW_REWARD_GLOBAL"] == 1
    finally:
        execution.close()
