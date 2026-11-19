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
