import json
import os

import pytest

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.recovery_env import Decision, ProbeDecision, RecoveryDecisionEnv
from gui_recover.sft_recovery import center, visible_target_boxes
from gui_recover.verification_env import TransitionExecution, teacher_evidence
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_study import prepare_references


pytestmark = [pytest.mark.integration, pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")]


def test_tail_selection_preserves_value_and_routes_only_confirmed_space(tmp_path):
    seed = 74002
    sample = TransitionExecution("enter-text", seed, None)
    try:
        goal = sample.obs.goal
        field = center(visible_target_boxes(sample.env, sample.obs)[1][0])
        submit = center(visible_target_boxes(sample.env, sample.obs, "submit")[1][0])
    finally:
        sample.close()
    (tmp_path / "preflight.json").write_text(json.dumps({"selected": [{"task": "enter-text", "goal": goal}]}))
    prepare_references(tmp_path)
    verifier = VisualTextVerifier(json.loads((tmp_path / "text-references.json").read_text()))

    class KnownPoints:
        def ground(self, picture, target):
            return {"point": submit if target == "Submit button" else field}

        def repair(self, view, mode, **options):
            return {"point": field}

    cases = (
        ("none", "visible_tail", Decision.CONTINUE),
        ("deferred-input", "visible_tail", Decision.CONTINUE),
        ("trailing-space-input", "blank_tail", Decision.RETRY_FIXED_ZOOM),
    )
    for regime, status, continuation in cases:
        policy = ActiveTextPolicy(verifier)
        env = RecoveryDecisionEnv("enter-text", KnownPoints(), regime=regime,
                                  input_repair="replace", enable_tail_probe=True)
        try:
            obs, _ = env.reset(seed=seed)
            assert obs["action_mask"].tolist() == [1, 0, 0, 0, 0]
            obs, reward, done, truncated, info = env.step(Decision.CONTINUE)
            assert (reward, done, truncated) == (0.0, False, False), (
                regime, info["outcome"], [e["execution_error"] for e in info["browser_events"]])
            if regime == "deferred-input":
                assert policy(obs) == Decision.WAIT
                obs, _, _, _, _ = env.step(Decision.WAIT)
            original = teacher_evidence(env.execution)["observed"]
            assert policy(obs) == ProbeDecision.PROBE_LAST_CHAR, (regime, policy.evidence)
            after, reward, done, truncated, info = env.step(ProbeDecision.PROBE_LAST_CHAR)
            assert (reward, done, truncated) == (0.0, False, False)
            assert after["probed"] and after["probe_restored"] and env.observation_space.contains(after)
            assert after["action_mask"].tolist() == [1, 0, 1, 1, 0]
            assert [e["event"] for e in info["browser_events"]] == ["probe_select_tail", "probe_collapse_selection"]
            assert teacher_evidence(env.execution)["observed"] == original
            assert policy(after) == continuation and policy.evidence[-1]["selection"]["status"] == status, policy.evidence
            _, reward, done, truncated, info = env.step(continuation)
            assert (reward, done, truncated) == (1.0, True, False)
            assert info["repairs"] == int(regime == "trailing-space-input")
        finally:
            env.close()
