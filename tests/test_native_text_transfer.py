import json
import os

import pytest

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.recovery_env import Decision, ProbeDecision, RecoveryDecisionEnv
from gui_recover.text_goals import expected_text
from gui_recover.verification_env import TransitionExecution, teacher_evidence
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_study import prepare_references


def test_visible_case_goals_resolve_before_typing():
    assert expected_text('Type "sara" in all upper case letters in the text input and press Submit.') == "SARA"
    assert expected_text('Type "LIAM" in all lower case letters in the text input and press Submit.') == "liam"
    assert expected_text('Enter "b7J" into the text field and press Submit.') == "b7J"
    assert expected_text("Submit all text.") is None


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")
@pytest.mark.parametrize("task", ["enter-text-2", "enter-text-dynamic"])
def test_tail_repair_generalizes_to_two_more_native_tasks(tmp_path, task):
    seed = 82000
    sample = TransitionExecution(task, seed, None)
    try:
        page = sample.env.env.unwrapped.page
        viewport = page.viewport_size

        def point(selector):
            box = page.locator(selector).bounding_box()
            return ((box["x"] + box["width"] / 2) / viewport["width"],
                    (box["y"] + box["height"] / 2) / viewport["height"])

        field, submit = point("#tt"), point("#subbtn")
        goal = sample.obs.goal
    finally:
        sample.close()
    (tmp_path / "preflight.json").write_text(json.dumps({"selected": [{"task": task, "goal": goal}]}))
    prepare_references(tmp_path)
    verifier = VisualTextVerifier(json.loads((tmp_path / "text-references.json").read_text()))

    class KnownPoints:
        def ground(self, picture, target):
            return {"point": submit if target == "Submit button" else field}

        def repair(self, view, mode, **options):
            return {"point": field}

    env = RecoveryDecisionEnv(task, KnownPoints(), regime="trailing-space-input", input_repair="replace",
                              enable_tail_probe=True, batched_probe=True)
    try:
        env.reset(seed=seed)
        observation, _, terminated, truncated, info = env.step(Decision.CONTINUE)
        assert not terminated and not truncated, info
        assert teacher_evidence(env.execution)["observed"] == expected_text(goal) + " "
        choose = ActiveTextPolicy(verifier)
        assert choose(observation) == ProbeDecision.PROBE_LAST_CHAR, choose.evidence
        observation, _, terminated, truncated, info = env.step(ProbeDecision.PROBE_LAST_CHAR)
        assert not terminated and not truncated and observation["probe_restored"], info
        assert choose(observation) == Decision.RETRY_FIXED_ZOOM, choose.evidence
        _, reward, terminated, truncated, info = env.step(Decision.RETRY_FIXED_ZOOM)
        assert (reward, terminated, truncated) == (1., True, False), info
    finally:
        env.close()
