import os
import re
import json

import pytest

from gui_recover.environment import MiniwobEnvironment
from gui_recover.actions import Action
from gui_recover.policies import OracleSmokePolicy
from gui_recover.runner import run_episode
from gui_recover.runner import is_success


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_real_miniwob_scaled_coordinate_click(tmp_path):
    env = MiniwobEnvironment("click-test")
    try:
        result = run_episode(env, OracleSmokePolicy(env), "click-test", 7, 2, tmp_path / "episode")
        assert result["success"], result
        assert result["steps"] == 1
        assert result["environment"]["screenshot_scale"] == 1.5
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_real_input_and_submit():
    # Privileged element lookup belongs only in this execution test, never a VLM policy.
    env = MiniwobEnvironment("enter-text")
    try:
        obs, _ = env.reset(4)
        page = env.env.unwrapped.page

        def position(selector):
            box = page.locator(selector).bounding_box()
            scale = page._bgym_scale_factor
            return ((box["x"] + box["width"] / 2) * scale / obs.screenshot.width,
                    (box["y"] + box["height"] / 2) * scale / obs.screenshot.height)

        value = re.search(r'Enter "(.*?)"', obs.goal).group(1)
        obs, _, done, _, _, _ = env.step(Action("INPUT", value[:2], position("#tt")))
        assert not done
        obs, _, done, _, _, _ = env.step(Action("TYPE", value[2:]))
        assert not done
        assert page.locator("#tt").input_value() == value
        _, reward, terminated, truncated, info, _ = env.step(Action("CLICK", position=position("#subbtn")))
        assert is_success(reward, terminated, truncated, info["task_info"])
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_dropped_click_still_advances_real_environment(tmp_path):
    env = MiniwobEnvironment("click-test")
    try:
        result = run_episode(env, OracleSmokePolicy(env), "click-test", 5, 2,
                             tmp_path / "episode", fault_kind="drop-click")
        assert result["success"] and result["injected"]
        assert result["steps"] == 2
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("fault", ["none", "drop-input"])
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_visual_input_guard_in_real_browser(tmp_path, fault):
    from gui_recover.grounding import ScriptedGroundingPolicy

    env = MiniwobEnvironment("enter-text")

    class DiagnosticBackend:
        # DOM lookup is limited to this grounder test double. The guard itself
        # receives screenshots and the proposed action only.
        config = {"diagnostic_grounder": True}
        last_metrics = {}

        def reset(self, seed):
            pass

        def ground(self, screenshot, query):
            page = env.env.unwrapped.page
            box = page.locator("#tt" if query == "text field" else "#subbtn").bounding_box()
            scale = page._bgym_scale_factor
            return json.dumps([(box["x"] + box["width"] / 2) * scale / screenshot.width,
                               (box["y"] + box["height"] / 2) * scale / screenshot.height])

    try:
        result = run_episode(env, ScriptedGroundingPolicy(DiagnosticBackend(), input_guard=True),
                             "enter-text", 7, 4, tmp_path / "episode",
                             fault_kind=fault, recovery="input-guard")
        assert result["success"], result
        assert result["input_guard"]["retries"] == int(fault == "drop-input")
        assert result["steps"] == 2 + int(fault == "drop-input")
    finally:
        env.close()
