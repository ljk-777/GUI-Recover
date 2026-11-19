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
@pytest.mark.parametrize("version,variant", [("v2", "default"), ("v3", "default"),
                                            ("v3", "shifted"), ("v3", "dark")])
@pytest.mark.parametrize("fault", ["none", "drop-input", "partial-input", "wrong-input"])
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_visual_input_guard_in_real_browser(tmp_path, fault, version, variant):
    from gui_recover.grounding import ScriptedGroundingPolicy

    env = MiniwobEnvironment("enter-text", visual_variant=variant)

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
        result = run_episode(env, ScriptedGroundingPolicy(DiagnosticBackend(), input_guard=True,
                                                        input_guard_version=version),
                             "enter-text", 7, 4, tmp_path / "episode",
                             fault_kind=fault, recovery="input-guard")
        assert result["success"] == (fault in ("none", "drop-input")), result
        assert result["input_guard"]["retries"] == int(fault == "drop-input")
        assert result["steps"] == 2 + int(fault == "drop-input")
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["shifted", "dark"])
@pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MINIWOB_URL to enable real browser tests")
def test_visual_layout_fixtures_keep_rewards_and_report_limits(variant):
    from gui_recover.input_guard import inspect_input

    env = MiniwobEnvironment("enter-text", visual_variant=variant)
    try:
        before, _ = env.reset(8)
        page = env.env.unwrapped.page
        scale = page._bgym_scale_factor
        box = page.locator("#tt").bounding_box()
        position = ((box["x"] + box["width"] / 2) * scale / before.screenshot.width,
                    (box["y"] + box["height"] / 2) * scale / before.screenshot.height)
        value = re.search(r'Enter "(.*?)"', before.goal).group(1)
        after, _, _, _, _, _ = env.step(Action("INPUT", value, position))
        signal = inspect_input(before.screenshot, after.screenshot, position, version="v2")
        assert signal["status"] == ("text_visible" if variant == "shifted" else "unknown")
        box = page.locator("#subbtn").bounding_box()
        submit = ((box["x"] + box["width"] / 2) * scale / after.screenshot.width,
                  (box["y"] + box["height"] / 2) * scale / after.screenshot.height)
        _, reward, terminated, truncated, info, _ = env.step(Action("CLICK", position=submit))
        assert is_success(reward, terminated, truncated, info["task_info"])
    finally:
        env.close()
