import os
import re

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
        obs, _, done, _, _, _ = env.step(Action("INPUT", value, position("#tt")))
        assert not done
        assert page.locator("#tt").input_value() == value
        _, reward, terminated, truncated, info, _ = env.step(Action("CLICK", position=position("#subbtn")))
        assert is_success(reward, terminated, truncated, info["task_info"])
    finally:
        env.close()
