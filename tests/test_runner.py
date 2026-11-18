import json

import pytest
from PIL import Image

from gui_recover.environment import VisualObservation
from gui_recover.runner import aggregate, is_success, render_replay, run_episode


@pytest.mark.parametrize("reward,terminated,truncated,info,expected", [
    (1, True, False, {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 1}, True),
    (1, False, False, {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 1}, False),
    (1, True, True, {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 1}, False),
    (0.5, True, False, {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 0.5}, False),
    (1, True, False, {}, False),
    (1, True, False, {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": -1}, False),
])
def test_success_requires_environment_evidence(reward, terminated, truncated, info, expected):
    assert is_success(reward, terminated, truncated, info) is expected


class FakeEnv:
    def reset(self, seed):
        return VisualObservation("<script>bad</script>", Image.new("RGB", (20, 10))), {}

    def metadata(self):
        return {}

    def step(self, action):
        return self.reset(0)[0], 0, False, False, {}, "mouse_click(10,5)"


class FixedPolicy:
    def __init__(self, raw):
        self.raw = raw

    def predict(self, obs, history):
        assert not hasattr(obs, "dom_object")
        return self.raw


@pytest.mark.parametrize("raw,reason,steps", [
    ('{"action":"ANSWER","value":"success","position":null}', "policy_stop", 0),
    ('not an action', "invalid_action", 0),
    ('{"action":"CLICK","value":null,"position":[0.5,0.5]}', "step_limit", 2),
])
def test_failures_stay_in_denominator_and_keep_evidence(tmp_path, raw, reason, steps):
    path = tmp_path / "episode"
    result = run_episode(FakeEnv(), FixedPolicy(raw), "fake", 0, 2, path)
    assert result["reason"] == reason
    assert result["steps"] == steps
    assert not result["success"]
    assert aggregate([result])["success_rate"] == 0
    events = [json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "reset"
    assert events[-1]["event"] == "episode_end"
    assert len(list(path.glob("*.png"))) == steps + 1
    render_replay(path)
    assert "<script>bad</script>" not in (path / "replay.html").read_text()
