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


def test_prediction_budget_counts_model_calls_separately(tmp_path):
    class BudgetedPolicy(FixedPolicy):
        def predict(self, obs, history):
            self.last_metrics = {"model_invocations": int(bool(history))}
            return super().predict(obs, history)

    click = '{"action":"CLICK","value":null,"position":[0.5,0.5]}'
    result = run_episode(FakeEnv(), BudgetedPolicy(click), "fake", 0, 5, tmp_path / "episode",
                         max_model_calls=1)
    assert result["reason"] == "model_call_limit"
    assert result["steps"] == result["predictions"] == 2
    assert result["model_calls"] == 1


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


def test_policy_stop_does_not_inflate_inference_calls(tmp_path):
    class ScriptedStop(FixedPolicy):
        last_metrics = {"model_invocations": 0}

    result = run_episode(FakeEnv(), ScriptedStop('{"action":"ANSWER","value":"done","position":null}'),
                         "fake", 0, 2, tmp_path / "episode")
    assert result["predictions"] == 1
    assert result["model_calls"] == 0


@pytest.mark.parametrize("budget,expected_calls,expected_steps,success", [
    (4, ["check", "navigate", "navigate", "check"], 3, True),
    (3, ["check", "navigate", "navigate"], 2, False),
])
def test_answer_check_shares_budget_and_visits_each_distinct_frame_once(
        tmp_path, budget, expected_calls, expected_steps, success):
    calls = []

    class Browser(FakeEnv):
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(action.action)
            finished = action.action == "ANSWER" and action.value == "seven"
            assert action.action in ("CLICK", "ANSWER")
            color = "black" if len(self.actions) >= 2 else "white"
            screen = VisualObservation("The question", Image.new("RGB", (20, 10), color))
            return screen, int(finished), finished, False, {}, "browser_action"

        def reset(self, seed):
            return VisualObservation("The question", Image.new("RGB", (20, 10), "white")), {}

    class Policy:
        last_metrics = {"model_invocations": 1}

        def check_answer(self, obs):
            calls.append("check")
            return "seven" if obs.screenshot.getpixel((0, 0)) == (0, 0, 0) else None

        def predict(self, obs, history):
            calls.append("navigate")
            assert len(history) == calls.count("navigate") - 1
            return '{"action":"CLICK","value":null,"position":[0.5,0.5]}'

    browser = Browser()
    out = tmp_path / "episode"
    result = run_episode(browser, Policy(), "fake", 0, budget, out, answer_check=True,
                         submit_answer=True, success_check=lambda reward, done, truncated, info: reward == 1 and done)
    assert calls == expected_calls
    assert result["predictions"] == result["model_calls"] == budget
    assert result["steps"] == expected_steps
    assert result["answer_checks"] == calls.count("check")
    assert result["success"] is success
    assert result["answers_submitted"] == result["answers_from_checks"] == int(success)
    assert result["invalid_predictions"] == 0
    events = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    assert events[1]["observation_sha256"] == events[0]["sha256"]
    assert events[1]["raw"] is None


@pytest.mark.parametrize("mode,budget,expected_predictions,expected_steps,expected_retries", [
    ("none", 3, 1, 0, 0),
    ("retry", 3, 3, 1, 1),
    ("feedback", 3, 3, 1, 1),
    ("feedback", 1, 1, 0, 0),
])
def test_invalid_action_retry_has_same_observation_and_shared_budget(
        tmp_path, mode, budget, expected_predictions, expected_steps, expected_retries):
    class RetryPolicy:
        last_metrics = {"model_invocations": 1}

        def __init__(self):
            self.calls = []
            self.feedback = None

        def set_feedback(self, message):
            self.feedback = message

        def predict(self, obs, history):
            self.calls.append((obs, len(history), self.feedback))
            if len(self.calls) == 2:
                return '{"action":"CLICK","value":null,"position":[0.5,0.5]}'
            return "invalid"

    policy = RetryPolicy()
    result = run_episode(FakeEnv(), policy, "fake", 0, budget, tmp_path / "episode",
                         invalid_action_retry=mode)
    assert result["reason"] == "invalid_action"
    assert result["predictions"] == result["model_calls"] == expected_predictions
    assert result["steps"] == expected_steps
    assert result["invalid_predictions"] == expected_predictions - expected_steps
    assert result["invalid_action_retries"] == result["repaired_predictions"] == expected_retries
    if expected_retries:
        first, retry, next_step = policy.calls
        assert first[0] is retry[0]
        assert first[1] == retry[1] == 0
        assert bool(retry[2]) == (mode == "feedback")
        assert next_step[1:] == (1, None)
