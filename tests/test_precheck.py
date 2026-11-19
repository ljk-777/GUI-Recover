import json

import pytest

from gui_recover.grounding import INPUT_TARGET_REQUERY, ScriptedGroundingPolicy
from gui_recover.input_guard import inspect_input_target
from gui_recover.runner import run_episode, write_json
from gui_recover.comparison import compare_precheck_runs
from test_input_guard import field, themed_field, TextEnv


@pytest.mark.parametrize("background,foreground", [((255, 255, 255), (0, 0, 0)),
                                                  ((51, 51, 51), (255, 255, 255))])
def test_target_requires_empty_region_and_point_inside_it(background, foreground):
    blank = themed_field("empty", background, foreground)
    assert inspect_input_target(blank, (0.5, 0.5))["status"] == "empty_region"
    assert inspect_input_target(blank, (0.09, 0.5))["reason"] == "point_outside_region_interior"
    assert inspect_input_target(themed_field("text", background, foreground), (0.5, 0.5))["status"] == "unknown"


@pytest.mark.parametrize("position", [None, [0.5], [float("nan"), 0.5], [-0.1, 0.5], [True, 0.5]])
def test_invalid_target_cannot_pass(position):
    assert inspect_input_target(field(), position)["reason"] == "invalid_position"


@pytest.mark.parametrize("background,foreground", [((255, 255, 255), (0, 0, 0)),
                                                  ((51, 51, 51), (255, 255, 255))])
def test_v2_accepts_an_inside_edge_point_without_expanding_outside_field(background, foreground):
    picture = themed_field("empty", background, foreground)
    near_inside_edge = (0.5, 31 / 80)
    assert inspect_input_target(picture, near_inside_edge, version="v1")["status"] == "unknown"
    assert inspect_input_target(picture, near_inside_edge, version="v2")["status"] == "empty_region"
    assert inspect_input_target(picture, (0.5, 29 / 80), version="v2")["status"] == "unknown"


class Grounder:
    config = {"model": "test"}
    last_metrics = {}

    def __init__(self, persistent=False):
        self.persistent = persistent

    def reset(self, seed):
        self.queries = []

    def ground(self, screenshot, query):
        self.queries.append(query)
        return "[0.5,0.85]" if self.persistent or len(self.queries) == 1 else "[0.5,0.5]"


class WrongTargetEnv(TextEnv):
    def reset(self, seed):
        self.executed = []
        return super().reset(seed)

    def step(self, action):
        self.executed.append(action)
        if action.position == (0.5, 0.85):
            return self.obs(), 0, True, False, {"task_info": {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": -1}}, "bad click"
        return super().step(action)


def run_case(path, enabled=True, persistent=False, fault="none", budget=4):
    env, backend = WrongTargetEnv(), Grounder(persistent)
    policy = ScriptedGroundingPolicy(backend, input_guard=True, input_guard_version="v3", input_precheck=enabled)
    result = run_episode(env, policy, "enter-text", 0, budget, path,
                         recovery="input-guard", fault_kind=fault)
    return result, env, backend, policy


def test_rejected_candidate_never_executes_and_costs_a_decision(tmp_path):
    result, env, backend, _ = run_case(tmp_path / "checked")
    assert result["success"] and result["model_calls"] == result["predictions"] == 3
    assert result["steps"] == 2 and len(env.executed) == 2
    assert backend.queries == ["text field", INPUT_TARGET_REQUERY, "Submit button"]
    events = [json.loads(line) for line in (tmp_path / "checked/trajectory.jsonl").read_text().splitlines()]
    assert [e["decision"] for e in events if e["event"] == "pre_execution_check"] == ["reground_and_retry", "execute"]
    assert [e["step"] for e in events if e["event"] == "transition"] == [1, 2]
    baseline, _, _, _ = run_case(tmp_path / "baseline", enabled=False)
    assert not baseline["success"] and baseline["reason"] == "task_failure"


def test_corrective_query_persists_for_post_action_missing_input_retry(tmp_path):
    result, _, backend, _ = run_case(tmp_path / "drop", fault="drop-input")
    assert result["success"] and result["injected"]
    assert result["model_calls"] == result["predictions"] == 4
    assert result["steps"] == 3
    assert backend.queries == ["text field", INPUT_TARGET_REQUERY, INPUT_TARGET_REQUERY, "Submit button"]
    assert result["input_precheck"]["regrounds"] == result["input_guard"]["retries"] == 1


def test_persistent_wrong_target_blocks_without_executing_or_injecting(tmp_path):
    result, env, _, _ = run_case(tmp_path / "blocked", persistent=True, fault="drop-input")
    assert not result["success"] and result["reason"] == "input_target_unverified"
    assert result["model_calls"] == result["predictions"] == 2
    assert result["steps"] == 0 and not result["injected"] and not env.executed
    assert result["input_precheck"]["blocked"] == 1


def test_one_decision_budget_does_not_hide_an_extra_grounding_call(tmp_path):
    result, env, backend, _ = run_case(tmp_path / "limit", budget=1)
    assert result["reason"] == "step_limit" and not result["success"]
    assert result["model_calls"] == result["predictions"] == 1 and result["steps"] == 0
    assert len(backend.queries) == 1 and not env.executed


def test_clean_target_adds_no_model_calls_and_episode_reset_clears_repair_state(tmp_path):
    class GoodGrounder(Grounder):
        def ground(self, screenshot, query):
            self.queries.append(query)
            return "[0.5,0.5]"
    policy = ScriptedGroundingPolicy(GoodGrounder(), input_guard=True, input_guard_version="v3", input_precheck=True)
    clean = run_episode(WrongTargetEnv(), policy, "enter-text", 0, 4, tmp_path / "clean")
    assert clean["success"] and clean["model_calls"] == clean["steps"] == 2
    assert clean["input_precheck"]["regrounds"] == 0
    policy = ScriptedGroundingPolicy(Grounder(), input_guard=True, input_guard_version="v3", input_precheck=True)
    for seed in (0, 1):
        result = run_episode(WrongTargetEnv(), policy, "enter-text", seed, 4, tmp_path / str(seed))
        assert result["success"] and result["input_precheck"]["regrounds"] == 1
        assert result["model_calls"] == 3


def paired_runs(tmp_path):
    paths = [tmp_path / "baseline", tmp_path / "checked"]
    for path, enabled in zip(paths, (False, True)):
        path.mkdir()
        result, _, _, policy = run_case(path / "enter-text-0", enabled=enabled)
        write_json(path / "manifest.json", {"policy": "showui-scripted", "diagnostic_only": False,
            "config": {"tasks": ["enter-text"], "seeds": [0], "max_steps": 4,
                       "out": str(path), "input_precheck": "visual" if enabled else "off"},
            "model_config": policy.config,
            "runtime": {"git_commit": "abc", "git_dirty": False, "packages": {},
                        "miniwob_commit": "def", "platform": "test"}})
        write_json(path / "summary.json", {"episodes": 1, "results": [result]})
    return paths


def test_precheck_comparison_checks_initial_state_and_full_denominator(tmp_path):
    baseline, checked = paired_runs(tmp_path)
    report = compare_precheck_runs(baseline, checked)
    assert report["matched_initial_pairs"] == report["rescued"] == 1
    assert report["degraded"] == 0
    assert report["arms"]["baseline"]["model_calls"] == 1
    assert report["arms"]["checked"]["model_calls"] == 3
    file = checked / "enter-text-0/trajectory.jsonl"
    events = [json.loads(line) for line in file.read_text().splitlines()]
    events[0]["sha256"] = "changed"
    file.write_text("".join(json.dumps(e) + "\n" for e in events))
    report = compare_precheck_runs(baseline, checked)
    assert report["mismatched_initial_pairs"] == 1 and report["rescued"] == 0
    assert report["arms"]["baseline"]["episodes"] == report["arms"]["checked"]["episodes"] == 1


@pytest.mark.parametrize("change", ["budget", "model", "dirty", "version"])
def test_precheck_comparison_rejects_unmatched_configuration(tmp_path, change):
    baseline, checked = paired_runs(tmp_path)
    file = checked / "manifest.json"
    manifest = json.loads(file.read_text())
    if change == "budget":
        manifest["config"]["max_steps"] = 5
    elif change == "model":
        manifest["model_config"]["model"] = "other"
    elif change == "version":
        manifest["model_config"]["input_precheck"]["signal"] = "other"
    else:
        manifest["runtime"]["git_dirty"] = True
    write_json(file, manifest)
    with pytest.raises(ValueError):
        compare_precheck_runs(baseline, checked)
