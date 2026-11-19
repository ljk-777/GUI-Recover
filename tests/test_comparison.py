import json

import pytest
from PIL import Image

from gui_recover.comparison import compare_runs
from gui_recover.environment import VisualObservation
from gui_recover.runner import run_episode, write_json


def make_runs(tmp_path):
    class Env:
        def reset(self, seed):
            return VisualObservation("Click", Image.new("RGB", (10, 10))), {}

        def metadata(self):
            return {}

        def step(self, action):
            done = action.action == "CLICK"
            return self.reset(0)[0], int(done), done, False, {"task_info": {"DONE_GLOBAL": done, "RAW_REWARD_GLOBAL": int(done)}}, "compiled"

    class Policy:
        def predict(self, obs, history):
            return "{'action':'CLICK','value':None,'position':[0.5,0.5]}"

    paths = []
    for label, fault, recovery in (("a", "none", "none"), ("b", "drop-click", "none"),
                                    ("c", "drop-click", "feedback"), ("d", "none", "feedback")):
        directory = tmp_path / label
        directory.mkdir()
        config = {"tasks": ["fake"], "seeds": [0], "max_steps": 2, "fault": fault,
                  "fault_seed": 0, "recovery": recovery}
        manifest = {"policy": "showui", "diagnostic_only": False, "config": config, "model_config": {},
                    "runtime": {"git_commit": "abc", "git_dirty": False, "packages": {},
                                "miniwob_commit": "def", "platform": "test"}}
        write_json(directory / "manifest.json", manifest)
        result = run_episode(Env(), Policy(), "fake", 0, 2, directory / "fake-0",
                             fault_kind=fault, recovery=recovery)
        write_json(directory / "summary.json", {"episodes": 1, "results": [result]})
        paths.append(directory)
    return paths


def test_pair_audit_and_no_false_recovery_gain(tmp_path):
    paths = make_runs(tmp_path)
    result = compare_runs(*paths)
    assert result["matched_injected_pairs"] == 1
    assert result["rescued"] == 0
    assert result["rescue_fraction"] is None
    assert result["arms"]["faulted"]["steps"] == 2
    assert result["arms"]["recovered"]["steps"] == 2


def test_pair_with_different_reset_image_is_excluded(tmp_path):
    paths = make_runs(tmp_path)
    file = paths[2] / "fake-0/trajectory.jsonl"
    events = [json.loads(line) for line in file.read_text().splitlines()]
    events[0]["sha256"] = "different"
    file.write_text("".join(json.dumps(e) + "\n" for e in events))
    result = compare_runs(*paths)
    assert result["matched_injected_pairs"] == 0
    assert result["mismatched_prefixes"] == 1


def test_different_budget_or_incomplete_run_is_rejected(tmp_path):
    paths = make_runs(tmp_path)
    path = paths[2] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["max_steps"] = 3
    write_json(path, manifest)
    with pytest.raises(ValueError, match="max_steps"):
        compare_runs(*paths)
    manifest["config"]["max_steps"] = 2
    manifest["config"]["seeds"] = [0, 1]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="did not complete"):
        compare_runs(*paths)
