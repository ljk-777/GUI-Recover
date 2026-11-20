import pytest

from gui_recover.actions import Action, ActionError
from gui_recover.multistep_recovery import BUTTONS, focus_probe, layout_fixture, validate_plan
from gui_recover.qwen_grounding import parse_pixel_point
from gui_recover.verification import pixel_hash
from gui_recover.verification_env import TransitionExecution


def test_transfer_task_must_be_entirely_held_out():
    plan = {"actor_calls": 0, "outcomes_observed": False, "selected": [
        {"seed": 1, "task": BUTTONS, "split": "train", "initial_pixel_hash": "a"},
        {"seed": 2, "task": BUTTONS, "split": "transfer", "initial_pixel_hash": "b"},
    ]}
    with pytest.raises(ValueError, match="Transfer task families"):
        validate_plan(plan)


@pytest.mark.parametrize("raw", ['[2,3]', '{"point_2d":[true,3]}', '{"point_2d":[10,5]}', '{"point_2d":[2,3],"tool":"click"}'])
def test_qwen_pixel_adapter_rejects_unusable_outputs(raw):
    with pytest.raises(ActionError):
        parse_pixel_point(raw, 10, 10)


def test_qwen_pixels_are_mapped_to_centers():
    assert parse_pixel_point('{"point_2d":[2,3]}', 10, 20) == (0.25, 0.175)


@pytest.mark.integration
def test_native_button_focus_probe_distinguishes_identical_observations():
    import os

    if not os.getenv("MINIWOB_URL"):
        pytest.skip("requires pinned native MiniWoB")
    marker = []
    for role in ("correct", "blank"):
        execution = TransitionExecution(BUTTONS, 97000, None)
        try:
            fixture, _ = layout_fixture(execution)
            initial = pixel_hash(execution.obs.screenshot)
            execution.proposed = Action("CLICK", position=tuple(fixture[role]))
            execution.first()
            assert initial == pixel_hash(execution.obs.screenshot)
            verdict, _, after = focus_probe(execution)
            assert verdict["restored"]
            assert pixel_hash(after) == initial
            marker.append(verdict["status"])
        finally:
            execution.close()
    assert marker == ["focused", "unfocused"]


def test_benchmark_adapter_can_submit_answers_without_changing_default(tmp_path):
    from PIL import Image

    from gui_recover.environment import VisualObservation
    from gui_recover.runner import run_episode

    class Environment:
        screen = VisualObservation("Answer the task", Image.new("RGB", (8, 8)))

        def __init__(self):
            self.actions = []

        def reset(self, seed):
            return self.screen, {}

        def step(self, action):
            self.actions.append(action)
            return self.screen, 1, True, False, {}, "benchmark_received_answer"

        def metadata(self):
            return {}

    class Model:
        def predict(self, obs, history):
            return '{"action":"ANSWER","value":"result","position":null}'

    environment = Environment()
    ordinary = run_episode(environment, Model(), "ordinary", 0, 1, tmp_path / "ordinary")
    submitted = run_episode(environment, Model(), "external", 0, 1, tmp_path / "external",
                            submit_answer=True, success_check=lambda reward, done, truncated, info:
                            reward == 1 and done and not truncated)
    assert ordinary["reason"] == "policy_stop"
    assert environment.actions == [Action("ANSWER", "result")]
    assert submitted["success"]
