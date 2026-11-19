import json

import pytest
from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.environment import VisualObservation
from gui_recover.grounding import ScriptedGroundingPolicy, parse_point, plan_from_goal, point_hits_boxes


@pytest.mark.parametrize("raw", ["[true,0]", "[True,0]", "[1.1,0]", "[float('nan'),0]",
    "[0.1,0.2],[0.2,0.3]", "{'x':0.1,'y':0.2}", "__import__('os').getcwd()"])
def test_point_boundary(raw):
    with pytest.raises(ActionError):
        parse_point(raw)


def test_plan_from_text_and_point_scoring():
    steps = plan_from_goal('Enter "Myron" into the text field and press Submit.')
    assert [(s.action, s.query, s.value) for s in steps] == [
        ("INPUT", "text field", "Myron"), ("CLICK", "Submit button", None)]
    assert parse_point("[0.2, 0.3]") == (0.2, 0.3)
    assert point_hits_boxes((0.2, 0.3), [[0.1, 0.2, 0.4, 0.5]])
    assert not point_hits_boxes((0.4, 0.5), [[0.1, 0.2, 0.4, 0.5]])
    assert plan_from_goal('Click on the "Yes" button.')[0].query == 'The "Yes" button'
    with pytest.raises(ValueError):
        plan_from_goal("Perform a general web task")


def test_scripted_policy_never_receives_boxes_or_dom():
    class Backend:
        config = {"model": "test"}
        last_metrics = {}
        calls = []

        def reset(self, seed):
            pass

        def ground(self, image, query):
            assert isinstance(image, Image.Image)
            self.calls.append(query)
            return "[0.2, 0.3]"

    backend = Backend()
    policy = ScriptedGroundingPolicy(backend)
    obs = VisualObservation('Enter "abc" into the text field and press Submit.', Image.new("RGB", (30, 20)))
    action = parse_action(policy.predict(obs, []))
    assert action.action == "INPUT" and action.value == "abc"
    history = [{"action": action.to_dict(), "execution_error": ""}]
    action = parse_action(policy.predict(obs, history))
    assert action.action == "CLICK"
    history.append({"action": action.to_dict(), "execution_error": ""})
    assert parse_action(policy.predict(obs, history)).action == "ANSWER"
    assert policy.last_metrics["model_invocations"] == 0
    assert policy.last_metrics["grounding_raw"] is None
    assert backend.calls == ["text field", "Submit button"]
    assert policy.config["uses_dom"] is False
