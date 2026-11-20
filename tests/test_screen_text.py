import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.environment import VisualObservation
from gui_recover.screen_text import from_readable_control, visible_rows


def test_ocr_regions_map_to_the_canvas_and_reject_unusable_rows():
    result = SimpleNamespace(
        boxes=np.array([[[30, 20], [70, 20], [70, 40], [30, 40]],
                        [[1, 1], [2, 1], [2, 2], [1, 2]]]),
        txts=[" Search ", "ignored"],
        scores=[0.99, 0.4],
    )
    rows = visible_rows(result, (100, 60), (50, 30))
    assert rows == [{"text": "Search", "bounds": [15, 10, 35, 20]}]
    action = from_readable_control('{"action":"CLICK","value":null,"control":0}', rows, (50, 30))
    assert action == Action("CLICK", position=(0.51, 15.5 / 30))
    for control in (-1, True, 1):
        with pytest.raises(ActionError):
            from_readable_control(json.dumps({"action": "CLICK", "value": None, "control": control}),
                                  rows, (50, 30))


def test_readable_candidate_still_needs_an_independent_read():
    class Actor:
        def __init__(self):
            self.responses = iter(('{"bbox_2d":[30,20,70,40],"evidence":["Blue"],"answer":"blue"}',
                                   '{"answer":"Blue"}'))

        def prepare_image(self, screen):
            return screen

        def ask(self, screen, prompt):
            self.last_metrics = {}
            return next(self.responses)

    reader = lambda screen, size: [{"text": "Blue", "bounds": [30, 20, 70, 40]}]
    actor = Actor()
    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=reader)
    observation = VisualObservation("What color is the shown item?", Image.new("RGB", (100, 60)))
    assert policy.predict(observation, []) is None
    assert actor.last_metrics["provider_format"] == "evidence_locate_read"
    assert parse_action(policy.predict(observation, [])) == Action("ANSWER", "blue")
    assert actor.last_metrics["provider_format"] == "evidence_verify_agreed"
