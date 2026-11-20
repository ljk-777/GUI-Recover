import json

import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy, evidence_answer, evidence_box, native_crop
from gui_recover.environment import VisualObservation


@pytest.mark.parametrize("box", [[-1, 4, 9, 10], [5, 4, 5, 10], [1, 3, 21, 8],
                                  [True, 1, 3, 4], [1, 2, 3.5, 4]])
def test_evidence_box_rejects_unusable_coordinates(box):
    with pytest.raises(ActionError):
        evidence_box(json.dumps({"bbox_2d": box}), (20, 20))


def test_evidence_crop_uses_native_pixels_at_the_prepared_box():
    screen = Image.new("RGB", (200, 120), "white")
    screen.putpixel((80, 60), (100, 50, 25))
    box = evidence_box('{"bbox_2d":[30,22,55,40]}', (100, 60))
    crop, (left, top, right, bottom) = native_crop(screen, (100, 60), box)
    assert crop.size == (right - left, bottom - top)
    assert crop.getpixel((80 - left, 60 - top)) == (100, 50, 25)


def test_price_range_needs_both_endpoints_in_visible_facts():
    goal = "What is the price range of the items in the second row?"
    assert evidence_answer('{"answer":"$500","evidence":["One tag says $500"]}', goal) == (None, "evidence_incomplete")
    assert evidence_answer('{"answer":"$500 to $1,200","evidence":["One tag says $500"]}', goal) == (None, "evidence_incomplete")
    assert evidence_answer('{"answer":"$500 to $1,200","evidence":["$500 and $1200.00"]}', goal) == (
        "$500 to $1,200", "evidence_answer")


def test_policy_localizes_without_submitting_and_reads_only_the_cropped_frame():
    class Actor:
        def __init__(self):
            self.calls = []

        def prepare_image(self, image):
            return image

        def ask(self, image, prompt):
            self.last_metrics = {"model_invocations": 1}
            self.calls.append(image)
            if len(self.calls) == 1:
                return '{"bbox_2d":[60,40,140,80]}'
            return '{"evidence":["Two stacks of three bowls"],"answer":"six"}'

        def predict(self, observation, history, *, allow_answer):
            assert not allow_answer
            return "navigation"

    actor = Actor()
    policy = EvidenceAnswerPolicy(actor)
    obs = VisualObservation("How many bowls?", Image.new("RGB", (200, 120)))
    assert policy.predict(obs, []) is None
    assert policy.last_metrics["evidence_stage"] == "locate"
    assert parse_action(policy.predict(obs, [])) == Action("ANSWER", "six")
    assert actor.calls[1].width < actor.calls[0].width
    assert policy.last_metrics["evidence_stage"] == "read"
    assert policy.predict(obs, []) == "navigation"
