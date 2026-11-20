import json

import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import (EvidenceAnswerPolicy, answer_signature, evidence_answer, evidence_box,
                                         evidence_clipped, evidence_navigation, native_crop)
from gui_recover.environment import VisualObservation
from gui_recover.qwen_grounding import QwenGroundingPolicy, parse_visible_answer


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


def test_native_region_can_carry_one_inert_label():
    raw = '[{"bbox_2d":[3,4,10,12],"label":"requested product"}]'
    assert evidence_box(raw, (20, 20)) == (3, 4, 10, 12)
    with pytest.raises(ActionError):
        evidence_box('[{"bbox_2d":[3,4,10,12]},{"bbox_2d":[6,5,7,8]}]', (20, 20))


def test_price_range_needs_both_endpoints_in_visible_facts():
    goal = "What is the price range of the items in the second row?"
    assert evidence_answer('{"answer":"$500","evidence":["One tag says $500"]}', goal) == (None, "evidence_incomplete")
    assert evidence_answer('{"answer":"$500 to $1,200","evidence":["One tag says $500"]}', goal) == (None, "evidence_incomplete")
    assert evidence_answer('{"answer":"$500 to $1,200","evidence":["$500 and $1200.00"]}', goal) == (
        "$500 to $1,200", "evidence_answer")


def test_clipped_row_requires_navigation_to_reveal_it():
    assert evidence_clipped("How many red items are in row three?", (20, 10, 70, 99), (100, 100))
    assert not evidence_clipped("How many red items are in row three?", (20, 10, 70, 80), (100, 100))
    assert not evidence_clipped("What brand is shown in the photo?", (20, 10, 70, 99), (100, 100))


def test_independent_answer_signature_matches_formatting_variants():
    assert answer_signature("$1,200.00") == answer_signature("1200")
    assert answer_signature("Blue and white.") == answer_signature("blue and white")
    assert answer_signature("$5 - $120") != answer_signature("$5")
    assert answer_signature("-3") != answer_signature("3")


def test_independent_reader_accepts_finite_numeric_json_answers():
    assert parse_visible_answer('{"answer":6}', allow_numeric=True) == "6"
    assert parse_visible_answer('{"answer":1200.0}', allow_numeric=True) == "1200.0"
    for output in ('{"answer":true}', '{"answer":NaN}', '{"answer":1e999}'):
        with pytest.raises(ActionError):
            parse_visible_answer(output, allow_numeric=True)
    with pytest.raises(ActionError):
        parse_visible_answer('{"answer":6}')


def test_native_click_and_type_use_one_validated_input():
    actor = QwenGroundingPolicy.__new__(QwenGroundingPolicy)
    actor.recovery_feedback = None
    actor.native_navigation = True
    actor.prepare_image = lambda image: image
    actor.last_metrics = {}
    obs = VisualObservation("Search for a product", Image.new("RGB", (20, 20)))
    first = '{"action":"click","coordinate":[5,8]}'
    second = '{"action":"type","text":"product"}'
    actor._generate = lambda messages: f"{first}\n{second}"
    assert parse_action(actor.predict(obs, [], allow_answer=False)) == Action("INPUT", "product", (0.275, 0.425))
    actor._generate = lambda messages: f"{first}\n{second}\n{first}"
    assert actor.predict(obs, [], allow_answer=False) == ""
    actor._generate = lambda messages: '{"action":"key","key":"Return"}'
    assert parse_action(actor.predict(obs, [], allow_answer=False)) == Action("ENTER")
    scroll = '{"action":"SCROLL","value":"down","position":null}'
    actor._generate = lambda messages: f"The target is below the viewport.\n\n{scroll}"
    assert parse_action(actor.predict(obs, [], allow_answer=False)) == Action("SCROLL", "down")
    actor._generate = lambda messages: f"{scroll}\n\n{scroll}"
    assert actor.predict(obs, [], allow_answer=False) == ""


def test_evidence_search_can_navigate_but_cannot_submit():
    reply = '{"action":"INPUT","value":"red boat","position":[5,8]}'
    assert evidence_navigation(reply, (20, 20)) == Action("INPUT", "red boat", (0.275, 0.425))
    with pytest.raises(ActionError):
        evidence_navigation('{"action":"ANSWER","value":"red","position":null}', (20, 20))


@pytest.mark.parametrize("confirm,reply,expected", [
    (False, None, "six"), (True, '{"answer":"six"}', "six"), (True, '{"answer":"four"}', None),
])
def test_policy_localizes_without_submitting_and_reads_only_the_cropped_frame(confirm, reply, expected):
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
            if len(self.calls) == 2:
                return '{"evidence":["Two stacks of three bowls"],"answer":"six"}'
            return reply

        def predict(self, observation, history, *, allow_answer):
            assert not allow_answer
            return "navigation"

    actor = Actor()
    policy = EvidenceAnswerPolicy(actor, confirm=confirm)
    obs = VisualObservation("How many bowls?", Image.new("RGB", (200, 120)))
    assert policy.predict(obs, []) is None
    assert policy.last_metrics["evidence_stage"] == "locate"
    candidate = policy.predict(obs, [])
    assert actor.calls[1].width < actor.calls[0].width
    assert policy.last_metrics["evidence_stage"] == "read"
    if confirm:
        assert candidate is None
        candidate = policy.predict(obs, [])
        assert policy.last_metrics["evidence_stage"] == "confirm"
    if expected is None:
        assert candidate is None
    else:
        assert parse_action(candidate) == Action("ANSWER", expected)
    assert policy.predict(obs, []) == "navigation"
