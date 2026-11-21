import json

import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import (EvidenceAnswerPolicy, answer_signature, evidence_answer, evidence_box,
                                         confirmation_image, confirmed_evidence_answer, evidence_clipped,
                                         evidence_navigation, native_crop, parse_independent_confirmation)
from gui_recover.environment import VisualObservation
from gui_recover.listing_cards import listing_breadcrumb_box, listing_title_heading, reference_title_crop, with_listing_breadcrumb
from gui_recover.price_discovery import PriceDiscovery, price_request
from gui_recover.qwen_grounding import QwenGroundingPolicy, normalize_qwen_action, parse_visible_answer
from gui_recover.verification import pixel_hash


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


def test_confirmation_cannot_submit_an_amount_outside_the_verified_price_tie():
    assert confirmed_evidence_answer("100.00", "$100", price_crop=True, expected_cents=10000) == "100"
    assert confirmed_evidence_answer("101.00", "$101", price_crop=True, expected_cents=10000) is None
    assert confirmed_evidence_answer("1,234.50", "$1234.5", price_crop=True) == "1234.50"
    assert confirmed_evidence_answer("100.00", "$100") == "100.00"


def test_ranked_price_crop_accepts_one_redundant_evidence_wrapper_only():
    goal = "Find the most expensive wheelchair lift with multiple people, and tell me how much it costs in dollars."
    nested = '{"evidence":[["2995.00 $"]],"answer":"2995.00"}'
    assert evidence_answer(nested, goal, price_crop=True) == ("2995.00", "evidence_answer")
    with pytest.raises(ActionError):
        evidence_answer(nested, goal)
    for evidence in ([[["2995.00 $"]]], [["2995.00 $"], ["other"]], [[2995]]):
        with pytest.raises(ActionError):
            evidence_answer(json.dumps({"evidence": evidence, "answer": "2995.00"}), goal, price_crop=True)


def test_hardcover_confirmation_requires_a_visible_matching_color_and_rigid_binding():
    request = price_request("Find the most expensive purple hard-case book, and tell me how much it costs in dollars.")
    assert parse_independent_confirmation(
        '{"answer":"1100.00","cover_color":"blue-violet","binding":"hardcover"}', request
    ) == "1100.00"
    assert parse_independent_confirmation(
        '{"answer":"1100.00","cover_color":"blue","binding":"hardcover"}', request
    ) is None
    assert parse_independent_confirmation(
        '{"answer":"1100.00","cover_color":"violet","binding":null}', request
    ) is None
    with pytest.raises(ActionError):
        parse_independent_confirmation('{"answer":"1100.00"}', request)


def test_listing_title_requires_all_heading_lines_to_agree_with_the_breadcrumb():
    rows = [
        {"text": "Classifieds > Jewelry > Vintage silver stone bracelet", "bounds": [112, 85, 480, 103]},
        {"text": "Vintage silver stone", "bounds": [107, 115, 480, 161]},
        {"text": "bracelet", "bounds": [106, 157, 260, 203]},
        {"text": "100.00 $", "bounds": [777, 124, 890, 157]},
        {"text": "Published date: 2023/11/16", "bounds": [115, 225, 276, 245]},
    ]
    assert listing_title_heading(rows, (1036, 588)) == ((106, 115, 480, 203), "Vintage silver stone bracelet")
    assert listing_title_heading([rows[0], *rows[2:]], (1036, 588)) is None
    assert listing_title_heading([dict(rows[0], text="Classifieds > Jewelry > Vintage silver..."), *rows[1:]], (1036, 588)) is None


def test_missing_breadcrumb_can_be_recovered_from_the_band_above_a_wrapped_heading():
    size = (1036, 588)
    rows = [
        {"text": "Classifieds", "bounds": [111, 5, 204, 27]},
        {"text": "Example wrapped heading", "bounds": [110, 119, 576, 160]},
        {"text": "Table", "bounds": [110, 158, 204, 199]},
        {"text": "Published date: yesterday", "bounds": [116, 228, 276, 245]},
    ]
    assert listing_title_heading(rows, size) is None
    band = listing_breadcrumb_box(rows, size)
    assert band == (90, 58, 981, 113)
    local = {"text": "Classifieds > Furniture > Example wrapped heading Table", "bounds": [24, 27, 430, 45]}
    recovered = with_listing_breadcrumb(rows, [local], band)
    assert listing_title_heading(recovered, size)[1] == "Example wrapped heading Table"
    assert listing_breadcrumb_box(recovered, size) is None
    assert with_listing_breadcrumb(rows, [local, local], band) is rows


@pytest.mark.parametrize(("reply", "answered"), [
    ('{"answer":"Silver ring size 6"}', True),
    ('{"answer":"Silver ring size"}', False),
    ('{"answer":"Silver ring size 8"}', False),
    ('{"answer":null}', False),
])
def test_reference_title_rereads_a_missing_number_from_title_pixels_only(reply, answered):
    class Actor:
        def prepare_image(self, image):
            return image

        def ask(self, image, prompt):
            assert "Silver ring" not in prompt and "size 6" not in prompt
            assert image.getpixel((158, 31)) == (255, 0, 0)
            assert {color for _, color in image.getcolors(maxcolors=2) or []} == {(255, 255, 255), (255, 0, 0)}
            self.last_metrics = {"model_invocations": 1}
            return reply

    rows = [
        {"text": "Classifieds > Jewelry > Silver ring size 6", "bounds": [80, 80, 260, 102]},
        {"text": "Silver ring size", "bounds": [81, 116, 205, 160]},
        {"text": "80.00 $", "bounds": [277, 124, 332, 159]},
        {"text": "Published date: yesterday", "bounds": [81, 187, 240, 203]},
    ]
    screen = Image.new("RGB", (480, 300), "white")
    screen.putpixel((231, 139), (255, 0, 0))
    screen.putpixel((133, 95), (0, 255, 0))
    screen.putpixel((290, 139), (0, 255, 0))
    assert listing_title_heading(rows, screen.size) is None
    assert reference_title_crop(rows, screen.size).bounds == (73, 108, 277, 168)
    altered_rows = [{**rows[0], "text": "Classifieds > Jewelry > Another silver ring size 6"}, *rows[1:]]
    assert reference_title_crop(altered_rows, screen.size) is None
    site = "http://localhost:19980"
    url = site + "/index.php?page=item&id=8"
    policy = EvidenceAnswerPolicy(Actor(), confirm=True, screen_reader=lambda *_: rows, listing_site=site,
                                  ranked_capture=lambda: pytest.fail("Already on the matched listing"))
    policy.reference_destination = url
    observation = VisualObservation("Find the exact item in the photo. What is its listing title?", screen,
                                    tabs=(("Silver ring size 6", url),), reference_images=(screen,))
    assert policy.predict(observation, []) is None
    assert policy.last_metrics["reference_title_missing_number"] is True
    result = policy.predict(observation, [])
    if answered:
        assert parse_action(result) == Action("ANSWER", "Silver ring size 6")
    else:
        assert result is None and policy.last_metrics["stop_after_evidence"] is True


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


def test_native_click_aliases_require_one_unambiguous_pixel_point():
    size = (20, 20)
    for reply in ({"action": "CLICK", "value": "[5, 8]"},
                  {"action": "CLICK", "value": [5, 8], "position": [5, 8]}):
        assert normalize_qwen_action(reply, size, native_navigation=True) == Action(
            "CLICK", position=(0.275, 0.425))
    for value, position in (([5, 8], [6, 8]), ([1, 8], [True, 8]), ([5, 8], [5.0, 8])):
        with pytest.raises(ActionError):
            normalize_qwen_action({"action": "CLICK", "value": value, "position": position},
                                  size, native_navigation=True)
    assert normalize_qwen_action({"action": "SCROLL", "value": "down"}, size,
                                 native_navigation=True) == Action("SCROLL", "down")


def test_evidence_search_can_navigate_but_cannot_submit():
    reply = '{"action":"INPUT","value":"red boat","position":[5,8]}'
    assert evidence_navigation(reply, (20, 20)) == Action("INPUT", "red boat", (0.275, 0.425))
    with pytest.raises(ActionError):
        evidence_navigation('{"action":"ANSWER","value":"red","position":null}', (20, 20))


def test_contact_navigation_can_ground_a_click_in_a_visible_box():
    raw = '{"action":"CLICK","value":{"bbox_2d":[3,4,11,12]}}'
    assert evidence_navigation(raw, (20, 20), allow_click_box=True) == Action("CLICK", position=(0.375, 0.425))
    for box in ('[3,4,21,12]', 'null'):
        with pytest.raises(ActionError):
            evidence_navigation('{"action":"CLICK","value":{"bbox_2d":' + box + '}}',
                                (20, 20), allow_click_box=True)


def test_only_a_visual_transition_after_a_click_becomes_context():
    before = VisualObservation("What color is the item?", Image.new("RGB", (200, 120), "yellow"))
    after = VisualObservation(before.goal, Image.new("RGB", (200, 120), "blue"))
    policy = EvidenceAnswerPolicy(object(), confirm=True, use_navigation_context=True)
    click = Action("CLICK", position=(0.25, 0.75))
    policy.observe_transition(before, before, click)
    policy.observe_transition(before, after, Action("SCROLL", "down"))
    assert policy.navigation_context is None
    policy.observe_transition(before, after, click)
    origin, position = policy.navigation_context
    assert position == click.position and origin.getpixel((0, 0)) == (255, 255, 0)
    crop = after.screenshot.crop((30, 40, 80, 100))
    plain = confirmation_image(after.screenshot, crop, (30, 40, 80, 100))
    linked = confirmation_image(after.screenshot, crop, (30, 40, 80, 100), policy.navigation_context)
    assert linked.height > plain.height


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


@pytest.mark.parametrize(("confirmed", "cue", "expected"), [
    ("12", "mirror_camera", True), ("12", "arms_length_portrait", True),
    (None, None, False), ("12", None, False), ("12", "hand_closeup", False),
    ("21", "mirror_camera", False),
])
def test_ranked_price_reader_transcribes_before_independent_photo_confirmation(confirmed, cue, expected):
    class Actor:
        def prepare_image(self, image):
            return image

        def ask(self, image, prompt):
            self.last_metrics = {}
            prompts.append(prompt)
            return next(replies)

    target = "ring that has an image taken as a selfie"
    goal = f"Find the most expensive {target}, and tell me how much it costs in dollars."
    screen = Image.new("RGB", (200, 120))
    obs = VisualObservation(goal, screen)
    prompts = []
    replies = iter((
        '{"evidence":["12.00 $"],"answer":"12.00"}',
        json.dumps({"answer": confirmed, "selfie_cue": cue}),
    ))
    actor = Actor()
    policy = EvidenceAnswerPolicy(actor, confirm=True)
    policy.price_discovery = PriceDiscovery(price_request(goal), "http://localhost:19980")
    bounds = (70, 20, 130, 60)
    policy.pending_crop = pixel_hash(screen), screen.crop(bounds), bounds, False

    assert policy.predict(obs, []) is None
    assert "Transcribe the single dollar price" in prompts[0]
    assert target not in prompts[0]
    answer = policy.predict(obs, [])
    assert target in prompts[1] and "upper-left overview" in prompts[1]
    if expected:
        assert parse_action(answer) == Action("ANSWER", "12")
    else:
        assert answer is None
