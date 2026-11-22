import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, compile_action, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy, parse_evidence_location
from gui_recover.environment import VisualObservation
from gui_recover.screen_text import RapidScreenReader, exact_initial_search, from_readable_control, visible_rows


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


def test_crop_reads_do_not_change_ocr_stages_for_later_screens(monkeypatch):
    class StatefulOCR:
        def __init__(self, **kwargs):
            self.use_det = self.use_cls = self.use_rec = True
            self.reads = []

        def __call__(self, image, *, use_det=None, use_cls=None, use_rec=None):
            if use_det is not None:
                self.use_det = use_det
            if use_cls is not None:
                self.use_cls = use_cls
            if use_rec is not None:
                self.use_rec = use_rec
            self.reads.append((self.use_det, self.use_cls, self.use_rec))
            return SimpleNamespace(boxes=None, txts=("27.00 $",), scores=[0.95])

    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=StatefulOCR))
    monkeypatch.setattr("importlib.metadata.version", lambda package: "test")
    reader = RapidScreenReader()
    screen = Image.new("RGB", (120, 80))
    reader(screen, screen.size)
    reader.read_upright(screen, screen.size)
    assert reader.read_line(screen, screen.size) == [{"text": "27.00 $", "bounds": [0, 0, 120, 80]}]
    reader.read_upright(screen, screen.size)
    reader(screen, screen.size)
    assert reader.engine.reads == [(True, True, True), (True, False, True), (False, False, True),
                                   (True, False, True), (True, True, True)]


@pytest.mark.parametrize(("texts", "scores"), (
    (None, ()), (("27.00 $",), None), (("27.00 $",), (0.64,)),
    (("27.00 $",), (float("nan"),)), (("27.", "7.00 $"), (0.9, 0.9)),
))
def test_whole_line_ocr_requires_one_confident_read(monkeypatch, texts, scores):
    reader = RapidScreenReader.__new__(RapidScreenReader)
    monkeypatch.setattr(reader, "_run", lambda *args, **kwargs: SimpleNamespace(txts=texts, scores=scores))
    screen = Image.new("RGB", (120, 80))
    assert reader.read_line(screen, screen.size) == []


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


def test_search_uses_only_a_recognized_empty_field_and_compiles_as_one_operation():
    rows = [{"text": "Keyword", "bounds": [10, 5, 20, 9]},
            {"text": "e.g., an item", "bounds": [10, 12, 40, 20]}]
    action = from_readable_control('{"action":"SEARCH","value":"dishwasher","control":1}', rows, (50, 30))
    assert action.action == "SEARCH"
    code = compile_action(action, 50, 30)
    assert code.splitlines()[1:] == ["keyboard_type('dishwasher')", "keyboard_press('Enter')"]
    with pytest.raises(ActionError):
        from_readable_control('{"action":"SEARCH","value":"dishwasher","control":0}', rows, (50, 30))


def test_untranscribed_fast_candidate_still_gets_a_crop():
    reply = '{"bbox_2d":[30,20,70,40],"evidence":["Bowls"],"answer":"5"}'
    rows = [{"text": "Bowls 150.00 $", "bounds": [30, 20, 70, 40]}]
    action, box, answer, status = parse_evidence_location(
        reply, "How many bowls?", (100, 60), confirm=True, rows=rows, guarded_ocr=True)
    assert (action, box, answer, status) == (None, (30, 20, 70, 40), None, "evidence_region")


def test_explicit_search_needs_visible_empty_form_and_exact_task_phrase():
    rows = [{"text": "Keyword", "bounds": [10, 5, 20, 9]},
            {"text": "e.g., an item", "bounds": [10, 12, 40, 20]},
            {"text": "Search", "bounds": [60, 12, 80, 20]}]
    goal = 'Search for "dishwasher" and find the brand.'
    action = exact_initial_search(goal, rows, (100, 60))
    assert action.action == "SEARCH" and action.value == "dishwasher"
    assert exact_initial_search(goal, rows[:-1], (100, 60)) is None
    assert exact_initial_search("What brand is the dishwasher?", rows, (100, 60)) is None


def test_policy_can_submit_a_literal_task_query_without_a_model_call():
    class Actor:
        def prepare_image(self, screen):
            return screen

        def ask(self, screen, prompt):
            raise AssertionError("An exact search should not use the model")

    rows = [{"text": "Keyword", "bounds": [10, 5, 20, 9]},
            {"text": "e.g., an item", "bounds": [10, 12, 40, 20]},
            {"text": "Search", "bounds": [60, 12, 80, 20]}]
    actor = Actor()
    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=lambda screen, size: rows)
    observation = VisualObservation('Search for "dishwasher" and find the brand.', Image.new("RGB", (100, 60)))
    assert parse_action(policy.predict(observation, [])).action == "SEARCH"
    assert actor.last_metrics["model_invocations"] == 0
