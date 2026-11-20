import json

import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.environment import VisualObservation
from gui_recover.listing_cards import RESULT_VIEW_HEIGHT, ranked_photo_request, read_card_checks
from gui_recover.ranked_cards import RankedCardEvidence
from gui_recover.verification import pixel_hash


SITE = "http://127.0.0.1:19980"
GOAL = 'Search for "poster" and return the links for the two most recent items that depict birds.'


def test_card_replies_bind_cues_to_each_index():
    assert read_card_checks('{"2":[false,null],"1":[true,"eagle"],"3":null}', 3) == [True, False, None]
    for raw in ('{"1":[true,null]}', '{"1":[true,"bird"],"1":[false,null]}', '{"2":[true,"eagle"]}'):
        with pytest.raises(ActionError):
            read_card_checks(raw, 1)


@pytest.mark.parametrize("first_reply, should_continue", [('{"1":[true,"eagle"],"2":[false,null]}', True),
                                                        ('{"1":[true,"eagle"],"2":null}', False)])
def test_ranked_cards_only_submits_after_contiguous_pages_and_confirmation(first_reply, should_continue):
    class Actor:
        last_metrics = {}
        replies = iter((first_reply, '{"1":[true,"owl"],"2":[true,"duck"]}',
                        '{"1":[true,"eagle"]}', '{"1":[true,"owl"]}'))

        def prepare_image(self, image):
            return image

        def ask(self, _image, request):
            assert SITE not in request
            self.last_metrics = {"peak_cuda_allocated_bytes": 1}
            return next(self.replies)

    actor = Actor()
    policy = RankedCardEvidence(SITE)
    request = ranked_photo_request(GOAL)
    assert request == ("poster", "birds", 2)
    first_url, second_url = [f"{SITE}/index.php?page=item&id={index}" for index in (1, 3)]

    def decide(first, height, color):
        screen = Image.new("RGB", (64, height), color)
        rows = [{"text": text} for text in ("Classifieds > Search results: poster", "Sort by: Newly listed",
                                            f"{first} - {first + 1} of 4 listings")]
        cards = [{"destination": f"{SITE}/index.php?page=item&id={index}",
                  "title": [2, index + 10, 30, index + 20], "image": [32, 5, 60, 32]}
                 for index in range(first, first + 2)]
        receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "cards": cards,
                   "grid": [0.5, 0.25], "next": [0.5, 0.75]}
        raw = policy.decide(VisualObservation(GOAL, screen), request, actor, lambda *_: rows, lambda: receipt)
        return parse_action(raw) if raw is not None else None

    assert decide(1, 72, "white") == Action("LISTING_GRID", position=(0.5, 0.25))
    next_action = decide(1, RESULT_VIEW_HEIGHT, "white")
    if should_continue:
        assert next_action == Action("CLICK", position=(0.5, 0.75))
        assert decide(3, RESULT_VIEW_HEIGHT, "pink") is None
        assert decide(3, RESULT_VIEW_HEIGHT, "pink") is None
        assert decide(3, RESULT_VIEW_HEIGHT, "pink") == Action("ANSWER", f"{first_url}\n{second_url}")
    else:
        assert next_action is None
        assert actor.last_metrics["stop_after_evidence"] is True
