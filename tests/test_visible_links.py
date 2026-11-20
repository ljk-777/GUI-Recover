import json

import pytest
from PIL import Image

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.environment import VisualObservation
from gui_recover.verification import pixel_hash
from gui_recover.visible_links import link_answer_count, listing_selection, preview_destinations, visible_listing_links


SITE = "http://127.0.0.1:19980"
URLS = [f"{SITE}/index.php?page=item&id={index}" for index in (42, 56)]
ROWS = [{"text": title, "bounds": [left, 10, left + 20, 20]}
        for title, left in [("Oak frame", 10), ("Blue canvas", 60), ("Other art", 80)]]


class Page:
    url = "http://127.0.0.1:19980/index.php?page=search"

    def evaluate(self, script, points):
        assert "elementFromPoint" in script
        assert points == [[0.15, 0.25], [0.5, 0.75], [0.8, 0.9]]
        return ["http://127.0.0.1:19980/index.php?page=item&id=42",
                "https://example.com/index.php?page=item&id=123",
                "http://127.0.0.1:19980/index.php?page=search"]


def test_link_preview_reports_public_item_destinations_under_supplied_points():
    result = visible_listing_links(Page(), "http://127.0.0.1:19980", [(0.15, 0.25), (0.5, 0.75), (0.8, 0.9)])
    assert [row["destination"] for row in result] == ["http://127.0.0.1:19980/index.php?page=item&id=42", None, None]


@pytest.mark.parametrize("positions", [[], [(True, 0)], [(0.5, 1)], [(0.2, float("nan"))],
                                        [(0.2, 0.3)] * 2, [(0.2, 0.3)] * 4])
def test_link_preview_validates_points_before_touching_the_browser(positions):
    with pytest.raises(ActionError):
        visible_listing_links(Page(), "http://127.0.0.1:19980", positions)


def test_link_selection_requires_the_complete_requested_count_and_in_bounds_titles():
    assert link_answer_count("Send the links for the three most recent listings.") == (3, 3)
    assert link_answer_count("What is the price of the two listings?") is None
    size = (100, 40)
    assert listing_selection('{"listings":[0,1]}', ROWS, size, (2, 2)) == ([0, 1], [(0.205, 0.3875), (0.705, 0.3875)])
    for value in ([0], [0, 0], [True, 1], [0, 3]):
        with pytest.raises(ActionError):
            listing_selection(json.dumps({"listings": value}), ROWS, size, (2, 2))


def test_aligned_price_selection_maps_to_the_clickable_title_once():
    rows = ROWS + [{"text": "20.00 $", "bounds": [9, 22, 26, 30]}]
    assert listing_selection('{"listings":[3,1]}', rows, (100, 40), (2, 2))[0] == [0, 1]
    with pytest.raises(ActionError, match="same listing"):
        listing_selection('{"listings":[0,3]}', rows, (100, 40), (2, 2))


@pytest.mark.parametrize("confirm,expected", [([1, 0], True), ([0, 2], False), (None, False)])
def test_listing_policy_submits_only_agreed_browser_destinations(confirm, expected):
    class Actor:
        replies = iter(([0, 1], confirm))

        def prepare_image(self, screen):
            return screen

        def ask(self, canvas, prompt):
            assert all(url not in prompt for url in URLS)
            self.last_metrics = {"model_invocations": 1}
            return json.dumps({"listings": next(self.replies)})

    screen = Image.new("RGB", (100, 40))

    def preview(points):
        return {"source_pixels": pixel_hash(screen), "links": [
            {"position": list(point), "destination": url} for point, url in zip(points, URLS, strict=True)
        ]}

    policy = EvidenceAnswerPolicy(Actor(), confirm=True, screen_reader=lambda *_: ROWS,
                                  listing_preview=preview, listing_site=SITE)
    observation = VisualObservation("Give me the links for the two listings shown in the first row.", screen)
    assert policy.predict(observation, []) is None
    assert policy.last_metrics["provider_format"] == "evidence_link_preview"
    raw = policy.predict(observation, [])
    assert policy.last_metrics["evidence_stage"] == "link_confirm"
    if expected:
        assert parse_action(raw) == Action("ANSWER", "\n".join(reversed(URLS)))
    else:
        assert raw is None


def test_browser_preview_rejects_stale_pixels_and_unregistered_destinations():
    point = (0.4, 0.6)
    receipt = {"source_pixels": "old", "links": [{"position": list(point), "destination": URLS[0]}]}
    with pytest.raises(ActionError, match="viewport"):
        preview_destinations(receipt, "new", [point], SITE)
    with pytest.raises(ActionError, match="local listing"):
        preview_destinations(receipt, "old", [point], "https://example.com")
    receipt["links"][0]["destination"] = None
    assert preview_destinations(receipt, "old", [point], SITE) is None
