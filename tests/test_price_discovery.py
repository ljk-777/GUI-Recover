from types import SimpleNamespace

import pytest
from PIL import Image

from gui_recover.actions import ActionError, category_search_terms, compile_action, parse_action
from gui_recover.environment import VisualObservation
from gui_recover.price_discovery import (
    PriceDiscovery, first_ranked_match, price_request, price_result_scope,
)


@pytest.mark.parametrize(("subject", "category", "query"), [
    ("purple hard-case book", "Books", "purple"),
    ("ring that has an image taken as a selfie", "Jewelry", "ring"),
    ("wheelchair lift that has multiple people in the image of the listing", "Beauty + health", "lift"),
])
def test_ranked_price_search_starts_on_a_visible_category_and_stops_if_it_stalls(subject, category, query):
    goal = f"Find the most expensive {subject}, and tell me how much it costs in dollars."
    request = price_request(goal)
    assert (request.category, request.query) == (category, query)
    rows = [{"text": label, "bounds": [2, 3, 10, 15]} for label in (
        "Select a category", "Category", "Keyword", "e.g., a blue used car", "Search",
    )]
    site = "http://localhost:19980"
    screen = Image.new("RGB", (50, 30))
    observation = VisualObservation(goal, screen, tabs=(("Classifieds", site + "/"),))
    actor = SimpleNamespace(prepare_image=lambda image: image)
    discovery = PriceDiscovery(request, site)
    raw, box = discovery.decide(observation, actor, lambda *_: rows, None)
    action = parse_action(raw)
    assert action.action == "SEARCH_CATEGORY" and box is None
    assert category_search_terms(action.value) == (category, query)
    with pytest.raises(ActionError, match="Classifieds adapter"):
        compile_action(action, *screen.size)
    assert discovery.decide(observation, actor, lambda *_: rows, None) == (None, None)
    assert actor.last_metrics["stop_after_evidence"] is True


def test_price_result_scope_checks_the_query_and_visible_category():
    task = price_request("Find the cheapest ring, and tell me how much it costs in dollars.")
    rows = [{"text": label} for label in (
        "Classifieds > Jewelry > Search results: ring", "1 - 12 of 20 listings",
    )]
    url = "http://localhost:19980/index.php?page=search&sPattern=ring&sCategory=1"
    assert price_result_scope(rows, task, "ring", url) == (1, 12, 20)
    assert price_result_scope(rows, task, "ring", url.replace("ring", "watch")) is None
    assert price_result_scope(rows[1:], task, "ring", url) is None


def test_price_match_requires_every_more_extreme_candidate_to_be_checked():
    assert first_ranked_match("[false,true,null]", 3) == (1, False)
    assert first_ranked_match("[null,true,false]", 3) == (None, False)
    assert first_ranked_match("[false,false,false]", 3) == (None, True)
    assert first_ranked_match("[0,true,false]", 3) == (None, False)
    assert first_ranked_match("[true]", 3) == (None, False)
