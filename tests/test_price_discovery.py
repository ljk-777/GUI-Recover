import json
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_recover.actions import ActionError, category_search_terms, compile_action, parse_action
from gui_recover.environment import VisualObservation
from gui_recover.price_discovery import (
    PriceDiscovery, _detail_price_box, first_ranked_match, price_request, price_result_scope,
)
from gui_recover.verification import pixel_hash


@pytest.mark.parametrize(("subject", "category", "query"), [
    ("purple hard-case book", "Books", ""),
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
    assert first_ranked_match('{"1":[false,"box"],"2":[true,"hand"],"3":null}', 3) == (1, False)
    assert first_ranked_match('{"1":null,"2":[true,"hand"],"3":[false,"box"]}', 3) == (None, False)
    assert first_ranked_match('{"1":[false,"table"],"2":[false,"box"]}', 2) == (None, True)
    assert first_ranked_match('{"1":[0,"table"],"2":[true,"hand"]}', 2) == (None, False)
    assert first_ranked_match('{"1":[true,"hand"]}', 3) == (None, False)
    assert first_ranked_match('[["false","box"],["true","hand"]]', 2) == (1, False)
    assert first_ranked_match('[["false","box"],["true","hand"]]', 3) == (None, False)
    assert first_ranked_match('[[false,"box"],["true",""]]', 2) == (None, False)


def test_price_discovery_keeps_order_across_batches_and_crops_the_opened_listing():
    site = "http://localhost:19980"
    goal = "Find the most expensive ring that has an image taken as a selfie, and tell me how much it costs in dollars."
    screen = Image.new("RGB", (400, 160), "white")
    cards = [
        {"destination": f"{site}/index.php?page=item&id={index + 1}", "image": [x, 40, x + 35, 80],
         "title": [x, 87, x + 37, 97]}
        for index, x in enumerate(range(10, 361, 55))
    ]
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "cards": cards, "grid": None, "next": None}
    rows = [{"text": label, "bounds": [1, 1, 25, 6]} for label in (
        "Classifieds > Jewelry > Search results: ring", "1 - 7 of 7 listings", "Sort by: Higher price first",
    )]
    skipped = json.dumps({str(i): [False, "box"] for i in range(1, 7)})
    replies = iter((skipped, '{"1":[true,"hand"]}'))
    actor = SimpleNamespace(prepare_image=lambda image: image, ask=lambda *_args, **_kwargs: next(replies), last_metrics={})
    discovery = PriceDiscovery(price_request(goal), site)
    url = site + "/index.php?page=search&sPattern=ring&sCategory=1&sOrder=i_price&iOrderType=desc"
    results = VisualObservation(goal, screen, tabs=(("Classifieds", url),))

    assert discovery.decide(results, actor, lambda *_: rows, lambda: receipt) == (None, None)
    assert actor.last_metrics["provider_format"] == "price_more_cards"
    raw, box = discovery.decide(results, actor, lambda *_: rows, lambda: receipt)
    assert parse_action(raw).action == "CLICK" and box is None
    assert discovery.destination == cards[6]["destination"]

    price_bounds = [310, 15, 365, 28]
    detail_rows = [
        {"text": "345.00 $", "bounds": price_bounds},
        {"text": "Published date: 2023/11/10", "bounds": [35, 46, 260, 54]},
        {"text": "Contact publisher", "bounds": [307, 56, 380, 64]},
    ]
    detail = VisualObservation(goal, screen, tabs=(("Selected item", discovery.destination),))
    assert discovery.decide(detail, actor, lambda *_: detail_rows, None) == (None, price_bounds)
    assert actor.last_metrics["provider_format"] == "price_region"


def test_price_discovery_rejects_duplicate_cards_after_turning_the_page():
    site = "http://localhost:19980"
    goal = "Find the most expensive ring that has an image taken as a selfie, and tell me how much it costs in dollars."
    screen = Image.new("RGB", (120, 110), "white")
    repeated = {"destination": site + "/index.php?page=item&id=1", "image": [10, 15, 60, 75],
                "title": [12, 77, 63, 91]}
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "cards": [repeated],
               "grid": None, "next": [0.75, 0.85]}
    actor = SimpleNamespace(prepare_image=lambda image: image, ask=lambda *_: '{"1":[false,"ring on a box"]}',
                            last_metrics={})
    discovery = PriceDiscovery(price_request(goal), site)
    url = site + "/index.php?page=search&sPattern=ring&sCategory=1&sOrder=i_price&iOrderType=desc"

    def rows(first):
        return [{"text": label, "bounds": [1, 1, 25, 6]} for label in (
            "Classifieds > Jewelry > Search results: ring", "Sort by: Higher price first",
            f"{first} - {first} of 2 listings",
        )]

    first_page = VisualObservation(goal, screen, tabs=(("Classifieds", url),))
    raw, box = discovery.decide(first_page, actor, lambda *_: rows(1), lambda: receipt)
    assert box is None and parse_action(raw).position == (0.75, 0.85)
    assert actor.last_metrics["provider_format"] == "price_next_page"
    next_page = VisualObservation(goal, screen, tabs=(("Classifieds", url + "&iPage=2"),))
    assert discovery.decide(next_page, actor, lambda *_: rows(2), lambda: receipt) == (None, None)
    assert actor.last_metrics["stop_after_evidence"] is True


def test_detail_price_crop_joins_the_visible_amount_and_currency_symbol():
    rows = [
        {"text": "8000.00", "bounds": [474, 79, 554, 99]},
        {"text": "$", "bounds": [553, 81, 567, 98]},
        {"text": "Published date: 2023/11/12", "bounds": [76, 143, 176, 154]},
        {"text": "Contact publisher", "bounds": [426, 148, 516, 162]},
        {"text": "2000.00 $", "bounds": [79, 690, 123, 704]},
    ]
    assert _detail_price_box(rows) == [474, 79, 567, 99]
    assert _detail_price_box([rows[0], *rows[2:]]) is None
    rows[1] = {"text": "$", "bounds": [588, 79, 603, 99]}
    assert _detail_price_box(rows) is None


@pytest.mark.parametrize("missing_known_listing", [False, True])
def test_tie_recovery_checks_exact_price_coverage_before_resuming_ranked_results(missing_known_listing):
    site = "http://localhost:19980"
    goal = "Find the most expensive book with a purple cover, and tell me how much it costs in dollars."
    screen = Image.new("RGB", (360, 150))
    url = site + "/index.php?page=search&sCategory=9&sOrder=i_price&iOrderType=desc&sShowAs=gallery"
    next_url = url + "&iPage=2"
    exact_url = (site + "/index.php?page=search&sCategory=9&sPriceMin=100&sPriceMax=100"
                 "&sOrder=dt_pub_date&iOrderType=desc&sShowAs=gallery")
    batches = []

    def ask(_panel, prompt):
        count = int(prompt.split("keys 1 through ")[1].split(".")[0])
        batches.append(count)
        return json.dumps({str(index): [False, "blue cover"] for index in range(1, count + 1)})

    actor = SimpleNamespace(prepare_image=lambda image: image, ask=ask, last_metrics={})
    discovery = PriceDiscovery(price_request(goal), site)

    def decide(address, result_range, listings, *, tie=False, next_page=False):
        order = "Newly listed" if tie else "Higher price first"
        rows = [{"text": value, "bounds": [1, 1, 12, 5]} for value in (
            "Classifieds > Books", result_range, f"Sort by: {order}", "Price", "Min.", "Max.", "Apply",
        )]
        cards = []
        for index, (listing_id, dollars) in enumerate(listings):
            x = 15 + 85 * index
            cards.append({"destination": f"{site}/index.php?page=item&id={listing_id}",
                          "image": [x, 20, x + 70, 76], "title": [x + 2, 78, x + 65, 90]})
            rows.append({"text": f"{dollars}.00 $", "bounds": [x + 3, 94, x + 36, 106]})
        receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "cards": cards,
                   "grid": None, "next": [0.8, 0.9] if next_page else None}
        observation = VisualObservation(goal, screen, tabs=(("Classifieds", address),))
        raw, box = discovery.decide(observation, actor, lambda *_: rows, lambda: receipt)
        assert box is None
        return parse_action(raw) if raw is not None else None

    assert decide(url, "1 - 2 of 5 listings", [(1, 120), (2, 100)], next_page=True).action == "CLICK"
    next_cards = [(2, 100), (3, 100), (4, 99)]
    filter_action = decide(next_url, "3 - 5 of 5 listings", next_cards)
    assert (filter_action.action, filter_action.value) == ("FILTER_TIED_PRICE", "100")
    assert batches == [2]
    group = [(2, 100), (5, 100), (6 if missing_known_listing else 3, 100)]
    returned = decide(exact_url, "1 - 3 of 3 listings", group, tie=True)
    assert batches == [2, 2]
    if missing_known_listing:
        assert returned is None and actor.last_metrics["stop_after_evidence"]
    else:
        assert returned.action == "RETURN_PRICE_RESULTS"
        assert actor.last_metrics["price_tie_count"] == 3
        assert decide(next_url, "3 - 5 of 5 listings", next_cards) is None
        assert batches == [2, 2, 1]
