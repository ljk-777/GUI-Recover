import pytest
from PIL import Image

from gui_recover.actions import ActionError
from gui_recover.listing_facts import LISTING_FACTS_ABSTAINED, ListingFacts, listing_fact_decision
from gui_recover.verification import pixel_hash


SITE = "http://localhost:19980"
ITEM = SITE + "/index.php?page=item&id=7"
DETAIL_GOAL = "What is the asking price in dollars and the city for the listing open in this tab?"
SEARCH_GOAL = 'Search for "lamp" and report the listing title and asking price in dollars of the most expensive result.'
SEARCH = SITE + "/index.php?page=search&sPattern=lamp"
SORTED = SEARCH + "&sOrder=i_price&iOrderType=desc&sShowAs=gallery"


def row(text, left, top, right, bottom):
    return {"text": text, "bounds": [left, top, right, bottom]}


def detail_rows():
    return [
        row("Classifieds > For sale > Brass lamp", 110, 96, 420, 113),
        row("Brass lamp", 110, 120, 300, 164),
        row("80.50", 791, 123, 884, 160),
        row("$", 883, 126, 909, 158),
        row("Published date: yesterday", 115, 187, 316, 203),
        row("Location: Oak Park, Illinois, United States", 116, 207, 452, 224),
        row("Contact publisher", 700, 375, 870, 400),
        row("120 $", 791, 600, 884, 618),
    ]


def test_current_listing_checks_price_and_city_from_the_header():
    screen = Image.new("RGB", (1000, 1000))
    rows = detail_rows()
    decision = listing_fact_decision(DETAIL_GOAL, ITEM, SITE, rows, screen.size, screen, None)
    assert decision.facts == ListingFacts("city", "Oak Park", 8050, (116, 123, 909, 224))
    verify = decision.facts.confirm
    assert verify('{"city":"Oak Park","dollars":"80.5"}') == ("Oak Park, 80.50 dollars", "evidence_verify_agreed")
    assert verify('{"city":"Illinois","dollars":"80.50"}') == (None, "evidence_verify_disagreed")
    assert verify('{"city":"Oak Park","dollars":"120"}') == (None, "evidence_verify_disagreed")
    assert verify('{"city":null,"dollars":"80.50"}') == (None, "evidence_verify_abstained")
    with pytest.raises(ActionError):
        verify('{"city":"Oak Park","dollars":80.5}')
    without_location = [line for line in rows if not line["text"].startswith("Location:")]
    assert listing_fact_decision(DETAIL_GOAL, ITEM, SITE, without_location, screen.size, screen, None).status == LISTING_FACTS_ABSTAINED


def test_keyword_results_bind_unique_highest_price_to_the_selected_detail():
    screen = Image.new("RGB", (1000, 1800))
    rows = [
        row("Classifieds > Search results: lamp", 110, 96, 420, 113),
        row("1 – 2 of 2 listings", 110, 120, 300, 141),
        row("Sort by: Newly listed", 710, 140, 940, 160),
        row("80.50 $", 100, 665, 143, 686),
        row("25 $", 320, 665, 363, 686),
    ]
    result = lambda url, capture=None: listing_fact_decision(SEARCH_GOAL, url, SITE, rows, screen.size, screen, capture)
    assert result(SITE + "/") is None
    assert result(SEARCH).action.action == "SORT_LISTINGS_PRICE"
    assert result(SEARCH + "&sCategory=3").status == LISTING_FACTS_ABSTAINED
    rows[2]["text"] = "Sort by: Higher price first"
    cards = [
        {"destination": ITEM, "image": [90, 400, 290, 610], "title": [91, 630, 220, 650]},
        {"destination": ITEM[:-1] + "8", "image": [310, 400, 510, 610], "title": [311, 630, 440, 650]},
    ]
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "cards": cards, "grid": [0.7, 0.1], "next": None}

    def capture():
        return receipt

    assert result(SORTED.replace("gallery", "list"), capture).action.action == "LISTING_GRID"
    select = result(SORTED, capture)
    assert select.action.action == "CLICK" and select.target == (ITEM, 8050)
    for target in (ITEM[:-1] + "8", ITEM):
        price = 8050 if target != ITEM else 2525
        decision = listing_fact_decision(SEARCH_GOAL, ITEM, SITE, detail_rows(), screen.size, screen, None, target=(target, price))
        assert decision.status == LISTING_FACTS_ABSTAINED
    confirmed = listing_fact_decision(SEARCH_GOAL, ITEM, SITE, detail_rows(), screen.size, screen, None, target=select.target)
    assert confirmed.facts.confirm('{"title":"Brass lamp","dollars":"80.50"}') == (
        "Brass lamp, 80.50 dollars", "evidence_verify_agreed",
    )
    rows[4]["text"] = "80.50 $"
    assert result(SORTED, capture).status == LISTING_FACTS_ABSTAINED
    receipt["cards"] = cards[1:]
    assert result(SORTED, capture).status == LISTING_FACTS_ABSTAINED
