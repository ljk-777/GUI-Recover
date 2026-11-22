from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from gui_recover.actions import Action, ActionError, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.environment import VisualObservation
from gui_recover.listing_facts import LISTING_FACTS_ABSTAINED, listing_fact_decision
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
    assert decision.facts.label_bounds == (116, 207, 452, 224)
    assert decision.facts.price_bounds == (791, 123, 909, 160)
    assert decision.facts.bounds == (116, 123, 909, 224)
    verify = decision.facts.confirm
    assert verify('{"city":"Oak Park","dollars":"80.5"}') == ("Oak Park, 80.50 dollars", "evidence_verify_agreed")
    assert verify('{"city":"Oak Park","dollars":80.50}') == ("Oak Park, 80.50 dollars", "evidence_verify_agreed")
    assert verify('{"city":"Oak Park","dollars":80.50000000000001}') == (None, "evidence_verify_disagreed")
    assert verify('{"city":"Illinois","dollars":"80.50"}') == (None, "evidence_verify_disagreed")
    assert verify('{"city":"Oak Park","dollars":"120"}') == (None, "evidence_verify_disagreed")
    assert verify('{"city":null,"dollars":"80.50"}') == (None, "evidence_verify_abstained")
    with pytest.raises(ActionError):
        verify('{"city":"Oak Park","dollars":true}')
    without_location = [line for line in rows if not line["text"].startswith("Location:")]
    assert listing_fact_decision(DETAIL_GOAL, ITEM, SITE, without_location, screen.size, screen, None).status == LISTING_FACTS_ABSTAINED


@pytest.mark.parametrize("reread_city", ("Oak Park", "Oak"))
def test_compound_disagreement_rereads_only_the_two_fields_once(reread_city):
    screen = Image.new("RGB", (1000, 770), "white")
    draw = ImageDraw.Draw(screen)
    draw.rectangle((213, 209, 248, 222), fill="blue")
    draw.rectangle((815, 130, 880, 153), fill="red")
    draw.rectangle((460, 168, 512, 196), fill="lime")
    actor = SimpleNamespace(prepare_image=lambda image: image)
    calls = []

    def ask(image, prompt, **context):
        calls.append(context)
        actor.last_metrics = {"model_invocations": 1}
        if len(calls) == 1:
            assert context["observation"].tabs
            return '{"city":"Oak","dollars":80.50}'
        assert not context and "entire upper strip" in prompt and "Oak" not in prompt
        assert image.getchannel("G").getextrema()[0] < 255
        assert image.getcolors(256) is not None
        assert (0, 255, 0) not in {color for _, color in image.getcolors(256)}
        return f'{{"text":"Location: {reread_city}, Illinois, United States","dollars":80.50}}'

    actor.ask = ask
    def reader(image, size):
        return detail_rows() if image is screen else [row("80.50 $", 10, 3, 150, 64)]

    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=reader,
                                  listing_site=SITE, ranked_capture=lambda: pytest.fail("Detail page needs no cards"))
    observation = VisualObservation(DETAIL_GOAL, screen, tabs=(("Detail", ITEM),))
    assert policy.predict(observation, []) is None
    assert policy.predict(observation, []) is None
    assert policy.last_metrics["listing_fact_reread_pending"] is True
    result = policy.predict(observation, [])
    assert len(calls) == 2 and policy.pending_fact_reread is None
    if reread_city == "Oak Park":
        assert parse_action(result) == Action("ANSWER", "Oak Park, 80.50 dollars")
    else:
        assert result is None and policy.last_metrics["stop_after_evidence"] is True


@pytest.mark.parametrize(("initial", "isolated", "localized_price", "native_price", "line_price", "submitted_price", "model_calls"), (
    ("80.50", "80.50", "80.50 $", None, None, "80.50", 2),
    ("90.50", "80.50", "80.50 $", None, None, "80.50", 2),
    ("80.50", "90.50", "80.50 $", None, "80.50 $", None, 2),
    ("80.50", "80.50", "120 $", None, "80.50 $", None, 2),
    ("80.50", "80.50", None, None, None, None, 1),
    ("90.50", "90.50", "90.50 $", None, "90.50 $", "90.50", 1),
    ("90.50", "80.50", "80. 0.50 $", "80.50 $", None, "80.50", 2),
    ("90.50", "90.50", "90. 0.50 $", "90.50 $", "80.50 $", "90.50", 1),
    ("90.50", None, "90. 0.50 $", None, "90.50 $", "90.50", 1),
    ("90.50", None, None, None, "80.50 $", None, 1),
    ("80.50", None, None, None, "90.50 $", None, 1),
    ("80.50", None, None, None, "80.50 $", None, 1),
    ("90.50", None, None, None, "", None, 1),
))
def test_city_price_requires_local_ocr_agreement(initial, isolated, localized_price, native_price, line_price,
                                                 submitted_price, model_calls):
    screen = Image.new("RGB", (1000, 770), "white")
    draw = ImageDraw.Draw(screen)
    draw.rectangle((820, 132, 871, 149), fill="red")
    draw.rectangle((246, 210, 294, 220), fill="blue")
    rows = detail_rows()
    rows[2]["text"] = "90.50"
    ocr_sizes = []

    def reader(image, size):
        ocr_sizes.append(image.size)
        if image is screen:
            return rows
        assert size == image.size == (252, 90)
        assert image.getpixel((100, 40)) == (255, 0, 0)
        assert not any(color == (0, 0, 255) for _, color in image.getcolors(256))
        return [row(localized_price, 10, 3, 150, 64)] if localized_price else []

    def read_upright(image, size):
        ocr_sizes.append(image.size)
        assert size == image.size == (134, 53)
        return [row(native_price, 5, 2, 75, 32)] if native_price else []

    def read_line(image, size):
        ocr_sizes.append(image.size)
        assert size == image.size == (134, 53)
        return [row(line_price, 0, 0, *size)] if line_price else []

    reader.read_upright = read_upright
    if line_price is not None:
        reader.read_line = read_line

    actor = SimpleNamespace(prepare_image=lambda image: image)
    calls = []

    def ask(image, prompt, **context):
        calls.append(context)
        actor.last_metrics = {"model_invocations": 1}
        return (f'{{"city":"Oak Park","dollars":"{initial}"}}' if len(calls) == 1 else
                f'{{"text":"Location: Oak Park, Illinois, United States","dollars":"{isolated}"}}')

    actor.ask = ask
    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=reader, listing_site=SITE,
                                  ranked_capture=lambda: pytest.fail("Detail page needs no cards"))
    observation = VisualObservation(DETAIL_GOAL, screen, tabs=(("Detail", ITEM),))
    assert policy.predict(observation, []) is None
    result = policy.predict(observation, [])
    assert policy.last_metrics["listing_price_check"]["bounds"] == [787, 119, 913, 164]
    retried = localized_price is None or " 0." in localized_price
    assert ("listing_price_upright_retry" in policy.last_metrics) is retried
    line_retried = retried and native_price is None and line_price is not None
    assert ("listing_price_line_retry" in policy.last_metrics) is line_retried
    if line_retried:
        assert policy.last_metrics["listing_price_line_retry"]["pixels_sha256"] == (
            policy.last_metrics["listing_price_upright_retry"]["pixels_sha256"])
    if model_calls == 2:
        assert result is None and policy.last_metrics["listing_fact_reread_pending"] is True
        result = policy.predict(observation, [])
    assert len(calls) == model_calls and len(ocr_sizes) == 2 + retried + line_retried
    if submitted_price:
        assert parse_action(result) == Action("ANSWER", f"Oak Park, {submitted_price} dollars")
    else:
        assert result is None and policy.last_metrics["stop_after_evidence"] is True


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
    assert confirmed.facts.confirm('{"letters":"B r a s s | l a m p","dollars":80.5}', reread=True) == (
        "Brass lamp, 80.50 dollars", "evidence_verify_agreed",
    )
    assert confirmed.facts.confirm('{"letters":"B r a s s l a m p","dollars":80.5}', reread=True) == (
        "Brass lamp, 80.50 dollars", "evidence_verify_agreed",
    )
    assert confirmed.facts.confirm('{"letters":"B r a s | s l a m p","dollars":80.5}', reread=True) == (
        None, "evidence_verify_disagreed",
    )
    rows[4]["text"] = "80.50 $"
    assert result(SORTED, capture).status == LISTING_FACTS_ABSTAINED
    receipt["cards"] = cards[1:]
    assert result(SORTED, capture).status == LISTING_FACTS_ABSTAINED
