from pathlib import Path

import pytest
from PIL import Image


def row(text, bounds):
    return {"text": text, "bounds": bounds}


def test_varied_price_order_preserves_first_hash_match_for_each_amount(monkeypatch):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_vwa_catalog_holdout import Card
    from prepare_vwa_price_retry_holdout import candidate_order

    cards = [Card(str(index), "Listing", dollars, None) for index, dollars in enumerate((25, 60, 25, 60, 80))]
    original = candidate_order(cards, "new-cohort")
    diverse = candidate_order(cards, "new-cohort", varied_prices=True)
    assert len({card.dollars for card in diverse[:3]}) == 3
    for amount in (25, 60, 80):
        assert next(card for card in diverse if card.dollars == amount) == next(card for card in original if card.dollars == amount)
    assert set(diverse) == set(original)


@pytest.mark.parametrize(("cropped_text", "group"), (("80. 0.50 $", "trigger"), ("80.50 $", "control")))
def test_screening_uses_enlarged_ocr_without_checking_native_retry(monkeypatch, cropped_text, group):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_vwa_price_retry_holdout import screen_price

    site = "http://localhost:19980"
    image = Image.new("RGB", (1000, 770), "white")
    whole = [
        row("80.00", [791, 123, 884, 160]),
        row("$", [883, 126, 909, 158]),
        row("Published date: yesterday", [115, 187, 316, 203]),
        row("Location: Oak Park, Illinois, United States", [116, 207, 452, 224]),
    ]
    sizes = []

    def reader(screen, size):
        sizes.append(screen.size)
        return whole if screen is image else [row(cropped_text, [10, 3, 150, 64])]

    screened = screen_price(image, site + "/index.php?page=item&id=7", site, "Oak Park", 80, reader, image.size)
    assert screened["outcome"] == group
    assert sizes == [image.size, (252, 90)]


@pytest.mark.parametrize(("enlarged", "upright", "group"), (
    ("80. 0.00 $", "80. 0.00 $", "trigger"),
    ("80. 0.00 $", "80.00 $", "upright_control"),
    ("80. 0.00 $", "81.00 $", "upright_ocr_disagrees"),
    ("80.00 $", None, "control"),
    ("81.00 $", None, "enlarged_ocr_disagrees"),
))
def test_whole_line_cohort_screens_detector_reads_only(monkeypatch, enlarged, upright, group):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_vwa_price_retry_holdout import screen_price

    site = "http://localhost:19980"
    image = Image.new("RGB", (1000, 770), "white")
    whole = [
        row("80.00", [791, 123, 884, 160]),
        row("$", [883, 126, 909, 158]),
        row("Published date: yesterday", [115, 187, 316, 203]),
        row("Location: Oak Park, Illinois, United States", [116, 207, 452, 224]),
    ]

    class Detector:
        def __call__(self, screen, size):
            return whole if screen is image else [row(enlarged, [10, 3, 150, 64])]

        def read_upright(self, screen, size):
            assert upright is not None and screen.size == size == (134, 53)
            return [row(upright, [5, 2, 75, 32])]

        def read_line(self, screen, size):
            pytest.fail("The candidate recognizer must never select the holdout")

    screened = screen_price(image, site + "/index.php?page=item&id=7", site, "Oak Park", 80, Detector(), image.size,
                            screen_upright=True)
    assert screened["outcome"] == group
    assert ("upright_ocr" in screened) is (upright is not None)
