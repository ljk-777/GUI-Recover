from pathlib import Path

import pytest
from PIL import Image


def row(text, bounds):
    return {"text": text, "bounds": bounds}


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
