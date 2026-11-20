import pytest

from gui_recover.actions import ActionError
from gui_recover.visible_links import visible_listing_links


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
