"""A bounded listing-link preview at screenshot positions."""

from math import isfinite
from urllib.parse import parse_qs, urlsplit

from .actions import ActionError


_TOOLTIP_DESTINATIONS = """
points => points.map(([x, y]) => {
    const node = document.elementFromPoint(x * window.innerWidth, y * window.innerHeight);
    const link = node && node.closest("a[href]");
    return link ? link.href : null;
})
"""


def normalized_points(positions):
    if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= 3:
        raise ActionError("Preview one to three visible listing locations")
    points = []
    for point in positions:
        if (not isinstance(point, (tuple, list)) or len(point) != 2
                or any(type(value) not in (int, float) or not 0 <= value < 1 or not isfinite(value) for value in point)):
            raise ActionError("Link preview needs a point inside the viewport")
        points.append(tuple(map(float, point)))
    if len(set(points)) != len(points):
        raise ActionError("Repeated link preview location")
    return points


def listing_destination(url, origin):
    if not isinstance(url, str) or len(url) > 2048:
        return None
    parsed = urlsplit(url)
    if ((parsed.scheme, parsed.netloc) != origin or parsed.username is not None
            or parsed.password is not None):
        return None
    fields = parse_qs(parsed.query)
    listing_id = fields.get("id", [])
    if fields.get("page") != ["item"] or len(listing_id) != 1 or not listing_id[0].isascii() or not listing_id[0].isdigit():
        return None
    return parsed.geturl()


def visible_listing_links(page, site, positions):
    """Return the href of a visible listing link under each supplied point."""
    origin_parts = urlsplit(site)
    origin = origin_parts.scheme, origin_parts.netloc
    current = urlsplit(page.url)
    if origin[0] not in ("http", "https") or (current.scheme, current.netloc) != origin:
        raise ActionError("Link preview requires the registered test origin")
    points = normalized_points(positions)
    raw = page.evaluate(_TOOLTIP_DESTINATIONS, [list(point) for point in points])
    if not isinstance(raw, list) or len(raw) != len(points):
        raise ActionError("Browser returned an unexpected number of links")
    return [{"position": list(point), "destination": listing_destination(url, origin)}
            for point, url in zip(points, raw, strict=True)]
