"""A bounded listing-link preview at screenshot positions."""

import json
import re
from math import isfinite
from urllib.parse import parse_qs, urlsplit

from .actions import Action, ActionError, parse_action
from .qwen_grounding import read_reply
from .screen_text import from_readable_control


_LINK = re.compile(r"\b(?:hyperlinks?|links?|urls?)\b", re.IGNORECASE)
_LINK_COUNT = re.compile(
    r"\b(?:the\s+)?(?:top\s+|first\s+|last\s+)?(one|two|three|[1-9])\s+"
    r"(?:(?:cheapest|newest|oldest|most|least|recently|listed|recent|expensive|matching|relevant|available)\s+)*"
    r"(?:listings?|items?|products?|results?|posts?|links?|urls?)\b", re.IGNORECASE,
)
_AMOUNT = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]{2})?"
_PRICE = re.compile(rf"^(?:\$\s*{_AMOUNT}|{_AMOUNT}\s*\$)$")


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
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
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


def link_answer_count(goal):
    """Return the supported count interval for an explicit link question."""
    request = _LINK.search(goal)
    if request is None:
        return None
    explicit = _LINK_COUNT.search(goal)
    if explicit:
        value = explicit[1].casefold()
        n = {"one": 1, "two": 2, "three": 3}[value] if value.isalpha() else int(value)
        return n, n
    return (2, 3) if request[0].casefold().endswith("s") else (1, 1)


def priced_titles(rows, size):
    """Pair a visible price with the single aligned text line immediately above it."""
    width, height = size
    matches = {}
    for price, row in enumerate(rows):
        if _PRICE.fullmatch(row["text"]) is None:
            continue
        left, top, _, _ = row["bounds"]
        candidates = [index for index, title in enumerate(rows) if
                      abs(left - title["bounds"][0]) <= max(4, width // 100)
                      and 0 <= top - title["bounds"][3] <= max(4, height // 30)
                      and _PRICE.fullmatch(title["text"]) is None]
        if len(candidates) == 1:
            matches[price] = candidates[0]
    return matches


def _title_text(value):
    return " ".join(value.casefold().split()).rstrip(".…").rstrip()


def listing_selection(raw, rows, size, count):
    """Resolve selected visible text centers to normalized browser points."""
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "listings" not in data:
        return None
    if set(data) != {"listings"}:
        raise ActionError("Expected only listing indices")
    selected = data["listings"]
    if selected is None:
        return [], []
    if not isinstance(selected, list) or not count[0] <= len(selected) <= min(count[1], 3):
        raise ActionError("Expected the required number of distinct visible titles")
    prices = priced_titles(rows, size)
    indices = []
    for value in selected:
        if type(value) is int and 0 <= value < len(rows):
            indices.append(prices.get(value, value))
            continue
        if isinstance(value, str) and 0 < len(value.strip()) <= 100:
            label = _title_text(value)
            matches = [index for index, row in enumerate(rows) if label and _title_text(row["text"]) == label]
            if len(matches) == 1:
                index = matches[0]
                indices.append(prices.get(index, index))
                continue
        raise ActionError("Listing choice is neither a visible index nor a unique rendered title")
    if len(set(indices)) != len(indices):
        raise ActionError("Multiple selections point to the same listing title")
    return indices, normalized_points([
        from_readable_control(json.dumps({"action": "CLICK", "value": None, "control": index}), rows, size).position
        for index in indices
    ])


def preview_destinations(receipt, frame, points, site):
    """Check provenance and preserve missing or duplicate destinations for abstention."""
    parts = urlsplit(site)
    origin = parts.scheme, parts.netloc
    if (not isinstance(receipt, dict) or set(receipt) != {"source_pixels", "links"}
            or receipt["source_pixels"] != frame
            or not isinstance(receipt["links"], list) or len(receipt["links"]) != len(points)):
        raise ActionError("Link preview does not belong to the selected viewport")
    destinations = []
    for row, point in zip(receipt["links"], points, strict=True):
        if (not isinstance(row, dict) or set(row) != {"position", "destination"}
                or normalized_points([row["position"]]) != [point]):
            raise ActionError("Preview result differs from the requested location")
        url = row["destination"]
        if url is not None and listing_destination(url, origin) != url:
            raise ActionError("Preview result is not a local listing URL")
        destinations.append(url)
    return destinations if None not in destinations and len(set(destinations)) == len(destinations) else None


def agreed_listing_answer(selected, destinations, confirmed):
    if not confirmed or set(confirmed) != set(selected):
        return None
    links = dict(zip(selected, destinations, strict=True))
    answer = Action("ANSWER", "\n".join(links[index] for index in confirmed))
    return parse_action(json.dumps(answer.to_dict()))
