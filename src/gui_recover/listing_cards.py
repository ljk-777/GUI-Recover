"""Read and inspect visible listing cards without exposing page contents to the model."""

import re
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .actions import Action, ActionError
from .qwen_grounding import read_reply
from .screen_text import narrow_keyword_search
from .verification import pixel_hash
from .visible_links import link_answer_count, listing_destination


MAX_VISIBLE_CARDS = 12
RESULT_VIEW_HEIGHT = 1800
_SEARCH = re.compile(r'^\s*search for\s+"([^"\n]{1,80})"\s+and\b', re.IGNORECASE)
_RECENT = re.compile(r"\b(?:most recent|most recently|newest|latest)\b", re.IGNORECASE)
_APPEARANCE = re.compile(r"\bthat\s+(?:feature|depict|show)\s+([^.!?]{1,100})[.!?]?\s*$", re.IGNORECASE)
_RANGE = re.compile(r"\b([\d,]+)\s*[-–]\s*([\d,]+)\s+of\s+([\d,]+)\s+listings\b", re.IGNORECASE)
_TITLE = re.compile(r"\b(?:title|listing name)\b", re.IGNORECASE)
_EXACT_ITEM = re.compile(r"\bexact (?:item|product|listing)\b", re.IGNORECASE)
_MULTIPLE_ITEMS = re.compile(r"\b(?:all listings|compare|both|every listing)\b", re.IGNORECASE)

_VISIBLE_CARDS = """
({width, height, limit}) => {
    const box = (rect) => rect.width > 0 && rect.height > 0 && rect.left >= 0 && rect.top >= 0
        && rect.right <= innerWidth && rect.bottom <= innerHeight
        ? [Math.floor(rect.left * width / innerWidth), Math.floor(rect.top * height / innerHeight),
           Math.ceil(rect.right * width / innerWidth), Math.ceil(rect.bottom * height / innerHeight)]
        : null;
    const hit = (rect) => document.elementFromPoint((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2);
    const position = (anchor) => {
        if (!anchor) return null;
        const rect = anchor.getBoundingClientRect();
        return box(rect) && hit(rect)?.closest('a[href]') === anchor
            ? [(rect.left + rect.right) / 2 / innerWidth, (rect.top + rect.bottom) / 2 / innerHeight] : null;
    };
    const cards = [];
    for (const card of document.querySelectorAll('li.listing-card')) {
        const anchor = card.querySelector('a.title[href]');
        if (!anchor) continue;
        const titleRect = anchor.getBoundingClientRect();
        const title = box(titleRect);
        if (!title || hit(titleRect)?.closest('a[href]') !== anchor) continue;
        const photo = card.querySelector('img');
        const rect = photo?.getBoundingClientRect();
        const image = rect && hit(rect) === photo ? box(rect) : null;
        cards.push({destination: anchor.href, title, image});
        if (cards.length > limit) break;
    }
    const current = new URL(location.href);
    const expected = Number(current.searchParams.get('iPage') || 1) + 1;
    current.searchParams.delete('iPage');
    current.searchParams.sort();
    current.hash = '';
    const next = Array.from(document.querySelectorAll('a[href]')).find(anchor => {
        if (anchor.textContent.trim() !== String(expected) || !position(anchor)) return false;
        const other = new URL(anchor.href);
        if (Number(other.searchParams.get('iPage')) !== expected) return false;
        other.searchParams.delete('iPage');
        other.searchParams.sort();
        other.hash = '';
        return other.href === current.href;
    });
    return {scroll_top: Math.round(scrollY), cards,
            grid: position(document.querySelector('a.grid-button')), next: position(next)};
}
"""


def ranked_photo_request(goal):
    """Read the literal query and the photo criterion of a supported ranked request."""
    search, appearance = _SEARCH.match(goal), _APPEARANCE.search(goal)
    count = link_answer_count(goal)
    if (search is not None and appearance is not None and _RECENT.search(goal)
            and count is not None and count[0] == count[1] <= 3):
        return search[1], appearance[1].strip(), count[0]
    return None


def visible_result_range(rows, query):
    labels = [row["text"].casefold() for row in rows]
    if (not any(label.endswith(f"search results: {query.casefold()}") for label in labels)
            or not any("sort by:" in label and label.endswith("newly listed") for label in labels)):
        return None
    return listing_range(rows)


def listing_range(rows):
    ranges = [match for row in rows if (match := _RANGE.search(row["text"]))]
    if len(ranges) != 1:
        return None
    first, last, total = (int(value.replace(",", "")) for value in ranges[0].groups())
    return (first, last, total) if 1 <= first <= last <= total else None


def _box(value, size):
    if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int for item in value):
        raise ActionError("Invalid visible card bounds")
    left, top, right, bottom = value
    if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
        raise ActionError("A visible card box must be inside the screenshot")
    return tuple(value)


def capture_listing_cards(page, site, screen):
    """Return bounded visible title and thumbnail geometry in screenshot pixels."""
    origin = urlsplit(site)
    current = urlsplit(page.url)
    if origin.scheme not in ("http", "https") or (current.scheme, current.netloc) != (origin.scheme, origin.netloc):
        raise ActionError("Card observation requires the registered test origin")
    data = page.evaluate(_VISIBLE_CARDS, {"width": screen.width, "height": screen.height, "limit": MAX_VISIBLE_CARDS})
    if len(data["cards"]) > MAX_VISIBLE_CARDS:
        data["cards"] = []
    data["cards"] = [card for card in data["cards"]
                     if listing_destination(card["destination"], (origin.scheme, origin.netloc))]
    return {"source_pixels": pixel_hash(screen), **data}


def inspect_card_receipt(receipt, frame, size, site):
    if (not isinstance(receipt, dict) or set(receipt) != {"source_pixels", "scroll_top", "cards", "grid", "next"}
            or receipt["source_pixels"] != frame or type(receipt["scroll_top"]) is not int
            or receipt["scroll_top"] < 0 or not isinstance(receipt["cards"], list)
            or len(receipt["cards"]) > MAX_VISIBLE_CARDS):
        raise ActionError("Card receipt does not describe the current screenshot")
    from .visible_links import normalized_points

    for key in ("grid", "next"):
        if receipt[key] is not None:
            normalized_points([receipt[key]])
    parts = urlsplit(site)
    seen = set()
    for card in receipt["cards"]:
        if not isinstance(card, dict) or set(card) != {"destination", "image", "title"}:
            raise ActionError("Invalid visible card")
        url = card["destination"]
        if listing_destination(url, (parts.scheme, parts.netloc)) != url or url is None or url in seen:
            raise ActionError("Expected distinct same-origin listing destinations")
        seen.add(url)
        _box(card["title"], size)
        if card["image"] is not None:
            _box(card["image"], size)
    return receipt["cards"]


def reference_title_request(goal):
    return bool(_TITLE.search(goal))


def title_point(card, size):
    left, top, right, bottom = card["title"]
    return ((left + right) / (2 * size[0]), (top + bottom) / (2 * size[1]))


def reference_result_destination(receipt, size, action):
    """Bind the selected visible title to its recorded destination."""
    matches = [card["destination"] for card in receipt["cards"] if title_point(card, size) == action.position]
    return matches[0] if len(matches) == 1 else None


def reference_result_action(screen, reference, receipt, site, rows, query, goal, *, refinement_size=None):
    """Expand, inspect or refine a verified first page of visible results."""
    if _EXACT_ITEM.search(goal) is None or _MULTIPLE_ITEMS.search(goal) is not None:
        return None
    scope = visible_result_range(rows, query)
    if scope is None or scope[0] != 1 or receipt["scroll_top"] != 0:
        return None
    cards = inspect_card_receipt(receipt, pixel_hash(screen), screen.size, site)
    if len(cards) != scope[1] or any(card["image"] is None for card in cards):
        if screen.height < RESULT_VIEW_HEIGHT and receipt["grid"] is not None:
            return Action("LISTING_GRID", position=tuple(receipt["grid"]))
        return None

    def pixels(image):
        # Classifieds thumbnails can stretch an entire photo to their fixed card aspect ratio.
        scaled = image.convert("RGB").resize((48, 48), Image.Resampling.LANCZOS)
        return np.asarray(scaled, dtype=np.float32) / 255

    wanted = pixels(reference)
    if float(wanted.std()) < 0.04:
        return None
    distances = [float(np.abs(wanted - pixels(screen.crop(card["image"]))).mean()) for card in cards]
    matches = [index for index, distance in enumerate(distances) if distance <= 0.055]
    if any(0.055 < distance < 0.105 for distance in distances):
        return None
    if not matches:
        return narrow_keyword_search(query, rows, refinement_size) if refinement_size is not None else None
    if len(matches) > 1 and _RECENT.search(goal) is None:
        return None
    return Action("CLICK", position=title_point(cards[matches[0]], screen.size))


def reference_title_region(screen, reference, receipt, site):
    """Localize a title only when the image has a strong, isolated visible match."""
    def pixels(image):
        return np.asarray(ImageOps.fit(image.convert("RGB"), (48, 48), method=Image.Resampling.LANCZOS),
                          dtype=np.float32) / 255

    wanted = pixels(reference)
    if float(wanted.std()) < 0.04:
        return None
    cards = inspect_card_receipt(receipt, pixel_hash(screen), screen.size, site)
    distances = [(float(np.abs(wanted - pixels(screen.crop(card["image"]))).mean()), index)
                 for index, card in enumerate(cards) if card["image"] is not None]
    distances.sort()
    if len(distances) < 2 or distances[0][0] > 0.055 or distances[1][0] - distances[0][0] < 0.05:
        return None
    card = cards[distances[0][1]]
    photo, title = card["image"], card["title"]
    # Keep the matched photo alongside its title so the independent reader can confirm identity.
    return (max(0, min(photo[0], title[0]) - 6), max(0, min(photo[1], title[1]) - 6),
            min(screen.width, max(photo[2], title[2]) + 6), min(screen.height, max(photo[3], title[3]) + 6))


def card_panel(items):
    """Lay out native screenshot crops with explicit independent labels."""
    width, height = 332, 250
    columns = min(len(items), 2)
    rows = (len(items) + columns - 1) // columns
    panel = Image.new("RGB", (width * columns, height * rows), "#eeeeee")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=20)
    for index, (screen, card) in enumerate(items):
        x, y = index % columns * width, index // columns * height
        draw.rectangle((x + 3, y + 3, x + width - 4, y + height - 4), fill="white")
        draw.text((x + 12, y + 9), f"Card {index + 1}", font=font, fill="black")
        photo = ImageOps.contain(screen.crop(card["image"]).convert("RGB"), (308, 170), Image.Resampling.LANCZOS)
        panel.paste(photo, (x + (width - photo.width) // 2, y + 37))
        title = ImageOps.contain(screen.crop(card["title"]).convert("RGB"), (308, 25), Image.Resampling.LANCZOS)
        panel.paste(title, (x + 12, y + 215))
    return panel


def read_card_checks(raw, count, *, allow_ordered=False):
    """Accept a result only when every numbered crop received its own bounded cue."""
    data = read_reply(raw)
    if (allow_ordered or count == 1) and isinstance(data, list) and len(data) == count:
        data = {str(index): verdict for index, verdict in enumerate(data, 1)}
    if not isinstance(data, dict) or set(data) != {str(index) for index in range(1, count + 1)}:
        raise ActionError("Each image must have one indexed judgment")
    matches = []
    for index in range(1, count + 1):
        value = data[str(index)]
        if value is None:
            matches.append(None)
            continue
        if not isinstance(value, list) or len(value) != 2:
            raise ActionError("A card judgment needs a boolean and visual cue, or null")
        match, cue = value
        if (allow_ordered or count == 1) and isinstance(match, str) and match in ("true", "false"):
            match = match == "true"
        if type(match) is not bool:
            raise ActionError("A card judgment needs a boolean")
        if not (cue is None and not match) and (not isinstance(cue, str) or not 0 < len(cue.strip()) <= 96):
            raise ActionError("A matching card needs a bounded visual cue")
        matches.append(match)
    return matches
