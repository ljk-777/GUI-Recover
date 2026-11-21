"""Ground compound Classifieds answers in a single matched listing."""

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from .actions import Action, ActionError
from .listing_cards import (
    RESULT_VIEW_HEIGHT, inspect_card_receipt, listing_range, listing_title_heading, title_point,
)
from .price_discovery import _card_prices, _detail_price_box, _displayed_cents, _point, _unique, same_origin
from .qwen_grounding import read_reply
from .verification import pixel_hash
from .visible_links import listing_destination


LISTING_FACTS_REGION = "evidence_listing_facts_region"
LISTING_FACTS_ABSTAINED = "evidence_listing_facts_abstained"
SEARCH_PRICE_NAVIGATION = "evidence_search_price_navigation"
LISTING_FACTS_STAGES = (LISTING_FACTS_REGION, LISTING_FACTS_ABSTAINED, SEARCH_PRICE_NAVIGATION)
_DETAIL_PRICE_CITY = re.compile(
    r"^\s*what is the (?:asking )?price in dollars and (?:the )?city for the listing "
    r"(?:open|shown) in (?:this|the current) tab[.?]?\s*$", re.I,
)
_SEARCH_HIGHEST = re.compile(
    r'^\s*search for "([^"\n]{1,80})" and (?:report|give me) the listing title and '
    r"(?:asking )?price in dollars of the most expensive result[.!?]?\s*$", re.I,
)


@dataclass(frozen=True)
class ListingFacts:
    field: str
    text: str
    cents: int
    bounds: tuple[int, int, int, int]

    def confirmation_prompt(self):
        description = ("city from the Location line, excluding state and country" if self.field == "city"
                       else "complete listing heading, preserving every word")
        return (
            f"Transcribe the {description} and the asking price from the enlarged crop on the right. "
            f'Return one JSON object with keys "{self.field}" and "dollars" containing strings. '
            "Use null for a field if it is unreadable. Write the dollar amount as digits with an optional decimal point."
        )

    def confirm(self, reply):
        data = read_reply(reply)
        if not isinstance(data, dict) or set(data) != {self.field, "dollars"}:
            raise ActionError("Expected the price and one listing field")
        text, dollars = data[self.field], data["dollars"]
        if text is None or dollars is None:
            return None, "evidence_verify_abstained"
        if not isinstance(text, str) or not isinstance(dollars, str):
            raise ActionError("Listing fields must be transcribed as strings")
        if (" ".join(text.casefold().split()) != self.text.casefold()
                or _displayed_cents(dollars.strip().strip("$").strip() + " $") != self.cents):
            return None, "evidence_verify_disagreed"
        whole, cents = divmod(self.cents, 100)
        amount = str(whole) if cents == 0 else f"{whole}.{cents:02d}"
        return f"{self.text}, {amount} dollars", "evidence_verify_agreed"


@dataclass(frozen=True)
class ListingFactDecision:
    status: str
    action: Action | None = None
    facts: ListingFacts | None = None
    target: tuple[str, int] | None = None


def main_listing_price(rows):
    bounds = _detail_price_box(rows)
    if bounds is None:
        return None
    lines = [row["text"] for row in rows
             if bounds[0] <= row["bounds"][0] < row["bounds"][2] <= bounds[2]
             and bounds[1] <= row["bounds"][1] < row["bounds"][3] <= bounds[3]]
    cents = _displayed_cents(" ".join(lines))
    return (tuple(bounds), cents) if cents is not None else None


def _detail_city(rows, size):
    location = _unique(rows, lambda text: text.startswith("location:"))
    published = _unique(rows, lambda text: text.startswith("published date:"))
    if location is None or published is None:
        return None
    start, top, _, end = location["bounds"]
    if (top < published["bounds"][3] or end - published["bounds"][3] > size[1] // 8
            or abs(start - published["bounds"][0]) > max(8, size[0] // 30)):
        return None
    parts = [part.strip() for part in location["text"].split(":", 1)[1].split(",")]
    if len(parts) != 3 or any(not part for part in parts) or not any(char.isalpha() for char in parts[0]):
        return None
    return tuple(location["bounds"]), " ".join(parts[0].split())


def _detail_facts(rows, size, field, expected=None):
    price = main_listing_price(rows)
    label = _detail_city(rows, size) if field == "city" else listing_title_heading(rows, size)
    if price is None or label is None or (expected is not None and price[1] != expected):
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    left, right = label[0], price[0]
    bounds = (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))
    return ListingFactDecision(LISTING_FACTS_REGION, facts=ListingFacts(field, label[1], price[1], bounds))


def _highest_result(query, active_url, site, rows, size, screen, capture):
    params = parse_qs(urlsplit(active_url).query)
    breadcrumb = f"classifieds > search results: {query.casefold()}"
    scope = listing_range(rows)
    if (params.get("page") != ["search"] or params.get("sPattern") != [query]
            or not set(params) <= {"page", "sPattern", "sOrder", "iOrderType", "sShowAs", "iPage"}
            or _unique(rows, lambda text: text == breadcrumb) is None or scope is None or scope[0] != 1):
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    menu = _unique(rows, lambda text: text.startswith("sort by:"))
    sorted_high = (params.get("sOrder") == ["i_price"] and params.get("iOrderType") == ["desc"]
                   and menu is not None and menu["text"].casefold() == "sort by: higher price first")
    if not sorted_high:
        action = Action("SORT_LISTINGS_PRICE", "Higher price first", _point(menu, size, toward_right=True)) if menu else None
        return (ListingFactDecision(SEARCH_PRICE_NAVIGATION, action=action) if action is not None
                else ListingFactDecision(LISTING_FACTS_ABSTAINED))
    receipt = capture()
    cards = inspect_card_receipt(receipt, pixel_hash(screen), screen.size, site)
    if receipt["scroll_top"] != 0:
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    if params.get("sShowAs") != ["gallery"] or screen.height < RESULT_VIEW_HEIGHT:
        if receipt["grid"] is not None:
            return ListingFactDecision(SEARCH_PRICE_NAVIGATION, action=Action("LISTING_GRID", position=tuple(receipt["grid"])))
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    if len(cards) != scope[1]:
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    top_count = min(2, scope[2])
    top = cards[:top_count]
    prices = _card_prices(rows, top, screen.size, size) if len(top) == top_count else None
    if not prices or (len(top) == 2 and prices[top[0]["destination"]] <= prices[top[1]["destination"]]):
        return ListingFactDecision(LISTING_FACTS_ABSTAINED)
    item = top[0]["destination"]
    return ListingFactDecision(SEARCH_PRICE_NAVIGATION, action=Action("CLICK", position=title_point(top[0], screen.size)),
                               target=(item, prices[item]))


def listing_fact_decision(goal, active_url, site, rows, size, screen, capture, *, target=None):
    if not same_origin(active_url, site):
        return None
    parts = urlsplit(active_url)
    if _DETAIL_PRICE_CITY.fullmatch(goal):
        if listing_destination(active_url, (parts.scheme, parts.netloc)) is None:
            return None
        return _detail_facts(rows, size, "city")
    requested = _SEARCH_HIGHEST.fullmatch(goal)
    if requested is None:
        return None
    if target is not None:
        return (_detail_facts(rows, size, "title", expected=target[1]) if active_url == target[0]
                else ListingFactDecision(LISTING_FACTS_ABSTAINED))
    if not parts.query and parts.path in ("", "/"):
        return None  # The existing visible keyword shortcut starts the search.
    return _highest_result(requested[1], active_url, site, rows, size, screen, capture)
