"""Find a candidate for an explicitly price-ranked Classifieds answer."""

import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from .actions import Action
from .listing_cards import (
    RESULT_VIEW_HEIGHT, card_panel, inspect_card_receipt, listing_range, read_card_checks, title_point,
)
from .ranked_cards import BATCH_SIZE
from .screen_text import empty_keyword_field, listing_detail, replace_keyword_search
from .verification import pixel_hash


PRICE_DISCOVERY = "price_discovery"
PRICE_REGION = "price_region"
SELFIE_PHOTO_REQUIREMENT = (
    "Look for the wearer photographing themselves in a mirror or an arm's-length self-portrait. "
    "A close-up of the item or of a hand wearing it does not establish a selfie."
)
_REQUEST = re.compile(
    r"^\s*find the (most expensive|cheapest|least expensive) ([^\n]{1,160}?),?\s+"
    r"and tell me how much it costs in dollars[.!?]?\s*$", re.IGNORECASE,
)
_QUALIFIER = re.compile(r"\s+(?:that|which|whose|with)\s+", re.IGNORECASE)
_CATEGORIES = {
    "Books": {"book", "books", "textbook", "textbooks"},
    "Jewelry": {"ring", "rings", "bracelet", "bracelets", "necklace", "necklaces", "earrings"},
    "Beauty + health": {"wheelchair", "wheelchairs", "walker", "walkers"},
}
_DOLLAR_PRICE = re.compile(r"(?:\$\s*[\d,.]+|[\d,.]+\s*\$)")
_PRICE_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d{2})?")


@dataclass(frozen=True)
class PriceRequest:
    category: str
    query: str
    alternate: str | None
    descending: bool
    visual_target: str

    @property
    def order_label(self):
        return "higher price first" if self.descending else "lower price first"

    @property
    def requires_selfie_cue(self):
        return re.search(r"\bselfie\b", self.visual_target, re.IGNORECASE) is not None


def price_request(goal):
    """Parse a supported goal without inspecting catalog labels or grading references."""
    match = _REQUEST.fullmatch(goal)
    if match is None:
        return None
    nouns = set(re.findall(r"[a-z]+", _QUALIFIER.split(match[2], maxsplit=1)[0].casefold()))
    categories = [(category, nouns & examples) for category, examples in _CATEGORIES.items() if nouns & examples]
    if len(categories) != 1 or len(categories[0][1]) != 1 or "or" in nouns:
        return None
    category, terms = categories[0]
    noun = next(iter(terms))
    if category == "Books":
        # The category already identifies books. Cover colors need not occur in listing text.
        query, alternate = "", None
    elif category == "Beauty + health" and "lift" in nouns:
        query, alternate = "lift", noun
    else:
        query, alternate = noun, None
    return PriceRequest(category, query, alternate, match[1].casefold() == "most expensive", match[2].rstrip(", "))


def same_origin(url, site):
    if url is None or site is None:
        return False
    current, registered = urlsplit(url), urlsplit(site)
    return (current.scheme, current.netloc) == (registered.scheme, registered.netloc)


def _point(row, size, *, toward_right=False):
    left, top, right, bottom = row["bounds"]
    width, height = size
    return ((left + (0.75 if toward_right else 0.5) * (right - left)) / width, (top + bottom) / (2 * height))


def _unique(rows, predicate):
    matches = [row for row in rows if predicate(row["text"].casefold())]
    return matches[0] if len(matches) == 1 else None


def price_category_search(request, rows, size):
    if empty_keyword_field(rows) is None:
        return None
    field = _unique(rows, lambda text: text == "select a category")
    if field is None:
        return None
    value = json.dumps({"category": request.category, "query": request.query}, ensure_ascii=False)
    return Action("SEARCH_CATEGORY", value, _point(field, size))


def price_result_scope(rows, request, query, active_url):
    params = parse_qs(urlsplit(active_url).query)
    expected_query = [query] if query else None
    if params.get("page") != ["search"] or params.get("sPattern") != expected_query or not params.get("sCategory"):
        return None
    prefix = f"classifieds > {request.category.casefold()}"
    if query:
        prefix += f" > search results: {query.casefold()}"
    if _unique(rows, lambda text: text == prefix) is None:
        return None
    return listing_range(rows)


def _price_order(rows, request, active_url):
    params = parse_qs(urlsplit(active_url).query)
    return (params.get("sOrder") == ["i_price"]
            and params.get("iOrderType") == ["desc" if request.descending else "asc"]
            and _unique(rows, lambda text: text == f"sort by: {request.order_label}") is not None)


def _detail_price_box(rows):
    if not listing_detail(rows):
        return None
    published = _unique(rows, lambda text: text.startswith("published date:"))
    if published is None:
        return None
    header = [row for row in rows
              if row["bounds"][3] <= published["bounds"][1]
              and row["bounds"][0] > published["bounds"][2]]
    price = [row["bounds"] for row in header if _DOLLAR_PRICE.fullmatch(row["text"])]
    # The OCR engine can put the currency symbol in its own box.
    for symbol in (row["bounds"] for row in header if row["text"] == "$"):
        for left, top, right, bottom in (row["bounds"] for row in header
                                         if _PRICE_AMOUNT.fullmatch(row["text"])):
            height = min(bottom - top, symbol[3] - symbol[1])
            overlap = min(bottom, symbol[3]) - max(top, symbol[1])
            gap = max(symbol[0] - right, left - symbol[2])
            if 2 * overlap >= height and -height / 2 <= gap <= height / 2:
                price.append([min(left, symbol[0]), min(top, symbol[1]),
                              max(right, symbol[2]), max(bottom, symbol[3])])
    return price[0] if len(price) == 1 else None


def first_ranked_match(reply, count):
    """Choose a match only after all earlier cards were explicitly rejected."""
    try:
        decisions = read_card_checks(reply, count, allow_ordered=True)
    except (ValueError, RecursionError):
        return None, False
    for index, value in enumerate(decisions):
        if value is None:
            return None, False
        if value:
            return index, False
    return None, True


class PriceDiscovery:
    """Navigate a bounded search and pass a grounded price crop to the ordinary reader."""

    def __init__(self, request, site):
        self.request = request
        self.site = site
        self.query = request.query
        self.visited = set()
        self.destination = None
        self.offset = 0
        self.page_frame = None
        self.first = 1
        self.total = None
        self.reviewed_destinations = set()

    def decide(self, observation, actor, screen_reader, capture):
        screen = observation.screenshot
        frame = pixel_hash(screen)
        active = observation.tabs[observation.active_tab][1] if observation.tabs else None
        metrics = {"evidence_stage": PRICE_DISCOVERY, "evidence_frame": frame, "provider_output": None,
                   "model_invocations": 0, "peak_cuda_allocated_bytes": 0}

        def finish(status, action=None, box=None):
            if action is not None:
                key = self.query, self.first, status
                if key in self.visited:
                    status, action = "price_abstained", None
                self.visited.add(key)
            if status == "price_abstained":
                metrics["stop_after_evidence"] = True
            metrics["provider_format"] = status
            if metrics["model_invocations"]:
                actor.last_metrics.update(metrics)
            else:
                actor.last_metrics = metrics
            return json.dumps(action.to_dict(), ensure_ascii=False) if action is not None else None, box

        if not same_origin(active, self.site):
            return finish("price_abstained")
        size = actor.prepare_image(screen).size
        rows = screen_reader(screen, size)
        metrics.update(price_screen_text=rows, price_canvas_size=list(size))
        if self.destination is not None:
            box = _detail_price_box(rows) if active == self.destination else None
            key = self.query, self.first, PRICE_REGION
            if box is None or key in self.visited:
                return finish("price_abstained")
            self.visited.add(key)
            return finish(PRICE_REGION, box=box)
        search = price_category_search(self.request, rows, size)
        if search is not None and not self.visited:
            return finish("price_search", search)
        scope = price_result_scope(rows, self.request, self.query, active)
        if scope is None or scope[0] != self.first or (self.total is not None and scope[2] != self.total):
            return finish("price_abstained")
        if not _price_order(rows, self.request, active):
            menu = _unique(rows, lambda text: text.startswith("sort by:"))
            if menu is not None:
                action = Action("SORT_LISTINGS_PRICE", self.request.order_label.capitalize(),
                                _point(menu, size, toward_right=True))
                return finish("price_sort", action)
            return finish("price_abstained")
        receipt = capture()
        metrics["price_card_receipt"] = receipt
        cards = inspect_card_receipt(receipt, frame, screen.size, self.site)
        if receipt["scroll_top"] != 0:
            return finish("price_abstained")
        if len(cards) != scope[1] - scope[0] + 1 or any(card["image"] is None for card in cards):
            if screen.height < RESULT_VIEW_HEIGHT and receipt["grid"] is not None:
                return finish("price_expand", Action("LISTING_GRID", position=tuple(receipt["grid"])))
            return finish("price_abstained")
        if self.offset and frame != self.page_frame:
            return finish("price_abstained")
        if not self.offset:
            destinations = {card["destination"] for card in cards}
            if destinations & self.reviewed_destinations:
                return finish("price_abstained")
            self.reviewed_destinations.update(destinations)
            self.total = scope[2]
        self.page_frame = frame
        batch = cards[self.offset:self.offset + BATCH_SIZE]
        panel = card_panel([(screen, card) for card in batch])
        example = '{"1":[false,"wooden table"],"2":[true,"person wearing ring"]}'
        condition = ""
        if self.request.requires_selfie_cue:
            condition = f"{SELFIE_PHOTO_REQUIREMENT}\n"
            example = '{"1":[false,"hand close-up"],"2":[true,"mirror self-portrait"]}'
        if len(batch) == 1:
            example = '{"1":[false,"wooden table"]}'
        reply = actor.ask(actor.prepare_image(panel), (
            f"Visual criterion: {self.request.visual_target}.\n"
            f"{condition}"
            "Inspect each numbered listing photo. First identify a concrete visual cue in the photo. "
            "Use that cue to decide whether the item visibly matches, recording false for a clearly unrelated photo. "
            f"Return only one JSON object with keys 1 through {len(batch)}. Every entry is "
            "[true or false, at most three words of visible evidence]. "
            f"For example, {example}."
        ))
        metrics.update(model_invocations=1, provider_output=reply, price_panel_pixels=pixel_hash(panel),
                       peak_cuda_allocated_bytes=actor.last_metrics.get("peak_cuda_allocated_bytes", 0))
        index, clear_miss = first_ranked_match(reply, len(batch))
        if index is not None:
            selected = batch[index]
            self.destination = selected["destination"]
            return finish("price_open", Action("CLICK", position=title_point(selected, screen.size)))
        if clear_miss and self.offset + len(batch) < len(cards):
            self.offset += len(batch)
            return finish("price_more_cards")
        if clear_miss and scope[1] < scope[2] and receipt["next"] is not None:
            self.first = scope[1] + 1
            self.offset = 0
            return finish("price_next_page", Action("CLICK", position=tuple(receipt["next"])))
        if clear_miss and self.request.alternate is not None and self.query != self.request.alternate:
            refinement = replace_keyword_search(self.query, self.request.alternate, rows, size)
            if refinement is not None:
                self.query = self.request.alternate
                self.offset = 0
                self.first = 1
                self.total = None
                self.reviewed_destinations.clear()
                return finish("price_refine", refinement)
        return finish("price_abstained")
