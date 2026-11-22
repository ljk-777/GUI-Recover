"""Locate answer evidence in a screenshot and read a native-pixel crop."""

import json
import re
from urllib.parse import urlsplit
from decimal import Decimal

from PIL import Image, ImageDraw, ImageOps

from .actions import Action, ActionError
from .environment import frame_key
from .listing_cards import (
    ReferenceTitleCrop, ambiguous_reference_title, exact_reference_title_request, listing_breadcrumb_box,
    listing_title_heading, ranked_photo_request,
    reference_result_action, reference_result_destination,
    reference_title_crop, reference_title_open_action, reference_title_region, reference_title_request, with_listing_breadcrumb,
)
from .listing_facts import LISTING_FACTS_ABSTAINED, ListingFacts, listing_fact_decision
from .price_discovery import SELFIE_PHOTO_REQUIREMENT, PriceDiscovery, cover_color_labels, price_request, same_origin
from .qwen_grounding import ask_observation, native_action_after_note, normalize_qwen_action, parse_visible_answer, read_reply
from .ranked_cards import RankedCardEvidence
from .screen_text import empty_keyword_field, exact_initial_search, from_readable_control, keyword_search, listing_detail
from .verification import pixel_hash
from .visible_links import agreed_listing_answer, link_answer_count, listing_selection, preview_destinations, priced_titles


_AMOUNT = re.compile(r"(?<![\w.])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?![\w.])")
_PRICE_QUESTION = re.compile(r"\b(?:cheapest|costliest|expensive|prices?|priced|budget|costs?)\b|\$", re.IGNORECASE)
_CONTACT = re.compile(r"\b(?:seller|lister|publisher)\b", re.IGNORECASE)
_PHOTO = re.compile(r"\b(?:image|photo|picture)\b", re.IGNORECASE)
_EMAIL_REQUEST = re.compile(r"\be-?mail\b", re.IGNORECASE)
_EMAIL_FIELD = re.compile(r"^e-?mail\s*:\s*", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_REFERENCE_QUERY = re.compile(r"[\w .&'-]{1,70}")
REFERENCE_SEARCH = "evidence_reference_search"
REFERENCE_SEARCH_ABSTAINED = "evidence_reference_search_abstained"
REFERENCE_RESULTS_EXPAND = "evidence_reference_results_expand"
REFERENCE_RESULTS_OPEN = "evidence_reference_results_open"
REFERENCE_RESULTS_REFINE = "evidence_reference_results_refine"
REFERENCE_CONTACT_REGION = "evidence_reference_contact_region"
REFERENCE_TITLE_OPEN = "evidence_reference_title_open"
REFERENCE_TITLE_AMBIGUOUS = "evidence_reference_title_ambiguous"
REFERENCE_TITLE_HEADING = "evidence_reference_title_heading"
REFERENCE_TITLE_CROP = "evidence_reference_title_crop"
REFERENCE_NAVIGATION = {
    "LISTING_GRID": REFERENCE_RESULTS_EXPAND,
    "CLICK": REFERENCE_RESULTS_OPEN,
    "SEARCH_REPLACE": REFERENCE_RESULTS_REFINE,
}


def reference_search_prompt(goal):
    return (
        f"Task: {goal}\nThe only image is the task reference. "
        "Give a short Classifieds search query for the requested subject or item. "
        "Use the most distinctive search term in the task's item description when given. "
        "Otherwise prefer an identifying brand or name visible on the item, then its object type. "
        "When asked about a person or character portrayed, use the subject's name. "
        "Multiple common words can broaden these search results. Use one distinctive word when possible, "
        "or a short recognizable name. Omit prices, dates, and background details. "
        'Return exactly {"query":"keywords"} or {"query":null} if unclear.'
    )


def reference_search_action(reply, rows, size):
    """Turn a reference-image query into an action on an observed empty field."""
    try:
        data = read_reply(reply)
        query = data["query"] if isinstance(data, dict) and set(data) == {"query"} else None
        if isinstance(query, str) and query.strip() and _REFERENCE_QUERY.fullmatch(query):
            return keyword_search(query.strip(), rows, size)
    except (ValueError, RecursionError):
        pass
    return None


def evidence_box(raw, size):
    data = read_reply(raw)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict) or set(data) not in ({"bbox_2d"}, {"bbox_2d", "label"}):
        raise ActionError("Expected one evidence box")
    if "label" in data and (not isinstance(data["label"], str) or len(data["label"]) > 200):
        raise ActionError("Expected a short evidence label")
    box = data["bbox_2d"]
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        raise ActionError("Evidence box must contain four integer pixels")
    left, top, right, bottom = box
    width, height = size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ActionError("Evidence box lies outside the prepared screenshot")
    return tuple(box)


def native_crop(screen, size, box, *, pad=True):
    """Map prepared-image coordinates to the original pixels with context."""
    left, top, right, bottom = box
    width, height = size
    pad_x = max(16, (right - left) // 8) if pad else 0
    pad_y = max(16, (bottom - top) // 8) if pad else 0
    bounds = (
        max(0, (left - pad_x) * screen.width // width),
        max(0, (top - pad_y) * screen.height // height),
        min(screen.width, ((right + pad_x) * screen.width + width - 1) // width),
        min(screen.height, ((bottom + pad_y) * screen.height + height - 1) // height),
    )
    return screen.crop(bounds), bounds


def breadcrumb_ocr_image(screen, size, box):
    crop, bounds = native_crop(screen, size, box, pad=False)
    return crop.resize((2 * crop.width, 2 * crop.height), Image.Resampling.LANCZOS), bounds


def listing_fact_field_crop(screen, size, box):
    padded = (max(0, box[0] - 4), max(0, box[1] - 4),
              min(size[0], box[2] + 4), min(size[1], box[3] + 4))
    return native_crop(screen, size, padded, pad=False)


def listing_price_ocr_image(screen, candidate, *, enlarge=True):
    crop, bounds = listing_fact_field_crop(screen, candidate.canvas_size, candidate.price_bounds)
    if enlarge:
        crop = crop.resize((2 * crop.width, 2 * crop.height), Image.Resampling.LANCZOS)
    return crop, bounds


def listing_upright_price_retry_image(screen, candidate):
    _, (left, top, right, bottom) = listing_price_ocr_image(screen, candidate, enlarge=False)
    bounds = (max(0, left - 4), max(0, top - 4), min(screen.width, right + 4), min(screen.height, bottom + 4))
    return screen.crop(bounds), bounds


def listing_price_ocr_receipt(screen, candidate, reader, *, upright_retry=False):
    image_for_ocr = listing_upright_price_retry_image if upright_retry else listing_price_ocr_image
    crop, bounds = image_for_ocr(screen, candidate)
    return {"bounds": list(bounds), "pixels_sha256": pixel_hash(crop), "rows": reader(crop, crop.size)}


def listing_fact_reread_image(screen, candidate):
    crops, bounds = [], []
    for box in (candidate.label_bounds, candidate.price_bounds):
        crop, native_bounds = listing_fact_field_crop(screen, candidate.canvas_size, box)
        crops.append(ImageOps.contain(crop.convert("RGB"), (1008, 100), Image.Resampling.LANCZOS))
        bounds.append(list(native_bounds))
    panel = Image.new("RGB", (max(crop.width for crop in crops) + 32, sum(crop.height for crop in crops) + 48), "white")
    panel.paste(crops[0], (16, 16))
    panel.paste(crops[1], (16, 32 + crops[0].height))
    return panel, bounds


def evidence_clipped(goal, box, size):
    """A range or ordinal row needs the visible extent of that row."""
    requires_row = re.search(r"\brows?\b", goal.casefold())
    return bool(requires_row and box[3] >= size[1] - 2)


def evidence_navigation(raw, size, *, compact_navigation=True, allow_click_box=False, tab_count=0):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        data = native_action_after_note(raw)
    if not isinstance(data, dict) or "action" not in data:
        return None
    if (allow_click_box and data["action"] == "CLICK" and set(data) == {"action", "value"}
            and isinstance(data["value"], dict) and set(data["value"]) == {"bbox_2d"}):
        box = evidence_box(json.dumps(data["value"]), size)
        if box is None:
            raise ActionError("A click box needs a visible region")
        left, top, right, bottom = box
        data = {"action": "CLICK", "value": None, "position": [(left + right) // 2, (top + bottom) // 2]}
    action = normalize_qwen_action(data, size, native_navigation=True, compact_navigation=compact_navigation,
                                   tab_count=tab_count)
    if action.action == "ANSWER":
        raise ActionError("An evidence search cannot submit an answer")
    return action


def evidence_answer(raw, goal, *, allow_string_abstention=False, price_crop=False):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "evidence"}:
        raise ActionError("Expected an answer and its visual evidence")
    answer, facts = data["answer"], data["evidence"]
    if price_crop and isinstance(facts, list) and len(facts) == 1 and isinstance(facts[0], list) and len(facts[0]) == 1:
        # A price crop has one displayed amount. Accept one redundant evidence wrapper.
        facts = facts[0]
    if (not isinstance(facts, list) or len(facts) > 6
            or any(not isinstance(fact, str) or not 0 < len(fact.strip()) <= 200 for fact in facts)):
        raise ActionError("Expected a bounded list of visual evidence descriptions")
    if answer is None or (allow_string_abstention and isinstance(answer, str) and answer.strip().casefold() == "null"):
        return None, "evidence_abstained"
    if not isinstance(answer, str) or not 0 < len(answer.strip()) <= 512:
        raise ActionError("Expected a nonempty short answer or null")
    if not facts:
        return None, "evidence_incomplete"
    if "price range" in goal.casefold():
        # A range answer needs both visible endpoints. Ignore formatting
        # differences such as thousands separators when matching amounts.
        bounds = [Decimal(value.replace(",", "")) for value in _AMOUNT.findall(answer)]
        observed = {Decimal(value.replace(",", "")) for value in _AMOUNT.findall(" ".join(facts))}
        if len(bounds) != 2 or not set(bounds) <= observed:
            return None, "evidence_incomplete"
    return answer.strip(), "evidence_answer"


def localized_candidate(raw, goal, size):
    try:
        data = read_reply(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != {"bbox_2d", "evidence", "answer"}:
        return None
    box = evidence_box(json.dumps({"bbox_2d": data["bbox_2d"]}), size)
    if box is None:
        raise ActionError("A candidate needs a selectable region")
    reply = json.dumps({"answer": data["answer"], "evidence": data["evidence"]})
    answer, _ = evidence_answer(reply, goal, allow_string_abstention=True)
    return box, answer


def has_transcribed_answer(answer, box, rows):
    text = " ".join(answer.casefold().split())
    pattern = re.compile(r"(?<![\w.])" + re.escape(text) + r"(?![\w.])")
    left, top, right, bottom = box
    for row in rows:
        x0, y0, x1, y1 = row["bounds"]
        if left <= (x0 + x1) / 2 < right and top <= (y0 + y1) / 2 < bottom:
            if pattern.search(row["text"].casefold()):
                return True
    return False


def parse_evidence_location(raw, goal, size, *, confirm, compact_navigation=True, rows=None, guarded_ocr=False,
                            navigation_only=False, allow_click_box=False, tab_count=0):
    """Return a browser action or a validated evidence region and optional candidate."""
    candidate = None
    localized = None
    if rows is not None:
        control = from_readable_control(raw, rows, size, guarded=guarded_ocr)
        if control is not None:
            return control, None, None, "evidence_ocr_navigation"
        localized = localized_candidate(raw, goal, size)
    if localized is not None:
        box, candidate = localized
        if guarded_ocr and candidate is not None and not has_transcribed_answer(candidate, box, rows):
            candidate = None
    else:
        navigation = evidence_navigation(raw, size, compact_navigation=compact_navigation,
                                         allow_click_box=allow_click_box, tab_count=tab_count) if confirm else None
        if navigation is not None:
            return navigation, None, None, "evidence_navigation"
        box = evidence_box(raw, size)
    if box is None:
        return None, None, None, "evidence_abstained"
    if navigation_only:
        raise ActionError("A listing link must come from a verified title preview")
    if evidence_clipped(goal, box, size):
        return None, None, None, "evidence_clipped"
    return None, box, candidate, "evidence_locate_read" if candidate is not None else "evidence_region"


def confirmation_image(screen, crop, bounds, navigation_context=None, *, focus_listing=False):
    overview_height = max(screen.height // 2, bounds[3]) if focus_listing else screen.height
    overview = screen.crop((0, 0, screen.width, overview_height)).convert("RGB")
    overview.thumbnail((448, 336), Image.Resampling.LANCZOS)
    ow, oh = overview.size
    scaled = (bounds[0] * ow // screen.width, bounds[1] * oh // overview_height,
              min(ow - 1, bounds[2] * ow // screen.width),
              min(oh - 1, bounds[3] * oh // overview_height))
    ImageDraw.Draw(overview).rectangle(scaled, outline="#176fde", width=2)
    previous = None
    if navigation_context is not None:
        before, (x, y) = navigation_context
        previous = before.convert("RGB")
        previous.thumbnail((448, 336), Image.Resampling.LANCZOS)
        px = min(previous.width - 1, int(x * previous.width))
        py = min(previous.height - 1, int(y * previous.height))
        ImageDraw.Draw(previous).ellipse((px - 7, py - 7, px + 7, py + 7), outline="#d85319", width=3)
    left_height = oh + (12 + previous.height if previous is not None else 0)
    panel = Image.new("RGB", (ow + 12 + crop.width, max(left_height, crop.height)), "white")
    panel.paste(overview)
    if previous is not None:
        panel.paste(previous, (0, oh + 12))
    panel.paste(crop, (ow + 12, 0))
    return panel


def answer_signature(answer):
    value = " ".join(answer.casefold().rstrip(".").split())
    if re.fullmatch(r"\$?[0-9][0-9,. ]*(?:[–—-]\s*\$?[0-9][0-9,. ]*)?", value):
        amounts = tuple(Decimal(amount.replace(",", "")) for amount in _AMOUNT.findall(value))
        if amounts:
            return amounts
    return value


def confirmed_evidence_answer(candidate, independent, *, price_crop=False, expected_cents=None):
    signature = answer_signature(candidate)
    if signature != answer_signature(independent):
        return None
    if expected_cents is not None and signature != (Decimal(expected_cents).scaleb(-2),):
        return None
    if not price_crop:
        return candidate
    if not isinstance(signature, tuple) or len(signature) != 1:
        return None
    amount, = signature
    whole = amount.to_integral_value()
    return format(whole if amount == whole else amount, "f")


def parse_selfie_price_confirmation(raw):
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "selfie_cue"}:
        raise ActionError("Expected the independently read price and a selfie cue")
    cue = data["selfie_cue"]
    if cue not in ("mirror_camera", "arms_length_portrait"):
        if cue is not None:
            raise ActionError("Unknown selfie cue")
        return None
    return parse_visible_answer(json.dumps({"answer": data["answer"]}), allow_numeric=True)


def parse_independent_confirmation(raw, request=None):
    if request is not None and request.requires_selfie_cue:
        return parse_selfie_price_confirmation(raw)
    color = request.hardcover_color if request is not None else None
    if color is None:
        return parse_visible_answer(raw, allow_numeric=True)
    data = read_reply(raw)
    if not isinstance(data, dict) or set(data) != {"answer", "cover_color", "binding"}:
        raise ActionError("Expected the independently read price, cover color and binding")
    visible = data["cover_color"]
    if (not isinstance(visible, str) or visible.strip().casefold() not in cover_color_labels(color)
            or data["binding"] != "hardcover"):
        return None
    return parse_visible_answer(json.dumps({"answer": data["answer"]}), allow_numeric=True)


def confirmation_outcome(raw, candidate, request=None, *, expected_cents=None, listing_fact_reread=False):
    try:
        if isinstance(candidate, ListingFacts):
            return candidate.confirm(raw, reread=listing_fact_reread)
        answer = (parse_visible_answer(raw) if isinstance(candidate, ReferenceTitleCrop)
                  else parse_independent_confirmation(raw, request))
    except (ValueError, RecursionError):
        return None, "evidence_invalid"
    if answer is None:
        return None, "evidence_verify_abstained"
    if isinstance(candidate, ReferenceTitleCrop):
        confirmed = candidate.title if " ".join(answer.casefold().split()) == candidate.title.casefold() else None
    else:
        confirmed = confirmed_evidence_answer(candidate, answer, price_crop=request is not None, expected_cents=expected_cents)
    return (confirmed, "evidence_verify_agreed") if confirmed is not None else (None, "evidence_verify_disagreed")


def checked_listing_confirmation(raw, candidate, price_rows):
    checked = candidate.with_price_from_crop(price_rows)
    if checked is None:
        return None, "evidence_verify_abstained", None
    answer, status = confirmation_outcome(raw, candidate)
    if answer is not None and checked.cents != candidate.cents:
        return None, "evidence_verify_disagreed", checked
    return answer, status, checked


def indexed_text(rows):
    return "\n".join(f"{index}: {json.dumps(row['text'], ensure_ascii=False)} at {row['bounds']}"
                     for index, row in enumerate(rows))


def listing_screen_text(rows, size):
    cards = priced_titles(rows, size)
    prices = "\n".join(f"Title {title} {json.dumps(rows[title]['text'], ensure_ascii=False)} costs {rows[price]['text']}"
                       for price, title in cards.items())
    text = f"Visible text:\n{indexed_text(rows)}"
    if prices:
        text += f"\nVisible prices directly below aligned text:\n{prices}"
    return text


def visible_contact_request(goal, rows):
    return bool(rows and _CONTACT.search(goal) and _PHOTO.search(goal)
                and any(row["text"].casefold() == "latest listings" for row in rows))


def reference_contact_box(goal, rows):
    """Find one visible email row in the publisher section of the matched item."""
    if not (_CONTACT.search(goal) and _EMAIL_REQUEST.search(goal)):
        return None
    headings = [row["bounds"] for row in rows if row["text"].casefold() == "contact publisher"]
    fields = [row for row in rows if _EMAIL_FIELD.match(row["text"])]
    if len(headings) != 1 or len(fields) != 1:
        return None
    heading, field = headings[0], fields[0]
    left, top, right, bottom = field["bounds"]
    if not (0 < top - heading[3] <= 12 * (heading[3] - heading[1]) and left < heading[2] and right > heading[0]):
        return None
    if _EMAIL.search(field["text"]):
        return field["bounds"]
    height = bottom - top
    following = [row["bounds"] for row in rows if _EMAIL.fullmatch(row["text"])
                 and abs(row["bounds"][0] - left) <= height and top < row["bounds"][1] <= bottom + height]
    if len(following) != 1:
        return None
    address = following[0]
    return [min(left, address[0]), top, max(right, address[2]), max(bottom, address[3])]


def contact_from_evidence(raw, goal, *, publisher_region=False):
    """Recover one literal contact fact from a validated evidence reply."""
    if not (_CONTACT.search(goal) and _EMAIL_REQUEST.search(goal)):
        return None
    data = read_reply(raw)
    if publisher_region and (data["answer"] is None or data["answer"].strip().casefold() == "null"):
        addresses = set()
        for fact in data["evidence"]:
            label = _EMAIL_FIELD.match(fact.strip())
            if label is not None:
                address = fact.strip()[label.end():]
                if _EMAIL.fullmatch(address):
                    addresses.add(address)
        if len(addresses) == 1:
            return addresses.pop()
    if data["answer"] is not None or len(data["evidence"]) != 1:
        return None
    address = data["evidence"][0].strip()
    return address if _EMAIL.fullmatch(address) else None


class EvidenceAnswerPolicy:
    """Use one model invocation per prediction, including crop localization."""

    def __init__(self, actor, *, confirm=False, use_navigation_context=False, screen_reader=None,
                 listing_preview=None, listing_site=None, ranked_capture=None):
        if screen_reader is not None and not confirm:
            raise ValueError("The OCR policy needs independent confirmation")
        if listing_preview is not None and (screen_reader is None or listing_site is None):
            raise ValueError("A listing preview needs visible text and a registered origin")
        self.actor = actor
        self.confirm = confirm
        self.use_navigation_context = use_navigation_context
        self.screen_reader = screen_reader
        self.price_retry_reader = getattr(screen_reader, "read_upright", screen_reader)
        self.listing_preview = listing_preview
        self.listing_site = listing_site
        self.ranked_capture = ranked_capture
        self.ranked_evidence = RankedCardEvidence(listing_site) if ranked_capture is not None else None
        self.ranked_request = None
        self.price_discovery = None
        self.reference_query = None
        self.reference_refined = False
        self.reference_destination = None
        self.reference_title_frame = None
        self.listing_fact_target = None
        self.checked_frames = set()
        self.pending_crop = None
        self.pending_answer = None
        self.pending_fact_reread = None
        self.pending_links = None
        self.navigation_context = None

    @property
    def last_metrics(self):
        return self.actor.last_metrics

    def reset(self, seed):
        self.actor.reset(seed)
        self.checked_frames.clear()
        self.pending_crop = None
        self.pending_answer = None
        self.pending_fact_reread = None
        self.pending_links = None
        self.navigation_context = None
        self.ranked_request = None
        self.price_discovery = None
        self.reference_query = None
        self.reference_refined = False
        self.reference_destination = None
        self.reference_title_frame = None
        self.listing_fact_target = None
        if self.ranked_capture is not None:
            self.ranked_evidence = RankedCardEvidence(self.listing_site)

    def set_feedback(self, message):
        self.actor.set_feedback(message)

    def observe_transition(self, before, after, action):
        if action.action == "TAB_FOCUS":
            self.navigation_context = None
        if (self.use_navigation_context and action.action == "CLICK"
                and pixel_hash(before.screenshot) != pixel_hash(after.screenshot)):
            self.navigation_context = before.screenshot.copy(), action.position

    def locate_listing_facts(self, observation, frame, rows, size):
        if not self.confirm or self.ranked_capture is None or observation.reference_images or not observation.tabs:
            return False, None
        active = observation.tabs[observation.active_tab][1]
        receipt = None

        def capture():
            nonlocal receipt
            receipt = self.ranked_capture()
            return receipt

        decision = listing_fact_decision(observation.goal, active, self.listing_site, rows, size,
                                         observation.screenshot, capture, target=self.listing_fact_target)
        if decision is None:
            return False, None
        self.actor.last_metrics = {
            "model_invocations": 0, "provider_output": None, "provider_format": decision.status,
            "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
            "provider_image_size": list(size), "peak_cuda_allocated_bytes": 0,
        }
        if receipt is not None:
            self.last_metrics["listing_fact_receipt"] = receipt
        if decision.facts is not None:
            crop, bounds = native_crop(observation.screenshot, size, decision.facts.bounds)
            self.pending_answer = frame, crop, bounds, decision.facts
            self.last_metrics["evidence_bounds"] = list(bounds)
        if decision.target is not None:
            self.listing_fact_target = decision.target
        if decision.status == LISTING_FACTS_ABSTAINED:
            self.last_metrics["stop_after_evidence"] = True
        raw = json.dumps(decision.action.to_dict(), ensure_ascii=False) if decision.action is not None else None
        return True, raw

    def locate(self, observation, frame, history):
        reference = None
        active_url = None
        if self.ranked_capture is not None and len(observation.reference_images) == 1 and observation.tabs:
            active_url = observation.tabs[observation.active_tab][1]
            here = urlsplit(active_url)
            site = urlsplit(self.listing_site)
            if (here.scheme, here.netloc) == (site.scheme, site.netloc):
                reference = observation.reference_images[0]
        exact_title = reference is not None and self.screen_reader is not None and exact_reference_title_request(observation.goal)
        if exact_title and self.reference_destination is None:
            receipt = self.ranked_capture()
            screen = observation.screenshot
            if ambiguous_reference_title(screen, reference, receipt, self.listing_site, observation.goal):
                self.actor.last_metrics = {
                    "model_invocations": 0, "provider_output": None, "provider_format": REFERENCE_TITLE_AMBIGUOUS,
                    "evidence_stage": "locate", "evidence_frame": frame, "reference_card_receipt": receipt,
                    "peak_cuda_allocated_bytes": 0, "stop_after_evidence": True,
                }
                return None
            action = reference_title_open_action(screen, reference, receipt, self.listing_site)
            if action is not None:
                self.reference_destination = reference_result_destination(receipt, screen.size, action)
                self.actor.last_metrics = {
                    "model_invocations": 0, "provider_output": None, "provider_format": REFERENCE_TITLE_OPEN,
                    "evidence_stage": "locate", "evidence_frame": frame, "reference_card_receipt": receipt,
                    "peak_cuda_allocated_bytes": 0,
                }
                return json.dumps(action.to_dict(), ensure_ascii=False)
        if reference is not None and not exact_title and reference_title_request(observation.goal):
            receipt = self.ranked_capture()
            screen = observation.screenshot
            bounds = reference_title_region(screen, reference, receipt, self.listing_site)
            if bounds is not None:
                self.pending_crop = frame, screen.crop(bounds), bounds, False
                self.actor.last_metrics = {
                    "model_invocations": 0, "provider_output": None, "provider_format": "evidence_reference_region",
                    "evidence_stage": "locate", "evidence_frame": frame, "evidence_bounds": list(bounds),
                    "reference_card_receipt": receipt, "peak_cuda_allocated_bytes": 0,
                }
                return None
        canvas = self.actor.prepare_image(observation.screenshot)
        width, height = canvas.size
        prompt = (
            f"Find the visual evidence needed to answer this question: {observation.goal}\n"
            "Point to the region containing the requested item and the relevant details. "
            "Include labels or row context that identify the target. For a price range include both ends. "
            "Select a useful region if it can be inspected more closely. "
            f"Use integer pixel coordinates in this {width} by {height} image with top-left origin. "
            'Return one JSON object like {"bbox_2d":[left,top,right,bottom]}. '
            'If the relevant region is not visible, return {"bbox_2d":null}. Do not answer the question.'
        )
        if self.confirm:
            prompt = (
                f"Task: {observation.goal}\n"
                f"Use the visible state of this {width} by {height} screenshot. "
                'If a specific visible region could support all requested details, return {"bbox_2d":[left,top,right,bottom]}. '
                "Choose a tight box with the relevant text or photo and enough context. "
                "If the user first needs a search, category, sort, or item opened, do one navigation action to reach it. "
                'Return {"action":"CLICK","value":null,"position":[x,y]} for a visible control, '
                '{"action":"INPUT","value":"query","position":[x,y]} to type into a visible search field, '
                '{"action":"ENTER","value":null,"position":null} to submit a filled search field, or '
                '{"action":"SCROLL","value":"down","position":null} to reveal lower content. '
                'Coordinates are integer pixels. Return {"bbox_2d":null} if neither option is clear. '
                "Return one JSON response and never answer the task directly."
            )
        rows = None
        navigation_rows = None
        link_count = link_answer_count(observation.goal) if self.listing_preview is not None else None
        if self.screen_reader is not None:
            rows = self.screen_reader(observation.screenshot, canvas.size)
            handled, raw = self.locate_listing_facts(observation, frame, rows, canvas.size)
            if handled:
                return raw
            if exact_title and self.reference_destination is not None and active_url == self.reference_destination:
                heading = listing_title_heading(rows, canvas.size)
                title_rows, breadcrumb_check = rows, None
                if heading is None:
                    box = listing_breadcrumb_box(rows, canvas.size)
                    if box is not None:
                        detail, native_bounds = breadcrumb_ocr_image(observation.screenshot, canvas.size, box)
                        local_rows = self.screen_reader(detail, (box[2] - box[0], box[3] - box[1]))
                        title_rows = with_listing_breadcrumb(rows, local_rows, box)
                        heading = listing_title_heading(title_rows, canvas.size)
                        breadcrumb_check = {"bounds": list(native_bounds), "pixels_sha256": pixel_hash(detail),
                                            "rows": local_rows}
                recovery = reference_title_crop(title_rows, canvas.size) if heading is None else None
                if heading is not None or recovery is not None:
                    if recovery is None:
                        box, title = heading
                        self.reference_title_frame = frame
                        provider_format = REFERENCE_TITLE_HEADING
                    else:
                        box, title = recovery.bounds, recovery
                        provider_format = REFERENCE_TITLE_CROP
                    crop, bounds = native_crop(observation.screenshot, canvas.size, box, pad=recovery is None)
                    self.pending_answer = frame, crop, bounds, title
                    self.actor.last_metrics = {
                        "model_invocations": 0, "provider_output": None, "provider_format": provider_format,
                        "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
                        "provider_image_size": list(canvas.size), "evidence_bounds": list(bounds),
                        "peak_cuda_allocated_bytes": 0,
                    }
                    if recovery is not None:
                        self.last_metrics["reference_title_missing_number"] = recovery.missing_number
                    if title_rows is not rows:
                        self.last_metrics["reference_title_breadcrumb"] = breadcrumb_check
                    return None
            if self.reference_destination is not None and active_url == self.reference_destination:
                box = reference_contact_box(observation.goal, rows)
                if box is not None:
                    crop, bounds = native_crop(observation.screenshot, canvas.size, box)
                    self.pending_crop = frame, crop, bounds, True
                    self.actor.last_metrics = {
                        "model_invocations": 0, "provider_output": None, "provider_format": REFERENCE_CONTACT_REGION,
                        "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
                        "provider_image_size": list(canvas.size), "evidence_bounds": list(bounds),
                        "peak_cuda_allocated_bytes": 0,
                    }
                    return None
            if reference is not None and self.reference_query is not None:
                receipt = self.ranked_capture()
                action = reference_result_action(observation.screenshot, reference, receipt, self.listing_site,
                                                 rows, self.reference_query, observation.goal,
                                                 refinement_size=None if self.reference_refined else canvas.size)
                if action is not None:
                    if action.action == "SEARCH_REPLACE":
                        self.reference_query = action.value
                        self.reference_refined = True
                    elif action.action == "CLICK":
                        self.reference_destination = reference_result_destination(
                            receipt, observation.screenshot.size, action)
                    self.actor.last_metrics = {
                        "model_invocations": 0, "provider_output": None,
                        "provider_format": REFERENCE_NAVIGATION[action.action],
                        "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
                        "provider_image_size": list(canvas.size), "reference_card_receipt": receipt,
                        "peak_cuda_allocated_bytes": 0,
                    }
                    return json.dumps(action.to_dict(), ensure_ascii=False)
            if not history:
                try:
                    shortcut = exact_initial_search(observation.goal, rows, canvas.size)
                except ActionError:
                    shortcut = None
                if shortcut is not None:
                    if self.ranked_capture is not None:
                        self.ranked_request = ranked_photo_request(observation.goal)
                        if exact_title:
                            self.reference_query = shortcut.value
                            self.reference_refined = True
                    self.actor.last_metrics = {
                        "model_invocations": 0, "provider_output": None, "provider_image_size": list(canvas.size),
                        "evidence_stage": "locate", "evidence_frame": frame, "screen_text": rows,
                        "provider_format": "evidence_exact_search", "peak_cuda_allocated_bytes": 0,
                    }
                    return json.dumps(shortcut.to_dict(), ensure_ascii=False)
                if reference is not None and empty_keyword_field(rows) is not None:
                    raw = self.actor.ask(self.actor.prepare_image(reference), reference_search_prompt(observation.goal))
                    action = reference_search_action(raw, rows, canvas.size)
                    if action is not None:
                        self.reference_query = action.value
                    status = REFERENCE_SEARCH if action is not None else REFERENCE_SEARCH_ABSTAINED
                    self.last_metrics.update(
                        provider_format=status, evidence_stage="locate", evidence_frame=frame, screen_text=rows,
                        search_canvas_size=list(canvas.size), reference_pixels=[pixel_hash(reference)],
                    )
                    return json.dumps(action.to_dict(), ensure_ascii=False) if action is not None else None
            navigation_rows = None if link_count is None and listing_detail(rows) else rows
        if navigation_rows is not None:
            controls = indexed_text(navigation_rows)
            recent = [{"action": item["action"]["action"],
                       "value": (item["action"]["value"] or "")[:80]}
                      for item in history[-4:]]
            prompt += (
                "\nUse the indexed visible text as screenshot evidence. "
                'For a button or field with an identified line, use {"action":"CLICK","value":null,"control":index} '
                'or {"action":"INPUT","value":"query","control":index}. Only use INPUT on a visible field or placeholder. '
                'On a page with an empty Keyword input showing an e.g. placeholder, '
                'use {"action":"SEARCH","value":"query","control":placeholder_index} to type and submit in one action. '
                "If the current page shows a search form and the target is elsewhere, search by the requested keyword "
                "or category instead of scrolling unrelated items. If the field already contains the query, submit it. "
                "After a SEARCH, inspect its results instead of typing the same query again. "
                "Use the visible row order and item photos to identify a requested listing. If publisher or item "
                "details are absent from the gallery, open a matching item. Do not scroll just to find details in a gallery. "
                'When the answer is already legible, you may return {"bbox_2d":[x0,y0,x1,y1],"evidence":["visible fact"],'
                '"answer":"candidate"}. Use a candidate only for literal text visible inside the region. For counts, colors '
                "and other visual details, return the box only. A separate read will confirm the answer. Do not guess. "
                f"Previous browser actions: {json.dumps(recent, ensure_ascii=False)}.\nVisible text:\n{controls}"
            )
        direct_contact = self.ranked_capture is not None and not history and visible_contact_request(observation.goal, rows)
        if direct_contact:
            prompt += ("\nLook at the photos in the visible Latest Listings cards before choosing the next action. "
                       "If a card's pictured subject and title identify a strong candidate, click that card now "
                       "to verify the listing details. Open a category only if a matching card is absent.")
        if link_count is not None:
            price_question = bool(_PRICE_QUESTION.search(observation.goal))
            selection_format = ('as an array in a JSON object keyed "listings". ' if price_question else
                                'such as {"listings":[10,12]}. ')
            row_scope = ("If the question limits the answer to a visible row, compare that row's cards in place. "
                         if price_question else "")
            seen = listing_screen_text(rows, canvas.size) if price_question else f"Visible text:\n{controls}"
            prompt = (
                f"Task: {observation.goal}\n"
                "Find all requested listings using the screenshot, their positions, and the indexed text. "
                f"The answer needs {link_count[0]} to {link_count[1]} distinct listing links. "
                "When the full requested group is visible, select the text index for each matching listing TITLE, "
                f"{selection_format}Exclude navigation, prices, labels, and unrelated cards. "
                "Check any required ordering, category, price, and result scope before selecting. "
                f"{row_scope}"
                "If the right results are not shown, issue one navigation command. "
                'Use {"action":"CLICK","value":null,"control":index} for a visible button or filter, '
                '{"action":"SEARCH","value":"short query","control":index} on the empty Keyword placeholder to search, '
                '{"action":"ENTER","value":null,"position":null} to submit a filled search field, or '
                '{"action":"SCROLL","value":"down","position":null} to see lower listings. '
                'Return {"listings":null} only if you cannot select or navigate. Never type or invent a URL. '
                f"Return one JSON object.\nPrevious actions: {json.dumps(recent, ensure_ascii=False)}.\n"
                f"{seen}"
            )
        raw = ask_observation(self.actor, canvas, prompt, observation, navigation=self.confirm)
        self.last_metrics.update(evidence_stage="locate", evidence_frame=frame, provider_format="evidence_invalid")
        if rows is not None:
            self.last_metrics["screen_text"] = rows
        try:
            selection = listing_selection(raw, rows, canvas.size, link_count) if link_count is not None else None
            if selection is None:
                navigation, box, candidate, status = parse_evidence_location(
                    raw, observation.goal, canvas.size, confirm=self.confirm, rows=navigation_rows, guarded_ocr=True,
                    navigation_only=link_count is not None, allow_click_box=direct_contact, tab_count=len(observation.tabs))
        except (ValueError, RecursionError):
            return None
        if selection is not None:
            indices, points = selection
            self.last_metrics["provider_format"] = "evidence_link_abstained"
            if indices:
                receipt = self.listing_preview(points)
                destinations = preview_destinations(receipt, frame, points, self.listing_site)
                self.last_metrics["listing_preview"] = receipt
                self.last_metrics["provider_format"] = "evidence_link_preview"
                if destinations is not None:
                    self.pending_links = frame, rows, indices, destinations, link_count
            return None
        self.last_metrics["provider_format"] = status
        if navigation is not None:
            return json.dumps(navigation.to_dict(), ensure_ascii=False)
        if box:
            crop, bounds = native_crop(observation.screenshot, canvas.size, box)
            self.last_metrics["evidence_bounds"] = list(bounds)
            if candidate is not None:
                self.pending_answer = frame, crop, bounds, candidate
            else:
                self.pending_crop = frame, crop, bounds, False
        return None

    def read(self, observation, frame, crop, bounds, *, publisher_region=False):
        canvas = self.actor.prepare_image(crop)
        visibility_check = ("If the requested details cannot be read, use null for answer. "
                            "A separate visual check will confirm that this is the correct item. "
                            if self.navigation_context is not None else
                            "If the requested subject, scope, or any required detail cannot be verified, use null for answer. ")
        prompt = (
            f"Question: {observation.goal}\n"
            "Inspect this crop. Identify the requested subject and list at most three brief, relevant facts. "
            "Omit unrelated text. For a count, count each visible group separately before adding. "
            "For a price range, give both endpoints. For a total price, read rates and pack sizes before calculating. "
            f"{visibility_check}"
            'Return one JSON object with exactly two keys: "evidence" (a list of short strings), '
            'then "answer" (short string or null).'
        )
        if self.price_discovery is not None:
            prompt = (
                "Transcribe the single dollar price shown in this crop. The item photo is checked separately. "
                'Return JSON with "evidence" containing the displayed price as a one-element list and '
                '"answer" containing the amount as a string of digits and an optional decimal point. '
                'If the printed price is unreadable, return {"evidence":[],"answer":null}.'
            )
        raw = ask_observation(self.actor, canvas, prompt, observation)
        self.last_metrics.update(evidence_stage="read", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_crop_sha256=pixel_hash(crop), provider_format="evidence_invalid")
        try:
            answer, status = evidence_answer(raw, observation.goal, allow_string_abstention=self.confirm,
                                             price_crop=self.price_discovery is not None)
        except (ValueError, RecursionError):
            return None
        if status == "evidence_abstained" and self.ranked_capture is not None and self.navigation_context is not None:
            address = contact_from_evidence(raw, observation.goal, publisher_region=publisher_region)
            if address is not None:
                answer, status = address, "evidence_contact_candidate"
        self.last_metrics["provider_format"] = status
        if answer is not None:
            if self.confirm or self.price_discovery is not None:
                self.pending_answer = frame, crop, bounds, answer
                return None
            return json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)
        return None

    def verify(self, observation, frame, crop, bounds, candidate):
        if isinstance(candidate, ReferenceTitleCrop):
            raw = self.actor.ask(self.actor.prepare_image(crop), (
                "Transcribe the full listing heading in this crop. Include every word and number. "
                'Return only JSON with the key "answer". Use {"answer":null} if any part is unreadable.'
            ))
            answer, status = confirmation_outcome(raw, candidate)
            self.last_metrics.update(evidence_stage="confirm", evidence_frame=frame, evidence_bounds=list(bounds),
                                     reference_title_crop_sha256=pixel_hash(crop), provider_format=status)
            if answer is None:
                self.last_metrics["stop_after_evidence"] = True
                return None
            return json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)
        request = self.price_discovery.request if self.price_discovery is not None else None
        color = request.hardcover_color if request is not None else None
        panel = confirmation_image(observation.screenshot, crop, bounds, self.navigation_context,
                                   focus_listing=color is not None)
        canvas = self.actor.prepare_image(panel)
        context_description = (
            "The bottom-left panel shows the previous viewport. Its orange circle marks the click that changed the view. "
            "Use it when the question depends on how this item was selected. "
            if self.navigation_context is not None else ""
        )
        prompt = (
            f"Question: {observation.goal}\n"
            "The upper-left panel is the current viewport and the blue outline marks the area enlarged on the right. "
            f"{context_description}"
            "Check the requested item's identity and the full scope in the overview, then read the enlargement. "
            "For a page, row, or search question, verify that the required results are visible. "
            "Give every requested component. A single price cannot answer a price range. "
            "If anything required is missing, return {\"answer\":null}. "
            "Otherwise return one JSON object with one key answer and a short string value."
        )
        if isinstance(candidate, ListingFacts):
            prompt = candidate.confirmation_prompt()
        elif self.reference_title_frame == frame:
            prompt = (
                "Transcribe the opened listing's full heading exactly from the enlarged image on the right. "
                'Keep every word even if the gallery card was shortened. Return only JSON with the key "answer". '
                'Use {"answer":null} if any heading text is unreadable.'
            )
        elif color is not None:
            shades = ", ".join(cover_color_labels(color))
            prompt = (
                "Inspect the current listing photo in the enlarged upper-left overview. Check the color of "
                f"the outer book cover itself. A match has one of these hues: {shades}. "
                "Confirm hardcover using the rigid cover or thick bound spine visible in the photo. "
                "Read the dollar amount from the crop on the right. Return JSON with exactly three keys: "
                '"cover_color" (visible hue or null), "binding" ("hardcover" if verified, otherwise null), '
                'and "answer" (the visible numeric amount as a string, or null if anything is unverified).'
            )
        elif request is not None and request.requires_selfie_cue:
            target = self.price_discovery.request.visual_target
            prompt = (
                f"Verify that the listing photo in the upper-left overview shows {target}. "
                f"{SELFIE_PHOTO_REQUIREMENT} "
                "Read the price independently from the enlargement on the right. "
                'Return only JSON with keys "answer" and "selfie_cue". Use "mirror_camera" if you can see '
                'the wearer holding their camera in the mirror, or "arms_length_portrait" for a visible '
                "arm's-length self-portrait of the wearer. "
                'If no cue is visible, return {"answer":null,"selfie_cue":null}. '
                'Otherwise put the transcribed dollar amount as a string in "answer", '
                "or null if the price is unreadable."
            )
        elif self.price_discovery is not None:
            target = self.price_discovery.request.visual_target
            prompt = (
                f"Check the listing photo in the upper-left overview. It must visibly show {target}. "
                "The blue outline marks the price field enlarged on the right. Read its amount independently. "
                'If the photo does not clearly match or the price is unreadable, return {"answer":null}. '
                'Otherwise return JSON with "answer" containing only the amount as a string of digits and '
                "an optional decimal point."
            )
        elif (self.ranked_capture is not None and self.navigation_context is not None
                and _EMAIL.fullmatch(candidate) and _EMAIL_REQUEST.search(observation.goal)):
            prompt = (
                f"Question: {observation.goal}\n"
                "The upper-left overview shows the opened listing. The blue outline is enlarged on the right. "
                f"{context_description}"
                "Check that the listing and its pictured subject are consistent with the question. "
                "Then read the displayed seller email verbatim from the right crop. If the listing clearly conflicts "
                "with the question or the field is unreadable, return {\"answer\":null}. "
                'Otherwise return one JSON object with key "answer" containing the exact visible email.'
            )
        raw = ask_observation(self.actor, canvas, prompt, observation)
        self.last_metrics.update(evidence_stage="confirm", evidence_frame=frame, evidence_bounds=list(bounds),
                                 evidence_panel_sha256=pixel_hash(panel), provider_format="evidence_invalid")
        if self.navigation_context is not None:
            previous, position = self.navigation_context
            self.last_metrics.update(evidence_context_sha256=pixel_hash(previous), evidence_context_position=list(position))
        tie = self.price_discovery.tie if self.price_discovery is not None else None
        checked = candidate
        if isinstance(candidate, ListingFacts) and candidate.field == "city":
            receipt = listing_price_ocr_receipt(observation.screenshot, candidate, self.screen_reader)
            self.last_metrics["listing_price_check"] = receipt
            confirmed, status, checked = checked_listing_confirmation(raw, candidate, receipt["rows"])
            if checked is None:
                # The price row is upright. Preserve its margins and skip orientation correction.
                retry = listing_price_ocr_receipt(observation.screenshot, candidate, self.price_retry_reader,
                                                   upright_retry=True)
                self.last_metrics["listing_price_upright_retry"] = retry
                confirmed, status, checked = checked_listing_confirmation(raw, candidate, retry["rows"])
        else:
            confirmed, status = confirmation_outcome(
                raw, candidate, request, expected_cents=tie.cents if tie is not None else None,
            )
        self.last_metrics["provider_format"] = status
        if confirmed is None and isinstance(candidate, ListingFacts):
            if checked is None:
                self.last_metrics["stop_after_evidence"] = True
            else:
                self.pending_fact_reread = frame, checked
                self.last_metrics["listing_fact_reread_pending"] = True
        return json.dumps(Action("ANSWER", confirmed).to_dict(), ensure_ascii=False) if confirmed is not None else None

    def reread_listing_facts(self, observation, frame, candidate):
        panel, bounds = listing_fact_reread_image(observation.screenshot, candidate)
        raw = self.actor.ask(self.actor.prepare_image(panel), candidate.reread_prompt())
        answer, status = confirmation_outcome(raw, candidate, listing_fact_reread=True)
        self.last_metrics.update(evidence_stage="confirm", evidence_frame=frame, provider_format=status,
                                 listing_fact_reread_sha256=pixel_hash(panel), listing_fact_reread_bounds=bounds)
        if answer is None:
            self.last_metrics["stop_after_evidence"] = True
            return None
        return json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)

    def verify_links(self, observation, frame, rows, selected, destinations, count):
        canvas = self.actor.prepare_image(observation.screenshot)
        seen = (listing_screen_text(rows, canvas.size) if _PRICE_QUESTION.search(observation.goal)
                else f"Visible text from the screenshot:\n{indexed_text(rows)}")
        raw = ask_observation(self.actor, canvas, (
            f"{seen}\n\n"
            f"Question: {observation.goal}\n"
            "Identify the complete set of matching listing titles using the screenshot. "
            "Check the scope, ordering, and visible card details. Exclude headings and controls. "
            'Return one JSON object keyed "listings". Its value must be an array of the actual text indices '
            f"from the listing titles, with {count[0]} to {count[1]} distinct entries. "
            "Use null if any required listing or condition is unverifiable."
        ), observation)
        self.last_metrics.update(evidence_stage="link_confirm", evidence_frame=frame,
                                 provider_format="evidence_link_disagreed")
        try:
            selection = listing_selection(raw, rows, canvas.size, count)
            answer = agreed_listing_answer(selected, destinations, selection[0] if selection is not None else None)
        except (ValueError, RecursionError):
            answer = None
        if answer is None:
            return None
        self.last_metrics["provider_format"] = "evidence_link_agreed"
        return json.dumps(answer.to_dict(), ensure_ascii=False)

    def predict(self, observation, history):
        if self.ranked_request is not None and history:
            first = history[0]["action"]
            if first["action"] == "SEARCH" and first["value"] == self.ranked_request[0]:
                return self.ranked_evidence.decide(observation, self.ranked_request, self.actor,
                                                   self.screen_reader, self.ranked_capture)
        frame = pixel_hash(observation.screenshot)
        if self.pending_fact_reread is not None:
            pending_frame, candidate = self.pending_fact_reread
            self.pending_fact_reread = None
            if frame == pending_frame:
                return self.reread_listing_facts(observation, frame, candidate)
        if self.pending_links is not None:
            pending_frame, rows, indices, destinations, count = self.pending_links
            self.pending_links = None
            if frame == pending_frame:
                return self.verify_links(observation, frame, rows, indices, destinations, count)
        if self.pending_answer is not None:
            pending_frame, crop, bounds, candidate = self.pending_answer
            self.pending_answer = None
            if frame == pending_frame:
                return self.verify(observation, frame, crop, bounds, candidate)
        if self.pending_crop is not None:
            pending_frame, crop, bounds, publisher_region = self.pending_crop
            self.pending_crop = None
            if frame == pending_frame:
                return self.read(observation, frame, crop, bounds, publisher_region=publisher_region)
        if (self.price_discovery is None and not history and self.ranked_capture is not None
                and len(observation.tabs) == 1 and not observation.reference_images):
            active = observation.tabs[0][1]
            request = price_request(observation.goal)
            if request is not None and same_origin(active, self.listing_site) and not urlsplit(active).query:
                self.price_discovery = PriceDiscovery(request, self.listing_site)
        if self.price_discovery is not None:
            raw, box = self.price_discovery.decide(observation, self.actor, self.screen_reader, self.ranked_capture)
            if box is not None:
                crop, bounds = native_crop(observation.screenshot, self.last_metrics["price_canvas_size"], box)
                self.last_metrics["evidence_bounds"] = list(bounds)
                self.pending_crop = frame, crop, bounds, False
            return raw
        key = frame_key(frame, observation.tabs, observation.active_tab)
        if key not in self.checked_frames:
            self.checked_frames.add(key)
            return self.locate(observation, frame, history)
        return self.actor.predict(observation, history, allow_answer=False)
