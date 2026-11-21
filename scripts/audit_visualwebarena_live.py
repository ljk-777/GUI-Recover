"""Reconstruct a live VisualWebArena study from its saved trajectories on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.answer_evidence import (
    REFERENCE_CONTACT_REGION, REFERENCE_NAVIGATION, REFERENCE_SEARCH, REFERENCE_SEARCH_ABSTAINED,
    REFERENCE_TITLE_AMBIGUOUS, REFERENCE_TITLE_CROP, REFERENCE_TITLE_HEADING, REFERENCE_TITLE_OPEN,
    breadcrumb_ocr_image, confirmation_image, confirmation_outcome, contact_from_evidence, evidence_answer,
    listing_fact_reread_image, listing_price_ocr_image, native_crop,
    parse_evidence_location, reference_contact_box, reference_search_action,
    visible_contact_request,
)
from gui_recover.environment import VisualObservation, frame_key
from gui_recover.listing_cards import (
    RESULT_VIEW_HEIGHT, ReferenceTitleCrop, ambiguous_reference_title, exact_reference_title_request,
    listing_breadcrumb_box, listing_title_heading, ranked_photo_request,
    reference_result_action, reference_result_destination, reference_title_crop, reference_title_open_action,
    reference_title_region, reference_title_request, with_listing_breadcrumb,
)
from gui_recover.listing_facts import (
    LISTING_FACTS_ABSTAINED, LISTING_FACTS_STAGES, SEARCH_PRICE_NAVIGATION, ListingFacts, listing_fact_decision,
)
from gui_recover.price_discovery import PRICE_DISCOVERY, PriceDiscovery, price_request, same_origin
from gui_recover.qwen_grounding import native_action_after_note, parse_visible_answer, read_reply
from gui_recover.ranked_cards import RankedCardEvidence
from gui_recover.runner import INVALID_ACTION_FEEDBACK, aggregate, write_json
from gui_recover.screen_text import empty_keyword_field, exact_initial_search, listing_detail
from gui_recover.verification import pixel_hash
from gui_recover.visible_links import agreed_listing_answer, link_answer_count, listing_selection, preview_destinations
from gui_recover.visual_text_audit import checked_file, read_json, require
from run_visualwebarena_live import (
    ANSWER_STUDIES, CONFIRM_MODES, EVIDENCE_STUDIES, HOLDOUT_PROTOCOL_SHA256, HOLDOUT_TASK_SET,
    FRESH_HOLDOUT_TASK_SET, INPUT_HOLDOUT_TASK_SET, INPUT_RETRIEVAL_TASK_SET, INPUT_TASK_SETS,
    LINK_MODES, NAVIGATION_CONTEXT_MODES, OCR_MODES, STUDY_MODES, SUPPLEMENTAL_TASK_ID, TASK_SETS,
    PRICE_DEV_MODEL_CALLS, PRICE_DEV_TASK_SET, category_search_code, exact_vwa_success, holdout_profile, input_fingerprints,
    replace_keyword_code, sort_listing_code, tied_price_filter_code, tied_price_return_code, url_origin,
)


def verify_ocr_rows(rows, size):
    require(isinstance(rows, list) and len(rows) <= 56, "Unexpected OCR row count")
    for line in rows:
        coords = line["bounds"]
        require(isinstance(line["text"], str) and 0 < len(line["text"]) <= 100
                and len(coords) == 4 and all(type(value) is int for value in coords)
                and 0 <= coords[0] < coords[2] <= size[0] and 0 <= coords[1] < coords[3] <= size[1],
                "Invalid OCR bounds")


def replay_price_ocr(screen_path, candidate, review, *, enlarge=True):
    with Image.open(screen_path) as screen:
        # The prior policy uses the same enlarged crop but has no scale argument.
        if enlarge:
            image, bounds = listing_price_ocr_image(screen, candidate)
        else:
            image, bounds = listing_price_ocr_image(screen, candidate, enlarge=False)
    require(review["bounds"] == list(bounds) and review["pixels_sha256"] == pixel_hash(image),
            "Price OCR reread differs from the saved screenshot")
    verify_ocr_rows(review["rows"], image.size)
    return review["rows"]


def validate_tab_state(state):
    require(isinstance(state, dict) and set(state) == {"tabs", "active_tab"}
            and isinstance(state["tabs"], list) and type(state["active_tab"]) is int
            and 0 <= state["active_tab"] < len(state["tabs"]) <= 256, "Malformed browser tabs")
    for tab in state["tabs"]:
        require(isinstance(tab, list) and len(tab) == 2 and all(isinstance(field, str) for field in tab)
                and len(tab[0]) <= 160 and len(tab[1]) <= 512, "Malformed browser tab title or address")


def check_reference_images(directory, references):
    hashes = []
    images = []
    for index, reference in enumerate(references, start=1):
        name = f"reference-{index:02d}.png"
        require(reference["filename"] == name, "Unexpected task reference filename")
        with Image.open(checked_file(directory / name, reference["sha256"])) as image:
            require(list(image.size) == reference["size"] and pixel_hash(image) == reference["pixels"],
                    "Task reference pixels differ from the recorded input")
            images.append(image.convert("RGB"))
        hashes.append(reference["pixels"])
    return hashes, images


def replay_ranked_decision(scan, request, screen, goal, prediction):
    metrics = prediction["metrics"]

    class RecordedActor:
        last_metrics = {}

        def prepare_image(self, image):
            return image.resize(tuple(metrics["ranked_canvas_size"])) if image is screen else image

        def ask(self, _image, _prompt):
            require(not self.last_metrics and isinstance(metrics["provider_output"], str),
                    "Ranked evidence needs exactly one recorded provider reply")
            self.last_metrics = {"peak_cuda_allocated_bytes": metrics["peak_cuda_allocated_bytes"]}
            return metrics["provider_output"]

    def ocr(_screen, size):
        rows = metrics["ranked_screen_text"]
        verify_ocr_rows(rows, size)
        return rows

    actor = RecordedActor()
    raw = scan.decide(VisualObservation(goal, screen), request, actor, ocr, lambda: metrics["ranked_receipt"])
    require(raw == prediction["raw"], "Ranked evidence does not reproduce the recorded action")
    keys = {key for key in metrics if key.startswith("ranked_") or key in (
        "model_invocations", "provider_output", "provider_format", "stop_after_evidence", "evidence_stage", "evidence_frame")}
    require({key: metrics[key] for key in keys} == {key: value for key, value in actor.last_metrics.items()
                                                   if key != "peak_cuda_allocated_bytes"},
            "Ranked evidence panel, viewport or reply differs from the recorded decision")
    return raw


def replay_price_decision(discovery, screen, goal, tabs, prediction):
    metrics = prediction["metrics"]

    class RecordedActor:
        last_metrics = {}

        def prepare_image(self, image):
            return image.resize(tuple(metrics["price_canvas_size"])) if image is screen else image

        def ask(self, _image, _prompt, **_kwargs):
            require(not self.last_metrics and isinstance(metrics["provider_output"], str),
                    "Price discovery needs exactly one recorded provider reply")
            self.last_metrics = {"peak_cuda_allocated_bytes": metrics["peak_cuda_allocated_bytes"]}
            return metrics["provider_output"]

    def ocr(_screen, size):
        rows = metrics["price_screen_text"]
        verify_ocr_rows(rows, size)
        return rows

    actor = RecordedActor()
    observation = VisualObservation(goal, screen, tabs=tuple(map(tuple, tabs["tabs"])), active_tab=tabs["active_tab"])
    raw, box = discovery.decide(observation, actor, ocr, lambda: metrics["price_card_receipt"])
    require(raw == prediction["raw"], "Price discovery does not reproduce the recorded action")
    keys = {key for key in metrics if key.startswith("price_") or key in (
        "model_invocations", "provider_output", "provider_format", "stop_after_evidence", "evidence_stage", "evidence_frame")}
    require({key: metrics[key] for key in keys} == {key: value for key, value in actor.last_metrics.items()
                                                   if key != "peak_cuda_allocated_bytes"},
            "Price discovery differs from the recorded scope, panel, or decision")
    return box


def audit_evidence(directory, predictions, frame_before_prediction, frames, goal, *,
                   confirm=False, click_origins=None, compact_navigation=False, use_screen_text=False,
                   guarded_ocr=False, direct_navigation=False, listing_site=None, use_ranked_cards=False,
                   tab_states=None, reference_images=()):
    visited = set()
    answers = set()
    stages = Counter()
    pending = None
    pending_confirmation = None
    listing_fact_reread_pending = False
    listing_price_checked = False
    pending_links = None
    link_count = link_answer_count(goal) if listing_site is not None else None
    ranked_request = ranked_photo_request(goal) if use_ranked_cards else None
    reference_query = None
    reference_refined = False
    reference_destination = None
    listing_fact_target = None
    scan = None
    price_scan = None
    for prediction in predictions:
        step = prediction["step"]
        index = frame_before_prediction[step]
        frame = frames[index]
        tabs = tab_states[index] if tab_states else {}
        key = frame_key(frame, tabs.get("tabs", ()), tabs.get("active_tab", 0))
        metrics = prediction["metrics"]
        raw = prediction["raw"]
        reply = metrics["provider_output"]
        if scan is not None:
            with Image.open(directory / f"{index:04d}.png") as picture:
                screen = picture.convert("RGB")
            reconstructed = replay_ranked_decision(scan, ranked_request, screen, goal, prediction)
            stages[metrics["provider_format"]] += 1
            if reconstructed is not None and parse_action(reconstructed).action == "ANSWER":
                answers.add(step)
            continue
        if metrics.get("evidence_stage") == PRICE_DISCOVERY:
            require(use_ranked_cards and use_screen_text and direct_navigation and tabs and not reference_images
                    and metrics["evidence_frame"] == frame and pending is None and pending_confirmation is None,
                    "Price discovery requires a registered visible Classifieds context")
            if price_scan is None:
                request = price_request(goal)
                start = tabs["tabs"][tabs["active_tab"]][1]
                require(step == 0 and len(tabs["tabs"]) == 1 and request is not None
                        and same_origin(start, listing_site) and not urlsplit(start).query,
                        "Price discovery must begin from the task's local Classifieds home page")
                price_scan = PriceDiscovery(request, listing_site)
            with Image.open(directory / f"{index:04d}.png") as picture:
                screen = picture.convert("RGB")
            box = replay_price_decision(price_scan, screen, goal, tabs, prediction)
            if box is not None:
                crop, bounds = native_crop(screen, metrics["price_canvas_size"], box)
                require(list(bounds) == metrics["evidence_bounds"], "Price evidence crop differs from the visible field")
                pending = frame, list(bounds), pixel_hash(crop), False
            stages[metrics["provider_format"]] += 1
            continue
        if pending_links is not None and frame == pending_links[0]:
            stage = "link_confirm"
        elif pending_confirmation is not None and frame == pending_confirmation[0]:
            stage = "confirm"
        elif pending is not None and frame == pending[0]:
            stage = "read"
        elif key not in visited:
            stage = "locate"
        else:
            stage = None
        require(metrics.get("evidence_stage") == stage, "Evidence check schedule disagrees with the observed frame")
        require("listing_price_check" not in metrics or (stage == "confirm" and not listing_fact_reread_pending),
                "Price OCR check must occur during the first compound confirmation")
        require("listing_price_native_retry" not in metrics or "listing_price_check" in metrics,
                "Native price OCR retry requires the enlarged price check")
        require("listing_price_ocr" not in metrics or (stage == "confirm" and listing_fact_reread_pending),
                "Price OCR reread is outside isolated compound confirmation")
        require("listing_preview" not in metrics or (stage == "locate" and link_count is not None),
                "Link preview is outside the registered link-localization stage")
        if stage != "link_confirm":
            pending_links = None
        if stage is None:
            if raw is None:
                try:
                    blocked = read_reply(reply)
                except json.JSONDecodeError:
                    blocked = native_action_after_note(reply)
                require(metrics["provider_format"] == "answer_blocked" and blocked["action"].upper() == "ANSWER",
                        "Unexpected non-action navigation decision")
            elif raw:
                require(parse_action(raw).action != "ANSWER", "Navigation bypassed the evidence check")
            continue
        require(metrics["evidence_frame"] == frame, "Evidence was attributed to the wrong screenshot")
        stages[stage] += 1
        status = "evidence_invalid"
        if stage == "locate":
            visited.add(key)
            pending = None
            if metrics["provider_format"] in LISTING_FACTS_STAGES:
                require(confirm and use_ranked_cards and use_screen_text and not reference_images and tabs and reply is None,
                        "Compound listing evidence requires a registered screenshot and active tab")
                rows, size = metrics["screen_text"], metrics["provider_image_size"]
                verify_ocr_rows(rows, size)
                captured = False

                def capture_cards():
                    nonlocal captured
                    captured = True
                    return metrics["listing_fact_receipt"]

                active = tabs["tabs"][tabs["active_tab"]][1]
                with Image.open(directory / f"{index:04d}.png") as picture:
                    screen = picture.convert("RGB")
                    decision = listing_fact_decision(goal, active, listing_site, rows, size, screen, capture_cards,
                                                     target=listing_fact_target)
                    require(decision is not None and decision.status == metrics["provider_format"],
                            "Compound listing evidence disagrees with the recorded screenshot")
                    require(captured == ("listing_fact_receipt" in metrics), "Unexpected listing receipt")
                    action = parse_action(raw) if raw is not None else None
                    require(action == decision.action, "Listing navigation differs from the visible evidence")
                    if decision.facts is not None:
                        _, bounds = native_crop(screen, size, decision.facts.bounds)
                        require(list(bounds) == metrics["evidence_bounds"], "Unexpected listing evidence crop")
                        pending_confirmation = frame, list(bounds), decision.facts
                if decision.target is not None:
                    listing_fact_target = decision.target
                require((metrics.get("stop_after_evidence") is True) == (decision.status == LISTING_FACTS_ABSTAINED),
                        "Compound listing evidence stop disagrees with the reconstructed decision")
                stages[decision.status.removeprefix("evidence_")] += 1
                continue
            if metrics["provider_format"] in (REFERENCE_TITLE_OPEN, REFERENCE_TITLE_AMBIGUOUS):
                require(use_ranked_cards and use_screen_text and exact_reference_title_request(goal)
                        and len(reference_images) == 1 and reference_destination is None and tabs
                        and url_origin(tabs["tabs"][tabs["active_tab"]][1]) == listing_site
                        and reply is None, "Reference title matching requires a reference photo and the registered site")
                receipt = metrics["reference_card_receipt"]
                with Image.open(directory / f"{index:04d}.png") as screen:
                    if metrics["provider_format"] == REFERENCE_TITLE_AMBIGUOUS:
                        require(raw is None and metrics.get("stop_after_evidence") is True
                                and ambiguous_reference_title(screen, reference_images[0], receipt, listing_site, goal),
                                "Title abstention requires duplicate matching photo pixels")
                        stages["reference_title_ambiguous"] += 1
                        continue
                    action = reference_title_open_action(screen, reference_images[0], receipt, listing_site)
                    require(raw is not None and action is not None and parse_action(raw) == action,
                            "Title navigation differs from the visible photo match")
                    reference_destination = reference_result_destination(receipt, screen.size, action)
                stages["reference_title_open"] += 1
                continue
            if metrics["provider_format"] in (REFERENCE_TITLE_HEADING, REFERENCE_TITLE_CROP):
                require(use_ranked_cards and use_screen_text and exact_reference_title_request(goal)
                        and reference_destination is not None and tabs
                        and tabs["tabs"][tabs["active_tab"]][1] == reference_destination
                        and raw is None and reply is None,
                        "Full title evidence requires the matched listing detail")
                rows, size = metrics["screen_text"], metrics["provider_image_size"]
                verify_ocr_rows(rows, size)
                if "reference_title_breadcrumb" in metrics:
                    review = metrics["reference_title_breadcrumb"]
                    band = listing_breadcrumb_box(rows, size)
                    require(band is not None, "Breadcrumb reread needs a missing breadcrumb above a visible heading")
                    with Image.open(directory / f"{index:04d}.png") as screen:
                        image, native_bounds = breadcrumb_ocr_image(screen, size, band)
                    require(review["bounds"] == list(native_bounds) and review["pixels_sha256"] == pixel_hash(image),
                            "Breadcrumb reread differs from the saved screenshot")
                    verify_ocr_rows(review["rows"], (band[2] - band[0], band[3] - band[1]))
                    recovered = with_listing_breadcrumb(rows, review["rows"], band)
                    require(recovered is not rows, "Breadcrumb reread did not produce a unique breadcrumb")
                    rows = recovered
                    stages["reference_title_breadcrumb"] += 1
                isolated = metrics["provider_format"] == REFERENCE_TITLE_CROP
                if isolated:
                    title = reference_title_crop(rows, size)
                    require(title is not None and metrics["reference_title_missing_number"] is title.missing_number,
                            "Isolated title crop differs from the visible heading and breadcrumb")
                    box = title.bounds
                else:
                    heading = listing_title_heading(rows, size)
                    require(heading is not None, "Full heading no longer matches the visible breadcrumb")
                    box, title = heading
                with Image.open(directory / f"{index:04d}.png") as screen:
                    _, bounds = native_crop(screen, size, box, pad=not isolated)
                require(list(bounds) == metrics["evidence_bounds"], "Unexpected title heading crop")
                pending_confirmation = frame, list(bounds), title
                stages["reference_title_crop" if isolated else "reference_title_heading"] += 1
                continue
            if metrics["provider_format"] == "evidence_reference_region":
                require(use_ranked_cards and reference_title_request(goal) and len(reference_images) == 1
                        and raw is None and reply is None, "Invalid reference-title evidence step")
                with Image.open(directory / f"{index:04d}.png") as screen:
                    bounds = reference_title_region(screen, reference_images[0], metrics["reference_card_receipt"], listing_site)
                    require(bounds is not None and list(bounds) == metrics["evidence_bounds"],
                            "Reference crop differs from the visible image match")
                    pending = frame, list(bounds), pixel_hash(screen.crop(bounds)), False
                stages["reference_region"] += 1
                continue
            if metrics["provider_format"] in (REFERENCE_SEARCH, REFERENCE_SEARCH_ABSTAINED):
                require(use_ranked_cards and use_screen_text and direct_navigation and step == 0
                        and len(reference_images) == 1 and tabs
                        and url_origin(tabs["tabs"][tabs["active_tab"]][1]) == listing_site,
                        "Reference search needs the registered site and initial reference image")
                size = metrics["search_canvas_size"]
                rows = metrics["screen_text"]
                verify_ocr_rows(rows, size)
                require(empty_keyword_field(rows) is not None and exact_initial_search(goal, rows, size) is None,
                        "Reference search requires an empty field and a nonliteral query")
                action = reference_search_action(reply, rows, size)
                status = REFERENCE_SEARCH if action is not None else REFERENCE_SEARCH_ABSTAINED
                require(metrics["provider_format"] == status
                        and (raw is None if action is None else raw is not None and parse_action(raw) == action),
                        "Search action differs from its reference-only query and visible field")
                if action is not None:
                    reference_query = action.value
                stages["reference_search"] += 1
                continue
            if metrics["provider_format"] == REFERENCE_CONTACT_REGION:
                require(use_ranked_cards and use_screen_text and reference_destination is not None
                        and tabs and tabs["tabs"][tabs["active_tab"]][1] == reference_destination
                        and raw is None and reply is None,
                        "Contact localization requires a visibly matched item detail")
                size = metrics["provider_image_size"]
                rows = metrics["screen_text"]
                verify_ocr_rows(rows, size)
                box = reference_contact_box(goal, rows)
                require(box is not None, "Expected a visible publisher email")
                with Image.open(directory / f"{index:04d}.png") as screen:
                    crop, bounds = native_crop(screen, size, box)
                    require(list(bounds) == metrics["evidence_bounds"], "Contact crop differs from the visible email row")
                    pending = frame, list(bounds), pixel_hash(crop), True
                stages["reference_contact_region"] += 1
                continue
            if metrics["provider_format"] in REFERENCE_NAVIGATION.values():
                require(use_ranked_cards and use_screen_text and reference_query is not None and len(reference_images) == 1
                        and reply is None and raw is not None and tabs
                        and url_origin(tabs["tabs"][tabs["active_tab"]][1]) == listing_site,
                        "Reference navigation requires the original image query and matching site")
                rows = metrics["screen_text"]
                size = metrics["provider_image_size"]
                receipt = metrics["reference_card_receipt"]
                verify_ocr_rows(rows, size)
                with Image.open(directory / f"{index:04d}.png") as screen:
                    action = reference_result_action(screen, reference_images[0], receipt, listing_site, rows,
                                                     reference_query, goal,
                                                     refinement_size=None if reference_refined else size)
                    require(action is not None and action == parse_action(raw),
                            "Result action differs from the visible cards")
                    if action.action == "CLICK":
                        reference_destination = reference_result_destination(receipt, screen.size, action)
                if action.action == "SEARCH_REPLACE":
                    reference_refined = True
                    reference_query = action.value
                status = REFERENCE_NAVIGATION[action.action]
                require(metrics["provider_format"] == status, "Unexpected reference navigation stage")
                stages[status.removeprefix("evidence_")] += 1
                continue
            size = metrics["provider_image_size"]
            require(("screen_text" in metrics) == use_screen_text, "Unexpected OCR mode")
            rows = metrics.get("screen_text")
            if rows is not None:
                verify_ocr_rows(rows, size)
            shortcut = None
            if direct_navigation and rows is not None and step == 0:
                try:
                    shortcut = exact_initial_search(goal, rows, size)
                except ActionError:
                    pass
            if shortcut is not None:
                require(metrics["provider_format"] == "evidence_exact_search" and reply is None
                        and parse_action(raw) == shortcut and "listing_preview" not in metrics,
                        "Search action differs from the visible field and task text")
                stages["grounded_search"] += 1
                if ranked_request is not None:
                    scan = RankedCardEvidence(listing_site)
                if use_ranked_cards and len(reference_images) == 1 and exact_reference_title_request(goal):
                    reference_query = shortcut.value
                    reference_refined = True
                continue
            if direct_navigation and link_count is None and rows is not None and listing_detail(rows):
                rows = None
            try:
                selected = listing_selection(reply, rows, size, link_count) if link_count is not None else None
                if selected is None:
                    navigation, box, candidate, status = parse_evidence_location(
                        reply, goal, size, confirm=confirm, compact_navigation=compact_navigation, rows=rows,
                        guarded_ocr=guarded_ocr, navigation_only=link_count is not None,
                        allow_click_box=use_ranked_cards and step == 0 and visible_contact_request(goal, rows),
                        tab_count=len(tabs.get("tabs", ())))
            except (ValueError, RecursionError):
                require(raw is None and "listing_preview" not in metrics,
                        "Unusable evidence localization generated a browser action or link preview")
            else:
                if selected is not None:
                    indices, points = selected
                    require(raw is None and bool(indices) == ("listing_preview" in metrics),
                            "Listing selection bypassed the independent check or disagrees with the recorded preview")
                    status = "evidence_link_abstained"
                    if indices:
                        destinations = preview_destinations(metrics["listing_preview"], frame, points, listing_site)
                        status = "evidence_link_preview"
                        if destinations is not None:
                            pending_links = frame, rows, indices, destinations, size
                elif navigation is not None:
                    require("listing_preview" not in metrics and raw is not None,
                            "Navigation generated an unrequested preview or failed to execute")
                    require(parse_action(raw) == navigation, "Evidence search executed an unexpected action")
                else:
                    require("listing_preview" not in metrics and raw is None,
                            "Evidence region bypassed the independent check or generated a link preview")
                    if box is not None:
                        with Image.open(directory / f"{index:04d}.png") as screen:
                            crop, bounds = native_crop(screen, size, box)
                        require(metrics["evidence_bounds"] == list(bounds), "Evidence crop bounds differ from localization")
                        if candidate is not None:
                            pending_confirmation = frame, list(bounds), candidate
                        else:
                            pending = frame, list(bounds), pixel_hash(crop), False
        elif stage == "read":
            _, bounds, crop_digest, publisher_region = pending
            pending = None
            require(metrics["evidence_bounds"] == bounds and metrics["evidence_crop_sha256"] == crop_digest,
                    "Evidence-read pixels differ from the recorded screenshot crop")
            candidate = None
            try:
                candidate, status = evidence_answer(reply, goal, allow_string_abstention=confirm,
                                                    price_crop=price_scan is not None)
            except (ValueError, RecursionError):
                pass
            if (status == "evidence_abstained" and use_ranked_cards and click_origins is not None
                    and click_origins.get(index) is not None):
                address = contact_from_evidence(reply, goal, publisher_region=publisher_region)
                if address is not None:
                    candidate, status = address, "evidence_contact_candidate"
            if candidate is None:
                require(raw is None, "Unsupported answer reached the browser")
            elif confirm:
                require(raw is None, "Candidate answer bypassed the independent check")
                pending_confirmation = frame, bounds, candidate
            else:
                action = parse_action(raw)
                require(action.action == "ANSWER" and action.value == candidate, "Answer differs from the crop read")
                answers.add(step)
        elif stage == "link_confirm":
            _, rows, indices, destinations, size = pending_links
            pending_links = None
            require(size == metrics["provider_image_size"], "Link verification used a different prepared viewport")
            try:
                selected = listing_selection(reply, rows, size, link_count)
                action = agreed_listing_answer(indices, destinations, selected[0] if selected is not None else None)
            except (ValueError, RecursionError):
                action = None
            if action is None:
                status = "evidence_link_disagreed"
                require(raw is None, "Link submission reached the browser without agreement")
            else:
                status = "evidence_link_agreed"
                require(raw is not None and parse_action(raw) == action, "Submission differs from the agreed link preview")
                answers.add(step)
        else:
            _, bounds, candidate = pending_confirmation
            pending_confirmation = None
            isolated_title = isinstance(candidate, ReferenceTitleCrop)
            isolated_facts = listing_fact_reread_pending
            listing_fact_reread_pending = False
            if isolated_title or isolated_facts:
                with Image.open(directory / f"{index:04d}.png") as screen:
                    if isolated_facts:
                        require(isinstance(candidate, ListingFacts), "Field reread requires the original compound candidate")
                        panel, crop_bounds = listing_fact_reread_image(screen, candidate)
                        require(metrics["listing_fact_reread_bounds"] == crop_bounds
                                and metrics["listing_fact_reread_sha256"] == pixel_hash(panel),
                                "Field reread must inspect the original price and label crops")
                    else:
                        require(metrics["evidence_bounds"] == bounds
                                and metrics["reference_title_crop_sha256"] == pixel_hash(screen.crop(bounds)),
                                "Title confirmation must inspect the recorded heading crop")
                require("evidence_panel_sha256" not in metrics and "evidence_context_sha256" not in metrics
                        and "listing_fact_reread_pending" not in metrics,
                        "Isolated confirmation must be bounded to its crop and may run only once")
                confirmed, status = confirmation_outcome(reply, candidate, listing_fact_reread=isolated_facts)
                if "listing_price_ocr" in metrics:
                    require(isolated_facts and not listing_price_checked and candidate.field == "city"
                            and status == "evidence_verify_disagreed",
                            "Price OCR reread requires a disagreeing price-and-city reread")
                    rows = replay_price_ocr(directory / f"{index:04d}.png", candidate, metrics["listing_price_ocr"])
                    confirmed, status = confirmation_outcome(
                        reply, candidate.with_price_from_crop(rows) or candidate, listing_fact_reread=True,
                    )
                    stages["listing_price_ocr"] += 1
                listing_price_checked = False
                require(metrics["provider_format"] == status, "Isolated confirmation status differs from the recorded reply")
                require((metrics.get("stop_after_evidence") is True) == (confirmed is None),
                        "An unresolved isolated confirmation must stop without submission")
                if confirmed is not None:
                    action = parse_action(raw)
                    require(action.action == "ANSWER" and action.value == confirmed, "Submission differs from the verified crop")
                    answers.add(step)
                else:
                    require(raw is None, "An unconfirmed crop read reached the browser")
                stages["listing_fact_reread" if isolated_facts else "reference_title_crop_confirm"] += 1
                continue
            require("reference_title_crop_sha256" not in metrics and "listing_fact_reread_sha256" not in metrics,
                    "Unexpected isolated confirmation")
            context = None
            origin = click_origins.get(index) if click_origins is not None else None
            if origin is not None:
                origin_index, position = origin
                with Image.open(directory / f"{origin_index:04d}.png") as previous:
                    context = previous.copy(), position
                require(metrics["evidence_context_sha256"] == frames[origin_index]
                        and metrics["evidence_context_position"] == position, "Context differs from the recorded click")
                stages["navigation_context"] += 1
            else:
                require("evidence_context_sha256" not in metrics, "Untracked navigation context reached confirmation")
            request = price_scan.request if price_scan is not None else None
            with Image.open(directory / f"{index:04d}.png") as screen:
                panel = confirmation_image(screen, screen.crop(bounds), bounds, context,
                                           focus_listing=request is not None and request.hardcover_color is not None)
            require(metrics["evidence_bounds"] == bounds and metrics["evidence_panel_sha256"] == pixel_hash(panel),
                    "Confirmation pixels differ from the recorded viewport and crop")
            tie = price_scan.tie if price_scan is not None else None
            checked = candidate
            listing_price_checked = "listing_price_check" in metrics
            if listing_price_checked:
                # Prior policies do not provide this helper and never record this receipt.
                from gui_recover.answer_evidence import checked_listing_confirmation

                require(isinstance(candidate, ListingFacts) and candidate.field == "city",
                        "Price OCR check requires a price-and-city candidate")
                rows = replay_price_ocr(directory / f"{index:04d}.png", candidate, metrics["listing_price_check"])
                confirmed, status, checked = checked_listing_confirmation(reply, candidate, rows)
                stages["listing_price_check"] += 1
                if "listing_price_native_retry" in metrics:
                    require(checked is None, "Native price OCR retry requires an unreadable enlarged crop")
                    rows = replay_price_ocr(directory / f"{index:04d}.png", candidate,
                                            metrics["listing_price_native_retry"], enlarge=False)
                    confirmed, status, checked = checked_listing_confirmation(reply, candidate, rows)
                    stages["listing_price_native_retry"] += 1
            else:
                confirmed, status = confirmation_outcome(
                    reply, candidate, request, expected_cents=tie.cents if tie is not None else None,
                )
            if metrics.get("listing_fact_reread_pending") is True:
                require(isinstance(checked, ListingFacts) and confirmed is None and not metrics.get("stop_after_evidence"),
                        "Only unconfirmed compound fields can schedule an isolated reread")
                pending_confirmation = frame, bounds, checked
                listing_fact_reread_pending = True
            elif isinstance(candidate, ListingFacts):
                require((metrics.get("stop_after_evidence") is True) == (confirmed is None),
                        "An unresolved compound answer must stop without submission")
            if status == "evidence_verify_agreed":
                action = parse_action(raw)
                require(action.action == "ANSWER" and action.value == confirmed, "Submission differs from agreed answer")
                answers.add(step)
            else:
                require(raw is None, "Answer reached the browser without agreement")
        require(metrics["provider_format"] == status,
                f"Evidence status at prediction {step}: reported {metrics['provider_format']}, reconstructed {status}")
    return stages, answers


def audit_episode(directory, budget, recovery_mode, retry_mode, *, answer_check=False, answer_study=False,
                  evidence_check=False, confirm=False, use_navigation_context=False, compact_navigation=False,
                  use_screen_text=False, guarded_ocr=False, direct_navigation=False, listing_site=None,
                  use_ranked_cards=False, use_browser_context=False, model_budget=None):
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    row = read_json(directory / "episode.json")
    require(events[0]["event"] == "reset" and events[-1] == {"event": "episode_end", **row},
            f"Incomplete trajectory: {directory.name}")
    require(row["reason"] != "runtime_error" and row["recovery"] == recovery_mode and row["fault_kind"] == "none",
            f"Unexpected runtime or intervention: {directory.name}")
    require(row["invalid_action_retry"] == retry_mode, f"Incorrect retry mode: {directory.name}")
    require(row.get("answer_check", False) == answer_check, f"Incorrect answer check mode: {directory.name}")
    by_event = {}
    for event in events:
        by_event.setdefault(event["event"], []).append(event)
    predictions = by_event["prediction"]
    if model_budget is not None:
        calls_before = 0
        for prediction in predictions:
            calls = prediction["metrics"]["model_invocations"]
            require(calls in (0, 1) and calls_before < model_budget, "Prediction exceeded the model call budget")
            calls_before += calls
        if row["reason"] == "model_call_limit":
            require(calls_before == model_budget and len(predictions) < budget, "Model budget stop disagrees with the trace")
    transitions = by_event.get("transition", [])
    tab_states = [event.get("browser_tabs") for event in [events[0], *transitions]]
    if use_browser_context:
        for state in tab_states:
            validate_tab_state(state)
    else:
        require(not any(tab_states) and "reference_images" not in events[0], "Unexpected browser context")
    reference_hashes, references = check_reference_images(directory, events[0].get("reference_images", []))
    require(0 < len(predictions) <= budget and [e["step"] for e in predictions] == list(range(len(predictions))),
            f"Invalid prediction budget: {directory.name}")
    stops = [p["step"] for p in predictions if p["metrics"].get("stop_after_evidence") is True]
    require(stops == ([len(predictions) - 1] if row["reason"] == "evidence_abstained" and use_ranked_cards else []),
            "Evidence abstention disagrees with the final decision")
    invalid_steps = []
    valid_steps = []
    formats = Counter()
    transition_by_step = {event["step"]: event for event in transitions}
    frame_before_prediction = {}
    checked_frames = set()
    checked_answers = set()
    observation = events[0]
    frame_index = 0
    for prediction in predictions:
        step = prediction["step"]
        metrics = prediction["metrics"]
        provider_format = metrics["provider_format"]
        formats[provider_format] += 1
        is_shortcut = direct_navigation and use_screen_text and provider_format == "evidence_exact_search" and step == 0
        reference_region = use_browser_context and use_ranked_cards and provider_format in (
            "evidence_reference_region", REFERENCE_CONTACT_REGION, REFERENCE_TITLE_OPEN, REFERENCE_TITLE_HEADING,
            REFERENCE_TITLE_AMBIGUOUS, REFERENCE_TITLE_CROP, *REFERENCE_NAVIGATION.values(), *LISTING_FACTS_STAGES,
        )
        ranked = use_ranked_cards and metrics.get("evidence_stage") in ("ranked_cards", PRICE_DISCOVERY)
        require(metrics["model_invocations"] in ((0, 1) if ranked else (0 if is_shortcut or reference_region else 1,)),
                "Unexpected model invocation count for the decision")
        digest = observation["sha256"]
        tabs = tab_states[frame_index] or {}
        key = frame_key(digest, tabs.get("tabs", ()), tabs.get("active_tab", 0))
        if metrics["model_invocations"] and use_browser_context:
            if provider_format in (REFERENCE_SEARCH, REFERENCE_SEARCH_ABSTAINED):
                require(use_ranked_cards and len(reference_hashes) == 1 and "browser_tabs" not in metrics
                        and metrics.get("reference_pixels") == reference_hashes,
                        "Reference-only model inputs differ from the registered image")
            elif metrics.get("evidence_stage") == PRICE_DISCOVERY:
                require(use_ranked_cards and not reference_hashes and "browser_tabs" not in metrics
                        and "price_panel_pixels" in metrics,
                        "Ranked price judgments must inspect only the reconstructed visible photo panel")
            elif "reference_title_crop_sha256" in metrics:
                require(use_ranked_cards and len(reference_hashes) == 1 and metrics.get("evidence_stage") == "confirm"
                        and "browser_tabs" not in metrics and "reference_pixels" not in metrics,
                        "Isolated title confirmation must exclude tab metadata and reference images")
            elif "listing_fact_reread_sha256" in metrics:
                require(use_ranked_cards and not reference_hashes and metrics.get("evidence_stage") == "confirm"
                        and "browser_tabs" not in metrics and "reference_pixels" not in metrics,
                        "Isolated field reread must exclude browser metadata")
            else:
                require(metrics.get("browser_tabs") == tabs and metrics.get("reference_pixels", []) == reference_hashes,
                        "Model inputs differ from the task reference images or active tabs")
        frame_before_prediction[step] = frame_index
        if step in transition_by_step:
            observation = transition_by_step[step]
            frame_index += 1
        checking = prediction.get("decision") == "answer_check"
        require(checking == (answer_check and key not in checked_frames),
                "Answer check schedule disagrees with the observed frame")
        if checking:
            require(prediction["observation_sha256"] == digest, "Answer check used an unexpected screenshot")
            checked_frames.add(key)
            try:
                candidate = parse_visible_answer(metrics["provider_output"])
            except (ValueError, RecursionError):
                candidate = None
                expected_format = "answer_invalid"
            else:
                expected_format = "answer_abstained" if candidate is None else "answer_json"
            require(provider_format == expected_format, "Provider answer status differs from its raw reply")
            if candidate is None:
                require(prediction["raw"] is None, "Missing answer generated a browser action")
                continue
            action = parse_action(prediction["raw"])
            require(action.action == "ANSWER" and action.value == candidate, "Answer differs from its raw reply")
            checked_answers.add(step)
        if evidence_check and prediction["raw"] is None:
            continue
        try:
            parse_action(prediction["raw"])
        except ActionError:
            invalid_steps.append(step)
        else:
            valid_steps.append(step)
    require([e["step"] for e in transitions] == valid_steps, "Unparsed action reached the browser")
    require([e["step"] for e in by_event.get("error", [])] == invalid_steps, "Invalid prediction log differs")
    for event in transitions:
        action = parse_action(predictions[event["step"]]["raw"]).to_dict()
        require(json.loads(json.dumps(action)) == event["action"] == event["proposed_action"],
                "Executed action differs from prediction")
        if action["action"] == "LISTING_GRID":
            require(use_ranked_cards and event["size"][1] == RESULT_VIEW_HEIGHT
                    and event["executed_code"].endswith("\nexpand_listing_view()"), "Grid viewport was not expanded")
        if action["action"] == "SEARCH_REPLACE":
            before = frame_before_prediction[event["step"]]
            require(use_ranked_cards and use_browser_context
                    and predictions[event["step"]]["metrics"]["provider_format"] in (
                        REFERENCE_NAVIGATION["SEARCH_REPLACE"], "price_refine")
                    and event["executed_code"] == replace_keyword_code(parse_action(predictions[event["step"]]["raw"]),
                                                                       [events[0], *transitions][before]["size"]),
                    "Search replacement did not execute the recorded keyword action")
        if action["action"] == "SEARCH_CATEGORY":
            require(use_ranked_cards and use_browser_context
                    and predictions[event["step"]]["metrics"]["provider_format"] == "price_search"
                    and event["executed_code"] == category_search_code(parse_action(predictions[event["step"]]["raw"])),
                    "Category search differs from the recorded on-screen action")
        if action["action"] == "SORT_LISTINGS_PRICE":
            before = frame_before_prediction[event["step"]]
            require(use_ranked_cards and use_browser_context
                    and predictions[event["step"]]["metrics"]["provider_format"] in ("price_sort", SEARCH_PRICE_NAVIGATION)
                    and event["executed_code"] == sort_listing_code(parse_action(predictions[event["step"]]["raw"]),
                                                                     [events[0], *transitions][before]["size"]),
                    "Price sort differs from the recorded on-screen action")
        if action["action"] == "FILTER_TIED_PRICE":
            before = frame_before_prediction[event["step"]]
            metrics = predictions[event["step"]]["metrics"]
            require(use_ranked_cards and use_browser_context and metrics["provider_format"] == "price_tie_open"
                    and metrics["price_tie_resume"] == tab_states[before]["tabs"][tab_states[before]["active_tab"]][1]
                    and metrics["price_tie_cents"] == 100 * int(action["value"])
                    and event["executed_code"] == tied_price_filter_code(parse_action(predictions[event["step"]]["raw"])),
                    "Tie recovery did not apply the recorded price filter")
        if action["action"] == "RETURN_PRICE_RESULTS":
            before = frame_before_prediction[event["step"]]
            metrics = predictions[event["step"]]["metrics"]
            destination = metrics["price_tie_resume"]
            require(use_ranked_cards and use_browser_context and metrics["provider_format"] == "price_tie_return"
                    and event["executed_code"] == tied_price_return_code(destination)
                    and tab_states[before + 1]["tabs"][tab_states[before + 1]["active_tab"]][1] == destination,
                    "Tie recovery did not return to the interrupted results")
        if action["action"] == "TAB_FOCUS":
            before = frame_before_prediction[event["step"]]
            target = int(action["value"]) - 1
            require(use_browser_context and target < len(tab_states[before]["tabs"])
                    and tab_states[before + 1]["active_tab"] == target and event["executed_code"] == f"tab_focus({target})",
                    "Tab focus did not reach the requested existing tab")

    retries = by_event.get("invalid_action_retry", [])
    repaired = by_event.get("invalid_action_repaired", [])
    first_invalid = invalid_steps[0] if invalid_steps else None
    should_retry = (retry_mode != "none" and first_invalid is not None and first_invalid + 1 < budget
                    and (model_budget is None or sum(p["metrics"]["model_invocations"]
                         for p in predictions[:first_invalid + 1]) < model_budget))
    require(len(retries) == int(should_retry), "Missing or extra invalid-action retry")
    if retries:
        retry = retries[0]
        next_step = retry["next_prediction"]
        expected_message = INVALID_ACTION_FEEDBACK if retry_mode == "feedback" else None
        require(retry["step"] == first_invalid and next_step == first_invalid + 1 < len(predictions)
                and retry["mode"] == retry_mode and retry["message"] == expected_message, "Invalid retry evidence")
        require([event["step"] for event in repaired] == ([next_step] if next_step in valid_steps else []),
                "Repair count differs from the immediate retry outcome")
    else:
        require(not repaired, "Repair without a retry")
    counts = {"predictions": len(predictions), "model_calls": sum(e["metrics"]["model_invocations"] for e in predictions),
              "steps": len(transitions), "invalid_predictions": len(invalid_steps),
              "invalid_action_retries": len(retries), "repaired_predictions": len(repaired)}
    require(all(row[key] == value for key, value in counts.items()), "Episode counters differ from trace")

    terminal = transitions[-1] if transitions else {"reward": 0, "terminated": False, "truncated": False}
    require(all(row[key] == terminal[key] for key in ("reward", "terminated", "truncated")),
            "Terminal reward differs from the final transition")
    require(row["success"] == exact_vwa_success(terminal["reward"], terminal["terminated"], terminal["truncated"], {}),
            "Success differs from the native reward")
    frames = []
    for index, event in enumerate([events[0], *transitions]):
        name = f"{index:04d}.png"
        require(event["screenshot"] == name, "Unexpected screenshot name")
        with Image.open(checked_file(directory / name, event["sha256"])) as picture:
            require(list(picture.size) == event["size"], "Recorded screenshot dimensions differ")
            frames.append(pixel_hash(picture))
    evidence_stages = {}
    if evidence_check:
        click_origins = None
        if use_navigation_context:
            click_origins = {}
            origin = None
            for index, event in enumerate(transitions):
                if event["action"]["action"] == "TAB_FOCUS":
                    origin = None
                if event["action"]["action"] == "CLICK" and frames[index] != frames[index + 1]:
                    origin = index, event["action"]["position"]
                click_origins[index + 1] = origin
        evidence_stages, checked_answers = audit_evidence(
            directory, predictions, frame_before_prediction, frames, events[0]["goal"],
            confirm=confirm, click_origins=click_origins, compact_navigation=compact_navigation,
            use_screen_text=use_screen_text, guarded_ocr=guarded_ocr, direct_navigation=direct_navigation,
            listing_site=listing_site, use_ranked_cards=use_ranked_cards,
            tab_states=tab_states if use_browser_context else None, reference_images=references)
    answer_outcome = None
    if answer_study:
        submissions = [event for event in transitions if event["action"]["action"] == "ANSWER"]
        counts.update(answer_checks=len(checked_frames) + evidence_stages.get("locate", 0),
                      answers_submitted=len(submissions), answers_from_checks=len(checked_answers))
        require(all(row[key] == counts[key] for key in ("answer_checks", "answers_submitted", "answers_from_checks")),
                "Answer counters differ from trace")
        if submissions:
            last = submissions[-1]
            source = "navigation"
            if last["step"] in checked_answers:
                source = "evidence-confirm" if confirm else ("evidence-check" if evidence_check else "answer-check")
                if use_navigation_context:
                    source = "evidence-links" if listing_site is not None else (
                        "evidence-ocr" if use_screen_text else "evidence-context")
                if use_ranked_cards:
                    source = "evidence-cards"
            answer_outcome = {"source": source, "prediction": last["step"] + 1, "native_reward": last["reward"]}
    before_retry = None
    if first_invalid is not None:
        before_retry = (first_invalid, frames[frame_before_prediction[first_invalid]],
                        predictions[first_invalid]["metrics"]["provider_output"])
    first_unchanged_scroll = None
    first_scroll_followup = None
    if recovery_mode != "none":
        scrolls = {event["step"]: frames[index + 1] for index, event in enumerate(transitions)
                   if event["action"]["action"] == "SCROLL" and frames[index] == frames[index + 1]}
        feedback = by_event.get("recovery_feedback", [])
        scroll_feedback = [event["step"] for event in feedback if event["scroll"]]
        require(len(feedback) == row["recovery_feedbacks"] <= 2
                and all(step in scrolls and step + 1 < len(predictions) for step in scroll_feedback)
                and (recovery_mode == "scroll-feedback" or not scroll_feedback), "Invalid scroll feedback")
        counts.update(scroll_detection_signals=len(scrolls), scroll_feedbacks=len(scroll_feedback))
        require(all(row[key] == counts[key] for key in ("scroll_detection_signals", "scroll_feedbacks")),
                "Scroll counters differ from trace")
        if scrolls:
            step = next(iter(scrolls))
            first_unchanged_scroll = (step, scrolls[step], predictions[step]["metrics"]["provider_output"])
            for index, event in enumerate(transitions):
                if event["step"] == step + 1:
                    first_scroll_followup = {"action": event["action"]["action"],
                                             "viewport_changed": frames[index] != frames[index + 1]}
                    break
    return {"row": row, "counts": counts, "formats": formats, "screenshots": len(frames),
            "reset": (events[0]["goal"], events[0]["seed"]), "reset_pixels": frames[0],
            "reset_context": (tab_states[0], reference_hashes),
            "first_invalid": before_retry, "first_unchanged_scroll": first_unchanged_scroll,
            "first_scroll_followup": first_scroll_followup, "answer_outcome": answer_outcome,
            "evidence_stages": dict(evidence_stages),
            "peak_cuda_allocated_bytes": max(e["metrics"]["peak_cuda_allocated_bytes"] for e in predictions)}


def audit(directory):
    summary = read_json(directory / "summary.json")
    task_set = summary["task_set"]
    task_ids = TASK_SETS[task_set]
    budget = summary["max_steps"]
    model_budget = summary.get("max_model_calls")
    study = summary["study"]
    compact_navigation = summary["model"].get("compact_navigation_replies", task_set == "evidence-navigation-diagnostic")
    require(study in ("format", "scroll", *ANSWER_STUDIES) and summary["budget_unit"] == "predictions"
            and 1 <= budget <= 12 and list(task_ids) == summary["task_ids"], "Unexpected study manifest")
    require(("listing_origin" in summary) == (study in LINK_MODES), "Unexpected listing preview mode")
    if task_set == HOLDOUT_TASK_SET:
        require(summary.get("protocol_sha256") == HOLDOUT_PROTOCOL_SHA256
                and holdout_profile(study, budget, summary["model"]), "Run differs from the locked holdout protocol")
    use_inputs = task_set in INPUT_TASK_SETS
    if use_inputs:
        protocol_hash, media_hash = input_fingerprints(task_set)
        if task_set == PRICE_DEV_TASK_SET:
            require(model_budget == PRICE_DEV_MODEL_CALLS or (model_budget is None and budget == 8),
                    "Unexpected ranked price development model call budget")
        require(summary.get("protocol_sha256") == protocol_hash
                and summary.get("reference_files_sha256") == media_hash
                and summary.get("browser_context") is True
                and holdout_profile(study, model_budget or budget, summary["model"], "evidence-cards"),
                "Run differs from the registered input holdout")
        require(set(summary["input_manifest"]) == {str(task_id) for task_id in task_ids},
                "Input manifest differs from the registered task set")
        if task_set == FRESH_HOLDOUT_TASK_SET:
            require("backend_origins" in summary, "Fresh holdout must record its actual backend origins")
        if "backend_origins" in summary:
            origins = summary["backend_origins"]
            registered_sites = {site for entry in summary["input_manifest"].values() for site in entry["start_sites"]}
            require(set(origins) == registered_sites and len(set(origins.values())) == len(origins),
                    "Registered benchmark sites require distinct origins")
    modes = STUDY_MODES[study]
    require(set(summary["strategies"]) == set(modes), "Unexpected study strategies")
    studies = {mode: [] for mode in modes}
    by_task = {}
    screenshots = 0
    for task_id in task_ids:
        paired = {}
        for mode in modes:
            name = f"visualwebarena.{task_id}"
            if study in ANSWER_STUDIES:
                recovery_mode, retry_mode = "scroll-feedback", "feedback"
            else:
                recovery_mode, retry_mode = ("none", mode) if study == "format" else (mode, "feedback")
            record = audit_episode(directory / f"{name}-{mode}", budget, recovery_mode, retry_mode,
                                   answer_check=mode == "answer-check", answer_study=study in ANSWER_STUDIES,
                                   evidence_check=study in EVIDENCE_STUDIES and mode != "answer-check",
                                   confirm=mode in CONFIRM_MODES, use_navigation_context=mode in NAVIGATION_CONTEXT_MODES,
                                   compact_navigation=compact_navigation, use_screen_text=mode in OCR_MODES,
                                   guarded_ocr=summary.get("screen_reader", {}).get("guarded_ocr", False),
                                   direct_navigation=summary.get("screen_reader", {}).get("direct_navigation", False),
                                   listing_site=summary["listing_origin"] if mode in LINK_MODES else None,
                                   use_ranked_cards=mode == "evidence-cards", use_browser_context=use_inputs,
                                   model_budget=model_budget)
            require(record["row"]["task"] == name and record["row"]["seed"] == record["reset"][1] == 0,
                    "Task or seed mismatch")
            studies[mode].append(record)
            paired[mode] = record
            screenshots += record["screenshots"]
        require(len({record["reset"] for record in paired.values()}) == 1, "Paired task goals or seeds differ")
        invalid = {record["first_invalid"] for record in paired.values()}
        scrolls = {record["first_unchanged_scroll"] for record in paired.values()}
        by_task[task_id] = {
            "identical_reset_pixels": len({record["reset_pixels"] for record in paired.values()}) == 1,
            "identical_first_invalid": len(invalid) == 1 and None not in invalid,
            "strategies": {mode: {**record["counts"], "success": record["row"]["success"],
                                  "reason": record["row"]["reason"]}
                           for mode, record in paired.items()}}
        if study == "scroll":
            by_task[task_id]["identical_first_unchanged_scroll"] = len(scrolls) == 1 and None not in scrolls
            by_task[task_id]["first_scroll_followup"] = {
                mode: record["first_scroll_followup"] for mode, record in paired.items()}
        if study in ANSWER_STUDIES:
            by_task[task_id]["answer_outcome"] = {mode: record["answer_outcome"] for mode, record in paired.items()}
        if study in EVIDENCE_STUDIES:
            by_task[task_id]["evidence_calls"] = paired[modes[-1]]["evidence_stages"]
        if use_inputs:
            first, second = paired.values()
            require(first["reset_context"] == second["reset_context"], "Paired reset tabs or references differ")
            state, pixels = first["reset_context"]
            expected = summary["input_manifest"][str(task_id)]
            require(pixels == expected["reference_pixels"] and len(state["tabs"]) >= expected["start_pages"],
                    "Reset did not expose the official references and start pages")
            if "backend_origins" in summary:
                wanted = [origins[site] for site in expected["start_sites"]]
                observed = [url_origin(tab[1]) for tab in state["tabs"][:len(wanted)]]
                require(observed == wanted, "Browser reset opened an unexpected origin")
            by_task[task_id]["reference_images"] = len(pixels)
            by_task[task_id]["initial_tabs"] = len(state["tabs"])
        if task_set == FRESH_HOLDOUT_TASK_SET:
            by_task[task_id]["source"] = "independent" if task_id == SUPPLEMENTAL_TASK_ID else "official"
    formats = {}
    for mode, records in studies.items():
        counts = {key: sum(record["counts"][key] for record in records) for key in records[0]["counts"]}
        actual = {**aggregate([record["row"] for record in records]), **counts}
        require(summary["strategies"][mode] == actual, f"Summary disagrees with {mode} trajectories")
        formats[mode] = dict(sum((record["formats"] for record in records), Counter()))
    report = {"study": study, "task_set": task_set, "prediction_budget": budget,
              "screenshots": screenshots, "strategies": summary["strategies"], "provider_formats": formats,
              "peak_cuda_allocated_bytes": max(record["peak_cuda_allocated_bytes"]
                                               for records in studies.values() for record in records),
              "by_task": by_task}
    if study in EVIDENCE_STUDIES:
        wrong = {mode: sum(outcome is not None and outcome["native_reward"] != 1.0 for outcome in (
            task["answer_outcome"][mode] for task in by_task.values())) for mode in modes}
        successes = {mode: summary["strategies"][mode]["successes"] for mode in modes}
        report["wrong_submissions"] = wrong
        if task_set in (HOLDOUT_TASK_SET, INPUT_HOLDOUT_TASK_SET, FRESH_HOLDOUT_TASK_SET):
            candidate = modes[-1]
            peak_calls = max(record["counts"]["model_calls"] for records in studies.values() for record in records)
            require(peak_calls <= 8, "Locked holdout exceeded the model call budget")
            gate = {"at_least_two_native_successes": successes[candidate] >= 2,
                    "more_native_successes_than_control": successes[candidate] > successes["answer-check"],
                    "fewer_wrong_submissions_than_control": wrong[candidate] < wrong["answer-check"]}
            report.update(protocol_sha256=summary["protocol_sha256"], acceptance_checks=gate,
                          max_observed_model_calls=peak_calls, target_met=all(gate.values()))
        if task_set == FRESH_HOLDOUT_TASK_SET:
            report["native_successes_by_source"] = {
                source: {mode: sum(entry["strategies"][mode]["success"] for entry in by_task.values()
                                   if entry["source"] == source) for mode in modes}
                for source in ("official", "independent")
            }
        if task_set in (INPUT_RETRIEVAL_TASK_SET, PRICE_DEV_TASK_SET):
            peak_calls = max(record["counts"]["model_calls"] for records in studies.values() for record in records)
            require(peak_calls <= 8, "Development replay exceeded the model call budget")
            report.update(scope="previously observed development tasks", max_observed_model_calls=peak_calls)
        if task_set in ("evidence-transfer", "evidence-confirm-transfer", "evidence-context-transfer"):
            candidate = modes[-1]
            report["target_met"] = (len(task_ids) == 8 and budget == 8
                                    and all(task["identical_reset_pixels"] for task in by_task.values())
                                    and successes[candidate] >= 2
                                    and successes[candidate] > successes["answer-check"]
                                    and wrong[candidate] < wrong["answer-check"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.run))
