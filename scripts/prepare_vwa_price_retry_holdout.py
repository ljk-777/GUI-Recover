"""Seal model-free price OCR triggers and readable controls from the catalog."""

import argparse
import os
import re
import tempfile
from pathlib import Path

from PIL import Image

from gui_recover.answer_evidence import listing_price_ocr_receipt
from gui_recover.listing_facts import listing_fact_decision
from gui_recover.screen_text import RapidScreenReader
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import read_json
from gui_recover.vwa_inputs import CATALOG_SERIES_IDS
from prepare_vwa_catalog_holdout import (
    CATALOG_POOL_PAGES, FAMILIES, OWN_INDEX, Catalog,
    catalog_pool, encoded, fingerprint, seal_catalog, task,
)
from run_vwa_catalog_holdout import PROFILE, browser_case


GOAL = "What is the asking price in dollars and the city for the listing open in this tab?"
SCREEN_LIMITS = {7: 96, 8: 144}
CASES_PER_GROUP = 3


def screen_price(screen, url, origin, city, dollars, reader, prepared_size):
    decision = listing_fact_decision(GOAL, url, origin, reader(screen, prepared_size), prepared_size, screen, None)
    facts = decision.facts if decision is not None else None
    if facts is None or facts.cents != 100 * dollars or facts.text.casefold() != city.casefold():
        return {"outcome": "whole_page_ocr_did_not_match"}
    enlarged = listing_price_ocr_receipt(screen, facts, reader)
    outcome = "trigger" if facts.with_price_from_crop(enlarged["rows"]) is None else "control"
    return {"outcome": outcome, "start_pixels_sha256": pixel_hash(screen), "enlarged_ocr": enlarged}


def candidate_order(pool, key, *, varied_prices=False):
    ordered = sorted(pool, key=lambda card: fingerprint(f"{key}\n{card.item}".encode()))
    if not varied_prices:
        return ordered
    first, repeated = [], []
    seen = set()
    for card in ordered:
        if card.dollars in seen:
            repeated.append(card)
        else:
            seen.add(card.dollars)
            first.append(card)
    return first + repeated


def freeze_retry_holdout(site, excluded, output, *, series=7, excluded_crops=None):
    from browsergym.core.observation import extract_screenshot
    from playwright.sync_api import TimeoutError as BrowserTimeout
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if output.exists():
        raise ValueError("Choose a fresh output directory for screening")
    catalog = Catalog(site)
    if catalog.origin != os.environ["VWA_CLASSIFIEDS"].rstrip("/"):
        raise ValueError("BrowserGym and the catalog source must use the same Classifieds origin")
    key = f"gui-recover-catalog-holdout-v{series}"
    ids = CATALOG_SERIES_IDS[key]
    screen_limit = SCREEN_LIMITS[series]
    pool, pool_digests = catalog_pool(catalog, excluded, CATALOG_POOL_PAGES[key])
    order = candidate_order(pool.values(), key, varied_prices=excluded_crops is not None)[:screen_limit]
    if not order:
        raise ValueError("The catalog contained no unseen price candidates")
    env = browser_case(task(ids[0], GOAL, order[0].item, ["10", "placeholder"], item=order[0].item), None, cards=True)
    reader = RapidScreenReader()
    selected = {"trigger": [], "control": []}
    seen_crops = set(excluded_crops or ())
    attempts = []
    try:
        env.reset(PROFILE["seed"])
        page = env.env.unwrapped.page
        for card in order:
            details = catalog.detail(card)
            record = {"listing_path": card.item}
            attempts.append(record)
            if not details or card.dollars < 10:
                record["outcome"] = "missing_catalog_evidence"
                continue
            city, detail_sha = details
            url = catalog.origin + card.item
            try:
                response = page.goto(url, wait_until="load", timeout=12000)
            except BrowserTimeout:
                record["outcome"] = "navigation_timeout"
                continue
            if response is None or not response.ok or page.url != url:
                record["outcome"] = "navigation_mismatch"
                continue
            screen = Image.fromarray(extract_screenshot(page)).convert("RGB")
            height, width = smart_resize(screen.height, screen.width, factor=28, min_pixels=56 * 56, max_pixels=784 * 784)
            receipt = screen_price(screen, url, catalog.origin, city, card.dollars, reader, (width, height))
            group = selected.get(receipt["outcome"])
            if excluded_crops is not None and group is not None:
                crop_sha = receipt["enlarged_ocr"]["pixels_sha256"]
                if crop_sha in seen_crops:
                    receipt["outcome"] = "repeated_price_crop"
                    group = None
                seen_crops.add(crop_sha)
            record.update(receipt)
            if group is not None and len(group) < CASES_PER_GROUP:
                group.append((card, city, detail_sha, screen, record))
            if len(attempts) % 10 == 0:
                print(f"Screened {len(attempts)} candidates, {len(selected['trigger'])} triggers selected", flush=True)
            if all(len(group) == CASES_PER_GROUP for group in selected.values()):
                break
    finally:
        env.close()

    specs, evidence, screens = [], [], {}
    for ident, (card, city, detail_sha, screen, record) in zip(ids, selected["trigger"] + selected["control"]):
        specs.append(task(ident, GOAL, card.item, [str(card.dollars), city], item=card.item))
        evidence.append({"task_id": ident, "family": FAMILIES[0], "listing_path": card.item, "source": {
            "detail_sha256": detail_sha, "visible": {"dollars": card.dollars, "city": city}, "price_retry_screening": record,
        }})
        screens[ident] = screen
    summary = {"model_calls": 0, "screen_limit": screen_limit, "ocr": reader.config,
               "ordered_candidates": [card.item for card in order], "attempts": attempts,
               "selected": {group: [row[0].item for row in rows] for group, rows in selected.items()}}
    if excluded_crops is not None:
        summary["excluded_crop_sha256"] = sorted(excluded_crops)
    raw = encoded(summary) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(specs) != len(ids):
        output.with_suffix(".screening.json").write_bytes(raw)
        raise ValueError(f"Only {len(selected['trigger'])} triggers and {len(selected['control'])} controls found")
    with tempfile.TemporaryDirectory(prefix=".price-retry-", dir=output.parent) as directory:
        staged = Path(directory) / "cohort"
        digest = seal_catalog(staged, key, specs, evidence, {}, excluded, pool_digests, screening_digest=fingerprint(raw))
        (staged / "price-retry-screening.json").write_bytes(raw)
        frames = staged / "screening-frames"
        frames.mkdir()
        for ident, screen in screens.items():
            screen.save(frames / f"{ident}.png")
        staged.rename(output)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--exclude-crops", type=Path)
    parser.add_argument("--series", type=int, choices=tuple(SCREEN_LIMITS), default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    excluded = set(read_json(args.exclude))
    if any(not isinstance(value, str) or not OWN_INDEX.fullmatch(value) for value in excluded):
        raise ValueError("The exclusion file needs verified listing paths")
    if (args.series == 8) != (args.exclude_crops is not None):
        parser.error("Series 8 requires the registered crop exclusions")
    excluded_crops = None
    if args.exclude_crops:
        excluded_crops = set(read_json(args.exclude_crops))
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in excluded_crops):
            raise ValueError("The crop exclusion file needs SHA-256 digests")
    print(freeze_retry_holdout(args.site, excluded, args.out, series=args.series, excluded_crops=excluded_crops), flush=True)


if __name__ == "__main__":
    main()
