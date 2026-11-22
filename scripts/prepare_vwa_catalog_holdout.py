"""Freeze a disjoint, locally scored Classifieds cohort from the rendered catalog."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from bs4 import BeautifulSoup
from PIL import Image

from gui_recover.listing_cards import REFERENCE_PHOTO_CLEAR_DISTANCE, reference_photo_distance, reference_thumbnail_pixels
from gui_recover.vwa_inputs import CATALOG_CAPTURE_PROFILES, CATALOG_SERIES_IDS

CATEGORIES = (9, 19, 6)
FAMILIES = ("detail-price-city", "gallery-reference", "search-reference", "search-highest-price")
PRICE = re.compile(r"([0-9][0-9,]*)\.00\s*\$")
RANGE = re.compile(r"([\d,]+)\s*-\s*([\d,]+)\s+of\s+([\d,]+)\s+listings\b", re.I)
SAFE_TITLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,'.&/-]{5,55}")
OWN_INDEX = re.compile(r"^/index\.php\?page=item&id=([1-9][0-9]*)$")
CITY_WORD_COUNTS = {"gui-recover-catalog-holdout-v4": (2, 1),
                    "gui-recover-catalog-holdout-v5": (1, 2, 1, 2, 1, 2),
                    "gui-recover-catalog-holdout-v6": (1, 2, 1, 2, 1, 2)}
DETAIL_PRICE_DIGITS = {"gui-recover-catalog-holdout-v6": (3, 2, 2, 3, 3, 2)}
PRICE_READ_DROPOUT_COHORT = "gui-recover-catalog-holdout-v10"
WHOLE_LINE_COHORTS = ("gui-recover-catalog-holdout-v9", "gui-recover-catalog-holdout-v11")
PRICE_RETRY_COHORTS = ("gui-recover-catalog-holdout-v7", "gui-recover-catalog-holdout-v8", *WHOLE_LINE_COHORTS)
CATALOG_POOL_PAGES = {"gui-recover-catalog-holdout-v4": 12, "gui-recover-catalog-holdout-v5": 20,
                      "gui-recover-catalog-holdout-v6": 20, PRICE_RETRY_COHORTS[0]: 20, PRICE_RETRY_COHORTS[1]: 32,
                      WHOLE_LINE_COHORTS[0]: 48, PRICE_READ_DROPOUT_COHORT: 48, WHOLE_LINE_COHORTS[1]: 64}


def price_retry_groups(key):
    return ("trigger", "control", "upright_control") if key in WHOLE_LINE_COHORTS else ("trigger", "control")


def catalog_families(key):
    if key in ("gui-recover-catalog-holdout-v5", "gui-recover-catalog-holdout-v6",
               PRICE_READ_DROPOUT_COHORT, *PRICE_RETRY_COHORTS):
        return FAMILIES[:1]
    return FAMILIES


def catalog_case_families(key):
    families = catalog_families(key)
    if len(families) == 1:
        return [families[0]] * len(CATALOG_SERIES_IDS[key])
    detail_count = len(CITY_WORD_COUNTS.get(key, (1, 1)))
    return [families[0]] * detail_count + [family for family in families[1:] for _ in range(2)]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class Card:
    item: str
    title: str
    dollars: int | None
    photo: str | None


class Catalog:
    def __init__(self, site):
        parts = urlsplit(site)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/") or parts.query:
            raise ValueError("Use the origin of the pinned Classifieds fixture")
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.opener = build_opener(NoRedirect())

    def read(self, path, limit=4 * 1024 * 1024):
        url = urljoin(self.origin + "/", path)
        if (urlsplit(url).scheme, urlsplit(url).netloc) != (urlsplit(self.origin).scheme, urlsplit(self.origin).netloc):
            raise ValueError("Catalog read escaped the fixture origin")
        with self.opener.open(url, timeout=10) as response:
            contents = response.read(limit + 1)
        if len(contents) > limit:
            raise ValueError("Catalog response exceeded the byte limit")
        return contents

    def html(self, path):
        contents = self.read(path)
        return BeautifulSoup(contents, "html.parser"), fingerprint(contents)

    def results(self, **query):
        path = "/index.php?" + urlencode({"page": "search", "sShowAs": "gallery", **query})
        try:
            soup, digest = self.html(path)
        except HTTPError as error:
            if error.code != 404 or "sPattern" not in query:
                raise
            return path, 0, [], None
        match = RANGE.search(soup.get_text(" ", strip=True))
        cards = []
        for row in soup.select("li.listing-card"):
            title = row.select_one("a.title[href]")
            if title is None:
                continue
            url = urlsplit(urljoin(self.origin, title["href"]))
            item = f"{url.path}?{url.query}"
            if url.netloc != urlsplit(self.origin).netloc or not OWN_INDEX.fullmatch(item):
                continue
            price = row.select_one(".currency-value")
            match_price = PRICE.fullmatch(price.get_text(" ", strip=True)) if price else None
            photo = row.select_one("img[src]")
            source = urlsplit(urljoin(self.origin, photo["src"])) if photo else None
            media_path = (source.path if source and source.netloc == url.netloc and not source.query
                          and source.path.startswith("/oc-content/uploads/") else None)
            # Use the whitespace that a reader sees under the theme's normal text layout.
            displayed_title = " ".join(title.get_text(" ", strip=True).split())
            cards.append(Card(item, displayed_title,
                              int(match_price[1].replace(",", "")) if match_price else None, media_path))
        total = int(match[3].replace(",", "")) if match else None
        return path, total, cards, digest

    def detail(self, card):
        soup, digest = self.html(card.item)
        price = soup.select_one(".price.isDesktop")
        location = soup.select_one("#item_location li")
        amount = PRICE.fullmatch(price.get_text(" ", strip=True)) if price else None
        if amount is None or int(amount[1].replace(",", "")) != card.dollars or location is None:
            return None
        match = re.fullmatch(r"Location:\s*([A-Za-z][A-Za-z .'-]{2,35}),\s*[^,]+,\s*United States",
                             location.get_text(" ", strip=True))
        return (match[1], digest) if match is not None else None

    @lru_cache(maxsize=64)
    def reference_photo(self, path):
        if path is None:
            return None
        try:
            blob = self.read(path)
        except HTTPError as error:
            if error.code != 404:
                raise
            return None
        try:
            with Image.open(io.BytesIO(blob)) as image:
                image.thumbnail((800, 800))
                return image.convert("RGB"), fingerprint(blob)
        except OSError:
            return None


def reference_photo_evidence(catalog, target, cards):
    """Compare a proposed reference photo with every other card in its result set."""
    photo = catalog.reference_photo(target.photo)
    if photo is None:
        return None
    image, digest = photo
    fitted = reference_thumbnail_pixels(image)
    stretched = reference_thumbnail_pixels(image, stretch=True)
    if min(float(fitted.std()), float(stretched.std())) < 0.04:
        return None
    comparisons = []
    for card in cards:
        if card.item == target.item:
            continue
        other = catalog.reference_photo(card.photo)
        if other is None or other[1] == digest:
            return None
        photo, other_digest = other
        distance = min(reference_photo_distance(fitted, reference_thumbnail_pixels(photo)),
                       reference_photo_distance(stretched, reference_thumbnail_pixels(photo, stretch=True)))
        if distance < REFERENCE_PHOTO_CLEAR_DISTANCE:
            return None
        comparisons.append({"item": card.item, "source_sha256": other_digest, "distance": distance})
    return {"source_sha256": digest, "comparisons": comparisons}


def task(task_id, goal, start, expected, *, item=None, image=None):
    rule = {"eval_types": (["url_match"] if item else []) + ["string_match"],
            "reference_answers": {"must_include": expected}, "reference_url": "__CLASSIFIEDS__" + item if item else "",
            "url_note": "EXACT"}
    return {"task_id": task_id, "intent": goal, "start_url": "__CLASSIFIEDS__" + start,
            "sites": ["classifieds"], "geolocation": None, "require_reset": False, "intent_template_id": task_id,
            "image": image, "eval": rule}


def catalog_pool(catalog, excluded, pages):
    pool = {}
    pool_digests = {}
    for category in CATEGORIES:
        for page in range(1, pages + 1):
            path, _, cards, digest = catalog.results(sCategory=category, sOrder="dt_pub_date", iOrderType="desc", iPage=page)
            pool_digests[path] = digest
            for card in cards:
                if card.item not in excluded and card.dollars and SAFE_TITLE.fullmatch(card.title):
                    pool[card.item] = card
    return pool, pool_digests


def freeze(site, excluded, output, *, series=1):
    key = f"gui-recover-catalog-holdout-v{series}"
    if key in PRICE_RETRY_COHORTS:
        raise ValueError("Use prepare_vwa_price_retry_holdout.py to screen the price retry cohort")
    ids = CATALOG_SERIES_IDS[key]
    city_word_counts = CITY_WORD_COUNTS.get(key)
    price_digits = DETAIL_PRICE_DIGITS.get(key)
    families = catalog_families(key)
    detail_count = catalog_case_families(key).count(FAMILIES[0])

    def sample_key(value):
        return fingerprint(f"{key}\n{value}".encode())

    catalog = Catalog(site)
    pool, pool_digests = catalog_pool(catalog, excluded, CATALOG_POOL_PAGES.get(key, 4))
    available = sorted(pool.values(), key=lambda card: sample_key(card.item))
    specs, evidence, media, used = [], [], {}, set(excluded)
    searched = {}
    selected_amounts = set()

    def record(family, start, goal, expected, card, source, *, pin_item=False, reference=False):
        case_id = ids[len(specs)]
        name = f"catalog-references/{case_id}.png" if reference else None
        if name:
            photo, _ = catalog.reference_photo(card.photo)
            stream = io.BytesIO()
            photo.save(stream, format="PNG")
            media[name] = stream.getvalue()
        specs.append(task(case_id, goal, start, expected, item=card.item if pin_item else None, image=name))
        evidence.append({"task_id": case_id, "family": family, "listing_path": card.item, "source": source})
        used.add(card.item)

    for card in available:
        if price_digits and len(str(card.dollars)) != price_digits[len(specs)]:
            continue
        if key == PRICE_READ_DROPOUT_COHORT and card.dollars in selected_amounts:
            continue
        verified = catalog.detail(card)
        if verified and card.dollars >= 10:
            city, digest = verified
            if city_word_counts and min(len(city.split()), 2) != city_word_counts[len(specs)]:
                continue
            record(FAMILIES[0], card.item, "What is the asking price in dollars and the city for the listing open in this tab?",
                   [str(card.dollars), city], card, {"detail_sha256": digest, "visible": {"dollars": card.dollars, "city": city}},
                   pin_item=True)
            selected_amounts.add(card.dollars)
        if len(specs) == detail_count:
            break
    if len(specs) != detail_count:
        raise ValueError(f"Need {detail_count} evidenced detail cases")

    phrases = set()
    for card in available:
        terms = re.findall(r"[a-z0-9]+", card.title.casefold())
        phrases.update(word for word in terms if len(word) >= 4)
        phrases.update(" ".join(pair) for pair in zip(terms, terms[1:]) if all(len(w) >= 3 for w in pair))
    for family in families[1:]:
        target_count = len(specs) + 2
        for phrase in sorted(phrases, key=sample_key):
            if phrase not in searched:
                if len(searched) >= 440:
                    raise ValueError("The registered search pool did not yield eight independent cases")
                searched[phrase] = catalog.results(sPattern=phrase)
            path, total, cards, digest = searched[phrase]
            if not 2 <= (total or 0) <= 12 or total != len(cards) or len({row.item for row in cards}) != total:
                continue
            if family == FAMILIES[3]:
                if any(row.dollars is None for row in cards):
                    continue
                ranked = sorted(cards, key=lambda row: row.dollars, reverse=True)
                card = ranked[0]
                if card.dollars <= ranked[1].dollars or not SAFE_TITLE.fullmatch(card.title) or card.item in used:
                    continue
                goal = f'Search for "{phrase}" and report the listing title and asking price in dollars of the most expensive result.'
                record(family, "/", goal, [card.title, str(card.dollars)], card,
                       {"search_url": path, "search_sha256": digest, "cards": [asdict(row) for row in ranked]},
                       pin_item=False)
            else:
                candidates = [row for row in cards[1:3] if row.item not in used and row.photo
                              and 8 <= len(row.title) <= 40 and SAFE_TITLE.fullmatch(row.title)
                              and row.title.casefold() != phrase]
                for card in sorted(candidates, key=lambda row: sample_key(row.item)):
                    photo_evidence = reference_photo_evidence(catalog, card, cards)
                    if photo_evidence is not None:
                        break
                else:
                    continue
                if family == FAMILIES[1]:
                    start = path
                    goal = "Find the exact item shown in the reference photo. What is its listing title?"
                else:
                    start = "/"
                    goal = f'Search for "{phrase}" and find the exact item in the reference photo. What is its listing title?'
                record(family, start, goal, [card.title], card,
                       {"search_url": path, "search_sha256": digest, "cards": [asdict(row) for row in cards],
                        "reference_photo": photo_evidence},
                       reference=True)
            phrases.remove(phrase)
            if len(specs) == target_count:
                break
        if len(specs) != target_count:
            raise ValueError(f"Insufficient unique cases for {family}")

    return seal_catalog(output, key, specs, evidence, media, excluded, pool_digests)


def seal_catalog(output, key, specs, evidence, media, excluded, pool_digests, *, screening_digest=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {name: {"bytes": len(blob), "sha256": fingerprint(blob)} for name, blob in media.items()}
    archive = output / "reference-media.tar.gz"
    with tarfile.open(archive, "w:gz") as pack:
        for name, blob in sorted(media.items()):
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            pack.addfile(member, io.BytesIO(blob))
    media_record = {"files": manifest, "archive_sha256": fingerprint(archive.read_bytes())}
    protocol = {
        "scope": "independently sampled Classifieds tasks",
        "sample_key": key, "task_ids": list(CATALOG_SERIES_IDS[key]), "families": list(catalog_families(key)),
        "seed": 0, "max_predictions": 12, "max_model_calls": 8,
        "strategies": ["answer-check", "evidence-cards"],
        "acceptance": {"minimum_successes": 2, "more_successes_than_control": True,
                       "fewer_wrong_submissions_than_control": True},
        "tasks_sha256": fingerprint(encoded(specs)), "evidence_sha256": fingerprint(encoded(evidence)),
        "reference_files_sha256": fingerprint(encoded(manifest)),
        "minimum_reference_photo_distance": REFERENCE_PHOTO_CLEAR_DISTANCE,
        "catalog_pool_sha256": fingerprint(encoded(pool_digests)),
        "exclusion_list_sha256": fingerprint(encoded(sorted(excluded))),
    }
    if key in CITY_WORD_COUNTS:
        protocol["detail_city_word_counts_capped_at_two"] = list(CITY_WORD_COUNTS[key])
    if key in DETAIL_PRICE_DIGITS:
        protocol["detail_price_digits"] = list(DETAIL_PRICE_DIGITS[key])
    if key in CATALOG_POOL_PAGES:
        protocol["catalog_pages_per_category"] = CATALOG_POOL_PAGES[key]
    if key in CATALOG_CAPTURE_PROFILES:
        protocol["capture_profile"] = CATALOG_CAPTURE_PROFILES[key]
    if screening_digest is not None:
        protocol["price_retry_screening_sha256"] = screening_digest
    for name, payload in (("cohort.json", protocol), ("tasks.json", specs), ("source-evidence.json", evidence),
                          ("reference-media.json", media_record)):
        (output / name).write_bytes(encoded(payload) + b"\n")
    return fingerprint((output / "cohort.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--series", type=int, choices=tuple(range(1, len(CATALOG_SERIES_IDS) + 1)), default=1)
    args = parser.parse_args()
    excluded = set(json.loads(args.exclude.read_text()))
    if any(not isinstance(item, str) or not OWN_INDEX.fullmatch(item) for item in excluded):
        raise ValueError("The exclusion file needs verified listing paths")
    print(freeze(args.site, excluded, args.out, series=args.series), flush=True)
