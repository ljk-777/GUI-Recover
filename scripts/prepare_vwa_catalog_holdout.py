"""Freeze a disjoint, locally scored Classifieds cohort from the rendered catalog."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from bs4 import BeautifulSoup
from PIL import Image

from gui_recover.vwa_inputs import CATALOG_SERIES_IDS

CATEGORIES = (9, 19, 6)
FAMILIES = ("detail-price-city", "gallery-reference", "search-reference", "search-highest-price")
PRICE = re.compile(r"([0-9][0-9,]*)\.00\s*\$")
RANGE = re.compile(r"([\d,]+)\s*-\s*([\d,]+)\s+of\s+([\d,]+)\s+listings\b", re.I)
SAFE_TITLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,'.&/-]{5,55}")
OWN_INDEX = re.compile(r"^/index\.php\?page=item&id=([1-9][0-9]*)$")


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
            cards.append(Card(item, title.get_text(" ", strip=True),
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


def task(task_id, goal, start, expected, *, item=None, image=None):
    rule = {"eval_types": (["url_match"] if item else []) + ["string_match"],
            "reference_answers": {"must_include": expected}, "reference_url": "__CLASSIFIEDS__" + item if item else "",
            "url_note": "EXACT"}
    return {"task_id": task_id, "intent": goal, "start_url": "__CLASSIFIEDS__" + start,
            "sites": ["classifieds"], "geolocation": None, "require_reset": False, "intent_template_id": task_id,
            "image": image, "eval": rule}


def freeze(site, excluded, output, *, series=1):
    key = f"gui-recover-catalog-holdout-v{series}"
    ids = CATALOG_SERIES_IDS[key]

    def sample_key(value):
        return fingerprint(f"{key}\n{value}".encode())

    catalog = Catalog(site)
    pool = {}
    pool_digests = {}
    for category in CATEGORIES:
        for page in range(1, 5):
            path, _, cards, digest = catalog.results(sCategory=category, sOrder="dt_pub_date", iOrderType="desc", iPage=page)
            pool_digests[path] = digest
            for card in cards:
                if card.item not in excluded and card.dollars and SAFE_TITLE.fullmatch(card.title):
                    pool[card.item] = card
    available = sorted(pool.values(), key=lambda card: sample_key(card.item))
    specs, evidence, media, used = [], [], {}, set(excluded)
    searched = {}

    def record(family, start, goal, expected, card, source, *, pin_item=False, reference=False):
        case_id = ids[len(specs)]
        name = f"catalog-references/{case_id}.png" if reference else None
        if name:
            with Image.open(io.BytesIO(catalog.read(card.photo))) as photo:
                photo.thumbnail((800, 800))
                stream = io.BytesIO()
                photo.convert("RGB").save(stream, format="PNG")
                media[name] = stream.getvalue()
        specs.append(task(case_id, goal, start, expected, item=card.item if pin_item else None, image=name))
        evidence.append({"task_id": case_id, "family": family, "listing_path": card.item, "source": source})
        used.add(card.item)

    for card in available:
        verified = catalog.detail(card)
        if verified and card.dollars >= 10:
            city, digest = verified
            record(FAMILIES[0], card.item, "What is the asking price in dollars and the city for the listing open in this tab?",
                   [str(card.dollars), city], card, {"detail_sha256": digest, "visible": {"dollars": card.dollars, "city": city}},
                   pin_item=True)
        if len(specs) == 2:
            break
    if len(specs) != 2:
        raise ValueError("Need two evidenced detail cases")

    phrases = set()
    for card in available:
        terms = re.findall(r"[a-z0-9]+", card.title.casefold())
        phrases.update(word for word in terms if len(word) >= 4)
        phrases.update(" ".join(pair) for pair in zip(terms, terms[1:]) if all(len(w) >= 3 for w in pair))
    for family in FAMILIES[1:]:
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
                              and row.title.casefold() != phrase
                              and sum(other.photo == row.photo for other in cards) == 1]
                if not candidates:
                    continue
                card = min(candidates, key=lambda row: sample_key(row.item))
                if family == FAMILIES[1]:
                    start = path
                    goal = "Find the exact item shown in the reference photo. What is its listing title?"
                else:
                    start = "/"
                    goal = f'Search for "{phrase}" and find the exact item in the reference photo. What is its listing title?'
                record(family, start, goal, [card.title], card,
                       {"search_url": path, "search_sha256": digest, "cards": [asdict(row) for row in cards]},
                       reference=True)
            phrases.remove(phrase)
            if len(specs) == target_count:
                break
        if len(specs) != target_count:
            raise ValueError(f"Insufficient unique cases for {family}")

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
        "sample_key": key, "task_ids": list(ids), "families": list(FAMILIES),
        "seed": 0, "max_predictions": 12, "max_model_calls": 8,
        "strategies": ["answer-check", "evidence-cards"],
        "acceptance": {"minimum_successes": 2, "more_successes_than_control": True,
                       "fewer_wrong_submissions_than_control": True},
        "tasks_sha256": fingerprint(encoded(specs)), "evidence_sha256": fingerprint(encoded(evidence)),
        "reference_files_sha256": fingerprint(encoded(manifest)),
        "catalog_pool_sha256": fingerprint(encoded(pool_digests)),
        "exclusion_list_sha256": fingerprint(encoded(sorted(excluded))),
    }
    for name, payload in (("cohort.json", protocol), ("tasks.json", specs), ("source-evidence.json", evidence),
                          ("reference-media.json", media_record)):
        (output / name).write_bytes(encoded(payload) + b"\n")
    return fingerprint((output / "cohort.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--series", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    excluded = set(json.loads(args.exclude.read_text()))
    if any(not isinstance(item, str) or not OWN_INDEX.fullmatch(item) for item in excluded):
        raise ValueError("The exclusion file needs verified listing paths")
    print(freeze(args.site, excluded, args.out, series=args.series), flush=True)
