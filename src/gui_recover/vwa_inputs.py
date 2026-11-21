"""Bounded task reference images and browser context for VisualWebArena."""

import base64
import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PIL import Image


CATALOG_SERIES_IDS = {
    "gui-recover-catalog-holdout-v1": tuple(range(10101, 10109)),
    "gui-recover-catalog-holdout-v2": tuple(range(10201, 10209)),
    "gui-recover-catalog-holdout-v3": tuple(range(10301, 10309)),
}


def reference_name(value, homepage="__HOMEPAGE__"):
    """Undo BrowserGym's homepage substitution for official relative assets."""
    prefix = homepage.rstrip("/") + "/"
    return "environment_docker/webarena-homepage/" + value[len(prefix):] if value.startswith(prefix) else value


class LocalReferenceArchive:
    """Resolve an official task's archived reference by its exact manifest name."""

    def __init__(self, directory, files_sha256=None, *, homepage="__HOMEPAGE__"):
        directory = Path(directory)
        record = json.loads((directory / "reference-media.json").read_bytes())
        canonical = json.dumps(record["files"], sort_keys=True, separators=(",", ":")).encode()
        if files_sha256 is not None and hashlib.sha256(canonical).hexdigest() != files_sha256:
            raise ValueError("Reference manifest differs from the registered one")
        archive = directory / "reference-media.tar.gz"
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["archive_sha256"]:
                raise ValueError("Reference archive differs from its manifest")
        self.images = {}
        self.homepage = homepage
        with tarfile.open(archive, "r:gz") as pack:
            members = pack.getmembers()
            if len(members) != len(record["files"]) or {m.name for m in members} != set(record["files"]):
                raise ValueError("Reference archive contains unexpected members")
            for member in members:
                expected = record["files"][member.name]
                if not member.isfile() or member.size != expected["bytes"] or member.size > 16 * 1024 * 1024:
                    raise ValueError("Reference member size or type differs from the manifest")
                with pack.extractfile(member) as stream:
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError("Reference image differs from its manifest")
                self.images[member.name] = data

    def __call__(self, name):
        name = reference_name(name, self.homepage)
        if name not in self.images:
            raise ValueError("Task reference is outside the registered archive")
        # Keep the image's format for BrowserGym, which writes its own local task input.
        return Image.open(io.BytesIO(self.images[name]))


def _inline_image(url):
    prefix, marker, encoded = url.partition(",")
    if marker != "," or prefix not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise ValueError("Expected a BrowserGym PNG or JPEG task reference")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        return image.convert("RGB")


def _visible_address(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or parsed.username is not None or parsed.password is not None:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))[:512]


def contextual_observation(base, observation, *, references=None):
    """Use BrowserGym's rendered tab metadata and the task's explicit goal references."""
    messages = observation.get("goal_object", ())
    images = tuple(_inline_image(message["image_url"]["url"]) for message in messages
                   if message["type"] == "image_url") if references is None else references
    if images and (not messages or messages[0]["type"] != "text"):
        raise ValueError("A reference image needs its original goal text")
    goal = messages[0]["text"] if images else base.goal
    tabs = tuple((" ".join(title.split())[:160], _visible_address(url)) for title, url in zip(
        observation["open_pages_titles"], observation["open_pages_urls"], strict=True))
    active = int(observation["active_page_index"][0])
    if not 0 <= active < len(tabs) <= 256:
        raise ValueError("Invalid active BrowserGym tab")
    return replace(base, goal=goal, reference_images=images, tabs=tabs, active_tab=active)
