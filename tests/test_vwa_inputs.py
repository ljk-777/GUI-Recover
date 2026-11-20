import base64
import hashlib
import io
import json
import tarfile

import pytest
from PIL import Image

from gui_recover.actions import ActionError, compile_action
from gui_recover.environment import VisualObservation, frame_key
from gui_recover.qwen_grounding import QwenGroundingPolicy, normalize_qwen_action
from gui_recover.verification import pixel_hash
from gui_recover.vwa_inputs import LocalReferenceArchive, contextual_observation, reference_name


def reference_png():
    stream = io.BytesIO()
    Image.new("RGB", (10, 8), "orange").save(stream, format="PNG")
    return stream.getvalue()


def test_official_reference_names_preserve_the_asset_suffix():
    suffix = "inputs/example.png"
    expected = "environment_docker/webarena-homepage/" + suffix
    assert reference_name("__HOMEPAGE__/" + suffix) == expected
    assert reference_name("http://localhost:5555/" + suffix, "http://localhost:5555/") == expected
    assert reference_name("http://elsewhere:5555/" + suffix, "http://localhost:5555/") != expected


def test_exact_archive_resolves_registered_asset_and_rejects_a_missing_reference(tmp_path):
    name = "images/fixture.png"
    data = reference_png()
    archive = tmp_path / "reference-media.tar.gz"
    with tarfile.open(archive, "w:gz") as pack:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        pack.addfile(member, io.BytesIO(data))
    manifest = tmp_path / "reference-media.json"
    manifest.write_text(json.dumps({"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                    "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}))
    loader = LocalReferenceArchive(tmp_path, hashlib.sha256(manifest.read_bytes()).hexdigest())
    with loader(name) as image:
        assert image.size == (10, 8) and image.format == "PNG"
    with pytest.raises(ValueError, match="outside"):
        loader("other.png")
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive differs"):
        LocalReferenceArchive(tmp_path)


def test_structured_goal_and_browser_tabs_reach_qwen_in_original_image_order():
    screen = Image.new("RGB", (16, 12), "white")
    task = {"goal_object": [
        {"type": "text", "text": "Which listing matches this photo?"},
        {"type": "text", "text": "Reference metadata and private path"},
        {"type": "image_url", "url": "data:image/png;base64," + base64.b64encode(reference_png()).decode()}],
        "open_pages_titles": ("Results", "Homepage"),
        "open_pages_urls": ("http://localhost:5555/search", "http://localhost:5555/"),
        "active_page_index": [1]}
    obs = contextual_observation(VisualObservation("legacy fallback", screen), task)
    actor = object.__new__(QwenGroundingPolicy)
    actor.prepare_image = lambda image: image
    actor._generate = lambda messages: messages
    actor.last_metrics = {}
    content = actor.ask(screen, obs.goal, observation=obs, navigation=True)[0]["content"]
    assert [message["image"].size for message in content if message["type"] == "image"] == [(16, 12), (10, 8)]
    assert "Which listing" in content[-1]["text"] and "private path" not in content[-1]["text"]
    assert "Active tab number: 2" in content[-1]["text"] and "TAB_FOCUS" in content[-1]["text"]
    assert actor.last_metrics["reference_pixels"] == [pixel_hash(obs.reference_images[0])]


def test_tab_actions_validate_the_observed_tab_count_and_are_not_generic_browser_scripts():
    data = {"action": "TAB_FOCUS", "value": "2", "position": None}
    action = normalize_qwen_action(data, (16, 12), native_navigation=True, tab_count=2)
    assert action.value == "2"
    with pytest.raises(ActionError):
        normalize_qwen_action(data, (16, 12), native_navigation=True, tab_count=1)
    with pytest.raises(ActionError):
        compile_action(action, 16, 12)
    tabs = (("Results", "http://localhost:5555/search"), ("Homepage", "http://localhost:5555/"))
    assert frame_key("same-pixels", tabs, 0) != frame_key("same-pixels", tabs, 1)
