import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from gui_recover.actions import ActionError, compile_action
from gui_recover.environment import VisualObservation, frame_key
from gui_recover.listing_cards import reference_title_region
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
    file_set = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    manifest.write_text(json.dumps({"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                    "files": file_set}))
    canonical = json.dumps(file_set, sort_keys=True, separators=(",", ":")).encode()
    loader = LocalReferenceArchive(tmp_path, hashlib.sha256(canonical).hexdigest())
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
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(reference_png()).decode()}}],
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
    assert normalize_qwen_action({"action": "TAB_FOCUS", "value": "2"}, (16, 12),
                                 native_navigation=True, tab_count=2) == action
    with pytest.raises(ActionError):
        normalize_qwen_action(data, (16, 12), native_navigation=True, tab_count=1)
    with pytest.raises(ActionError):
        compile_action(action, 16, 12)
    tabs = (("Results", "http://localhost:5555/search"), ("Homepage", "http://localhost:5555/"))
    assert frame_key("same-pixels", tabs, 0) != frame_key("same-pixels", tabs, 1)


def test_reference_region_uses_the_unique_visible_match_and_its_title():
    reference = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(reference).rectangle((10, 5, 36, 20), fill="red")
    screen = Image.new("RGB", (100, 70), "black")
    screen.paste(reference, (2, 4))
    site = "http://localhost:5555"
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "grid": None, "next": None, "cards": [
        {"destination": site + "/index.php?page=item&id=1", "image": [2, 4, 42, 34], "title": [2, 36, 42, 46]},
        {"destination": site + "/index.php?page=item&id=2", "image": [54, 4, 94, 34], "title": [54, 36, 94, 46]},
    ]}
    assert reference_title_region(screen, reference, receipt, site) == (0, 0, 48, 52)
    screen.paste(reference, (54, 4))
    receipt["source_pixels"] = pixel_hash(screen)
    assert reference_title_region(screen, reference, receipt, site) is None


def test_staged_holdout_keeps_complete_pairs_and_freezes_the_next_phase(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_visualwebarena_live as live

    monkeypatch.setenv("VWA_CLASSIFIEDS", "http://localhost:19980")
    monkeypatch.setenv("VWA_HOMEPAGE", "http://localhost:19981")
    monkeypatch.setattr(live.importlib.metadata, "version", lambda _name: "1")
    monkeypatch.setattr(live, "holdout_profile", lambda *args: True)
    monkeypatch.setattr(live, "RapidScreenReader", lambda: SimpleNamespace(config={}))
    monkeypatch.setattr(live, "EvidenceAnswerPolicy", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "VisualWebArenaEnvironment", lambda task_id, **_options: SimpleNamespace(
        task=f"visualwebarena.{task_id}", preview_listing_links=None, capture_cards=None, close=lambda: None))
    monkeypatch.setattr(live, "render_replay", lambda _pending: None)

    recorded = []

    def fake_episode(_env, _policy, task, seed, _budget, pending, **options):
        recorded.append((task, options["answer_check"]))
        pending.mkdir()
        failed = task == "visualwebarena.36" and len(recorded) == 16
        row = dict.fromkeys(("predictions", "model_calls", "steps", "invalid_predictions", "invalid_action_retries",
                             "repaired_predictions", "scroll_feedbacks", "scroll_detection_signals", "answer_checks",
                             "answers_submitted", "answers_from_checks"), 0)
        row.update(task=task, seed=seed, success=False, reason="runtime_error" if failed else "step_limit")
        live.write_json(pending / "episode.json", row)
        return row

    monkeypatch.setattr(live, "run_episode", fake_episode)
    actor = SimpleNamespace(config={"revision": "fixed"})
    out = tmp_path / "study"
    run = dict(out=out, study="evidence-cards", task_set=live.INPUT_HOLDOUT_TASK_SET, max_steps=8,
               reference_archive=object(), input_manifest={})
    with pytest.raises(ValueError, match="before the cross-site phase"):
        live.run_study(actor, **run, input_phase="finish")
    live.run_study(actor, **run, input_phase="classifieds")
    assert len(recorded) == 14
    assert all(task != "visualwebarena.36" for task, _mode in recorded)
    assert not (out / "summary.json").exists()
    with pytest.raises(ValueError, match="original code, model"):
        live.run_study(SimpleNamespace(config={"revision": "changed"}), **run, input_phase="finish")
    with pytest.raises(RuntimeError, match="trace retained"):
        live.run_study(actor, **run, input_phase="finish")
    assert not (out / "summary.json").exists()
    live.run_study(actor, **run, input_phase="finish")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["task_ids"] == list(live.TASK_SETS[live.INPUT_HOLDOUT_TASK_SET])
    assert recorded[14:] == [("visualwebarena.36", True), ("visualwebarena.36", False), ("visualwebarena.36", False)]
    assert all(row["episodes"] == 8 for row in summary["strategies"].values())
