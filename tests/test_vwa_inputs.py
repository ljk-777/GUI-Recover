import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from gui_recover.actions import ActionError, compile_action, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy, contact_from_evidence, reference_contact_box
from gui_recover.environment import VisualObservation, frame_key
from gui_recover.listing_cards import reference_result_action, reference_title_region
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


def test_catalog_title_uses_rendered_whitespace_instead_of_source_spacing(monkeypatch):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_vwa_catalog_holdout import Catalog

    catalog = Catalog("http://localhost:19980")
    html = b"""<p>1 - 1 of 1 listings</p>
        <li class="listing-card"><a class="title" href="/index.php?page=item&id=8">
            Silver  ring
            size 6
        </a></li>"""
    monkeypatch.setattr(catalog, "read", lambda path: html)
    _, total, cards, _ = catalog.results(sPattern="ring")
    assert total == 1
    assert cards[0].title == "Silver ring size 6"


def test_catalog_rejects_a_reference_with_a_reencoded_photo_on_another_card(monkeypatch):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_vwa_catalog_holdout import Card, Catalog, reference_photo_evidence
    from run_vwa_catalog_holdout import validate_reference_photo_evidence

    image = Image.new("RGB", (100, 64), "navy")
    ImageDraw.Draw(image).rectangle((18, 10, 77, 51), fill="orange")
    different = Image.new("RGB", (100, 64), "green")
    photos = {}
    for name, picture, encoding in (("ring.png", image, "PNG"), ("alias.jpg", image, "JPEG"),
                                     ("different.png", different, "PNG")):
        stream = io.BytesIO()
        picture.save(stream, format=encoding)
        photos["/" + name] = stream.getvalue()
    catalog = Catalog("http://localhost:19980")
    monkeypatch.setattr(catalog, "read", photos.__getitem__)
    target, alias, other = [Card(f"/index.php?page=item&id={index}", "Ring", 20, path)
                            for index, path in enumerate(photos, 1)]
    assert photos[target.photo] != photos[alias.photo]
    assert reference_photo_evidence(catalog, target, [target, other, alias]) is None
    proof = {"listing_path": target.item, "source": {"cards": [{"item": card.item} for card in (target, other)],
              "reference_photo": reference_photo_evidence(catalog, target, [target, other])}}
    validate_reference_photo_evidence(proof)
    proof["source"]["reference_photo"]["comparisons"] = []
    with pytest.raises(ValueError, match="does not cover"):
        validate_reference_photo_evidence(proof)


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
    monkeypatch.setenv("VWA_SHOPPING", "http://localhost:19981")
    monkeypatch.setenv("VWA_REDDIT", "http://localhost:19982")
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
        failed = task == "visualwebarena.36" and len(recorded) == 14
        row = dict.fromkeys(("predictions", "model_calls", "steps", "invalid_predictions", "invalid_action_retries",
                             "repaired_predictions", "scroll_feedbacks", "scroll_detection_signals", "answer_checks",
                             "answers_submitted", "answers_from_checks"), 0)
        row.update(task=task, seed=seed, success=False, reason="runtime_error" if failed else "step_limit")
        live.write_json(pending / "episode.json", row)
        return row

    monkeypatch.setattr(live, "run_episode", fake_episode)
    actor = SimpleNamespace(config={"revision": "fixed"})
    out = tmp_path / "study"
    sites = {str(task): {"start_sites": ["classifieds"]} for task in live.TASK_SETS[live.INPUT_HOLDOUT_TASK_SET]}
    sites["36"]["start_sites"].append("shopping")
    sites["233"]["start_sites"].append("reddit")
    run = dict(out=out, study="evidence-cards", task_set=live.INPUT_HOLDOUT_TASK_SET, max_steps=8,
               reference_archive=object(), input_manifest=sites)
    with pytest.raises(ValueError, match="before the cross-site phase"):
        live.run_study(actor, **run, input_phase="finish")
    live.run_study(actor, **run, input_phase="classifieds")
    assert len(recorded) == 12
    assert all(task not in ("visualwebarena.36", "visualwebarena.233") for task, _mode in recorded)
    assert not (out / "summary.json").exists()
    with pytest.raises(ValueError, match="original code, model"):
        live.run_study(SimpleNamespace(config={"revision": "changed"}), **run, input_phase="finish")
    with pytest.raises(RuntimeError, match="trace retained"):
        live.run_study(actor, **run, input_phase="finish")
    assert not (out / "summary.json").exists()
    live.run_study(actor, **run, input_phase="finish")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["task_ids"] == list(live.TASK_SETS[live.INPUT_HOLDOUT_TASK_SET])
    assert recorded[12:] == [("visualwebarena.36", True), ("visualwebarena.36", False), ("visualwebarena.36", False),
                             ("visualwebarena.233", True), ("visualwebarena.233", False)]
    assert all(row["episodes"] == 8 for row in summary["strategies"].values())


def test_registered_start_pages_reveal_sites_missing_from_login_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_visualwebarena_live as live

    sites = live.registered_start_sites("__CLASSIFIEDS__/index.php?page=item |AND| __REDDIT__/f/demo/1")
    assert sites == ["classifieds", "reddit"]
    manifest = {"fixture": {"start_sites": sites}}
    monkeypatch.setenv("VWA_CLASSIFIEDS", "http://localhost:19980")
    monkeypatch.setenv("VWA_REDDIT", "http://localhost:19980")
    with pytest.raises(ValueError, match="distinct origins"):
        live.holdout_backend_origins(manifest)
    monkeypatch.setenv("VWA_REDDIT", "http://localhost:19982")
    assert live.holdout_backend_origins(manifest)["reddit"] == "http://localhost:19982"


def test_reference_search_sends_only_reference_to_model_and_grounds_action_on_screen():
    site = "http://localhost:19980"
    screen = Image.new("RGB", (200, 100), "blue")
    reference = Image.new("RGB", (64, 48), "red")
    rows = [
        {"text": "Keyword", "bounds": [5, 3, 40, 20]},
        {"text": "e.g., a blue used car", "bounds": [10, 24, 90, 36]},
        {"text": "Search", "bounds": [110, 24, 147, 37]},
    ]
    observation = VisualObservation("Find this exact item and tell me the price.", screen,
                                    reference_images=(reference,), tabs=(("Classifieds", site),))
    actor = SimpleNamespace(prepare_image=lambda image: image, last_metrics={})

    def ask(image, prompt):
        assert image is reference
        assert "only image is the task reference" in prompt
        actor.last_metrics = {"provider_output": '{"query":"red jacket"}', "model_invocations": 1}
        return actor.last_metrics["provider_output"]

    actor.ask = ask
    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=lambda image, size: rows,
                                  listing_site=site, ranked_capture=lambda: None)
    action = parse_action(policy.predict(observation, []))
    assert action.action == "SEARCH" and action.value == "red jacket"
    assert action.position == (50.5 / 200, 30.5 / 100)
    assert policy.last_metrics["reference_pixels"] == [pixel_hash(reference)]
    assert policy.last_metrics["evidence_frame"] == pixel_hash(screen)
    assert "browser_tabs" not in policy.last_metrics


def test_reference_result_navigation_requires_complete_scope_and_only_breaks_ties_for_latest():
    site = "http://localhost:19980"
    screen = Image.new("RGB", (300, 160), "white")
    reference = Image.new("RGB", (64, 45), "navy")
    ImageDraw.Draw(reference).rectangle((13, 8, 48, 37), fill="orange")
    thumbnail = reference.resize((80, 67))
    screen.paste(thumbnail, (5, 20))
    screen.paste(thumbnail, (105, 20))
    screen.paste(Image.new("RGB", (80, 67), "green"), (205, 20))
    cards = [
        {"destination": f"{site}/index.php?page=item&id={index + 1}", "image": [x, 20, x + 80, 87],
         "title": [x, 90, x + 80, 100]}
        for index, x in enumerate((5, 105, 205))
    ]
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "next": None, "grid": [0.9, 0.1],
               "cards": cards[:2]}
    rows = [{"text": text} for text in ("Search results: jacket", "Sort by: Newly listed", "1 - 3 of 5 listings")]
    latest = "Find the most recent lister of this exact item."
    action = reference_result_action(screen, reference, receipt, site, rows, "jacket", latest)
    assert action.action == "LISTING_GRID"
    receipt["cards"] = cards
    action = reference_result_action(screen, reference, receipt, site, rows, "jacket", latest)
    assert action.action == "CLICK" and action.position == (45 / 300, 95 / 160)
    assert reference_result_action(screen, reference, receipt, site, rows, "jacket",
                                   "Find the lister of this exact item.") is None
    assert reference_result_action(screen, reference, receipt, site, rows, "boots", latest) is None


def test_exact_photo_title_stops_on_duplicate_matching_pixels_without_invoking_the_model():
    site = "http://localhost:19980"
    screen = Image.new("RGB", (300, 160), "white")
    reference = Image.new("RGB", (80, 67), "navy")
    ImageDraw.Draw(reference).rectangle((13, 8, 48, 37), fill="orange")
    screen.paste(reference, (5, 20))
    screen.paste(reference, (105, 20))
    cards = [
        {"destination": f"{site}/index.php?page=item&id={index + 1}", "image": [x, 20, x + 80, 87],
         "title": [x, 90, x + 80, 100]}
        for index, x in enumerate((5, 105))
    ]
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "next": None, "grid": None, "cards": cards}
    goal = "Find the exact item shown in the reference photo. What is its listing title?"
    observation = VisualObservation(goal, screen, tabs=(("Search", site + "/index.php?page=search"),),
                                    reference_images=(reference,))
    actor = SimpleNamespace(prepare_image=lambda image: pytest.fail("No model needed"))
    policy = EvidenceAnswerPolicy(actor, confirm=True, screen_reader=lambda *_: pytest.fail("No OCR needed"),
                                  listing_site=site, ranked_capture=lambda: receipt)
    assert policy.predict(observation, []) is None
    assert policy.last_metrics["provider_format"] == "evidence_reference_title_ambiguous"
    assert policy.last_metrics["stop_after_evidence"] is True
    assert policy.last_metrics["model_invocations"] == 0


def test_reference_result_refines_only_a_complete_clear_miss_with_a_visible_keyword_field():
    site = "http://localhost:19980"
    screen = Image.new("RGB", (300, 160), "white")
    reference = Image.new("RGB", (64, 45), "navy")
    ImageDraw.Draw(reference).rectangle((13, 8, 48, 37), fill="orange")
    screen.paste(Image.new("RGB", (80, 67), "green"), (205, 20))
    rows = [{"text": text, "bounds": [10, 10 + index * 15, 100, 20 + index * 15]}
            for index, text in enumerate(("Search results: brand model type", "Sort by: Newly listed",
                                          "1 - 1 of 30 listings", "Your search", "brand model type", "Apply"))]
    receipt = {"source_pixels": pixel_hash(screen), "scroll_top": 0, "next": None, "grid": None,
               "cards": [{"destination": f"{site}/index.php?page=item&id=1", "title": [205, 90, 285, 100],
                          "image": [205, 20, 285, 87]}]}
    goal = "Find the most recent lister of this exact item."
    args = screen, reference, receipt, site, rows, "brand model type", goal
    assert reference_result_action(*args) is None
    action = reference_result_action(*args, refinement_size=screen.size)
    assert (action.action, action.value, action.position) == ("SEARCH_REPLACE", "brand", (55 / 300, 75 / 160))
    assert parse_action(json.dumps(action.to_dict())) == action
    with pytest.raises(ActionError, match="Classifieds adapter"):
        compile_action(action, *screen.size)
    rows.pop()
    assert reference_result_action(*args, refinement_size=screen.size) is None


def test_reference_contact_localization_requires_one_aligned_publisher_email():
    heading = {"text": "Contact publisher", "bounds": [30, 10, 60, 20]}
    email = {"text": "E-mail: demo@example.com", "bounds": [30, 75, 85, 85]}
    goal = "Tell me the email of the lister of this exact item."
    assert reference_contact_box(goal, [heading, email]) == email["bounds"]
    assert reference_contact_box(goal, [email]) is None
    assert reference_contact_box(goal, [heading, email, email]) is None
    assert reference_contact_box(goal, [heading, {**email, "bounds": [2, 75, 25, 85]}]) is None
    label = {"text": "E-mail:", "bounds": [30, 75, 50, 85]}
    address = {"text": "demo@example.com", "bounds": [30, 84, 85, 94]}
    assert reference_contact_box(goal, [heading, label, address]) == [30, 75, 85, 94]
    assert reference_contact_box(goal, [heading, label, {**address, "bounds": [30, 105, 85, 115]}]) is None


def test_contact_candidate_uses_one_labeled_email_only_after_publisher_crop():
    goal = "Tell me the email of the lister of this exact item."
    facts = ["Name: Sample Publisher", "E-mail: demo@example.com", "$20"]
    raw = json.dumps({"answer": "null", "evidence": facts})
    assert contact_from_evidence(raw, goal) is None
    assert contact_from_evidence(raw, goal, publisher_region=True) == "demo@example.com"
    ambiguous = json.dumps({"answer": "null", "evidence": [*facts, "Email: second@example.com"]})
    assert contact_from_evidence(ambiguous, goal, publisher_region=True) is None
