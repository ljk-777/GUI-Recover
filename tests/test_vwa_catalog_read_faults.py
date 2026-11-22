import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_recover.actions import parse_action
from gui_recover.environment import VisualObservation
from gui_recover.verification import pixel_hash


@pytest.fixture
def faults(monkeypatch):
    pytest.importorskip("bs4")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_vwa_catalog_read_faults

    return run_vwa_catalog_read_faults


def test_reply_faults_preserve_the_original_amount(faults):
    city = faults.changed_reply("shorten-city", '{"city":"Borough of Oak Park","dollars":80.50}')
    title = faults.changed_reply("change-title-letter", '{"title":"Xylophone X2","dollars":"90.25"}')
    assert json.loads(city) == {"city": "Borough", "dollars": 80.5}
    assert json.loads(title) == {"title": "yylophone X2", "dollars": "90.25"}
    with pytest.raises(ValueError, match="at least two words"):
        faults.changed_reply("shorten-city", '{"city":"Rockville","dollars":30}')


def test_ocr_fault_applies_once_at_the_registered_frame(faults):
    screen = Image.new("RGB", (200, 140), "white")
    crumb = {"text": "Classifieds > Furniture > Desk", "bounds": [10, 20, 151, 31]}
    heading = {"text": "Desk", "bounds": [10, 40, 48, 58]}
    rows = [crumb, heading]
    target = {"evidence_frame": pixel_hash(screen), "screen_text": rows}
    case = faults.FaultCase({"task_id": 10403}, "omit-breadcrumb", {}, target)
    policy = faults.FaultPolicy(None, lambda *_: rows, case, confirm=True)
    assert policy.read_screen(Image.new("RGB", screen.size, "black"), screen.size) == rows
    assert policy.injected is None
    assert policy.read_screen(screen, screen.size) == [heading]
    assert policy.injected == {"fault": "omit-breadcrumb", "omitted_row": crumb}
    assert policy.read_screen(screen, screen.size) == rows


def test_price_ocr_fault_changes_only_the_registered_amount(faults):
    price = {"text": "90.00", "bounds": [791, 123, 884, 160]}
    rows = [
        price,
        {"text": "$", "bounds": [883, 126, 909, 158]},
        {"text": "Published date: yesterday", "bounds": [115, 187, 316, 203]},
        {"text": "Location: Oak Park, Illinois, United States", "bounds": [116, 207, 452, 224]},
        {"text": "120 $", "bounds": [791, 600, 884, 618]},
    ]
    screen = Image.new("RGB", (1000, 770), "white")
    target = {"evidence_frame": pixel_hash(screen), "screen_text": rows}
    case = faults.FaultCase({"task_id": 10401}, "change-ocr-price-digit", {}, target)
    policy = faults.FaultPolicy(None, lambda *_: rows, case, confirm=True)
    altered = policy.read_screen(screen, screen.size)
    assert altered[0] == {**price, "text": "10.00"}
    assert altered[1:] == rows[1:]
    assert policy.injected == {"fault": case.kind, "original_row": price, "injected_row": altered[0]}
    assert policy.read_screen(screen, screen.size) == rows


def test_price_dropout_erases_only_detector_rows_and_preserves_whole_line_read(faults):
    screen = Image.new("RGB", (1000, 770), "white")
    rows = [
        {"text": "90.00", "bounds": [791, 123, 884, 160]},
        {"text": "$", "bounds": [883, 126, 909, 158]},
        {"text": "Published date: yesterday", "bounds": [115, 187, 316, 203]},
        {"text": "Location: Oak Park, Illinois, United States", "bounds": [116, 207, 452, 224]},
    ]

    def crop_rows(_, size):
        return [{"text": "90.00 $", "bounds": [0, 0, *size]}]

    def reader(image, size):
        return rows if image is screen else crop_rows(image, size)

    reader.read_upright = reader.read_line = crop_rows
    actor = SimpleNamespace(prepare_image=lambda image: image)

    def ask(*_, **__):
        actor.last_metrics = {"model_invocations": 1}
        return '{"city":"Oak Park","dollars":90}'

    actor.ask = ask
    frame = pixel_hash(screen)
    case = faults.FaultCase({"task_id": 11001}, faults.LOCAL_PRICE_DROPOUT, {}, {"evidence_frame": frame})
    site = "http://localhost:19980"
    policy = faults.FaultPolicy(actor, reader, case, confirm=True, listing_site=site, ranked_capture=lambda: None)
    observation = VisualObservation("What is the asking price in dollars and the city for the listing open in this tab?",
                                    screen, tabs=(("Detail", site + "/index.php?page=item&id=7"),))
    assert policy.predict(observation, []) is None
    assert parse_action(policy.predict(observation, [])).value == "Oak Park, 90 dollars"
    for key in faults.PRICE_DETECTOR_RECEIPTS:
        receipt = policy.last_metrics[key]
        recorded = policy.injected["reads"][key]
        assert receipt["rows"] == []
        assert recorded["original_rows"][0]["text"] == "90.00 $"
        assert receipt["pixels_sha256"] == recorded["pixels_sha256"]
    assert policy.read_screen(screen, screen.size) == rows
    assert policy.price_retry_reader is crop_rows


@pytest.mark.parametrize(("stage", "key"), (("initial", "city"), ("isolated", "text")))
def test_correlated_price_fault_changes_only_the_model_amount(faults, stage, key):
    reply = json.dumps({key: "Oak Park", "dollars": 25.5})
    assert json.loads(faults.matching_price_reply(stage, reply, 2550, 3550)) == {
        key: "Oak Park", "dollars": "35.50",
    }
    with pytest.raises(ValueError, match="originally read the unmodified price"):
        faults.matching_price_reply(stage, reply, 2500, 3500)


def test_model_price_fault_supports_visible_thousands_and_currency(faults):
    reply = '{"city":"Oak Park","dollars":"1,025.50 $"}'
    assert json.loads(faults.matching_price_reply("initial", reply, 102550, 202550))["dollars"] == "2025.50"


@pytest.mark.parametrize(("dollars", "frame"), ((25, "start"), (35, "start"), (25, "different")))
def test_late_price_fault_anchors_amount_without_requiring_city_agreement(faults, tmp_path, dollars, frame):
    rows = [
        {"text": "25.00", "bounds": [791, 123, 884, 160]},
        {"text": "$", "bounds": [883, 126, 909, 158]},
        {"text": "Published date: yesterday", "bounds": [115, 187, 316, 203]},
        {"text": "Location: Borough of Oak Park, Illinois, United States", "bounds": [116, 207, 452, 224]},
    ]
    reply = json.dumps({"city": "Oak Park", "dollars": dollars})
    steps = [
        {"provider_format": faults.LISTING_FACTS_REGION, "screen_text": rows, "evidence_frame": "start"},
        {"provider_format": "evidence_verify_disagreed", "evidence_stage": "confirm",
         "evidence_frame": frame, "provider_output": reply},
    ]
    (tmp_path / "trajectory.jsonl").write_text(
        "".join(json.dumps({"event": "prediction", "metrics": step}) + "\n" for step in steps)
    )
    with pytest.raises(faults.FaultAnchorUnavailable):
        faults.clean_target(tmp_path, "match-ocr-price-in-reread")
    if dollars != 25 or frame != "start":
        with pytest.raises(faults.FaultAnchorUnavailable):
            faults.clean_target(tmp_path, faults.AMOUNT_ANCHORED_REREAD, expected_cents=2500)
        return
    target = faults.clean_target(tmp_path, faults.AMOUNT_ANCHORED_REREAD, expected_cents=2500)
    assert target["clean_confirmation"] == reply
    reread = '{"text":"Location: Borough of Oak Park, Illinois, United States","dollars":25}'
    assert json.loads(faults.inject_matching_price(faults.AMOUNT_ANCHORED_REREAD, target, reread)) == {
        "text": "Location: Borough of Oak Park, Illinois, United States", "dollars": "35.00",
    }


def test_offline_audit_rejects_an_injected_change_to_the_price(faults, monkeypatch, tmp_path):
    original = '{"city":"Oak Park","dollars":30}'
    altered = faults.changed_reply("shorten-city", original)
    target = {"evidence_frame": "frame", "provider_output": original}
    baseline = {"reset": ("goal", 0), "reset_context": ("tabs", ()), "reset_pixels": "frame"}
    stats = {"success": True, "reason": "success", "answers_submitted": 1, "predictions": 3, "model_calls": 2}
    monkeypatch.setattr(faults, "audit_evidence",
                        lambda *_: {**baseline, "row": stats, "evidence_stages": {"listing_fact_reread": 1}})
    case = faults.FaultCase({"task_id": 10401}, "shorten-city", baseline, target)

    def write_artifact(reply):
        faults.write_json(tmp_path / "injected-fault.json", {
            "fault": "shorten-city", "original_reply": original, "injected_reply": reply,
        })
        event = {"event": "prediction", "metrics": {"evidence_frame": "frame",
                 "evidence_stage": "confirm", "provider_output": reply}}
        (tmp_path / "trajectory.jsonl").write_text(json.dumps(event) + "\n")

    write_artifact(altered)
    assert faults.audit_fault(tmp_path, case, {})["success"]
    write_artifact(altered.replace("30", "31"))
    with pytest.raises(ValueError, match="changed confirmation field"):
        faults.audit_fault(tmp_path, case, {})


def test_finalized_cases_resume_without_initializing_a_model(faults, monkeypatch, tmp_path):
    site = "http://localhost:19980"
    monkeypatch.setenv("VWA_CLASSIFIEDS", site)
    monkeypatch.setattr(faults, "QwenGroundingPolicy", lambda *a, **k: pytest.fail("No model needed for resume"))
    monkeypatch.setattr(faults, "probe_instance", lambda: pytest.fail("No browser needed for resume"))
    checked = []
    stats = {"success": True, "answers_submitted": 1, "model_calls": 2}

    def audit_episode(directory, case, plan):
        checked.append(directory.name)
        assert directory.is_dir()
        return stats

    monkeypatch.setattr(faults, "audit_fault", audit_episode)
    out = tmp_path / "recorded"
    plan = {"cohort_sha256": "sealed", "listing_origin": site, "format": faults.FORMAT}
    faults.run_faults(out, plan, [], None, None)
    (out / "audit.json").unlink()  # An interrupted run can finish its last episode before writing totals.
    case = faults.FaultCase({"task_id": 10401}, "shorten-city", {}, {})
    (out / case.episode).mkdir()
    report = faults.run_faults(out, plan, [case], None, None)
    assert checked == [case.episode, case.episode]
    assert report["native_successes"] == 1 and report["model_calls"] == 2


def test_restart_recovers_an_incomplete_plan_write(faults, monkeypatch, tmp_path):
    site = "http://localhost:19980"
    monkeypatch.setenv("VWA_CLASSIFIEDS", site)
    plan = {"cohort_sha256": "sealed", "listing_origin": site, "format": faults.FORMAT}
    temporary_plan = tmp_path / "run-plan.json.partial"
    temporary_plan.write_text('{"format":')
    unrelated = tmp_path / "other-result.json"
    unrelated.write_text('{"marker":true}')
    with pytest.raises(ValueError, match="requires an empty output directory"):
        faults.run_faults(tmp_path, plan, [], None, None)
    assert json.loads(unrelated.read_text()) == {"marker": True}
    assert not (tmp_path / "run-plan.json").exists()
    assert temporary_plan.read_text() == '{"format":'

    unrelated.unlink()
    faults.run_faults(tmp_path, plan, [], None, None)
    assert not temporary_plan.exists()
    assert json.loads((tmp_path / "run-plan.json").read_text())["cohort_sha256"] == "sealed"


def test_concurrent_writer_cannot_replace_the_plan(faults, monkeypatch, tmp_path):
    fcntl = pytest.importorskip("fcntl")
    site = "http://localhost:19980"
    monkeypatch.setenv("VWA_CLASSIFIEDS", site)
    plan = {"cohort_sha256": "sealed", "listing_origin": site, "format": faults.FORMAT}
    with (tmp_path / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="already writing"):
            faults.run_faults(tmp_path, plan, [], None, None)
        assert not (tmp_path / "run-plan.json").exists()

    faults.run_faults(tmp_path, plan, [], None, None)
    assert (tmp_path / "run-plan.json").is_file()
