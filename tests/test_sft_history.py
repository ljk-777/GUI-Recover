import json

from PIL import Image
import pytest

from gui_recover.sft import HISTORY_PROFILES, file_sha256, grounding_query
from gui_recover.sft_history import STUDY_HEADS, reuse_development_data, select_history_prompt


@pytest.mark.parametrize("profile", HISTORY_PROFILES)
def test_history_profiles_validate_full_action_before_summarizing(profile):
    action = {"action": "INPUT", "value": "example", "position": [0.21, 0.32]}
    assert grounding_query("Submit button", history_profile=profile) == "Submit button"
    query = grounding_query("Submit button", action, history_profile=profile)
    assert "Submit button" in query
    if profile == "action-only":
        assert "position" not in query and "0.21" not in query
    else:
        assert '[0.21,0.32]' in query
    with pytest.raises(ValueError):
        grounding_query("Submit button", {**action, "reward": 1}, history_profile=profile)


def test_reordered_profile_preserves_legacy_lines_and_changes_target_position():
    action = {"action": "CLICK", "value": None, "position": [0.1, 0.2]}
    legacy = grounding_query("target", action)
    assert legacy == ('target\nPrevious proposed action: {"action":"CLICK","value":null,"position":[0.1,0.2]}'
                      '\nRecheck the current screenshot and return the target location.')
    reordered = grounding_query("target", action, history_profile="target-last")
    assert reordered.splitlines() == legacy.splitlines()[1:] + legacy.splitlines()[:1]
    with pytest.raises(ValueError):
        grounding_query("target", history_profile="unknown")


def prediction(kind, hits, split="validation"):
    return {"split": split, "kind": kind, "task": "enter-text", "heads": {
        h: {"hit": h in hits, "point": [0.2, 0.3]} for h in STUDY_HEADS}}


def test_recovery_only_gain_cannot_win_selection_with_normal_regressions():
    rows = [prediction("normal_submit", {"base_plain", "history_plain", "history_current_target"}),
            prediction("correction", {"history_target_last", "history_current_target"}),
            prediction("correction", {"history_target_last"})]
    choice = select_history_prompt(rows)
    assert choice["selected_head"] == "history_current_target"
    assert choice["eligible_candidates"] == ["history_current_target"]


def test_selection_falls_back_without_eligible_repair_and_rejects_test_labels():
    rows = [prediction("normal_submit", {"base_plain", "history_plain"}),
            prediction("correction", {"history_target_last"})]
    assert select_history_prompt(rows)["selected_head"] == "history_plain"
    assert not select_history_prompt(rows)["prompt_restoration_passed_validation"]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_history_prompt(rows)


def test_development_reuse_copies_exact_evidence_and_excludes_old_test(tmp_path):
    reference, out = tmp_path / "reference", tmp_path / "out"
    reference.mkdir()
    out.mkdir()
    (reference / "traces").mkdir()
    groups = {}
    cohort = []
    for i, split in enumerate(("train", "validation", "test")):
        image = reference / f"{split}.png"
        Image.new("RGB", (10, 10), (i * 50, 0, 0)).save(image)
        row = {"id": split, "split": split, "task": "click-test", "seed": i,
            "kind": "normal_entry", "image": image.name, "image_sha256": file_sha256(image),
            "query": "target", "target_boxes": [[0.2, 0.2, 0.8, 0.8]],
            "target_point": [0.5, 0.5], "answer": "[0.5,0.5]"}
        (reference / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        (reference / f"traces/click-test-{i}.json").write_text(json.dumps({"initial": {"image": image.name}}))
        groups[split] = [{"task": "click-test", "seeds": [i]}]
        cohort.append({"task": "click-test", "seed": i, "split": split})
    config = {"splits": groups}
    (reference / "manifest.json").write_text(json.dumps({"config": config,
        "runtime": {"git_commit": "original"}, "source_cohort": cohort, "filtered": []}))
    rows, provenance = reuse_development_data(reference, out, config)
    assert set(rows) == {"train", "validation"}
    for split in rows:
        assert file_sha256(out / f"{split}.jsonl") == file_sha256(reference / f"{split}.jsonl")
        assert file_sha256(out / f"{split}.png") == file_sha256(reference / f"{split}.png")
    assert not (out / "test.png").exists() and not (out / "test.jsonl").exists()
    assert not (out / "traces/click-test-2.json").exists()
    assert len(provenance["source_cohort"]) == 2
    changed = {"splits": {**groups, "validation": [{"task": "click-test", "seeds": [99]}]}}
    with pytest.raises(ValueError, match="scenes differ"):
        reuse_development_data(reference, out, changed)
