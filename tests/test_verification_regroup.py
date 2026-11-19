from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from gui_recover.actions import Action
from gui_recover.sft import file_sha256
from gui_recover.verification_data import load_transitions, observation_rows
from gui_recover.verification_regroup import regroup_development_data, validate_regroup


def configs():
    original = {"splits": {s: [{"task": "focus-text", "seeds": seeds}]
        for s, seeds in (("train", [1, 2, 3]), ("validation", [4]), ("test", [5]))},
        "click_faults": ["drop-click", "offset-click"], "input_faults": ["drop-input"]}
    revised = deepcopy(original)
    revised["splits"]["train"][0]["seeds"][2] = 4
    revised["splits"]["validation"][0]["seeds"][0] = 3
    return original, revised


def test_regroup_rejects_test_changes_and_changed_fault_assignments():
    original, revised = configs()
    validate_regroup(original, revised)
    revised["splits"]["test"][0]["seeds"] = [6]
    with pytest.raises(ValueError, match="test is immutable"):
        validate_regroup(original, revised)
    original, revised = configs()
    revised["splits"]["train"][0]["seeds"] = [1, 4, 3]
    revised["splits"]["validation"][0]["seeds"] = [2]
    with pytest.raises(ValueError, match="assigned faults"):
        validate_regroup(original, revised)


def test_regroup_reindexes_vectors_but_keeps_test_images_and_traces_identical(tmp_path):
    original, revised = configs()
    source, out = tmp_path/"raw", tmp_path/"regrouped"
    (source/"images").mkdir(parents=True)
    (source/"traces").mkdir()
    expected = {}
    for split, groups in original["splits"].items():
        rows = []
        for index, seed in enumerate(groups[0]["seeds"]):
            scene = f"focus-text-{seed}"
            image = source/"images"/f"{scene}.png"
            Image.new("RGB", (12, 12), (1 if seed == 4 else seed, 20, 30)).save(image)
            frame = {"image": str(image.relative_to(source)), "sha256": file_sha256(image)}
            for regime in ("none", original["click_faults"][index % 2]):
                row = {"id": scene+"-"+regime, "scene_id": scene, "split": split,
                    "task": "focus-text", "seed": seed, "regime": regime, "deferred": False,
                    "status": "ok", "reference_achieved": True, "goal": "Focus textbox", "target": "text field",
                    "proposed_action": Action("CLICK", position=(0.5, 0.5)).to_dict(), "before": frame,
                    "immediate": {**frame, "effect_achieved": True, "terminated": False, "truncated": False}}
                trace = source/"traces"/f"{row['id']}.json"
                trace.write_text(json.dumps({"record": row}))
                row.update(trace=str(trace.relative_to(source)), trace_sha256=file_sha256(trace))
                rows.append(row)
        (source/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        views, collisions = observation_rows(source, rows)
        (source/f"{split}-views.json").write_text(json.dumps(views))
        (source/f"{split}-collisions.json").write_text(json.dumps(collisions))
        ids = [v["transition_id"]+":"+v["phase"] for v in views]
        values = np.asarray([[int(v["scene_id"].split("-")[-1]), i] for i, v in enumerate(views)], dtype=np.float32)
        expected.update(zip(ids, values))
        for b in ("pixels", "frozen"):
            np.savez_compressed(source/f"{split}-{b}.npz", features=values, ids=np.asarray(ids))
    m = regroup_development_data(source, out, original, revised, "original-commit", {"git_commit": "assembly-commit"})
    assert len(m["regroup"]["moves"]) == 2 and m["new_actor_generations_during_regroup"] == 0
    assert m["audit"]["cross_split_scene_and_initial_image_overlap"] == 0
    for split in revised["splits"]:
        load_transitions(out, split)
        with np.load(out/f"{split}-frozen.npz", allow_pickle=False) as a:
            for key, vector in zip(a["ids"], a["features"]):
                assert np.array_equal(vector, expected[key])
    for name, sha in m["regroup"]["source_files_sha256"].items():
        if name.startswith(("images/", "traces/", "test")):
            assert file_sha256(out/name) == sha
