import hashlib
import json

import pytest
from PIL import Image

from gui_recover.sft import audit_dataset, completion_labels, grounding_messages, load_records


def test_only_assistant_tokens_are_supervised_and_boundary_is_verified():
    assert completion_labels([1, 9, 9, 2, 30, 31, 3], [1, 9, 9, 2]) == [-100, -100, -100, -100, 30, 31, 3]
    for full, prompt in (([1, 3, 4], [1, 2]), ([1, 2], [1, 2]), ([1], [])):
        with pytest.raises(ValueError, match="boundary"):
            completion_labels(full, prompt)


def test_grounding_prompt_has_no_teacher_metadata_or_answer_in_user_turn():
    picture = Image.new("RGB", (10, 10))
    messages = grounding_messages(picture, "text field", "[0.5,0.5]")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"][1]["image"] is picture
    assert messages[0]["content"][-1] == {"type": "text", "text": "text field"}
    assert "[0.5,0.5]" not in str(messages[0])


def make_dataset(root):
    for index, split in enumerate(("train", "validation", "test")):
        image = root / f"{split}.png"
        Image.new("RGB", (10, 10), (index * 50, 0, 0)).save(image)
        row = {"id": split, "split": split, "task": "seen" if split != "test" else "new",
               "seed": index, "image": image.name, "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
               "query": "field", "target_boxes": [[0.2, 0.2, 0.8, 0.8]],
               "target_point": [0.5, 0.5], "answer": "[0.5,0.5]"}
        (root/f"{split}.jsonl").write_text(json.dumps(row)+"\n")


def test_dataset_audit_checks_split_groups_and_visible_label_alignment(tmp_path):
    make_dataset(tmp_path)
    assert audit_dataset(tmp_path)["unseen_test_tasks"] == ["new"]
    file = tmp_path / "test.jsonl"
    row = json.loads(file.read_text())
    row["answer"] = "[0.9,0.9]"
    row["target_point"] = [0.9, 0.9]
    file.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="outside"):
        load_records(tmp_path, "test")


def test_identical_images_cannot_cross_splits(tmp_path):
    make_dataset(tmp_path)
    file = tmp_path / "test.jsonl"
    row = json.loads(file.read_text())
    row["image"] = "train.png"
    row["image_sha256"] = hashlib.sha256((tmp_path/"train.png").read_bytes()).hexdigest()
    file.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="leaks"):
        audit_dataset(tmp_path)
