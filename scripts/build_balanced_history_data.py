"""Reweight development data without mining or reading any new test scene."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from gui_recover.cli import fingerprint
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256, load_records
from gui_recover.sft_balanced import balanced_training_records
from gui_recover.sft_history import reuse_development_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze clean training recipe")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, reused = reuse_development_data(args.reference_data, out, config)
    train = balanced_training_records(rows["train"], config["balanced_training"]["submit_repeats"])
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    validation = [{**r, "history_input": r.get("previous_action") is not None} for r in rows["validation"]]
    (out / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in validation))
    shutil.copyfile(Path(args.reference_data) / "MINIWOB_LICENSE.txt", out / "MINIWOB_LICENSE.txt")
    a, b = load_records(out, "train"), load_records(out, "validation")
    assert not {r["image_sha256"] for r in a} & {r["image_sha256"] for r in b}
    write_json(out / "manifest.json", {"runtime": runtime, "config": config, **reused,
        "purpose": "Training-only dataset; test split deliberately absent; repeats are not new scenes",
        "records": {"train": len(train), "validation": len(validation)},
        "unique_train_scenes": len({(r["task"], r["seed"]) for r in train}),
        "train_by_kind": dict(Counter(r["kind"] for r in train)),
        "train_by_context": dict(Counter(str(r["history_input"]) for r in train)),
        "split_sha256": {s: file_sha256(out / f"{s}.jsonl") for s in ("train", "validation")}})


if __name__ == "__main__":
    main()
