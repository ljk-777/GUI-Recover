"""Prepare the frozen seven official tasks and one independently sampled task."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_vwa_input_holdout import prepare_reference_archive
from run_visualwebarena_live import (
    FRESH_HOLDOUT_TASK_SET, SUPPLEMENTAL_TASK_ID, TASK_SETS, registered_input_entries,
)


def prepare(upstream, supplemental_path, output):
    task_ids = TASK_SETS[FRESH_HOLDOUT_TASK_SET]
    official_bytes = (upstream / "config_files/vwa/test_classifieds.raw.json").read_bytes()
    official = json.loads(official_bytes)
    supplemental_bytes = supplemental_path.read_bytes()
    supplemental = json.loads(supplemental_bytes)
    if supplemental["task_id"] != SUPPLEMENTAL_TASK_ID or any(
            task["task_id"] == SUPPLEMENTAL_TASK_ID for task in official):
        raise ValueError("Expected one independent task outside the official registry")
    entries = registered_input_entries([*official, supplemental], task_ids)
    if tuple(task["task_id"] for task in entries) != task_ids:
        raise ValueError("Some registered task rows are absent or duplicated")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    count, media_sha256 = prepare_reference_archive(upstream, entries, output)
    protocol = {
        "task_ids": list(task_ids),
        "official_task_ids": list(task_ids[:-1]),
        "supplemental_task_id": SUPPLEMENTAL_TASK_ID,
        "upstream_config_sha256": hashlib.sha256(official_bytes).hexdigest(),
        "entries_sha256": hashlib.sha256(canonical).hexdigest(),
        "reference_files_sha256": media_sha256,
        "seed": 0,
        "max_predictions": 8,
        "policy_comparison": ["answer-check", "evidence-cards"],
    }
    encoded = (json.dumps(protocol, indent=2) + "\n").encode()
    (output / "cohort.json").write_bytes(encoded)
    (output / "supplemental-task.json").write_bytes(supplemental_bytes)
    return count, hashlib.sha256(encoded).hexdigest(), media_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count, protocol_hash, media_hash = prepare(args.upstream, args.supplemental, args.out)
    print(f"Prepared {count} registered reference images.")
    print(f"Protocol SHA-256 {protocol_hash}")
    print(f"Reference manifest SHA-256 {media_hash}")
