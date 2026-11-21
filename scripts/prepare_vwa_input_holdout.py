"""Assemble the registered VisualWebArena holdout references from the upstream checkout."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from run_visualwebarena_live import INPUT_MEDIA_SHA256, INPUT_PROTOCOL_SHA256


def prepare(upstream, protocol_path, output):
    protocol_bytes = protocol_path.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != INPUT_PROTOCOL_SHA256:
        raise ValueError("Expected the registered input holdout protocol")
    protocol = json.loads(protocol_bytes)
    data = (upstream / "config_files/vwa/test_classifieds.raw.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != protocol["upstream_config_sha256"]:
        raise ValueError("Upstream tasks differ from the registered checkout")
    tasks = [task for task in json.loads(data) if task["task_id"] in protocol["task_ids"]]
    reference_count, _ = prepare_reference_archive(upstream, tasks, output, expected_sha256=INPUT_MEDIA_SHA256)
    (output / "cohort.json").write_bytes(protocol_bytes)
    return reference_count


def prepare_reference_archive(upstream, tasks, output, *, expected_sha256=None):
    names = set()
    for task in tasks:
        images = task.get("image") or []
        names.update([images] if isinstance(images, str) else images)
    image_root = (upstream / "environment_docker/webarena-homepage").resolve()
    assets = {}
    for name in sorted(names):
        path = (upstream / name).resolve(strict=True)
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError("Task reference must be an upstream homepage asset")
        assets[name] = path.read_bytes()
    references = {name: {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
                  for name, blob in assets.items()}
    canonical = json.dumps(references, sort_keys=True, separators=(",", ":")).encode()
    reference_sha256 = hashlib.sha256(canonical).hexdigest()
    if expected_sha256 is not None and reference_sha256 != expected_sha256:
        raise ValueError("Upstream image bytes differ from the registered file set")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "reference-media.tar.gz"
    with tarfile.open(archive, "w:gz") as pack:
        for name, blob in assets.items():
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            pack.addfile(member, io.BytesIO(blob))
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "reference-media.json").write_text(json.dumps(
        {"files": references, "archive_sha256": archive_digest}, indent=2) + "\n")
    return len(assets), reference_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.upstream, args.protocol, args.out)} registered reference images.")
