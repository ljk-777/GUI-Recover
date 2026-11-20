"""Inspect public VisualWebArena screenshots with a frozen offline recheck."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
from PIL import Image

from gui_recover.qwen_grounding import QwenGroundingPolicy
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text_audit import checked_file


def download(root, config, endpoint):
    plan = json.loads(config.read_text())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for case in plan["cases"]:
        task = int(case["task_id"])
        (root / str(task)).mkdir()
        for number, digest in enumerate(case["image_sha256"]):
            name = f"screenshot_step_{number}.png"
            relative = f"screenshots/visualwebarena/{plan['agent']}/visualwebarena.{task}/{name}"
            url = f"{endpoint.rstrip('/')}/datasets/{plan['dataset']}/resolve/{plan['dataset_revision']}/{relative}"
            with urlopen(url, timeout=30) as source:
                content = source.read(8 * 1024 * 1024 + 1)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"VisualWebArena screenshot hash mismatch for {task}/{number}")
            (root / str(task) / name).write_bytes(content)
    (root / "protocol.json").write_bytes(config.read_bytes())


def inspect_frames(first, second):
    if first.size != second.size:
        raise ValueError("Recorded viewport dimensions changed")
    before, after = (np.asarray(picture, dtype=np.int16) for picture in (first, second))
    changed = np.max(np.abs(before - after), axis=2) >= 20
    return {"size": list(first.size), "changed_fraction": float(changed.mean()),
            "can_verify_action_effect_from_pixels_alone": False}


def answer(actor, picture, goal):
    prompt = ("Answer the user's question from visible evidence on this captured web page. "
              "Return only the requested short answer. Return UNKNOWN if the evidence is insufficient.\n"
              f"Question: {goal}")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": picture}, {"type": "text", "text": prompt}]}]
    started = perf_counter()
    result = actor._generate(messages).strip()
    return {"answer": result, "seconds": perf_counter() - started, **actor.last_metrics}


def recheck_image(first, last):
    # Use a second recorded visual observation at the same 2:1 split in every case.
    width, height = first.size
    canvas = Image.new("RGB", (width, height * 2), "white")
    canvas.paste(last, (0, 0))
    canvas.paste(first, (0, height))
    return canvas


def accepted(raw, alternatives):
    return raw.strip().strip("'\"").casefold() in {value.casefold() for value in alternatives}


def replay(root, out, actor=None):
    root = Path(root)
    plan = json.loads((root / "protocol.json").read_text())
    if plan["kind"] != "recorded_visualwebarena" or len({case["task_id"] for case in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Expected unique task IDs from a recorded VisualWebArena split")
    summaries = []
    for case in plan["cases"]:
        images = []
        for number, digest in enumerate(case["image_sha256"]):
            path = root / str(int(case["task_id"])) / f"screenshot_step_{number}.png"
            with Image.open(checked_file(path, digest)) as picture:
                images.append(picture.convert("RGB"))
        first, last = images
        result = {"task_id": case["task_id"], "goal": case["goal"], "recorded_action": case["recorded_action"],
            "image_sha256": case["image_sha256"], "historical_environment_reward": case["historical_environment_reward"],
            "observations": inspect_frames(first, last)}
        if actor is not None:
            plain = answer(actor, last, case["goal"])
            extra = plain["answer"].strip(".").casefold() == "unknown"
            checked = answer(actor, recheck_image(first, last), case["goal"]) if extra else plain
            alternatives = case["accepted_answers"]
            result.update(last_frame={**plain, "matches": accepted(plain["answer"], alternatives)},
                          with_recheck={**checked, "matches": accepted(checked["answer"], alternatives),
                                        "extra_model_requests": int(extra)})
        summaries.append(result)
    report = {"kind": "offline_visualwebarena_answer_recheck", "live_browser_rollouts": 0,
        "protocol_sha256": file_sha256(root / "protocol.json"), "cases": summaries}
    if actor is not None:
        report.update(model=actor.config,
            single_frame_matches=sum(r["last_frame"]["matches"] for r in summaries),
            recheck_matches=sum(r["with_recheck"]["matches"] for r in summaries),
            extra_model_requests=sum(r["with_recheck"]["extra_model_requests"] for r in summaries))
    write_json(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", type=Path, metavar="PROTOCOL", help="Fetch only the six registered screenshots")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--qwen", type=Path, help="Optional local Qwen checkpoint for answer rechecking")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh report path")
    if args.download is not None:
        download(args.recordings, args.download, args.endpoint)
    actor = QwenGroundingPolicy(args.qwen, max_new_tokens=32, max_pixels=896 * 896,
                                device=args.device) if args.qwen else None
    print(json.dumps(replay(args.recordings, args.out, actor), indent=2))


if __name__ == "__main__":
    main()
