"""Generate privileged teacher labels offline; all model inputs stay visual."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.runner import write_json
from gui_recover.sft import REQUERY, audit_dataset


def visible_boxes(page, selector, picture):
    boxes = []
    scale = page._bgym_scale_factor
    locator = page.locator(selector) if isinstance(selector, str) else selector
    for element in locator.all():
        b = element.bounding_box()
        if b is None:
            continue
        box = [max(0, b["x"] * scale / picture.width), max(0, b["y"] * scale / picture.height),
               min(1, (b["x"] + b["width"]) * scale / picture.width),
               min(1, (b["y"] + b["height"]) * scale / picture.height)]
        if (box[2] - box[0]) * picture.width >= 4 and (box[3] - box[1]) * picture.height >= 4:
            boxes.append(box)
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    shutil.copyfile("vendor/miniwob-plusplus/LICENSE", out / "MINIWOB_LICENSE.txt")
    manifest = {"config": config, "runtime": fingerprint(),
                "teacher": "DOM boxes are labels only; no DOM in model inputs",
                "scope": "Grounding and query reformulation, not demonstrated recovery or autonomous planning",
                "filtered": [], "environment": None}
    for split, groups in config["splits"].items():
        rows = []
        for group in groups:
            task = group["task"]
            for seed in group["seeds"]:
                env = MiniwobEnvironment(task)
                try:
                    obs, _ = env.reset(seed)
                    page = env.env.unwrapped.page
                    manifest["environment"] = env.metadata()
                    if task == "click-test":
                        targets = [("button", "Click Me! button", "#subbtn")]
                    elif task == "enter-text":
                        targets = [("input", "text field", "#tt"), ("submit", "Submit button", "#subbtn"),
                                   ("requery", REQUERY, "#tt")]
                    elif task == "focus-text":
                        targets = [("input", "text field", "#tt")]
                    elif task == "click-button":
                        text = re.fullmatch(r'Click on the "(.*)" button\.', obs.goal).group(1)
                        # Exact accessible name matches all correct duplicate buttons.
                        candidates = page.get_by_role("button", name=text, exact=True)
                        targets = [("button", f'The "{text}" button', candidates)]
                    else:
                        raise ValueError("Unsupported teacher task")
                    image = f"images/{task}-{seed}.png"
                    obs.screenshot.save(out / image)
                    sha = hashlib.sha256((out / image).read_bytes()).hexdigest()
                    for kind, query, selector in targets:
                        boxes = visible_boxes(page, selector, obs.screenshot)
                        if not boxes:
                            manifest["filtered"].append({"task": task, "seed": seed, "kind": kind,
                                                         "reason": "no_visible_target_box_at_least_4px"})
                            continue
                        b = boxes[0]
                        point = [round((b[0] + b[2]) / 2, 3), round((b[1] + b[3]) / 2, 3)]
                        rows.append({"id": f"{task}-{seed}-{kind}", "split": split,
                            "task": task, "seed": seed, "kind": kind, "query": query,
                            "image": image, "image_sha256": sha, "target_boxes": boxes,
                            "target_point": point, "answer": json.dumps(point, separators=(",", ":"))})
                finally:
                    env.close()
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"Collected {split}: {len(rows)} supervised pairs", flush=True)
    manifest["audit"] = audit_dataset(out)
    for split in config["splits"]:
        manifest.setdefault("split_sha256", {})[split] = hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
