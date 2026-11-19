"""Same-image navigation vs grounding probe. Labels used only after generation."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from PIL import Image

from gui_recover.actions import ActionError, parse_action
from gui_recover.cli import fingerprint
from gui_recover.environment import VisualObservation
from gui_recover.grounding import parse_point, point_hits_boxes
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.prompts import GROUNDING
from gui_recover.runner import write_json


def summarize(rows):
    result = {}
    for variant in sorted({r["variant"] for r in rows}):
        groups = {"all": [r for r in rows if r["variant"] == variant]}
        for task in sorted({r["task"] for r in rows}):
            for phase in sorted({r["phase"] for r in rows if r["task"] == task}):
                groups[f"{task}:{phase}"] = [r for r in groups["all"] if r["task"] == task and r["phase"] == phase]
        result[variant] = {key: {"cases": len(group), "hits": sum(r["hit"] for r in group),
            "invalid": sum(bool(r["error"]) for r in group), "targets_visible": sum(r["target_visible"] for r in group),
            "mean_seconds": statistics.mean(r["seconds"] for r in group)}
            for key, group in groups.items() if group}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = Path(args.cases)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((cases / "inputs.json").read_text())
    model = ShowUIPolicy(model=args.model, revision=MODEL_REVISION)
    write_json(directory / "manifest.json", {"runtime": fingerprint(), "model_config": model.config,
        "grounding_prompt": GROUNDING, "cases_manifest": json.loads((cases / "manifest.json").read_text()),
        "inputs_sha256": hashlib.sha256((cases / "inputs.json").read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256((cases / "evaluator-labels.json").read_bytes()).hexdigest(),
        "metric": "normalized point inside at least one visible target box; not task success"})
    rows = []
    for item in inputs:
        screenshot = cases / item["screenshot"]
        if hashlib.sha256(screenshot.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Screenshot checksum mismatch")
        image = Image.open(screenshot).convert("RGB")
        variants = ["grounding-subgoal"]
        if item["phase"] == 0:
            variants = ["navigation-full", "grounding-full", "grounding-subgoal"]
        for variant in variants:
            model.reset(item["seed"])
            start = time.perf_counter()
            if variant == "navigation-full":
                raw = model.predict(VisualObservation(item["goal"], image), [])
            else:
                query = item["goal"] if variant == "grounding-full" else item["query"]
                raw = model.ground(image, query)
            elapsed = time.perf_counter() - start
            point, error = None, None
            try:
                point = parse_action(raw, mode="mixed").position if variant == "navigation-full" else parse_point(raw)
            except ActionError as exc:
                error = str(exc)
            # Evaluator labels are read only after the model invocation; never passed to it.
            boxes = json.loads((cases / "evaluator-labels.json").read_text())[item["id"]]["boxes"]
            row = {"id": item["id"], "task": item["task"], "seed": item["seed"], "phase": item["phase"],
                   "variant": variant, "raw": raw, "point": point, "error": error, "seconds": elapsed,
                   "target_visible": bool(boxes), "hit": point_hits_boxes(point, boxes), "metrics": model.last_metrics}
            rows.append(row)
            with (directory / "predictions.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            write_json(directory / "summary.json", summarize(rows))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
