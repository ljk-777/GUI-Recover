"""Supplement strict geometric scores with real hit-testing, no model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from gui_recover.actions import Action
from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.runner import is_success, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases, probe = Path(args.cases), Path(args.probe)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {r["id"]: r for r in json.loads((cases / "inputs.json").read_text())}
    labels = json.loads((cases / "evaluator-labels.json").read_text())
    rows = [json.loads(line) for line in (probe / "predictions.jsonl").read_text().splitlines()]
    results = []
    write_json(directory / "manifest.json", {"runtime": fingerprint(), "model_calls": 0,
              "purpose": "Reexecute saved click-button grounding points; keep strict box scores unchanged",
              "probe_manifest": json.loads((probe / "manifest.json").read_text())})
    for row in rows:
        if row["task"] != "click-button" or row["variant"] != "grounding-subgoal":
            continue
        env = MiniwobEnvironment(row["task"])
        try:
            obs, _ = env.reset(row["seed"])
            image_path = directory / f"{row['id']}.png"
            obs.screenshot.save(image_path)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if sha != inputs[row["id"]]["sha256"]:
                raise ValueError("Reset screenshot differs from original probe")
            point = row["point"]
            if point is None:
                raise ValueError("Invalid saved model output")
            page = env.env.unwrapped.page
            scale = page._bgym_scale_factor
            x, y = point[0] * obs.screenshot.width / scale, point[1] * obs.screenshot.height / scale
            element = page.evaluate("""([x,y]) => {
                const e=document.elementFromPoint(x,y);
                return e ? {tag:e.tagName,text:e.textContent} : null;
            }""", [x, y])
            _, reward, terminated, truncated, info, code = env.step(Action("CLICK", position=tuple(point)))
            results.append({"id": row["id"], "point": point, "boxes": labels[row["id"]]["boxes"],
                "strict_box_hit": row["hit"], "css_point": [x, y], "hit_test_element": element,
                "actual_success": is_success(reward, terminated, truncated, info["task_info"]),
                "executed_code": code, "task_info": info["task_info"], "reset_sha256": sha})
        finally:
            env.close()
        write_json(directory / "results.json", results)
    summary = {"cases": len(results), "strict_box_hits": sum(r["strict_box_hit"] for r in results),
               "actual_successes": sum(r["actual_success"] for r in results),
               "box_miss_actual_success": [r["id"] for r in results if not r["strict_box_hit"] and r["actual_success"]]}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
