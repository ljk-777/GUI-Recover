"""Replay saved actions for evaluator-only input state labels, no model calls.

DOM values are used here for offline auditing, never in policy/guard decisions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import statistics

from PIL import Image

from gui_recover.actions import parse_action
from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.grounding import plan_from_goal
from gui_recover.input_guard import inspect_input
from gui_recover.runner import is_success, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", {"runtime": fingerprint(), "runs": args.runs,
        "model_calls": 0, "labels": "DOM input value read during offline replay only"})
    labels, episodes = [], []
    for source in args.runs:
        source = Path(source)
        source_manifest = json.loads((source / "manifest.json").read_text())
        signal_version = source_manifest["model_config"]["input_guard"]["signal"].rsplit("_", 1)[-1]
        summary = json.loads((source / "summary.json").read_text())
        for result in summary["results"]:
            path = source / f"{result['task']}-{result['seed']}" / "trajectory.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            observations = {e["step"]: e for e in events if e["event"] == "policy_observation"}
            env = MiniwobEnvironment(result["task"], visual_variant=source_manifest["config"].get("visual_variant", "default"))
            try:
                obs, _ = env.reset(result["seed"])
                image_path = directory / f"{source.name}-{result['seed']}-reset.png"
                obs.screenshot.save(image_path)
                sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if sha != events[0]["sha256"]:
                    raise ValueError("Replay initial screenshot differs")
                expected = plan_from_goal(obs.goal)[0].value
                replay_success = False
                previous_screenshot = events[0]["screenshot"]
                for event in events:
                    if event["event"] != "transition":
                        continue
                    action = parse_action(json.dumps(event["action"]))
                    obs, reward, terminated, truncated, info, code = env.step(action)
                    replay_success = is_success(reward, terminated, truncated, info["task_info"])
                    if event["proposed_action"]["action"] == "INPUT":
                        value = env.env.unwrapped.page.locator("#tt").input_value()
                        observation = observations.get(event["step"])
                        signal = (observation or {}).get("signal", {})
                        detector_seconds = None
                        if signal:
                            before = Image.open(path.parent / previous_screenshot).convert("RGB")
                            after = Image.open(path.parent / event["screenshot"]).convert("RGB")
                            start = time.perf_counter()
                            reproduced = inspect_input(before, after, event["proposed_action"]["position"], version=signal_version)
                            detector_seconds = time.perf_counter() - start
                            if json.loads(json.dumps(reproduced)) != signal:
                                raise ValueError("Original screenshot signal could not be reproduced")
                        labels.append({"run": source.name, "task": result["task"], "seed": result["seed"],
                            "step": event["step"], "actual_empty": value == "",
                            "matches_expected": value == expected, "value_length": len(value),
                            "guard_status": signal.get("status"),
                            "offline_detector_seconds": detector_seconds,
                            "decision": (observation or {}).get("decision")})
                    previous_screenshot = event["screenshot"]
                if replay_success != result["success"]:
                    raise ValueError("Replay outcome differs from original")
                episodes.append({"run": source.name, "seed": result["seed"], "success": replay_success,
                                 "reset_sha256": sha})
            finally:
                env.close()
            write_json(directory / "labels.json", labels)
            write_json(directory / "episodes.json", episodes)
    checked = [r for r in labels if r["guard_status"] is not None]
    tp = sum(r["actual_empty"] and r["guard_status"] == "empty" for r in checked)
    fp = sum(not r["actual_empty"] and r["guard_status"] == "empty" for r in checked)
    fn = sum(r["actual_empty"] and r["guard_status"] != "empty" for r in checked)
    totals = {"replayed_episodes": len(episodes), "input_states": len(labels), "guard_checks": len(checked),
              "empty_detection_tp": tp, "empty_detection_fp": fp, "empty_detection_fn": fn,
              "unknown": sum(r["guard_status"] == "unknown" for r in checked),
              "precision": tp / (tp + fp) if tp + fp else None,
              "recall": tp / (tp + fn) if tp + fn else None}
    timings = [r["offline_detector_seconds"] for r in checked]
    totals["offline_detector_mean_seconds"] = statistics.mean(timings) if timings else None
    totals["offline_detector_max_seconds"] = max(timings) if timings else None
    write_json(directory / "summary.json", totals)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
