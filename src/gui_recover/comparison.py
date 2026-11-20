"""Validate matched injection experiments before reporting any recovery result."""

import json
import copy
from pathlib import Path


def _load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = {(r["task"], r["seed"]): r for r in summary["results"]}
    if len(rows) != summary["episodes"]:
        raise ValueError("Duplicate/incomplete episode identifiers")
    config = manifest["config"]
    expected = {(task, seed) for task in config["tasks"] for seed in config["seeds"]}
    if set(rows) != expected:
        raise ValueError("Run did not complete the configured task/seed set")
    return directory, manifest, rows


def compare_runs(clean, faulted, recovered, clean_recovered=None):
    directories = [clean, faulted, recovered]
    if clean_recovered is not None:
        directories.append(clean_recovered)
    runs = [_load(d) for d in directories]
    _, reference, identifiers = runs[0]
    if reference["policy"] not in ("showui", "showui-scripted") or reference["diagnostic_only"]:
        raise ValueError("Recovery comparisons require nonprivileged model runs")
    recovery_mode = "input-guard" if reference["policy"] == "showui-scripted" else "feedback"

    def model_config(manifest):
        config = copy.deepcopy(manifest["model_config"])
        if reference["policy"] == "showui-scripted":
            guard = config.get("input_guard")
            if not isinstance(guard, dict) or guard.get("enabled") != (manifest["config"]["recovery"] == "input-guard"):
                raise ValueError("Input guard configuration does not match experiment arm")
            # This enable switch is the intervention; all other guard parameters
            # and the entire base model/planner configuration must remain identical.
            guard["enabled"] = False
        return config
    shared = ("policy", "tasks", "seeds", "max_steps", "model", "revision", "prompt_profile", "parser", "fault_seed",
              "input_guard_version", "visual_variant", "adapter")
    for directory, manifest, rows in runs:
        if set(rows) != set(identifiers):
            raise ValueError("Mismatched task/seed sets")
        for key in shared:
            if manifest["config"].get(key) != reference["config"].get(key):
                raise ValueError(f"Different comparison configuration: {key}")
        if manifest["runtime"]["git_commit"] != reference["runtime"]["git_commit"] or manifest["runtime"]["git_dirty"]:
            raise ValueError("Comparisons require identical, clean code revisions")
        if model_config(manifest) != model_config(reference):
            raise ValueError("Different model configurations")
        for key in ("packages", "miniwob_commit", "platform"):
            if manifest["runtime"][key] != reference["runtime"][key]:
                raise ValueError(f"Different runtime configuration: {key}")
    expected_modes = [("none", "none"), (runs[1][1]["config"]["fault"], "none"),
                      (runs[1][1]["config"]["fault"], recovery_mode)]
    if expected_modes[1][0] == "none":
        raise ValueError("Faulted run must enable injection")
    if clean_recovered is not None:
        expected_modes.append(("none", recovery_mode))
    for (_, manifest, _), modes in zip(runs, expected_modes):
        if (manifest["config"]["fault"], manifest["config"]["recovery"]) != modes:
            raise ValueError("Unexpected experiment arm")

    def events(directory, key):
        return [json.loads(line) for line in (directory / f"{key[0]}-{key[1]}" / "trajectory.jsonl").read_text().splitlines()]

    pairs = []
    for key in identifiers:
        a, b, c = (run[2][key] for run in runs[:3])
        record = {"task": key[0], "seed": key[1], "clean_success": a["success"],
                  "faulted_success": b["success"], "recovered_success": c["success"]}
        if not b["injected"] or not c["injected"]:
            record["status"] = "not_injected"
        else:
            b_events, c_events = events(runs[1][0], key), events(runs[2][0], key)

            def prefix(items):
                screens = []
                predictions = []
                for e in items:
                    if e["event"] == "diagnostic" and e.get("fault"):
                        # Transition at injected step has already been recorded;
                        # exclude its screenshot to compare the pre-injection state.
                        return screens[:-1], predictions, e["fault"]
                    if "screenshot" in e:
                        screens.append(e["sha256"])
                    if e["event"] == "prediction":
                        predictions.append(e["raw"])
                raise ValueError("Injected result has no fault event")

            bp, cp = prefix(b_events), prefix(c_events)
            record["status"] = "matched" if bp == cp else "mismatched_prefix"
        pairs.append(record)

    matched = [p for p in pairs if p["status"] == "matched"]
    b_failed = [p for p in matched if not p["faulted_success"]]
    b_succeeded = [p for p in matched if p["faulted_success"]]
    rescued = sum(p["recovered_success"] for p in b_failed)
    degraded = sum(not p["recovered_success"] for p in b_succeeded)
    labels = ["clean", "faulted", "recovered", "clean_recovered"]
    summaries = {}
    for label, (_, _, rows) in zip(labels, runs):
        n = len(rows)
        summaries[label] = {"episodes": n, "successes": sum(r["success"] for r in rows.values()),
            "success_rate": sum(r["success"] for r in rows.values()) / n,
            "predictions": sum(r["predictions"] for r in rows.values()),
            "steps": sum(r["steps"] for r in rows.values()),
            "feedbacks": sum(r["recovery_feedbacks"] for r in rows.values()),
            "model_calls": (sum(r["model_calls"] for r in rows.values())
                            if all(r.get("model_calls") is not None for r in rows.values()) else None),
            "input_guard_checks": sum((r.get("input_guard") or {}).get("checks", 0) for r in rows.values()),
            "input_guard_retries": sum((r.get("input_guard") or {}).get("retries", 0) for r in rows.values()),
            "inference_seconds": sum(r["inference_seconds"] for r in rows.values())}
    return {"arms": summaries, "matched_injected_pairs": len(matched),
            "not_injected": sum(p["status"] == "not_injected" for p in pairs),
            "mismatched_prefixes": sum(p["status"] == "mismatched_prefix" for p in pairs),
            "rescued": rescued, "faulted_failures": len(b_failed),
            "rescue_fraction": rescued / len(b_failed) if b_failed else None,
            "degraded": degraded, "faulted_successes": len(b_succeeded),
            "degradation_fraction": degraded / len(b_succeeded) if b_succeeded else None,
            "pairs": pairs}
