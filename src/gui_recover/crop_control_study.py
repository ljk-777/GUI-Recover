"""Matched repair execution on the same newly collected natural failures."""

import json
from pathlib import Path

from .contextual_repair import ContextualRepairPolicy
from .repair_study import aggregate_episodes
from .runner import write_json
from .verification_data import load_transitions
from .verification_runner import run_verification_case


def crop_comparison(data, grounder, out, config):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = load_transitions(data, "test")
    if any(r.get(phase) and r[phase]["task_info"].get("error")
           for r in sources for phase in ("immediate", "after_wait")):
        raise RuntimeError("Source MiniWoB execution reports an environment error")
    rows = [r for r in sources if r["status"] == "ok" and not r["immediate"]["effect_achieved"]]
    fraction = config["crop_fraction"]
    methods = {
        "plain": ("plain", None),
        "feedback": ("feedback", None),
        "adaptive_zoom": ("feedback_zoom", None),
        "fixed_zoom": ("feedback_zoom", (0, 0, fraction, fraction)),
        "content_crop": ("feedback_zoom", config["content_crop"]),
    }
    episodes = []
    with (out/"episodes.jsonl").open("x") as log:
        for row in rows:
            for label, (mode, bounds) in methods.items():
                directory = out/"online"/f"{row['id']}-{label}"
                policy = ContextualRepairPolicy(grounder, mode, fraction, directory/"repair-input.png",
                                                fixed_crop=bounds)
                result = run_verification_case(row, "oracle", grounder, None, out, repairer=policy, run_name=label)
                if result["repairs"]:
                    prediction, = (event["prediction"] for event in
                        map(json.loads, (directory/"trajectory.jsonl").read_text().splitlines())
                        if event["event"] == "repair_prediction")
                    left, top, right, bottom = prediction["crop"]
                    width, height = prediction["original_size"]
                    visible = row["teacher_boxes"]["immediate"]
                    result["visible_targets"] = len(visible)
                    result["target_center_in_crop"] = any(
                        left <= (x1+x2)*width/2 < right and top <= (y1+y2)*height/2 < bottom
                        for x1, y1, x2, y2 in visible)
                episodes.append(result)
                log.write(json.dumps(result)+"\n")
                log.flush()
                if (result["reason"] in ("runtime_error", "prefix_mismatch")
                        or result.get("task_info", {}).get("error")):
                    raise RuntimeError(f"{row['id']} {label}: {result['reason']} {result.get('error', '')}")
                print("crop", row["id"], label, result["reason"], flush=True)
    by_case = {r["id"]: {} for r in rows}
    aggregate = aggregate_episodes(episodes)
    for episode in episodes:
        by_case[episode["id"]][episode["strategy"]] = episode["success"]
    for label, counts in aggregate.items():
        subset = [ep for ep in episodes if ep["strategy"] == label and ep["repairs"]]
        counts.update(attempts_with_visible_target=sum(ep["visible_targets"] > 0 for ep in subset),
                      attempts_with_target_center_in_crop=sum(ep["target_center_in_crop"] for ep in subset))
    paired = {
        fixed: {
            "both_succeed": sum(r["adaptive_zoom"] and r[fixed] for r in by_case.values()),
            "adaptive_only": sum(r["adaptive_zoom"] and not r[fixed] for r in by_case.values()),
            "fixed_only": sum(r[fixed] and not r["adaptive_zoom"] for r in by_case.values()),
            "neither_succeeds": sum(not r[fixed] and not r["adaptive_zoom"] for r in by_case.values()),
        } for fixed in ("fixed_zoom", "content_crop")
    }
    summary = {
        "source_layouts": len(sources), "invalid_sources": sum(r["status"] != "ok" for r in sources),
        "natural_failures": len(rows),
        "terminal_natural_failures": sum(r["immediate"]["terminated"] or r["immediate"]["truncated"] for r in rows),
        "all": aggregate, "paired_against_adaptive": paired, "case_outcomes": by_case,
        "by_task": {task: aggregate_episodes([r for r in episodes if r["task"] == task])
                    for task in sorted({r["task"] for r in sources})},
    }
    write_json(out/"summary.json", summary)
    return summary
