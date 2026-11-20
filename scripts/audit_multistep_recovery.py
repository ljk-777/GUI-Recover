"""Verify a frozen multistep evaluation and render a browser-free report."""

import argparse
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from gui_recover.multistep_recovery import EXTRA_BUDGET, METHODS, regimes, summarize, validate_plan
from gui_recover.probe_learning import ProbeLearner, adjusted_return
from gui_recover.recovery_env import ProbeDecision
from gui_recover.runner import is_success, write_json
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import checked_file


def read(path):
    return json.loads(path.read_text())


def audit(out):
    out = Path(out).resolve()
    receipt = read(out / "before-rollouts.json")
    plan = read(checked_file(out / "preflight.json", receipt["plan_sha256"]))
    checked_file(out / "text-references.json", receipt["bank_sha256"])
    validate_plan(plan)
    scenes = {(r["task"], r["seed"]): r for r in plan["selected"] if r["split"] in ("test", "transfer")}
    expected = {(task, seed, condition, method) for task, seed in scenes
                for condition in regimes(task) for method in METHODS}
    evaluation = out / "evaluation"
    rows = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    names = set()
    frames = 0
    for row in rows:
        task, seed, regime, method = (row[k] for k in ("task", "seed", "regime", "method"))
        name = task, seed, regime, method
        relative = f"rollouts/{task}-{seed}-{regime}-{method}/record.json"
        if name not in expected or name in names or row["trace"] != relative:
            raise ValueError("Incomplete or invalid matched trial index")
        names.add(name)
        path = checked_file(evaluation / relative, row["trace_sha256"])
        trace = read(path)
        index = {key: value for key, value in row.items() if key not in ("trace", "trace_sha256")}
        if trace != {**index, "trace": "record.json"}:
            raise ValueError(f"Index disagrees with trial {relative}")
        scene = scenes[task, seed]
        if (row["goal"], row["split"], row["frames"]["initial"]["pixel_hash"]) != (
                scene["goal"], scene["split"], scene["initial_pixel_hash"]):
            raise ValueError("Trial reset is inconsistent with the frozen layout")
        for label, frame in row["frames"].items():
            if frame["image"] != f"{label}.png":
                raise ValueError("Unexpected frame name")
            with Image.open(checked_file(path.parent / frame["image"], frame["sha256"])) as picture:
                if pixel_hash(picture) != frame["pixel_hash"]:
                    raise ValueError("Frame pixels do not match the record")
            frames += 1
        events = row["events"]
        terminal = events[-1] if events else {"reward": 0, "terminated": False, "truncated": False, "task_info": {}}
        if (row["reward"], row["terminated"], row["truncated"]) != (
                terminal["reward"], terminal["terminated"], terminal["truncated"]):
            raise ValueError("Summary terminal state differs from recorded event")
        if bool(row["success"]) != is_success(terminal["reward"], terminal["terminated"],
                                               terminal["truncated"], terminal["task_info"]):
            raise ValueError("Exact native reward does not reproduce")
        # There is one scripted first action and no reset in the branch log.
        suffix = next((i for i, event in enumerate(events) if event["event"] == "scripted_continuation"), len(events))
        repair_steps = events[1:suffix]
        if repair_steps and repair_steps[-1]["event"] == "pending_delivered_before_continuation":
            continued_delivery = int(
                row["extra_operations"] != len(repair_steps) + row["focus_probes"])
        else:
            continued_delivery = 0
        if (len(repair_steps) - continued_delivery + row["focus_probes"] != row["extra_operations"]
                or row["extra_operations"] > EXTRA_BUDGET):
            raise ValueError("Recovery step count exceeds or differs from the budget")
        repairs = [e for e in repair_steps if e["event"] in ("retry_click", "idempotent_input_repair")]
        if len(repairs) != row["repairs"] or len(row["repair_evidence"]) != row["repairs"]:
            raise ValueError("Repair evidence count differs")
        if any(item["observation"] not in row["frames"] for item in row["repair_evidence"]):
            raise ValueError("Repair input does not have a saved native frame")
    if names != expected:
        raise ValueError("Missing planned rollout")
    metrics = summarize(rows)
    saved = read(evaluation / "summary.json")
    if metrics != saved["metrics"] or len(rows) != saved["episodes"]:
        raise ValueError("Saved aggregate cannot be reconstructed")
    return {"episodes": len(rows), "layouts": len(scenes), "screenshots": frames,
            "metrics": metrics, "gate": saved["gate"]}, rows


def audit_probe_learning(out):
    out = Path(out)
    root = out / "probe-training"
    receipt = read(root / "before-evaluation.json")
    if receipt["test_started"] is not False:
        raise ValueError("Invalid checkpoint freeze")
    records = checked_file(root / "train.jsonl", receipt["train_index_sha256"])
    checkpoint = read(checked_file(root / "probe-q.json", receipt["probe_q_sha256"]))
    settings = read(out / "protocol.json")["config"]["learning"]
    learner = ProbeLearner(seed=settings["seed"], epsilon=settings["epsilon"], min_visits=settings["minimum_action_visits"])
    plan = read(out / "preflight.json")
    allowed = {(row["task"], row["seed"]) for row in plan["selected"] if row["split"] == "train"}
    queried = 0
    for line in records.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "train" or (row["task"], row["seed"]) not in allowed:
            raise ValueError("Probe-learning trace crossed a split")
        skipped = any(entry["status"] == "skipped_probe" for entry in row["decisions"])
        if row["probes"] or skipped:
            expected = ProbeDecision.PROBE_LAST_CHAR if row["probes"] else ProbeDecision.CONTINUE
            if learner.choose({"waited": bool(row["waits"])}) != expected:
                raise ValueError("Seeded probe exploration differs from its trace")
            queried += 1
        score = adjusted_return({**row, "return": float(row["success"])})
        if score != row["adjusted_return"]:
            raise ValueError("Probe reward or action costs differ")
        learner.finish_episode(score)
    if learner.checkpoint() != checkpoint:
        raise ValueError("Frozen probe values do not reproduce")
    return {"training_episodes": learner.episodes, "sampled_probe_decisions": queried,
            "checkpoint_sha256": receipt["probe_q_sha256"], "checkpoint_reproduced": True}


def html(summary, rows):
    columns = ("Method", "Exact success", "Healthy rewrites", "Natural failures recovered", "Extra operations")
    metrics = summary["metrics"]["all"]
    header = "".join(f"<th>{label}</th>" for label in columns)
    results = "".join(
        f"<tr><th>{escape(method)}</th><td>{m['exact_successes']}/{m['episodes']}</td>"
        f"<td>{m['healthy_repairs']}/{m['healthy_live']}</td>"
        f"<td>{m['natural_recovered']}/{m['natural_failures']}</td><td>{m['extra_operations']}</td></tr>"
        for method, m in metrics.items()
    )
    details = []
    for row in rows:
        directory = "evaluation/" + str(Path(row["trace"]).parent)
        images = " ".join(
            f'<a href="{quote(directory + "/" + frame["image"])}">{escape(label)}</a>'
            for label, frame in row["frames"].items()
        )
        details.append(
            f"<tr><td>{escape(row['task'])}</td><td>{row['seed']}</td><td>{escape(row['regime'])}</td>"
            f"<td>{escape(row['method'])}</td><td>{escape(row['outcome'])}</td><td>{row['extra_operations']}</td>"
            f"<td>{images}</td></tr>"
        )
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>GUI-Recover multistep evaluation</title>
<style>body{font:16px system-ui,sans-serif;margin:2rem;color:#17212a}
table{border-collapse:collapse;margin-block:1rem;width:100%}
td,th{text-align:left;padding:.55rem;border-bottom:1px solid #bdc5cb}a{color:#145287}</style>
<h1>Multi-step recovery evaluation</h1>""" + (
        f"<table><thead><tr>{header}</tr></thead><tbody>{results}</tbody></table>"
        "<h2>Browser rollouts</h2><table><thead><tr><th>Task</th><th>Seed</th><th>Condition</th><th>Method</th>"
        "<th>Outcome</th><th>Extra operations</th><th>Native frames</th></tr></thead>"
        f"<tbody>{''.join(details)}</tbody></table></html>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", action="store_true", help="Write report.html into the audited run")
    parser.add_argument("--verify-training", action="store_true", help="Recompute the frozen probe-learning values")
    args = parser.parse_args()
    summary, rows = audit(args.run)
    if args.verify_training:
        summary["probe_learning_audit"] = audit_probe_learning(args.run)
    if args.report:
        path = args.run / "report.html"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(html(summary, rows))
        write_json(args.run / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
