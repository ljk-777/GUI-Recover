"""Replay recorded visual decisions and verify the bounded tail probe on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.recovery_env import Decision, ProbeDecision, exact_success
from gui_recover.recovery_rollouts import baseline_decision
from gui_recover.visual_text import ContentRecoveryPolicy, VisualTextVerifier
from gui_recover.visual_text_audit import checked_file, load_observation, read_json, require

from run_active_text_verification import POLICIES, summarize


def audit_branch(root, row, scene, choose, *, relative=None):
    if relative is None:
        relative = Path("rollouts") / f"{row['task']}-{row['seed']}-{row['regime']}-{row['policy']}" / "transitions.jsonl"
    trace = (root / relative).resolve()
    require(row["trace"] == relative.as_posix() and trace.relative_to(root) == relative, "Unexpected rollout path")
    name = trace.parent.name
    steps = [json.loads(line) for line in checked_file(trace, row["trace_sha256"]).read_bytes().splitlines()]
    require(1 <= len(steps) <= 4, f"Invalid decision count: {name}")
    first = steps[0]["observation"]["current"]["sha256"]
    require(first == scene["initial_sha256"] == row["initial_sha256"], f"Reset screenshot differs: {name}")
    images, decisions = {}, Counter()
    previous = None
    requests = operations = restored_probes = 0
    for index, step in enumerate(steps, 1):
        before, after = step["observation"], step["next_observation"]
        require(step["schema_version"] == 2 and step["step"] == index, f"Invalid transition version or order: {name}")
        require(previous is None or before == previous, f"Observation chain differs: {name}")
        require(before["goal"] == after["goal"] == scene["goal"], f"Goal changed: {name}")
        obs = load_observation(before, trace.parent, images)
        load_observation(after, trace.parent, images)
        for state in (before, after):
            selection = state["selected"]
            require(bool(selection) == bool(state["probed"]), f"Inconsistent selection state: {name}")
            if selection:
                loaded = load_observation({"before": selection, "current": selection}, trace.parent, images)["current"]
                if state is before:
                    obs["selected"] = loaded
        decision = choose(obs) if choose else baseline_decision(obs, "fixed_zoom")
        require((step["action"], step["decision"]) == (decision, decision.name)
                and before["action_mask"][decision], f"Decision differs: {name}/{index}")
        events = step["evidence"]["browser_events"]
        require(not any(e["execution_error"] or e["task_info"].get("error") for e in events), f"Browser error: {name}")
        if decision == ProbeDecision.PROBE_LAST_CHAR:
            require([e["executed_code"] for e in events] ==
                    ["keyboard_press('Shift+ArrowLeft')", "keyboard_press('ArrowRight')"], f"Unexpected probe keys: {name}")
            restored = before["current"]["sha256"] == after["current"]["sha256"]
            require(bool(after["probe_restored"]) == restored and after["probed"], f"Probe restore differs: {name}")
            require(not after["action_mask"][ProbeDecision.PROBE_LAST_CHAR]
                    and not after["action_mask"][Decision.WAIT], f"Probe was not bounded: {name}")
            restored_probes += int(restored)
        last = events[-1] if events else None
        success = bool(step["terminated"] and not step["truncated"] and last and exact_success(
            last["terminated"], last["truncated"], last["reward"], last["task_info"]))
        require(step["reward"] == float(success) and step["evidence"]["success"] is success, f"Reward differs: {name}")
        require(sum((step["terminated"], step["truncated"])) == int(index == len(steps)), f"Episode ending differs: {name}")
        operations += len(events)
        requests += len(step["evidence"]["predictions"])
        require(requests == step["evidence"]["actor_requests"], f"Actor request count differs: {name}")
        previous = after
        decisions[decision] += 1
    evidence = read_json(checked_file(trace.with_name("visual-verification.json"), row["visual_verification_sha256"]))
    require(json.loads(json.dumps(choose.evidence if choose else [])) == evidence, f"Visual evidence differs: {name}")
    values = {
        "success": success, "return": step["reward"], "decisions": len(steps),
        "waits": decisions[Decision.WAIT], "repairs": decisions[Decision.RETRY_FIXED_ZOOM],
        "probes": decisions[ProbeDecision.PROBE_LAST_CHAR], "probe_seconds": step["evidence"]["probe_seconds"],
        "actor_requests": requests, "browser_operations": operations,
        "selection_statuses": [item["selection"]["status"] for item in evidence if item.get("selection")],
    }
    require(all(row[key] == value for key, value in values.items()) and values["probes"] <= 1,
            f"Episode index differs: {name}")
    return {
        "decisions": len(steps), "browser_operations": operations, "screenshots": len(images),
        "restored_probes": restored_probes,
    }, steps[0]["next_observation"]


def audit(root):
    root = root.resolve()
    protocol, freeze, summary = (read_json(root / name) for name in ("protocol.json", "before-rollouts.json", "summary.json"))
    plan = read_json(checked_file(root / "preflight.json", summary["plan_sha256"]))
    bank = read_json(checked_file(root / "text-references.json", summary["bank_sha256"]))
    require(freeze["plan_sha256"] == summary["plan_sha256"] and freeze["bank_sha256"] == summary["bank_sha256"],
            "Prepared reference hashes differ")
    require(all(record["actor_calls"] == 0 and record["outcomes_observed"] is False for record in (plan, freeze)),
            "References must precede inference and outcomes")
    profile = protocol["config"][protocol["profile"]]
    scenes, regimes = plan["selected"], profile["regimes"]
    require(len(scenes) == profile["layouts"] == len({row["initial_sha256"] for row in scenes}),
            "Layout registry differs from the profile")
    prior = {row["initial_sha256"] for row in read_json(root / "development-layouts.json")}
    require(not prior & {row["initial_sha256"] for row in scenes}, "Selected layout overlaps the engineering probe")
    index = checked_file(root / "episodes.jsonl", summary["index_sha256"])
    rows = [json.loads(line) for line in index.read_bytes().splitlines()]
    expected = [(scene, regime, policy) for scene in scenes for regime in regimes for policy in POLICIES]
    require(len(rows) == len(expected) == summary["episodes"], "Recorded branch count differs")
    verifier = VisualTextVerifier(bank)
    counts = Counter()
    firsts = {}
    for row, (scene, regime, policy) in zip(rows, expected, strict=True):
        require((row["task"], row["seed"], row["regime"], row["policy"], row["split"])
                == (scene["task"], scene["seed"], regime, policy, "test"), "Run index order differs from preflight")
        choose = None
        if policy == "glyph_check":
            choose = ContentRecoveryPolicy(verifier)
        elif policy == "active_tail":
            choose = ActiveTextPolicy(verifier)
        totals, first = audit_branch(root, row, scene, choose)
        counts.update(totals)
        starting_point = (first["proposed_action"], first["current"]["sha256"])
        key = row["seed"], regime
        require(firsts.setdefault(key, starting_point) == starting_point, "Paired source screenshot or prediction differs")
    require(summary["conditions"] == summarize(rows, regimes), "Paired aggregate differs")
    return {"status": "passed", "episodes": len(rows), "layouts": len(scenes), **counts,
            "normal_trailing_alias_pairs": sum(firsts[s["seed"], "none"][1] == firsts[s["seed"], "trailing-space-input"][1]
                                               for s in scenes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.directory)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Active-probe audit failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
