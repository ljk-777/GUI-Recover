"""Verify saved visual-text rollouts on CPU, including interrupted runs."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .recovery_env import Decision, FRAME_SIZE, Phase, SUPPORTED_REGIMES, exact_success
from .sft import file_sha256
from .visual_text import ContentRecoveryPolicy, VisualTextVerifier

from .visual_text_study import POLICIES, presence_wait, summarize


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(path, digest):
    require(file_sha256(path) == digest, f"Checksum mismatch: {path}")
    return path


def load_observation(state, directory, images):
    loaded = dict(state)
    for name in ("before", "current"):
        image = state[name]
        filename = (directory / image["image"]).resolve()
        require(filename.parent == directory, "Screenshot outside its rollout directory")
        if filename not in images:
            with Image.open(checked_file(filename, image["sha256"])) as picture:
                require(picture.mode == "RGB" and picture.size == FRAME_SIZE, f"Invalid screenshot: {filename}")
                images[filename] = (image["sha256"], np.array(picture))
        digest, loaded[name] = images[filename]
        require(digest == image["sha256"], f"Inconsistent screenshot checksum: {filename}")
    return loaded


def audit_episode(root, row, layout, verifier):
    name = f'{row["task"]}-{row["seed"]}-{row["regime"]}-{row["policy"]}'
    relative = Path("rollouts") / name / "transitions.jsonl"
    trace = (root / relative).resolve()
    require(row["trace"] == relative.as_posix() and trace.relative_to(root) == relative, f"Invalid trace path: {name}")
    steps = [json.loads(line) for line in checked_file(trace, row["trace_sha256"]).read_bytes().splitlines()]
    evidence = read_json(checked_file(trace.with_name("visual-verification.json"), row["visual_verification_sha256"]))
    require(1 <= len(steps) <= 3, f"Invalid decision count: {name}")
    require(row["initial_sha256"] == layout["initial_sha256"], f"Unregistered initial screenshot: {name}")
    require(steps[0]["observation"]["current"]["sha256"] == layout["initial_sha256"], f"Reset differs: {name}")
    require(steps[0]["observation"]["phase"] == Phase.SOURCE, f"Invalid initial phase: {name}")

    glyph = ContentRecoveryPolicy(verifier)
    choose = glyph if row["policy"] == "glyph_check" else presence_wait
    images = {}
    decisions = Counter()
    actor_requests = 0
    previous = None
    for number, step in enumerate(steps, 1):
        before, after = step["observation"], step["next_observation"]
        require(step["schema_version"] == 1 and step["step"] == number, f"Invalid step index: {name}/{number}")
        require(previous is None or before == previous, f"Broken observation chain: {name}/{number}")
        require(before["goal"] == after["goal"] == layout["goal"], f"Goal changed: {name}/{number}")
        obs = load_observation(before, trace.parent, images)
        load_observation(after, trace.parent, images)
        decision = choose(obs)
        require((step["action"], step["decision"]) == (decision, decision.name)
                and obs["action_mask"][decision], f"Decision changed: {name}/{number}")
        decisions[decision] += 1
        events = step["evidence"]["browser_events"]
        require(not any(event.get("execution_error") or event["task_info"].get("error") for event in events),
                f"Browser execution error: {name}/{number}")
        last_event = events[-1] if events else None
        success = bool(step["terminated"] and not step["truncated"] and last_event and exact_success(
            last_event["terminated"], last_event["truncated"], last_event["reward"], last_event["task_info"]))
        require(step["reward"] == float(success) and step["evidence"]["success"] is success,
                f"Reward differs from terminal browser evidence: {name}/{number}")
        terminal = number == len(steps)
        require(sum((step["terminated"], step["truncated"])) == int(terminal),
                f"Invalid termination flags: {name}/{number}")
        actor_requests += len(step["evidence"]["predictions"])
        previous = after

    require(previous["phase"] == Phase.CLOSED and not any(previous["action_mask"]), f"Rollout still open: {name}")
    require(json.loads(json.dumps(glyph.evidence)) == evidence, f"Visual evidence changed: {name}")
    result = {
        "decisions": len(steps), "waits": decisions[Decision.WAIT], "repairs": decisions[Decision.RETRY_FIXED_ZOOM],
        "actor_requests": actor_requests, "return": step["reward"], "success": success,
        "outcome": step["evidence"]["outcome"], "terminated": step["terminated"], "truncated": step["truncated"],
        "visual_statuses": [entry["status"] for entry in evidence],
    }
    require(all(row[key] == value for key, value in result.items()), f"Index disagrees with trace: {name}")
    return len(steps), len(images)


def audit(root):
    root = root.resolve()
    protocol = read_json(root / "protocol.json")
    freeze = read_json(root / "before-rollouts.json")
    plan = read_json(checked_file(root / "preflight.json", freeze["plan_sha256"]))
    bank = read_json(checked_file(root / "text-references.json", freeze["bank_sha256"]))
    require(all(item["actor_calls"] == 0 and item["outcomes_observed"] is False for item in (plan, freeze)),
            "Reference inputs were not frozen before rollouts")
    profile = protocol["profile"]
    split = "demo" if profile == "quickstart" else "test"
    layouts = plan["selected"]
    require(len(layouts) == protocol["config"][profile]["layouts"]
            == len({layout["initial_sha256"] for layout in layouts})
            == len({layout["seed"] for layout in layouts}), "Invalid preflight layout registry")
    expected = [(layout, regime, policy) for layout in layouts for regime in SUPPORTED_REGIMES for policy in POLICIES]
    verifier = VisualTextVerifier(bank)

    summary_path = root / "summary.json"
    index = root / "episodes.jsonl"
    complete = summary_path.exists()
    contents = index.read_bytes() if index.exists() else b""
    # Each completed row ends in a newline. A stopped writer may leave a partial final line.
    lines = contents.split(b"\n")
    tail = lines.pop()
    if complete:
        require(not tail and len(lines) == len(expected), "Completed run has an incomplete episode index")
    require(len(lines) <= len(expected), "Episode index contains extra rows")
    rows = []
    verified_steps = verified_screenshots = 0
    for line, (layout, regime, policy) in zip(lines, expected):
        row = json.loads(line)
        require((row["task"], row["seed"], row["regime"], row["policy"], row["split"])
                == (layout["task"], layout["seed"], regime, policy, split), "Episode index differs from preflight order")
        steps, screenshots = audit_episode(root, row, layout, verifier)
        verified_steps += steps
        verified_screenshots += screenshots
        rows.append(row)

    result = {
        "status": "complete" if complete else "incomplete",
        "profile": profile,
        "layouts": len(layouts),
        "expected_episodes": len(expected),
        "verified_episodes": len(rows),
        "verified_paired_conditions": len(rows) // len(POLICIES),
        "verified_decisions": verified_steps,
        "verified_screenshots": verified_screenshots,
        "uncommitted_index_bytes": len(tail),
    }
    if complete:
        summary = read_json(summary_path)
        checked_file(index, summary["index_sha256"])
        expected_summary = {**summarize(rows), "episodes": len(expected), "profile": profile,
                            "plan_sha256": freeze["plan_sha256"], "bank_sha256": freeze["bank_sha256"]}
        require(all(summary[key] == value for key, value in expected_summary.items()), "Summary disagrees with traces")
        result.update(gate=summary["gate"], conditions=summary["conditions"])
    return result
