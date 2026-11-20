"""Verify timing pairs, native screenshots and unchanged recovery decisions."""

import argparse
from collections import Counter
import json
from pathlib import Path

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.active_text_audit import audit_branch
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_audit import checked_file, read_json, require

from run_tail_probe_latency import POLICIES, compare


def signature(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    return [(step["action"], step["reward"], step["terminated"], step["truncated"],
             step["observation"]["current"]["sha256"], step["next_observation"]["current"]["sha256"],
             step["next_observation"]["selected"], step["next_observation"]["probe_restored"]) for step in rows]


def audit(root):
    protocol, summary, freeze = [read_json(root / name) for name in
                                 ("protocol.json", "summary.json", "before-rollouts.json")]
    plan = read_json(checked_file(root / "preflight.json", summary["plan_sha256"]))
    bank = read_json(checked_file(root / "text-references.json", summary["bank_sha256"]))
    require(freeze["plan_sha256"] == summary["plan_sha256"] and freeze["bank_sha256"] == summary["bank_sha256"]
            and not freeze["outcomes_observed"] and freeze["actor_calls"] == 0, "Plan or references changed")
    rows = [json.loads(line) for line in checked_file(root / "episodes.jsonl", summary["index_sha256"]).read_bytes().splitlines()]
    profile = protocol["config"][protocol["profile"]]
    scenes = plan["selected"]
    require(len(scenes) == profile["layouts"] == len({s["initial_sha256"] for s in scenes}), "Invalid layout registry")
    expected = [(s, regime, policy) for i, s in enumerate(scenes) for j, regime in enumerate(profile["regimes"])
                for policy in (POLICIES[::-1] if (i+j) % 2 else POLICIES)]
    require(len(rows) == len(expected) == summary["episodes"], "Incomplete timing study")
    verifier = VisualTextVerifier(bank)
    counts = Counter()
    seen = {}
    for row, (scene, regime, policy) in zip(rows, expected, strict=True):
        require((row["task"], row["seed"], row["regime"], row["policy"], row["split"])
                == (scene["task"], scene["seed"], regime, policy, "test"), "Timing episode order differs")
        totals, _ = audit_branch(root, row, scene, ActiveTextPolicy(verifier))
        counts.update(totals)
        identity = scene["seed"], regime
        seen.setdefault(identity, {})[policy] = (signature(root / row["trace"]), row)
    matched = sum(p["active_tail"][0] == p["batched_tail"][0] for p in seen.values())
    expected_summary = compare(rows, protocol["config"]["minimum_median_paired_latency_reduction"])
    require(summary["comparison"] == expected_summary, "Pair comparison does not reproduce")
    for pairs in seen.values():
        slow, fast = pairs["active_tail"][1], pairs["batched_tail"][1]
        require(slow["browser_operations"] - fast["browser_operations"] == slow["probes"],
                "Each batched probe must remove exactly one Gym step")
    return {"status": "passed", "layouts": len(scenes), "rollouts": len(rows), "matching_visual_pairs": matched,
            **counts, "comparison": summary["comparison"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.directory.resolve())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Timing audit failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
