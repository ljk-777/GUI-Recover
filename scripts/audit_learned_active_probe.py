"""Reproduce the online probe checkpoint and verify held-out decisions on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from gui_recover.active_text import ActiveTextPolicy
from gui_recover.active_text_audit import audit_branch
from gui_recover.probe_learning import FrozenProbeChoice, ProbeLearner, adjusted_return, skip_after_wait
from gui_recover.recovery_env import Decision
from gui_recover.recovery_rollouts import iter_learning_episodes
from gui_recover.verification_env import DEFERRED_TAIL_REGIME
from gui_recover.visual_text import VisualTextVerifier
from gui_recover.visual_text_audit import checked_file, read_json, require

from run_learned_active_probe import POLICIES, summarize


def audit(root):
    root = root.resolve()
    protocol, freeze, summary = [read_json(root / name) for name in
                                 ("protocol.json", "before-test.json", "summary.json")]
    bank = read_json(checked_file(root / "text-references.json", summary["bank_sha256"]))
    before_rollouts = read_json(root / "before-rollouts.json")
    plan = read_json(checked_file(root / "preflight.json", summary["plan_sha256"]))
    require(before_rollouts["bank_sha256"] == summary["bank_sha256"]
            and before_rollouts["plan_sha256"] == summary["plan_sha256"]
            and not before_rollouts["outcomes_observed"] and before_rollouts["actor_calls"] == 0,
            "Plan or references were altered after preparation")
    checkpoint = read_json(checked_file(root / "probe-q.json", summary["q_sha256"]))
    require(freeze["q_sha256"] == summary["q_sha256"] and freeze["test_started"] is False
            and freeze["train_index_sha256"] == summary["index_sha256"]["train"], "Invalid checkpoint freeze receipt")
    config = protocol["config"]
    profile = config[protocol["profile"]]
    scenes = read_json(root / "splits.json")
    require([{k: v for k, v in row.items() if k != "split"} for row in scenes] == plan["selected"], "Layout registry changed")
    require(len({row["initial_sha256"] for row in scenes}) == len(scenes), "Repeated train or test layout")

    verifier = VisualTextVerifier(bank)
    learner = ProbeLearner(seed=config["seed"], epsilon=config["epsilon"], min_visits=config["min_visits"])
    frozen = FrozenProbeChoice(checkpoint)
    selectors = dict(explore_probe=learner.choose, always_probe=None, skip_after_wait=skip_after_wait, learned_probe=frozen)
    cycle = [(s, g, "explore_probe") for s in scenes if s["split"] == "train" for g in config["train_regimes"]]
    train_queue, rng = [], Random(config["seed"])
    for _ in range(profile["passes"]):
        rng.shuffle(cycle)
        train_queue.extend(cycle)
    test_queue = [(s, g, p) for s in scenes if s["split"] == "test" for g in profile["test_regimes"] for p in POLICIES]
    counts = Counter()
    seen = {"train": set(), "test": set()}
    sequences, waited = {}, {}
    for split, queue in (("train", train_queue), ("test", test_queue)):
        path = checked_file(root / f"{split}.jsonl", summary["index_sha256"][split])
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        require(len(rows) == len(queue) == summary["episodes_by_split"][split], f"Incorrect {split} episode count")
        for sequence, (row, (scene, regime, policy)) in enumerate(zip(rows, queue, strict=True)):
            require((row["task"], row["seed"], row["regime"], row["policy"], row["split"])
                    == (scene["task"], scene["seed"], regime, policy, split), "Collection order differs from the protocol")
            relative = Path("rollouts") / split / f"{sequence:03d}-{scene['seed']}-{regime}-{policy}" / "transitions.jsonl"
            rule = ActiveTextPolicy(verifier, selectors[policy])
            totals, _ = audit_branch(root, row, scene, rule, relative=relative)
            counts.update(totals)
            require(adjusted_return(row) == row["adjusted_return"], "Cost-adjusted return differs")
            steps = [json.loads(line) for line in (root / relative).read_bytes().splitlines()]
            seen[split].update(s["observation"]["current"]["sha256"] for s in steps)
            if split == "train":
                learner.finish_episode(row["adjusted_return"])
            else:
                sequences[row["seed"], regime, policy] = tuple(s["action"] for s in steps)
                if policy == "always_probe":
                    for step in steps:
                        if step["action"] == Decision.WAIT:
                            waited[row["seed"], regime] = step["next_observation"]["current"]["sha256"]
        if split == "train":
            require(learner.checkpoint() == checkpoint, "Replaying train did not produce the frozen table")
            require(sum(1 for _ in iter_learning_episodes(root, "train")) == len(rows), "Version 2 learner export differs")
        else:
            require(summarize(rows, profile["test_regimes"]) == summary["conditions"], "Held-out aggregate differs")
    require(not seen["train"] & seen["test"], "Train/test decision screenshots overlap")
    require(dict(frozen.counts) == summary["learned_routing"], "Frozen routing differs")
    test_layouts = [row["seed"] for row in scenes if row["split"] == "test"]
    comparisons = [(seed, regime) for seed in test_layouts for regime in profile["test_regimes"]]
    return {"status": "passed", "checkpoint_replayed": True, "episodes_by_split": summary["episodes_by_split"],
            **counts, "test_episode_matches_skip_rule": sum(
                sequences[s, r, "learned_probe"] == sequences[s, r, "skip_after_wait"] for s, r in comparisons),
            "test_episodes_per_policy": len(comparisons), "delayed_tail_alias_pairs": sum(
                (s, DEFERRED_TAIL_REGIME) in waited and waited.get((s, "deferred-input")) == waited[s, DEFERRED_TAIL_REGIME]
                for s in test_layouts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.directory), indent=2))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Learned-probe audit failed: {exc}\n")


if __name__ == "__main__":
    main()
