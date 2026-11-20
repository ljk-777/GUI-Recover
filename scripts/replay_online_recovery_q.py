"""Reproduce a completed online Q checkpoint from its recorded train split on CPU."""

import argparse
import json
from pathlib import Path

from gui_recover.online_recovery_q import OnlineRecoveryQ
from gui_recover.recovery_rollouts import iter_learning_episodes
from gui_recover.sft import file_sha256
from gui_recover.visual_text import VisualTextVerifier


def replay(directory):
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    freeze = json.loads((directory / "before-test.json").read_text())
    bank_path = directory / "text-references.json"
    checkpoint_path = directory / "online-q.json"
    if (file_sha256(bank_path) != summary["bank_sha256"]
            or file_sha256(checkpoint_path) != summary["q_sha256"]
            or freeze["q_sha256"] != summary["q_sha256"]
            or freeze["train_index_sha256"] != summary["index_sha256"]["train"]
            or freeze["test_started"] is not False):
        raise ValueError("Training or checkpoint provenance differs from the recorded freeze")

    verifier = VisualTextVerifier(json.loads(bank_path.read_text()))
    config = protocol["config"]
    learner = OnlineRecoveryQ(verifier, seed=config["seed"], epsilon=config["epsilon"])
    decisions = 0
    for (_, seed, regime), steps in iter_learning_episodes(directory, "train"):
        for step in steps:
            obs = step["observation"]
            action = learner.choose(obs)
            if action != step["action"]:
                raise ValueError(f"Recorded training choice could not be reproduced for {seed} / {regime}")
            learner.observe(obs, action, step["reward"], step["next_observation"],
                            step["terminated"], step["truncated"])
            decisions += 1
    expected = json.loads(checkpoint_path.read_text())
    actual = learner.checkpoint()
    if actual != expected or actual["episodes"] != summary["episodes_by_split"]["train"]:
        raise ValueError("Training replay produced a different checkpoint")
    return {
        "status": "passed",
        "train_episodes": actual["episodes"],
        "replayed_live_decisions": decisions,
        "q_entries": sum(len(values) for values in actual["q"].values()),
        "checkpoint_sha256": summary["q_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = replay(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Checkpoint replay failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
