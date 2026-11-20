"""Record and compare actual Gymnasium recovery transitions."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .recovery_env import Decision, FRAME_SIZE, Phase, transition_view
from .runner import write_json
from .sft import file_sha256
from .verification import appearance_verdict


POLICIES = ("continue", "plain", "fixed_zoom", "visual_fixed")


def baseline_decision(observation, policy):
    """A frozen baseline sees only the same observation as a learned policy."""
    if policy not in POLICIES:
        raise ValueError("Unknown fixed policy")
    if observation["phase"] != Phase.REVIEW or policy == "continue":
        return Decision.CONTINUE
    if policy == "plain":
        return Decision.RETRY_PLAIN
    if policy == "fixed_zoom":
        return Decision.RETRY_FIXED_ZOOM
    status = appearance_verdict(transition_view(observation))["status"]
    if status == "unknown" and observation["action_mask"][Decision.WAIT]:
        return Decision.WAIT
    if status == "not_achieved":
        return Decision.RETRY_FIXED_ZOOM
    return Decision.CONTINUE


def _save_state(directory, index, obs, before):
    image = directory/f"{index:03d}.png"
    Image.fromarray(obs["current"]).save(image)
    current = {"image": image.name, "sha256": file_sha256(image)}
    if before is None:
        before = current
    return {**{name: obs[name] for name in ("goal", "target", "proposed_action", "phase",
                                           "waited", "remaining_decisions")},
            "action_mask": obs["action_mask"].tolist(), "before": before, "current": current}


def record_rollout(env, scene, policy, directory, *, choose=None, on_transition=None):
    """Write a sequence of observation/action/reward/next-observation records."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    observation, _ = env.reset(seed=scene["seed"])
    state = _save_state(directory, 0, observation, None)
    if state["current"]["sha256"] != scene["initial_sha256"]:
        raise RuntimeError("Reset screenshot differs from the registered layout")
    rollout = directory/"transitions.jsonl"
    with rollout.open("x") as stream:
        for step in range(1, env.max_decisions+1):
            action = baseline_decision(observation, policy) if choose is None else Decision(choose(observation))
            if not observation["action_mask"][action]:
                raise ValueError("The rollout policy selected a masked decision")
            previous = observation
            observation, reward, terminated, truncated, info = env.step(action)
            following = _save_state(directory, step, observation, state["before"])
            stream.write(json.dumps({"schema_version": 1, "step": step, "observation": state,
                "action": int(action), "decision": action.name, "reward": reward, "terminated": terminated,
                "truncated": truncated, "next_observation": following, "evidence": info})+"\n")
            stream.flush()
            if on_transition is not None:
                on_transition(previous, action, reward, observation, terminated, truncated)
            state = following
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Rollout escaped the decision budget")
    result = {"task": scene["task"], "seed": scene["seed"], "split": scene["split"], "regime": env.regime,
        "policy": policy, "success": info["success"], "return": reward,
        "terminated": terminated, "truncated": truncated, "outcome": info["outcome"],
        "decisions": info["decisions"], "actor_requests": info["actor_requests"],
        "waits": info["waits"], "repairs": info["repairs"],
        "initial_sha256": scene["initial_sha256"], "trace_sha256": file_sha256(rollout)}
    write_json(directory/"episode.json", result)
    if result["outcome"] == "browser_error":
        raise RuntimeError("Browser execution error retained in rollout")
    return result


def summarize_rollouts(episodes):
    held_out = [r for r in episodes if r["split"] == "test"]
    metrics = {}
    for regime in ("none", "drop-input"):
        selected = [r for r in held_out if r["regime"] == regime]
        by_layout = {}
        for row in selected:
            pair = by_layout.setdefault((row["task"], row["seed"]), {})
            if row["policy"] in pair:
                raise ValueError("Duplicate test episode")
            pair[row["policy"]] = row
        if any(set(pair) != set(POLICIES) for pair in by_layout.values()):
            raise ValueError("Unpaired test episode")
        metrics[regime] = {}
        for p in POLICIES:
            rows = [pair[p] for pair in by_layout.values()]
            metrics[regime][p] = {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
                "gains": sum(pair[p]["success"] and not pair["continue"]["success"] for pair in by_layout.values()),
                "losses": sum(not pair[p]["success"] and pair["continue"]["success"] for pair in by_layout.values()),
                "decisions": sum(r["decisions"] for r in rows), "repairs": sum(r["repairs"] for r in rows),
                "waits": sum(r["waits"] for r in rows), "actor_requests":sum(r["actor_requests"] for r in rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows))}
    primary = "visual_fixed"
    clean, injected = [metrics[regime][primary] for regime in ("none", "drop-input")]
    return {"test": metrics, "episodes_by_split": dict(Counter(r["split"] for r in episodes)),
            "effect_gate": {"primary": primary, "clean_preserved": clean["episodes"] > 0 and clean["losses"] == 0,
                            "clean_gain": clean["gains"] > 0, "injected_gain": injected["gains"] > 0,
                            "injected_preserved": injected["episodes"] > 0 and injected["losses"] == 0}}


def iter_learning_episodes(directory, split="train"):
    """Yield paired-branch identity and audited inputs without evaluator evidence."""
    if split not in ("train", "validation"):
        raise ValueError("The learner reader accepts train and validation only")
    root = Path(directory).resolve()
    split_root = (root/"rollouts"/split).resolve()
    index = root/f"{split}.jsonl"
    manifest = json.loads((root/"summary.json").read_text())
    if file_sha256(index) != manifest["index_sha256"][split]:
        raise ValueError("Split index checksum changed")

    def load_state(state, trace):
        result = dict(state)
        for key in ("before", "current"):
            image = state[key]
            path = (trace.parent/image["image"]).resolve()
            if path.parent != trace.parent or file_sha256(path) != image["sha256"]:
                raise ValueError("Observation screenshot path or checksum invalid")
            with Image.open(path) as picture:
                if picture.mode != "RGB" or picture.size != FRAME_SIZE:
                    raise ValueError("Invalid observation screenshot")
                result[key] = np.array(picture)
        result["action_mask"] = np.asarray(state["action_mask"], dtype=np.int8)
        return result

    with index.open() as stream:
        for line in stream:
            episode = json.loads(line)
            trace = (root/episode["trace"]).resolve()
            if episode["split"] != split or not trace.is_relative_to(split_root) or not split_root.is_relative_to(root):
                raise ValueError("Trace is outside the requested split")
            if file_sha256(trace) != episode["trace_sha256"]:
                raise ValueError("Trace checksum changed")
            transitions = []
            for item in map(json.loads, trace.read_text().splitlines()):
                if item["schema_version"] != 1:
                    raise ValueError("Unsupported transition schema")
                transitions.append({**{key: item[key] for key in ("action", "reward", "terminated", "truncated")},
                    **{key: load_state(item[key], trace) for key in ("observation", "next_observation")}})
            yield (episode["task"], episode["seed"], episode["regime"]), transitions


def iter_learning_transitions(directory, split="train"):
    """Yield learner inputs without branch identity or evaluator evidence."""
    for _, transitions in iter_learning_episodes(directory, split):
        yield from transitions
