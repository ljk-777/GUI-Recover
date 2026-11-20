"""Benchmark three bounded recovery controllers and train the text-probe chooser."""

import argparse
import json
import os
from pathlib import Path
from random import Random

from gui_recover.contextual_repair import RepairGrounder
from gui_recover.multistep_recovery import (
    EXTRA_BUDGET, FORM, METHODS, Recovery, paired_rollouts, register, run_trial, summarize, validate_plan,
)
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.portable_labels import PortableCheckboxStateVerifier
from gui_recover.probe_learning import FrozenProbeChoice, ProbeLearner, adjusted_return
from gui_recover.qwen_grounding import QWEN_REVISION, QwenGroundingPolicy
from gui_recover.recovery_env import ProbeDecision
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.visual_text import VisualTextVerifier, render_text_references
from gui_recover.verification_env import task_request


def prior_hashes(paths):
    from PIL import Image
    from gui_recover.verification import pixel_hash

    return {
        pixel_hash(frame)
        for path in paths
        for name in ("before.png", "000.png", "initial.png")
        for file in path.rglob(name)
        for frame in (Image.open(file),)
    }


def prepare_bank(out, plan):
    values = [task_request(FORM, r["goal"])[2] for r in plan["selected"] if r["task"] == FORM]
    if not values:
        raise ValueError("The multistep protocol requires form references")
    # The standalone Chromium reference renderer must own its sync event loop.
    import subprocess
    import sys

    request = json.dumps(values)
    code = ("import json,sys; from gui_recover.visual_text import render_text_references; "
            "print(json.dumps(render_text_references(json.load(sys.stdin))))")
    report = subprocess.run([sys.executable, "-c", code], input=request, text=True, capture_output=True, check=True)
    bank = json.loads(report.stdout)
    write_json(out / "text-references.json", bank)


def create_actor(args, config):
    if args.actor == "showui":
        if file_sha256(args.model / "pytorch_model.bin") != config["showui_weights_sha256"]:
            raise ValueError("ShowUI weights differ from the protocol")
        return ShowUIPolicy(str(args.model), revision=MODEL_REVISION, max_new_tokens=32)
    actor = QwenGroundingPolicy(args.model, revision=QWEN_REVISION)
    weights = sorted(args.model.glob("*.safetensors"))
    if len(weights) != 2:
        raise ValueError("The pinned Qwen checkpoint requires two weight shards")
    actor.config["weights_sha256"] = {p.name: file_sha256(p) for p in weights}
    return actor


def learn_probes(out, scenes, actor, text, checkbox, settings):
    out.mkdir()
    eligible = [r for r in scenes if r["task"] == FORM and r["split"] == "train"]
    cases = [(s, r) for s in eligible for r in ("healthy", "deferred", "trailing", "deferred_tail")]
    learner = ProbeLearner(seed=settings["seed"], epsilon=settings["epsilon"],
                           min_visits=settings["minimum_action_visits"])
    random = Random(settings["seed"])
    recovery = Recovery(actor, text, checkbox, choose_probe=lambda waited:
                        learner.choose({"waited": waited}) == ProbeDecision.PROBE_LAST_CHAR)

    def score(row):
        row["adjusted_return"] = adjusted_return({**row, "return": float(row["success"])})
        return row["adjusted_return"]

    with (out / "train.jsonl").open("x") as stream:
        number = 0
        for _ in range(settings["passes"]):
            random.shuffle(cases)
            for scene, regime in cases:
                trace = out / f"{number:03d}-{scene['seed']}-{regime}"
                row = run_trial(scene, regime, "visual", actor, recovery, trace)
                learner.finish_episode(score(row))
                row.update(trace=str(trace.relative_to(out) / "record.json"),
                           trace_sha256=file_sha256(trace / "record.json"))
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                number += 1
    checkpoint = learner.checkpoint()
    write_json(out / "probe-q.json", checkpoint)
    write_json(out / "before-evaluation.json", {"test_started": False, "train_index_sha256": file_sha256(out / "train.jsonl"),
                                               "probe_q_sha256": file_sha256(out / "probe-q.json")})
    return checkpoint


def compare_probes(out, scenes, actor, text, checkbox, checkpoint, rules):
    frozen = FrozenProbeChoice(checkpoint)
    recovery = Recovery(actor, text, checkbox, choose_probe=lambda waited:
                        frozen({"waited": waited}) == ProbeDecision.PROBE_LAST_CHAR)
    conditions = ("healthy", "deferred", "trailing", "deferred_tail")
    candidates = paired_rollouts([s for s in scenes if s["task"] == FORM], actor, recovery, out,
                                 methods=["visual"], allowed_regimes=conditions)
    original = [r for r in rules if r["method"] == "visual" and r["task"] == FORM and r["regime"] in conditions]
    pairs = [{**row, "method": "learned_probe"} for row in candidates] + original
    summary = summarize(pairs, ("visual", "learned_probe"))
    challenge = [r for r in candidates if r["regime"] == "deferred_tail"]
    verdicts = {
        "preserved_rule_successes": all(a["success"] or not b["success"] for a, b in zip(
            sorted(candidates, key=lambda r: (r["seed"], r["regime"])),
            sorted(original, key=lambda r: (r["seed"], r["regime"])), strict=True)),
        "delayed_tail_challenge_passed": bool(challenge) and all(r["success"] for r in challenge),
        "fewer_probes": sum(r["probes"] for r in candidates) < sum(r["probes"] for r in original),
    }
    result = {"summary": summary, "gate": verdicts, "by_regime": {condition: {
        method: {"exact_successes": sum(r["success"] for r in pairs if r["method"] == method and r["regime"] == condition),
                 "episodes": sum(r["method"] == method and r["regime"] == condition for r in pairs)}
        for method in ("visual", "learned_probe")
    } for condition in conditions}}
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "model", "font", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--actor", choices=("showui", "qwen"), default="showui")
    parser.add_argument("--prior-evidence", action="append", default=[], type=Path)
    parser.add_argument("--transfer-from", type=Path)
    parser.add_argument("--layouts-per-task", type=int, default=2)
    parser.add_argument("--skip-probe-learning", action="store_true")
    args = parser.parse_args()
    if len(args.prior_evidence) and not all(p.is_dir() for p in args.prior_evidence):
        parser.error("Prior evidence roots must exist")
    if args.layouts_per_task < 1:
        parser.error("Expected at least one transfer layout per task")
    config = json.loads(args.config.read_text())
    if config["minimum_extra_operations_budget"] != EXTRA_BUDGET:
        raise ValueError("Protocol and runtime operation budgets differ")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Expose only the assigned GPU 0")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    if args.transfer_from:
        source = args.transfer_from
        plan = json.loads((source / "preflight.json").read_text())
        receipt = json.loads((source / "before-rollouts.json").read_text())
        if receipt["plan_sha256"] != file_sha256(source / "preflight.json") or receipt["bank_sha256"] != file_sha256(
                source / "text-references.json"):
            raise ValueError("Source layout or reference freeze has changed")
        from collections import Counter
        counts = Counter()
        chosen = []
        for row in plan["selected"]:
            if row["split"] in ("test", "transfer") and counts[row["task"]] < args.layouts_per_task:
                chosen.append(row)
                counts[row["task"]] += 1
        plan["selected"] = chosen
        (out / "text-references.json").write_bytes((source / "text-references.json").read_bytes())
    else:
        groups = [{**g, "seeds": range(g["start"], g["start"] + g["reserve"])}
                  for g in config["profiles"][args.profile]]
        known = prior_hashes(args.prior_evidence)
        plan = register(groups, known)
        prepare_bank(out, plan)
    validate_plan(plan)
    write_json(out / "preflight.json", plan)
    write_json(out / "before-rollouts.json", {"plan_sha256": file_sha256(out / "preflight.json"),
        "bank_sha256": file_sha256(out / "text-references.json"), "actor_calls": 0, "outcomes_observed": False})
    print("registered", len(plan["selected"]), "layouts", flush=True)

    text = VisualTextVerifier(json.loads((out / "text-references.json").read_text()))
    checkbox = PortableCheckboxStateVerifier(args.font)
    raw_actor = create_actor(args, config)
    actor = RepairGrounder(raw_actor, out / "inference")
    write_json(out / "protocol.json", {"config": config, "source_config_sha256": file_sha256(args.config),
        "profile": args.profile, "model": raw_actor.config, "font_sha256": file_sha256(args.font),
        "comparison_scope": "first-action recovery with a shared scripted continuation",
        "transfer_from": bool(args.transfer_from)})
    recovery = Recovery(actor, text, checkbox)
    scenes = plan["selected"]
    if not args.transfer_from:
        development = [s for s in scenes if s["split"] in ("train", "validation")]
        dev = paired_rollouts(development, actor, recovery, out / "development", methods=("visual",))
        write_json(out / "development" / "summary.json", summarize(dev, ("visual",)))
    checkpoint = None
    if args.actor == "showui" and not args.transfer_from and not args.skip_probe_learning:
        checkpoint = learn_probes(out / "probe-training", scenes, actor, text, checkbox, config["learning"])
    heldout = [s for s in scenes if s["split"] in ("test", "transfer")]
    rows = paired_rollouts(heldout, actor, recovery, out / "evaluation")
    metrics = summarize(rows)
    per_split = {split: summarize([r for r in rows if r["split"] == split]) for split in ("test", "transfer")}
    totals = metrics["all"]
    active = totals["visual"]
    gate = {"healthy_repair_fraction_passed": bool(active["healthy_live"]) and
            active["healthy_repairs"] / active["healthy_live"] <= config["maximum_healthy_repair_fraction"],
            "natural_recovery_gain_over_best_fixed": active["natural_recovered"] > max(
                totals[name]["natural_recovered"] for name in METHODS[:2])}
    early = {"metrics": metrics, "by_split": per_split, "gate": gate, "episodes": len(rows)}
    write_json(out / "evaluation" / "summary.json", early)
    learning = compare_probes(out / "probe-evaluation", heldout, actor, text, checkbox, checkpoint, rows) if checkpoint else None
    write_json(out / "summary.json", {**early, "learning": learning, "inference": actor.counters(),
        "peak_cuda_reserved_bytes": raw_actor.torch.cuda.max_memory_reserved()})
    print("completed", out.name, flush=True)


if __name__ == "__main__":
    main()
