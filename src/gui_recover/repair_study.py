"""Fixed repair comparisons, conservative validation selection and diagnostics."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .actions import parse_action
from .contextual_repair import ContextualRepairPolicy, REPAIR_MODES
from .grounding import point_hits_boxes
from .runner import write_json
from .sft import file_sha256
from .verification import TransitionView, appearance_verdict, fit_classifier, probabilities
from .verification_backbone import FrozenFeatures
from .verification_data import load_transitions, observation_rows, view_from_record
from .verification_runner import run_verification_case
from .verification_training import LearnedVerifier, choose_threshold, probability_verdicts, routing_metrics


def probe_view(root, row, probe):
    root = Path(root)
    return TransitionView(Image.open(root/row["before"]["image"]).convert("RGB"),
        Image.open(root/row[probe["phase"]]["image"]).convert("RGB"),
        parse_action(json.dumps(row["proposed_action"])), row["goal"], probe["target"], probe["phase"])


def prime_sources(grounder, data, split):
    for row in load_transitions(data, split):
        if row["status"] == "ok":
            grounder.prime(Image.open(Path(data)/row["before"]["image"]).convert("RGB"),
                row["target"], row["raw"], {"source_id": row["id"], "source_trace_sha256": row["trace_sha256"]})


def evaluate_queries(data, split, grounder, out, config, selected=None):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {r["id"]: r for r in load_transitions(data, split)}
    probes = json.loads((data/f"{split}-probes.json").read_text())
    predictions = []
    for probe in probes:
        modes = REPAIR_MODES if split != "test" or probe["kind"] == "natural_error" else tuple(dict.fromkeys(("plain", selected)))
        view = probe_view(data, rows[probe["source_id"]], probe)
        heads = {}
        for mode in modes:
            value = grounder.repair(view, mode, config["crop_fraction"])
            heads[mode] = {**value, "hit": point_hits_boxes(value["point"], probe["target_boxes"])}
        predictions.append({**probe, "heads": heads})
        write_json(out/"predictions.json", predictions)
        print("queried", split, probe["id"], flush=True)
    summary = summarize_queries(predictions)
    write_json(out/"summary.json", summary)
    return predictions, summary


def summarize_queries(predictions):
    def group(rows):
        modes = sorted({h for r in rows for h in r["heads"]})
        return {"queries": len(rows), "heads": {h: {
            "evaluated": sum(h in r["heads"] for r in rows),
            "hits": sum(r["heads"][h]["hit"] for r in rows if h in r["heads"]),
            "invalid": sum(r["heads"][h]["point"] is None for r in rows if h in r["heads"]),
            "newly_hit_vs_plain": sum(r["heads"][h]["hit"] and not r["heads"]["plain"]["hit"] for r in rows if h in r["heads"]),
            "degraded_vs_plain": sum(not r["heads"][h]["hit"] and r["heads"]["plain"]["hit"] for r in rows if h in r["heads"]),
        } for h in modes}}
    return {"all": group(predictions),
            **{k: group([r for r in predictions if r["kind"] == k]) for k in ("healthy_recheck", "normal_submit", "natural_error")},
            "by_task": {task: group([r for r in predictions if r["task"] == task]) for task in sorted({r["task"] for r in predictions})}}


def aggregate_episodes(rows):
    result = {}
    for strategy in sorted({r["strategy"] for r in rows}):
        group = [r for r in rows if r["strategy"] == strategy]
        result[strategy] = {"episodes": len(group), "successes": sum(r["success"] for r in group),
            **{k: sum(r.get(k, 0) for r in group) for k in ("repairs", "waits", "logical_actor_calls", "new_actor_generations", "environment_steps", "verifier_calls")},
            "runtime_errors": sum(r["reason"] == "runtime_error" for r in group),
            "prefix_mismatches": sum(r["reason"] == "prefix_mismatch" for r in group)}
    return result


def run_conditionals(data, split, grounder, out, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [r for r in load_transitions(data, split) if r["status"] == "ok" and not r["immediate"]["effect_achieved"]]
    episodes = []
    write_json(out/"episodes.json", episodes)
    comparisons = [("oracle", mode, "oracle_"+mode) for mode in REPAIR_MODES]+[("wait_retry", "plain", "wait_plain")]
    for row in rows:
        for trigger, mode, label in comparisons:
            path = out/"online"/f"{row['id']}-{label}"/"repair-input.png"
            policy = ContextualRepairPolicy(grounder, mode, config["crop_fraction"], path)
            result = run_verification_case(row, trigger, grounder, None, out, repairer=policy, run_name=label)
            episodes.append(result)
            write_json(out/"episodes.json", episodes)
            print("conditional", split, row["id"], label, result["reason"], flush=True)
    summary = aggregate_episodes(episodes)
    write_json(out/"scope.json", {"conditional_privileged_error_cohort": True,
        "source_ids": [r["id"] for r in rows], "all_natural_errors_retained": True,
        "trigger_oracle_is_not_a_visual_score": True})
    write_json(out/"summary.json", summary)
    if any(r["reason"] in ("runtime_error", "prefix_mismatch") for r in episodes):
        raise ValueError("Conditional infrastructure failure retained")
    return episodes, summary


def select_repair(predictions, conditional, config):
    if not predictions or any(r["split"] != "validation" for r in predictions):
        raise ValueError("Repair selection accepts validation predictions only")
    summary = summarize_queries(predictions)
    if any(summary[k]["queries"] == 0 for k in ("healthy_recheck", "normal_submit", "natural_error")):
        raise ValueError("Validation requires healthy, Submit and natural-error controls")
    floors = {k: summary[k]["heads"]["plain"]["hits"] for k in ("healthy_recheck", "normal_submit")}
    base = conditional["oracle_plain"]["successes"]
    eligible = [mode for mode in config["candidate_order"] if
        all(summary[k]["heads"][mode]["hits"] >= floors[k] for k in floors)
        and summary["all"]["heads"][mode]["invalid"] <= summary["all"]["heads"]["plain"]["invalid"]
        and conditional["oracle_"+mode]["successes"] > base]
    selected = max(eligible, key=lambda mode: (conditional["oracle_"+mode]["successes"],
        summary["healthy_recheck"]["heads"][mode]["hits"], -config["candidate_order"].index(mode))) if eligible else "plain"
    return {"selected_mode": selected, "eligible_modes": eligible, "normal_floors": floors,
        "validation_repair_improved": bool(eligible), "baseline_conditional_successes": base,
        "rule": "healthy recheck and Submit hits >= plain; invalids <= plain; strictly more actual natural-error recoveries; fixed ties; otherwise plain fallback"}


def train_action_phase(reference, out, runtime):
    reference, out = Path(reference), Path(out)
    manifest = json.loads((reference/"manifest.json").read_text())
    profile = manifest["config"]["classifier"]
    out.mkdir(parents=True, exist_ok=False)
    views = {s: json.loads((reference/f"{s}-views.json").read_text()) for s in ("train", "validation")}
    arrays = {}
    for split in views:
        if file_sha256(reference/f"{split}-pixels.npz") != manifest["features_sha256"][split]["pixels"]:
            raise ValueError("Reference detector features changed")
        with np.load(reference/f"{split}-pixels.npz", allow_pickle=False) as a:
            if a["ids"].tolist() != [v["transition_id"]+":"+v["phase"] for v in views[split]]:
                raise ValueError("Detector feature IDs differ")
            arrays[split] = a["features"][:, -8:].astype(np.float64)
    fit = [i for i, row in enumerate(views["train"]) if row["fit_eligible"]]
    counts = Counter(views["train"][i]["scene_id"] for i in fit)
    weights = [1/counts[views["train"][i]["scene_id"]] for i in fit]
    labels = [views["train"][i]["label"] for i in fit]
    records = load_transitions(reference, "validation")
    best, trials = None, []
    for l2 in profile["l2"]:
        model = fit_classifier(arrays["train"][fit], labels, weights, l2=l2, rank=profile["rank"], steps=profile["steps"])
        chosen, thresholds = choose_threshold(records, views["validation"], probabilities(model, arrays["validation"]), profile)
        r = chosen["routing"]
        score = (r["injected_failure_recall"]["rate"] or 0, -r["healthy_false_trigger"]["rate"],
                 -(r["pending_premature"]["rate"] or 0), l2, chosen["threshold"])
        trials.append({"l2": l2, "chosen": chosen, "thresholds": thresholds})
        if best is None or score > best[0]:
            best = score, model, l2, chosen
    _, model, l2, chosen = best
    np.savez_compressed(out/"classifier.npz", **model)
    selection = {"runtime": runtime, "backend": "action_phase", "l2": l2, "threshold": chosen["threshold"],
        "classifier_sha256": file_sha256(out/"classifier.npz"),
        "reference_manifest_sha256": file_sha256(reference/"manifest.json"),
        "fit_views": len(fit), "fit_scene_groups": len(counts), "validation_routing": chosen["routing"],
        "features": "action type one-hot, proposed x/y, value length, own observation phase; no image or goal features"}
    write_json(out/"selection.json", selection)
    write_json(out/"trials.json", trials)
    return selection


class StudyEncoder:
    def __init__(self, actor):
        self.frozen = FrozenFeatures(actor)

    def encode(self, view, backend):
        if backend == "action_phase":
            from .verification import pixel_features
            return pixel_features(view)[-8:].astype(np.float32)
        return self.frozen.encode(view, backend)


def evaluate_detectors(data, inherited, action_phase, encoder, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records = load_transitions(data, "test")
    views, collisions = observation_rows(data, records)
    by_id = {r["id"]: r for r in records}
    verifiers = {b: LearnedVerifier(Path(inherited)/b, encoder) for b in ("pixels", "frozen")}
    verifiers["action_phase"] = LearnedVerifier(action_phase, encoder)
    verdicts = {b: {} for b in (*verifiers, "appearance")}
    predictions = []
    for v in views:
        view = view_from_record(data, by_id[v["transition_id"]], v["phase"])
        values = {b: model.verify(view) for b, model in verifiers.items()}
        values["appearance"] = appearance_verdict(view)
        predictions.append({**v, "predictions": values})
        for b, value in values.items():
            verdicts[b][(v["transition_id"], v["phase"])] = value
    summary = {b: routing_metrics(records, values) for b, values in verdicts.items()}
    write_json(out/"predictions.json", predictions)
    write_json(out/"collisions.json", collisions)
    write_json(out/"summary.json", summary)
    return summary, verifiers["frozen"]


def run_mixed(data, grounder, verifier, out, config, selected):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = load_transitions(data, "test")
    fixed = []
    for task in sorted({r["task"] for r in sources}):
        fixed.extend(sorted([r for r in sources if r["task"] == task], key=lambda r: r["seed"])[:config["online_scenes_per_task"]])
    comparisons = [("none", "plain", "none"), ("wait_retry", "plain", "wait_plain"),
                   ("verifier", "plain", "verifier_plain"), ("oracle", selected, "oracle_selected")]
    if selected != "plain":
        comparisons.append(("verifier", selected, "verifier_selected"))
    episodes = []
    write_json(out/"episodes.json", episodes)
    for row in fixed:
        for trigger, mode, label in comparisons:
            path = out/"online"/f"{row['id']}-{label}"/"repair-input.png"
            policy = ContextualRepairPolicy(grounder, mode, config["crop_fraction"], path)
            result = run_verification_case(row, trigger, grounder, verifier, out, repairer=policy, run_name=label)
            episodes.append(result)
            write_json(out/"episodes.json", episodes)
            print("mixed", row["id"], label, result["reason"], flush=True)
    summary = {"all": aggregate_episodes(episodes),
        "natural_errors": aggregate_episodes([r for r in episodes if next(s for s in sources if s["id"] == r["id"]).get("immediate") and not next(s for s in sources if s["id"] == r["id"])["immediate"]["effect_achieved"]]),
        "fixed_source_ids": [r["id"] for r in fixed], "selection_mode": selected}
    write_json(out/"summary.json", summary)
    if any(r["reason"] in ("runtime_error", "prefix_mismatch") for r in episodes):
        raise ValueError("Mixed-run infrastructure failure retained")
    return episodes, summary
