"""Freeze development selection, then collect and evaluate an untouched test."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from gui_recover.cli import fingerprint
from gui_recover.contextual_repair import RepairGrounder
from gui_recover.policies import ShowUIPolicy
from gui_recover.repair_study import (StudyEncoder, aggregate_episodes, evaluate_detectors, evaluate_queries,
    prime_sources, run_conditionals, run_mixed, select_repair, train_action_phase)
from gui_recover.repair_study_data import collect_test, prepare_development
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "test"), required=True)
    for name in ("config", "reference", "inherited-detectors", "model", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--development")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Commit the clean experiment implementation before running")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Use the explicitly assigned GPU 0 only")
    config = json.loads(Path(args.config).read_text())
    reference, inherited, out = Path(args.reference), Path(args.inherited_detectors), Path(args.out)
    if file_sha256(reference/"manifest.json") != config["reference_manifest_sha256"]:
        raise ValueError("Frozen reference source changed")
    if file_sha256(inherited/"frozen/classifier.npz") != config["inherited_frozen_classifier_sha256"]:
        raise ValueError("Inherited verifier weights changed")
    for backend, sha in config["inherited_detector_selections_sha256"].items():
        if file_sha256(inherited/backend/"selection.json") != sha:
            raise ValueError("Inherited detector threshold metadata changed")
    if file_sha256(Path(args.model)/"pytorch_model.bin") != config["base_weights_sha256"]:
        raise ValueError("Frozen actor weights changed")
    selection = None
    if args.stage == "test":
        if not args.development:
            raise ValueError("Test requires a frozen development selection")
        development = Path(args.development)
        selection = json.loads((development/"selection.json").read_text())
        expected = {"config_sha256": file_sha256(args.config),
            "development_manifest_sha256": file_sha256(development/"data/manifest.json"),
            "validation_predictions_sha256": file_sha256(development/"validation-queries/predictions.json"),
            "validation_episodes_sha256": file_sha256(development/"validation-execution/episodes.json"),
            "action_phase_classifier_sha256": file_sha256(development/"action-phase/classifier.npz"),
            "action_phase_selection_sha256": file_sha256(development/"action-phase/selection.json")}
        if any(selection[k] != value for k, value in expected.items()) or selection["runtime"]["git_commit"] != runtime["git_commit"]:
            raise ValueError("Frozen validation ledger differs")
        recomputed = select_repair(json.loads((development/"validation-queries/predictions.json").read_text()),
            aggregate_episodes(json.loads((development/"validation-execution/episodes.json").read_text())), config)
        if any(selection[k] != value for k, value in recomputed.items()):
            raise ValueError("Selection does not reproduce from validation evidence")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"config.json", config)
    write_json(out/"runtime.json", runtime)
    if args.stage == "development":
        prepare_development(reference, out/"data", config, runtime)
        train_action_phase(reference, out/"action-phase", runtime)
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) != 0:
        raise RuntimeError("Assigned GPU 0 became busy; no model loaded, do not switch GPUs")
    actor = ShowUIPolicy(args.model, max_new_tokens=32)
    grounder = RepairGrounder(actor, out/"inference")
    if args.stage == "development":
        for split in ("train", "validation"):
            prime_sources(grounder, out/"data", split)
        _, development_summary = evaluate_queries(out/"data", "train", grounder, out/"development-queries", config)
        val_predictions, validation_summary = evaluate_queries(out/"data", "validation", grounder, out/"validation-queries", config)
        _, conditional = run_conditionals(out/"data", "validation", grounder, out/"validation-execution", config)
        selection = {**select_repair(val_predictions, conditional, config), "runtime": runtime,
            "config_sha256": file_sha256(args.config),
            "development_manifest_sha256": file_sha256(out/"data/manifest.json"),
            "validation_predictions_sha256": file_sha256(out/"validation-queries/predictions.json"),
            "validation_episodes_sha256": file_sha256(out/"validation-execution/episodes.json"),
            "action_phase_classifier_sha256": file_sha256(out/"action-phase/classifier.npz"),
            "action_phase_selection_sha256": file_sha256(out/"action-phase/selection.json"),
            "new_test_collected": False}
        write_json(out/"selection.json", selection)
        write_json(out/"summary.json", {"development_queries": development_summary,
            "validation_queries": validation_summary, "conditional_validation": conditional,
            "selection": selection, "inference": grounder.counters()})
        print("FROZEN SELECTION", json.dumps(selection), flush=True)
    else:
        selection_sha = file_sha256(development/"selection.json")
        collect_test([reference, *args.exclude], out/"data", config, runtime, grounder, selection_sha)
        _, queries = evaluate_queries(out/"data", "test", grounder, out/"queries", config, selection["selected_mode"])
        encoder = StudyEncoder(actor)
        detectors, verifier = evaluate_detectors(out/"data", inherited, development/"action-phase", encoder, out/"detectors")
        _, conditional = run_conditionals(out/"data", "test", grounder, out/"conditional", config)
        _, mixed = run_mixed(out/"data", grounder, verifier, out/"mixed", config, selection["selected_mode"])
        write_json(out/"summary.json", {"selection_sha256_before_collection": selection_sha,
            "selected_mode": selection["selected_mode"], "queries": queries,
            "detectors": detectors, "conditional": conditional, "mixed": mixed,
            "inference": grounder.counters(), "feature_counters": encoder.frozen.counters(),
            "peak_cuda_allocated_bytes": actor.torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved()})
        print("FINAL", json.dumps({"selection": selection["selected_mode"], "conditional": conditional, "mixed": mixed}), flush=True)


if __name__ == "__main__":
    main()
