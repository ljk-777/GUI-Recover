"""Predeclared history-prompt interventions and validation-only selection."""

import json
from pathlib import Path
import shutil

from .sft import file_sha256, grounding_query, load_records
from .sft_ablation import AblationGrounder
from .sft_recovery import audit_corrections


STUDY_HEADS = {
    "base_plain": ("base_plain", "legacy"),
    "history_plain": ("history_plain", "legacy"),
    "base_legacy": ("base_history", "legacy"),
    "history_legacy": ("history_history", "legacy"),
    "base_target_last": ("base_history", "target-last"),
    "history_target_last": ("history_history", "target-last"),
    "base_current_target": ("base_history", "current-target"),
    "history_current_target": ("history_history", "current-target"),
    "base_action_only": ("base_history", "action-only"),
    "history_action_only": ("history_history", "action-only"),
}
CANDIDATES = ("history_target_last", "history_current_target", "history_action_only")


def study_query(query, previous, head):
    underlying, profile = STUDY_HEADS[head]
    return grounding_query(query, None if underlying.endswith("plain") else previous,
                           history_profile=profile)


class HistoryStudyGrounder(AblationGrounder):
    def predict(self, picture, query, previous, head):
        underlying, profile = STUDY_HEADS[head]
        return super().predict(picture, query, previous, underlying, history_profile=profile)


def summarize_predictions(rows, heads):
    return {"queries": len(rows), "heads": {h: {
        "hits": sum(r["heads"][h]["hit"] for r in rows),
        "invalid": sum(r["heads"][h]["point"] is None for r in rows),
        "newly_hit_vs_base_plain": sum(r["heads"][h]["hit"] and not r["heads"]["base_plain"]["hit"] for r in rows),
        "degraded_vs_base_plain": sum(not r["heads"][h]["hit"] and r["heads"]["base_plain"]["hit"] for r in rows),
    } for h in heads}}


def study_summary(predictions, heads):
    def group(rows):
        return summarize_predictions(rows, heads)
    return {
        "all": group(predictions),
        "normal": group([r for r in predictions if r["kind"] != "correction"]),
        "normal_submit": group([r for r in predictions if r["kind"] == "normal_submit"]),
        "correction": group([r for r in predictions if r["kind"] == "correction"]),
        "distractor_entry": group([r for r in predictions if r["kind"] == "normal_entry" and r["task"] == "click-button"]),
        "by_task": {t: group([r for r in predictions if r["task"] == t]) for t in sorted({r["task"] for r in predictions})},
    }


def select_history_prompt(validation_predictions):
    if not validation_predictions or any(r["split"] != "validation" for r in validation_predictions):
        raise ValueError("Selection accepts validation predictions only")
    summary = study_summary(validation_predictions, STUDY_HEADS)
    if not summary["normal_submit"]["queries"] or not summary["correction"]["queries"]:
        raise ValueError("Selection requires both normal Submit and actual corrections")
    # Prevent recovery-only optimization. Compare normal/Submit to both plain controls.
    floors = {group: max(summary[group]["heads"][h]["hits"] for h in ("base_plain", "history_plain"))
              for group in ("normal", "normal_submit")}
    eligible = [h for h in CANDIDATES if all(summary[g]["heads"][h]["hits"] >= n for g, n in floors.items())]
    chosen = max(eligible, key=lambda h: (
        summary["correction"]["heads"][h]["hits"], summary["normal"]["heads"][h]["hits"],
        -summary["all"]["heads"][h]["invalid"], -CANDIDATES.index(h))) if eligible else "history_plain"
    return {"selected_head": chosen, "eligible_candidates": eligible, "normal_floors": floors,
            "prompt_restoration_passed_validation": bool(eligible),
            "rule": "normal/Submit >= both plain controls; maximize correction, normal, fewer invalids, fixed candidate order; otherwise history_plain fallback"}


def reuse_development_data(reference, out, config):
    """Copy only train/validation evidence; never copy reference test rows/images."""
    reference, out = Path(reference).resolve(), Path(out)
    source = json.loads((reference / "manifest.json").read_text())
    rows = {}
    paths = set()
    for split in ("train", "validation"):
        if source["config"]["splits"][split] != config["splits"][split]:
            raise ValueError("Reused development scenes differ from frozen config")
        rows[split] = load_records(reference, split)
        paths.add(f"{split}.jsonl")
        paths.update(r["image"] for r in rows[split])
    cohort = [r for r in source["source_cohort"] if r["split"] in rows]
    for item in cohort:
        path = f"traces/{item['task']}-{item['seed']}.json"
        trace = json.loads((reference / path).read_text())
        paths.add(path)
        paths.update(trace[key]["image"] for key in ("initial", "after_action", "current") if key in trace)
    for path in sorted(paths):
        src = (reference / path).resolve()
        dst = (out / path).resolve()
        if not src.is_relative_to(reference) or not dst.is_relative_to(out.resolve()):
            raise ValueError("Reused evidence path escapes dataset")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return rows, {"source_cohort": cohort,
        "filtered": [r for r in source["filtered"] if r["split"] in rows],
        "corrections": {s: audit_corrections(out, r) for s, r in rows.items()},
        "reused_development": {"source_manifest_sha256": file_sha256(reference / "manifest.json"),
            "source_code": source["runtime"]["git_commit"],
            "split_sha256": {s: file_sha256(out / f"{s}.jsonl") for s in rows}}}
