"""Audit a price OCR retry comparison without loading Qwen."""

import argparse
import json
import os
from pathlib import Path

from audit_vwa_catalog_price_transfer import totals
from gui_recover.actions import parse_action
from gui_recover.visual_text_audit import read_json, require
from prepare_vwa_catalog_holdout import WHOLE_LINE_COHORT, encoded, fingerprint, price_retry_groups
from run_vwa_catalog_holdout import PROFILE, audit_run, load_cohort
from run_vwa_catalog_read_faults import prediction_metrics


def exact_answers(directory, expected):
    answers = []
    for line in (directory / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "prediction" and event["raw"]:
            action = parse_action(event["raw"])
            if action.action == "ANSWER":
                answers.append(action.value)
    return answers == [f"{expected['city']}, {expected['dollars']} dollars"]


def compare(cohort, digest, protocol_file, runs):
    protocol = read_json(protocol_file)
    index, cases, references = load_cohort(cohort, digest, os.getenv("VWA_HOMEPAGE", "__HOMEPAGE__"))
    series = int(index["sample_key"].rsplit("-v", 1)[1])
    evidence = {str(proof["task_id"]): proof["source"] for proof in read_json(cohort / "source-evidence.json")}
    screening = read_json(cohort / "price-retry-screening.json")
    selection = protocol["selection"]
    whole_line = index["sample_key"] == WHOLE_LINE_COHORT
    require(protocol["cohort"] == index["sample_key"]
            and all(protocol[key] == PROFILE[key] for key in ("seed", "max_predictions", "max_model_calls"))
            and screening["screen_limit"] == selection["maximum_screened_listings"]
            and index["catalog_pages_per_category"] == selection["catalog_pages_per_category"]
            and len(screening["selected"]["trigger"]) == selection["triggers"]
            and len(screening["selected"]["control"]) == selection["controls"]
            and selection["one_x_ocr_used_for_selection"] is whole_line
            and selection["model_inference_used_for_selection"] is False,
            "The screening or inference budget differs from the preregistration")
    if whole_line:
        require(len(screening["selected"]["upright_control"]) == selection["upright_controls"]
                and screening["upright_ocr_used_for_selection"] is True
                and screening["whole_line_ocr_used_for_selection"] is selection["whole_line_ocr_used_for_selection"] is False,
                "Whole-line screening must register both detector reads without using the candidate recognizer")
    preflight = (cohort / "preflight.log").read_text().splitlines()
    require(all(preflight.count(f"{task} native preflight passed") == 1 for task in evidence),
            "Expected native preflight for every selected case")
    if selection.get("exclude_repeated_crop_pixels"):
        crop_exclusions = read_json(protocol_file.with_name(f"vwa-price-crop-exclusions-{series:03d}.json"))
        require(screening["excluded_crop_sha256"] == sorted(crop_exclusions), "The crop exclusions differ from registration")
    if selection.get("exclude_previous_screening_pool"):
        listing_exclusions = set(read_json(protocol_file.with_name(f"vwa-catalog-exclusions-{series:03d}.json")))
        require(index["exclusion_list_sha256"] == fingerprint(encoded(sorted(listing_exclusions)))
                and not listing_exclusions.intersection(screening["ordered_candidates"]),
                "The ordered candidate pool overlaps the registered listing exclusions")

    by_arm, policy_hashes, receipt_matches, frame_matches = {}, {}, {}, {}
    upright_matches = {}
    model = None
    origin = None
    for arm in ("prior", "current"):
        directory = runs / f"visualwebarena-catalog-price-retry-challenge-{arm}-{series:03d}"
        plan = read_json(directory / "run-plan.json")
        if model is None:
            model, origin = plan["model"], plan["listing_origin"]
        require(plan["model"] == model and plan["listing_origin"] == origin and plan["strategies"] == ["evidence-cards"],
                "The arms must share the frozen checkpoint and site")
        files = protocol.get("policy_files", ("answer_evidence.py", "listing_facts.py"))
        policy_hashes[arm] = {name: plan["source_files"][name] for name in files}
        if whole_line:
            require(policy_hashes[arm] == protocol["policy_file_sha256"][arm],
                    "The run does not contain the preregistered policy files")
        audited = audit_run(directory, cases, references, digest)
        require(audited == read_json(directory / "audit.json"), "Saved arm results differ from the trajectories")
        by_arm[arm], frame_matches[arm], receipt_matches[arm] = {}, [], []
        upright_matches[arm] = []
        for task, pair in audited["by_task"].items():
            episode = directory / f"visualwebarena.{task}-evidence-cards"
            baseline = evidence[task]["price_retry_screening"]
            row = by_arm[arm][task] = pair["strategies"]["evidence-cards"]
            if pair["reset_pixels"] == baseline["start_pixels_sha256"]:
                frame_matches[arm].append(task)
            steps = tuple(prediction_metrics(episode))
            receipts = [step["listing_price_check"] for step in steps if "listing_price_check" in step]
            if receipts == [baseline["enlarged_ocr"]]:
                receipt_matches[arm].append(task)
            if "upright_ocr" in baseline:
                retries = [step["listing_price_upright_retry"] for step in steps if "listing_price_upright_retry" in step]
                if retries == [baseline["upright_ocr"]]:
                    upright_matches[arm].append(task)
            require(not row["success"] or exact_answers(episode, evidence[task]["visible"]),
                    "A native success must match the sealed city and exact price")

    groups = {group: sorted(task for task, proof in evidence.items() if proof["price_retry_screening"]["outcome"] == group)
              for group in price_retry_groups(index["sample_key"])}
    scores = {group: {arm: totals({task: rows[task] for task in identifiers}, include_native_retries=True)
                      for arm, rows in by_arm.items()} for group, identifiers in groups.items()}
    modes = protocol.get("retry_receipts", {"current": "listing_price_native_retry"})
    require("current" in modes and set(modes) <= set(by_arm)
            and set(modes.values()) <= {"listing_price_native_retry", "listing_price_upright_retry"},
            "Unknown registered price retry receipt")
    if "listing_price_upright_retry" in modes.values():
        for group_scores in scores.values():
            for score in group_scores.values():
                score.setdefault("upright_price_ocr_retries", 0)
    targets = protocol["acceptance"]
    added = scores["trigger"]["current"]["native_successes"] - scores["trigger"]["prior"]["native_successes"]
    checks = {
        "all_pairs_match_frozen_screening_frames": all(set(matched) == set(evidence) for matched in frame_matches.values()),
        "both_arms_reproduce_the_enlarged_ocr_trigger": (
            all(set(groups["trigger"]) <= set(matched) for matched in receipt_matches.values())
            and all(by_arm[arm][task]["evidence_stages"].get(receipt) == 1
                    for arm, receipt in modes.items() for task in groups["trigger"])
        ),
        "minimum_additional_trigger_successes": added >= targets["minimum_additional_trigger_successes"],
        "zero_wrong_submissions": all(arm["wrong_submissions"] == 0 for group in scores.values() for arm in group.values()),
        "all_readable_control_successes_preserved": all(not by_arm["prior"][task]["success"] or by_arm["current"][task]["success"]
                                                      for task in groups["control"]),
        "control_model_calls_do_not_increase": scores["control"]["current"]["model_calls"] <= scores["control"]["prior"]["model_calls"],
        "no_native_scale_retry_on_readable_controls": all(
            not by_arm[arm][task]["evidence_stages"].get(receipt)
            for arm, receipt in modes.items() for task in groups["control"]
        ),
    }
    if "minimum_current_trigger_successes" in targets:
        checks["minimum_current_trigger_successes"] = scores["trigger"]["current"]["native_successes"] >= targets["minimum_current_trigger_successes"]
    if "all_prior_trigger_successes_preserved" in targets:
        checks["all_prior_trigger_successes_preserved"] = all(
            not by_arm["prior"][task]["success"] or by_arm["current"][task]["success"] for task in groups["trigger"]
        )
    if whole_line:
        require(set(modes) == set(by_arm) and set(modes.values()) == {"listing_price_upright_retry"},
                "Both whole-line comparison policies must use the upright detection retry")
        detector_retries = set(groups["trigger"] + groups["upright_control"])
        line_retries = {arm: {task: row["evidence_stages"].get("listing_price_line_retry", 0)
                              for task, row in tasks.items()} for arm, tasks in by_arm.items()}
        controls = ("control", "upright_control")
        checks.update(
            both_arms_match_all_registered_ocr_receipts=(
                all(set(matched) == set(evidence) for matched in receipt_matches.values())
                and all(set(matched) == detector_retries for matched in upright_matches.values())),
            one_line_read_per_trigger_only=(
                all(count == 0 for count in line_retries["prior"].values())
                and all(count == (task in groups["trigger"]) for task, count in line_retries["current"].items())),
            all_upright_control_successes_preserved=all(
                not by_arm["prior"][task]["success"] or by_arm["current"][task]["success"]
                for task in groups["upright_control"]),
            minimum_prior_control_successes_per_group=all(
                scores[group]["prior"]["native_successes"] >= targets["minimum_prior_control_successes_per_group"]
                for group in controls),
            upright_control_model_calls_do_not_increase=(
                scores["upright_control"]["current"]["model_calls"] <= scores["upright_control"]["prior"]["model_calls"]),
            control_local_ocr_reads_do_not_increase=all(
                scores[group]["current"]["localized_ocr_reads"] <= scores[group]["prior"]["localized_ocr_reads"]
                for group in controls),
            trigger_model_calls_do_not_increase=(
                scores["trigger"]["current"]["model_calls"] <= scores["trigger"]["prior"]["model_calls"]),
        )
        for group_scores in scores.values():
            for score in group_scores.values():
                score.setdefault("whole_line_ocr_retries", 0)
    require(set(checks) == set(targets), "The acceptance checks differ from the preregistration")
    report = {"cohort_sha256": digest, "model_calls_during_screening": screening["model_calls"],
              "listings_screened": len(screening["attempts"]), "groups": groups, "frame_matches": frame_matches,
              "enlarged_ocr_receipt_matches": receipt_matches, "policy_file_sha256": policy_hashes,
              "acceptance_checks": checks, "target_met": all(checks.values()), "scores": scores, "by_task": by_arm}
    if whole_line:
        report["upright_ocr_receipt_matches"] = upright_matches
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.cohort, args.sha256, args.protocol, args.runs), indent=2))


if __name__ == "__main__":
    main()
