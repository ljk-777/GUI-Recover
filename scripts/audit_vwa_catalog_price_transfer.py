"""Recheck the frozen price-and-city transfer comparison from browser traces."""

import argparse
import json
import os
from pathlib import Path

from gui_recover.actions import parse_action
from gui_recover.qwen_grounding import read_reply
from prepare_vwa_catalog_holdout import encoded, fingerprint
from run_vwa_catalog_holdout import PROFILE, audit_run, load_cohort
from run_vwa_catalog_read_faults import LOCAL_PRICE_DROPOUT, audit_fault_run, load_study, prediction_metrics


def read_json(path):
    return json.loads(path.read_bytes())


def exact_answers(directory, expected):
    answers = []
    for line in (directory / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "prediction" and event["raw"]:
            action = parse_action(event["raw"])
            if action.action == "ANSWER":
                answers.append(action.value)
    return answers == [f"{expected['city']}, {expected['dollars']} dollars"]


def totals(rows, *, include_native_retries=False):
    scores = {
        "native_successes": sum(row["success"] for row in rows.values()),
        "wrong_submissions": sum(row["answers_submitted"] > 0 and not row["success"] for row in rows.values()),
        "abstentions": sum(row["reason"] == "evidence_abstained" for row in rows.values()),
        "model_calls": sum(row["model_calls"] for row in rows.values()),
        "localized_ocr_reads": sum(row["evidence_stages"].get(key, 0) for row in rows.values()
                                   for key in ("listing_price_check", "listing_price_ocr")),
    }
    if include_native_retries:
        retries = sum(row["evidence_stages"].get("listing_price_native_retry", 0) for row in rows.values())
        scores["native_scale_ocr_retries"] = retries
        scores["localized_ocr_reads"] += retries
    for stage, key in (("listing_price_upright_retry", "upright_price_ocr_retries"),
                       ("listing_price_line_retry", "whole_line_ocr_retries")):
        retries = sum(row["evidence_stages"].get(stage, 0) for row in rows.values())
        if retries:
            scores[key] = retries
            scores["localized_ocr_reads"] += retries
    return scores


def comparable_injection(path):
    receipt = read_json(path)
    if "reply" in receipt:
        for field in ("original_reply", "injected_reply"):
            receipt["reply"][field] = read_reply(receipt["reply"][field])
    return receipt


def compare(cohort, digest, protocol_path, fault_path, runs):
    protocol = read_json(protocol_path)
    registered = read_json(fault_path)
    cohort_protocol, cases, references = load_cohort(cohort, digest, os.getenv("VWA_HOMEPAGE", "__HOMEPAGE__"))
    identifiers = {str(case["task_id"]) for case in cases}
    if (protocol["cohort"] != cohort_protocol["sample_key"] or set(protocol["faults"]) != identifiers
            or registered != {"cohort_sha256": digest, "cases": protocol["faults"]}
            or any(protocol[key] != PROFILE[key] for key in ("seed", "max_predictions", "max_model_calls"))):
        raise ValueError("The cohort, fault plan, or execution budget differs from the preregistration")
    series = int(cohort_protocol["sample_key"].rsplit("-v", 1)[1])
    clean_runs = {arm: runs / f"visualwebarena-catalog-price-transfer-clean-{arm}-{series:03d}" for arm in ("prior", "current")}
    fault_runs = {arm: runs / f"visualwebarena-catalog-price-transfer-fault-{arm}-{series:03d}" for arm in clean_runs}
    plan, fault_cases, _ = load_study(cohort, digest, fault_path, clean_runs["prior"], allow_unexecuted=True)
    dropout = set(protocol["faults"].values()) == {LOCAL_PRICE_DROPOUT}
    if dropout:
        excluded = set(read_json(protocol_path.with_name(f"vwa-catalog-exclusions-{series:03d}.json")))
        sealed = read_json(cohort / "source-evidence.json")
        paths = {row["listing_path"] for row in sealed}
        amounts = {row["source"]["visible"]["dollars"] for row in sealed}
        if (cohort_protocol["exclusion_list_sha256"] != fingerprint(encoded(sorted(excluded)))
                or not paths.isdisjoint(excluded) or len(amounts) != len(cases)
                or cohort_protocol["catalog_pages_per_category"] != protocol["selection"]["catalog_pages_per_category"]):
            raise ValueError("The dropout cohort differs from the registered catalog sample")
    clean_reports, fault_reports, policy_hashes = {}, {}, {}
    for arm in clean_runs:
        clean_plan = read_json(clean_runs[arm] / "run-plan.json")
        fault_plan = read_json(fault_runs[arm] / "run-plan.json")
        files = [Path(path).name for path in protocol["policy_files"]]
        policy_hashes[arm] = {name: clean_plan["source_files"][name] for name in files}
        if (clean_plan["strategies"] != ["evidence-cards"]
                or any(clean_plan[key] != plan[key] or fault_plan[key] != plan[key] for key in ("model", "listing_origin"))
                or any(fault_plan["source_files"][name] != digest for name, digest in policy_hashes[arm].items())):
            raise ValueError("The checkpoint, origin, or policy changed within the comparison")
        if dropout and policy_hashes[arm] != protocol["policy_file_sha256"][arm]:
            raise ValueError("The dropout run does not use the registered policy revision")
        clean_reports[arm] = audit_run(clean_runs[arm], cases, references, digest)
        if clean_reports[arm] != read_json(clean_runs[arm] / "audit.json"):
            raise ValueError(f"The {arm} clean audit differs from the episode traces")
        fault_reports[arm] = audit_fault_run(fault_runs[arm], plan, fault_cases)
    clean = {arm: {task: row["strategies"]["evidence-cards"] for task, row in report["by_task"].items()}
             for arm, report in clean_reports.items()}
    fault = {arm: report["by_task"] for arm, report in fault_reports.items()}
    paired_clean = sorted(task for task in identifiers
                          if clean_reports["prior"]["by_task"][task]["reset_pixels"]
                          == clean_reports["current"]["by_task"][task]["reset_pixels"])
    paired_fault = []
    for task in sorted(fault["prior"].keys() & fault["current"].keys()):
        prior, current = (fault[arm][task] for arm in fault)
        if prior["fault_delivery"] == current["fault_delivery"] == "complete":
            receipts = [comparable_injection(fault_runs[arm] / f"visualwebarena.{task}/injected-fault.json")
                        for arm in fault]
            if receipts[0] == receipts[1]:
                paired_fault.append(task)
    include_retries = "native_scale_ocr_retries" in protocol["report"]
    clean_scores = {arm: totals(rows, include_native_retries=include_retries) for arm, rows in clean.items()}
    fault_scores = {arm: totals(rows, include_native_retries=include_retries) for arm, rows in fault.items()}
    preflight = (cohort / "preflight.log").read_text().splitlines()
    clean_losses = sorted(task for task in identifiers if clean["prior"][task]["success"] and not clean["current"][task]["success"])
    native_check = "all_registered_tasks_pass_native_preflight" if dropout else "all_six_tasks_pass_native_preflight"
    checks = {
        native_check: all(preflight.count(f"{task} native preflight passed") == 1 for task in identifiers),
        "all_clean_and_fault_pairs_audited_with_identical_start_frames": set(paired_clean) == set(paired_fault) == identifiers,
        "all_prior_clean_successes_preserved": not clean_losses,
        "clean_wrong_submissions_do_not_increase": clean_scores["current"]["wrong_submissions"] <= clean_scores["prior"]["wrong_submissions"],
        "more_native_successes_under_faults": fault_scores["current"]["native_successes"] > fault_scores["prior"]["native_successes"],
    }
    if dropout:
        expected = {str(row["task_id"]): row["source"]["visible"] for row in read_json(cohort / "source-evidence.json")}
        replies = {arm: {task: next(step["provider_output"] for step in prediction_metrics(
                   fault_runs[arm] / f"visualwebarena.{task}") if "listing_price_check" in step)
                   for task in rows} for arm, rows in fault.items()}
        checks.update(
            zero_wrong_submissions=all(
                score["wrong_submissions"] == 0 for group in (clean_scores, fault_scores) for score in group.values()),
            every_native_success_has_the_exact_answer=all(
                not row["success"] or exact_answers(directory / f"visualwebarena.{task}{suffix}", expected[task])
                for group, dirs, suffix in ((clean, clean_runs, "-evidence-cards"), (fault, fault_runs, ""))
                for arm, directory in dirs.items() for task, row in group[arm].items()),
            minimum_current_fault_successes=fault_scores["current"]["native_successes"] >= protocol["minimum_current_fault_successes"],
            original_fault_model_replies_match=replies["prior"] == replies["current"],
            fault_model_calls_do_not_increase=fault_scores["current"]["model_calls"] <= fault_scores["prior"]["model_calls"],
            whole_line_read_only_under_dropout=(
                all(not row["evidence_stages"].get("listing_price_line_retry") for group in clean.values() for row in group.values())
                and all(not row["evidence_stages"].get("listing_price_line_retry") for row in fault["prior"].values())
                and all(row["evidence_stages"].get("listing_price_line_retry") == 1 for row in fault["current"].values())),
        )
    else:
        checks["fewer_wrong_submissions_under_faults"] = fault_scores["current"]["wrong_submissions"] < fault_scores["prior"]["wrong_submissions"]
    checks["clean_model_calls_do_not_increase"] = clean_scores["current"]["model_calls"] <= clean_scores["prior"]["model_calls"]
    if "no_current_wrong_submissions_under_faults" in protocol["acceptance"]:
        checks["no_current_wrong_submissions_under_faults"] = fault_scores["current"]["wrong_submissions"] == 0
    if set(checks) != set(protocol["acceptance"]) or not all(value is True for value in protocol["acceptance"].values()):
        raise ValueError("Acceptance checks differ from the preregistration")
    return {
        "cohort_sha256": digest, "registered_tasks": sorted(identifiers), "acceptance_checks": checks,
        "target_met": all(checks.values()), "paired_clean_tasks": paired_clean, "paired_fault_tasks": paired_fault,
        "clean_losses": clean_losses, "policy_file_sha256": policy_hashes,
        "clean": clean_scores, "fault": fault_scores, "by_task": {"clean": clean, "fault": fault},
        "unexecuted_registered_cases": {arm: outcome["unexecuted"] for arm, outcome in fault_reports.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--fault-plan", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.cohort, args.sha256, args.protocol, args.fault_plan, args.runs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
