"""Run or audit the paired, independently sampled Classifieds holdout."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from audit_visualwebarena_live import audit_episode
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.listing_cards import REFERENCE_PHOTO_CLEAR_DISTANCE
from gui_recover.qwen_grounding import QwenGroundingPolicy
from gui_recover.runner import render_replay, run_episode, write_json
from gui_recover.screen_text import RapidScreenReader
from gui_recover.verification import pixel_hash
from gui_recover.vwa_inputs import CATALOG_SERIES_IDS, LocalReferenceArchive
from prepare_vwa_catalog_holdout import (
    CATALOG_POOL_PAGES, CITY_WORD_COUNTS, DETAIL_PRICE_DIGITS, FAMILIES, OWN_INDEX, PRICE_RETRY_COHORTS,
    catalog_case_families, catalog_families, encoded, fingerprint,
)
from run_visualwebarena_live import VisualWebArenaEnvironment, exact_vwa_success, holdout_profile, probe_instance


STRATEGIES = ("answer-check", "evidence-cards")
SCOPE = "independently sampled Classifieds tasks"
PROFILE = {
    "scope": SCOPE, "families": list(FAMILIES),
    "seed": 0, "max_predictions": 12, "max_model_calls": 8, "strategies": list(STRATEGIES),
    "acceptance": {"minimum_successes": 2, "more_successes_than_control": True,
                   "fewer_wrong_submissions_than_control": True},
}


def registered_url(value, *, search_allowed):
    if not isinstance(value, str) or not value.startswith("__CLASSIFIEDS__/"):
        return False
    path = value.removeprefix("__CLASSIFIEDS__")
    if path == "/" or OWN_INDEX.fullmatch(path):
        return True
    parts = urlsplit(path)
    query = parse_qs(parts.query)
    return (search_allowed and parts.path == "/index.php" and not (parts.netloc or parts.fragment)
            and set(query) == {"page", "sShowAs", "sPattern"} and query["page"] == ["search"]
            and query["sShowAs"] == ["gallery"] and len(query["sPattern"]) == 1)


def validate_reference_photo_evidence(proof):
    reference = proof["source"]["reference_photo"]
    digest = reference["source_sha256"]
    comparisons = reference["comparisons"]
    paths = [card["item"] for card in proof["source"]["cards"]]
    target = proof["listing_path"]
    if (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or len(paths) != len(set(paths)) or target not in paths or len(paths) < 2
            or [row["item"] for row in comparisons] != [path for path in paths if path != target]
            or any(type(row["distance"]) not in (int, float)
                   or not REFERENCE_PHOTO_CLEAR_DISTANCE <= row["distance"] <= 1
                   or type(row["source_sha256"]) is not str
                   or re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is None
                   or row["source_sha256"] == digest for row in comparisons)):
        raise ValueError("Reference photo screening does not cover distinct photos across the result set")


def validate_price_retry_screening(directory, protocol, evidence):
    raw = (directory / "price-retry-screening.json").read_bytes()
    if fingerprint(raw) != protocol["price_retry_screening_sha256"]:
        raise ValueError("The price retry screening receipt changed")
    report = json.loads(raw)
    ordered = report["ordered_candidates"]
    attempts = report["attempts"]
    paths = [record["listing_path"] for record in attempts]
    if (report["model_calls"] != 0 or len(set(ordered)) != len(ordered) or len(ordered) > report["screen_limit"]
            or paths != ordered[:len(paths)]):
        raise ValueError("The OCR-only search differs from its ordered candidate pool")
    count = len(evidence) // 2
    groups = {outcome: [attempt for attempt in attempts if attempt["outcome"] == outcome][:count]
              for outcome in ("trigger", "control")}
    chosen = [record for group in groups.values() for record in group]
    if len(chosen) != len(evidence):
        raise ValueError("The price retry cohort requires balanced OCR triggers and controls")
    if report["selected"] != {outcome: [record["listing_path"] for record in rows] for outcome, rows in groups.items()}:
        raise ValueError("The screening groups differ from the frozen candidate order")
    if "excluded_crop_sha256" in report:
        hashes = [record["enlarged_ocr"]["pixels_sha256"] for record in chosen]
        if len(set(hashes)) != len(hashes) or set(hashes).intersection(report["excluded_crop_sha256"]):
            raise ValueError("The fresh cohort contains an already screened price crop")
    for proof, record in zip(evidence, chosen, strict=True):
        frame = directory / "screening-frames" / f"{proof['task_id']}.png"
        with Image.open(frame) as image:
            image_sha = pixel_hash(image)
        if (proof["listing_path"] != record["listing_path"] or proof["source"]["price_retry_screening"] != record
                or image_sha != record["start_pixels_sha256"]):
            raise ValueError("The selected price retry case differs from its frozen OCR screen")


def load_cohort(directory, digest, homepage):
    raw = (directory / "cohort.json").read_bytes()
    if fingerprint(raw) != digest:
        raise ValueError("Cohort digest differs from its registration")
    protocol = json.loads(raw)
    sample_key = protocol["sample_key"]
    ids = CATALOG_SERIES_IDS.get(sample_key)
    expected_profile = {**PROFILE, "families": list(catalog_families(sample_key))}
    if ids is None or protocol.get("task_ids") != list(ids) or any(protocol.get(key) != value for key, value in expected_profile.items()):
        raise ValueError("Unexpected independent evaluation profile")
    if protocol.get("catalog_pages_per_category") != CATALOG_POOL_PAGES.get(protocol["sample_key"]):
        raise ValueError("Unexpected catalog sampling depth")
    minimum_photo_distance = protocol.get("minimum_reference_photo_distance")
    if ((ids[0] >= 10301 or minimum_photo_distance is not None)
            and minimum_photo_distance != REFERENCE_PHOTO_CLEAR_DISTANCE):
        raise ValueError("Unexpected reference photo screening threshold")
    cases = json.loads((directory / "tasks.json").read_bytes())
    evidence = json.loads((directory / "source-evidence.json").read_bytes())
    if (fingerprint(encoded(cases)) != protocol["tasks_sha256"]
            or fingerprint(encoded(evidence)) != protocol["evidence_sha256"]):
        raise ValueError("Independent task definitions or source evidence changed")
    if tuple(case["task_id"] for case in cases) != ids or tuple(row["task_id"] for row in evidence) != ids:
        raise ValueError("Expected the preregistered case IDs")
    if [row["family"] for row in evidence] != catalog_case_families(sample_key):
        raise ValueError("Unexpected distribution of preregistered task families")
    city_word_counts = CITY_WORD_COUNTS.get(sample_key)
    if city_word_counts and (
        protocol.get("detail_city_word_counts_capped_at_two") != list(city_word_counts)
        or tuple(min(len(row["source"]["visible"]["city"].split()), 2)
                 for row in evidence[:len(city_word_counts)]) != city_word_counts
    ):
        raise ValueError("Expected the registered mix of single-word and multiword city names")
    price_digits = DETAIL_PRICE_DIGITS.get(sample_key)
    if price_digits and (protocol.get("detail_price_digits") != list(price_digits)
                         or tuple(len(str(row["source"]["visible"]["dollars"]))
                                  for row in evidence[:len(price_digits)]) != price_digits):
        raise ValueError("Expected the registered mix of two- and three-digit prices")
    if sample_key in PRICE_RETRY_COHORTS:
        validate_price_retry_screening(directory, protocol, evidence)
    references = LocalReferenceArchive(directory, protocol["reference_files_sha256"], homepage=homepage)
    used_images = set()
    for case, proof in zip(cases, evidence, strict=True):
        evaluation = case["eval"]
        image = case["image"]
        expected = evaluation.get("reference_answers", {})
        required_types = ["url_match", "string_match"] if proof["family"] == FAMILIES[0] else ["string_match"]
        expected_image = (f"catalog-references/{case['task_id']}.png" if proof["family"] in FAMILIES[1:3]
                          else None)
        expected_start = ("__CLASSIFIEDS__" + proof["listing_path"] if proof["family"] == FAMILIES[0]
                          else "__CLASSIFIEDS__" + proof["source"]["search_url"] if proof["family"] == FAMILIES[1]
                          else "__CLASSIFIEDS__/")
        expected_reference_url = expected_start if proof["family"] == FAMILIES[0] else ""
        expected_count = 1 if expected_image else 2
        if (case["sites"] != ["classifieds"] or case["require_reset"] is not False or image != expected_image
                or not isinstance(case["intent"], str) or not 1 <= len(case["intent"]) <= 220
                or not registered_url(case["start_url"], search_allowed=True)
                or case["start_url"] != expected_start or evaluation["reference_url"] != expected_reference_url
                or evaluation.get("url_note") != "EXACT" or evaluation["eval_types"] != required_types
                or set(expected) != {"must_include"} or not isinstance(expected["must_include"], list)
                or len(expected["must_include"]) != expected_count
                or any(not isinstance(value, str) or not 1 <= len(value) <= 64 or " |OR| " in value
                       for value in expected["must_include"])):
            raise ValueError("Unexpected independent case scope or native rules")
        if image:
            if minimum_photo_distance is not None:
                validate_reference_photo_evidence(proof)
            used_images.add(image)
    if used_images != set(references.images):
        raise ValueError("Reference archive does not match the registered questions")
    return protocol, cases, references


def browser_case(case, references, *, cards):
    return VisualWebArenaEnvironment(case["task_id"], enable_card_view=cards, enable_browser_context=True,
                                     reference_archive=references, start_sites=["classifieds"], catalog_case=case)


def preflight(cases, references):
    """Check each task in BrowserGym without showing a solution to the model."""
    from gui_recover.actions import Action

    probe_instance()
    for case in cases:
        env = browser_case(case, references, cards=False)
        try:
            observation, _ = env.reset(0)
            start = case["start_url"].replace("__CLASSIFIEDS__", os.environ["VWA_CLASSIFIEDS"].rstrip("/"))
            if observation.goal != case["intent"] or len(observation.tabs) != 1 or observation.tabs[0][1] != start:
                raise ValueError("Native task did not expose exactly the registered question and start page")
            expected_pixels = []
            if case["image"]:
                with references(case["image"]) as reference:
                    expected_pixels.append(pixel_hash(reference.convert("RGB")))
            if [pixel_hash(image) for image in observation.reference_images] != expected_pixels:
                raise ValueError("Native task reference pixels differ from the registered image")
            answer = ", ".join(case["eval"]["reference_answers"]["must_include"])
            _, reward, terminated, truncated, info, _ = env.step(Action("ANSWER", answer))
            if not exact_vwa_success(reward, terminated, truncated, info.get("task_info", {})):
                raise ValueError(f"Native preflight rejected the verified answer for {case['task_id']}")
            print(f"{case['task_id']} native preflight passed", flush=True)
        finally:
            env.close()


def audit_pair(run, case, reference_pixels, listing_site, model, strategies=STRATEGIES):
    task_id = case["task_id"]
    checked = {}
    for mode in strategies:
        evidence = mode == "evidence-cards"
        directory = run / f"visualwebarena.{task_id}-{mode}"
        checked[mode] = audit_episode(
            directory, PROFILE["max_predictions"], "scroll-feedback", "feedback", answer_check=not evidence,
            answer_study=True, evidence_check=evidence, confirm=evidence, use_navigation_context=evidence,
            compact_navigation=model.get("compact_navigation_replies", False), use_screen_text=evidence,
            guarded_ocr=True, direct_navigation=True, listing_site=listing_site if evidence else None,
            use_ranked_cards=evidence, use_browser_context=True, model_budget=PROFILE["max_model_calls"],
        )
        row = checked[mode]
        if row["reset"] != (case["intent"], PROFILE["seed"]) or row["reset_context"][1] != reference_pixels:
            raise ValueError("Native replay differs from the frozen question or reference")
    outcomes = {mode: {"success": row["row"]["success"], "reason": row["row"]["reason"],
                       "answers_submitted": row["row"]["answers_submitted"],
                       "model_calls": row["counts"]["model_calls"], "predictions": row["counts"]["predictions"],
                       "screenshots": row["screenshots"], "evidence_stages": row["evidence_stages"]}
                for mode, row in checked.items()}
    evidence = checked["evidence-cards"]
    if len(strategies) == 1:
        return {"reset_pixels": evidence["reset_pixels"], "strategies": outcomes}
    control = checked["answer-check"]
    if control["reset_context"] != evidence["reset_context"]:
        raise ValueError("The paired reset tabs or references differ")
    return {"identical_reset_pixels": control["reset_pixels"] == evidence["reset_pixels"], "strategies": outcomes}


def audit_run(run, cases, references, cohort_digest):
    plan = json.loads((run / "run-plan.json").read_bytes())
    if plan["cohort_sha256"] != cohort_digest or not holdout_profile(
        "evidence-cards", PROFILE["max_model_calls"], plan["model"], "evidence-cards"
    ):
        raise ValueError("The checkpoint or registration differs from the paired protocol")
    strategies = tuple(plan.get("strategies", STRATEGIES))
    if strategies not in (STRATEGIES, ("evidence-cards",)):
        raise ValueError("Unknown catalog strategy selection")
    pairs = {}
    for case in cases:
        image = case["image"]
        pixels = []
        if image:
            with references(image) as ref:
                pixels = [pixel_hash(ref.convert("RGB"))]
        pairs[str(case["task_id"])] = audit_pair(run, case, pixels, plan["listing_origin"], plan["model"], strategies)
    scores = {mode: {"native_successes": 0, "wrong_submissions": 0} for mode in strategies}
    for pair in pairs.values():
        for mode, row in pair["strategies"].items():
            scores[mode]["native_successes"] += int(row["success"])
            scores[mode]["wrong_submissions"] += int(row["answers_submitted"] > 0 and not row["success"])
    if len(strategies) == 1:
        return {"scope": SCOPE, "cohort_sha256": cohort_digest, "task_count": len(pairs),
                "max_model_calls": PROFILE["max_model_calls"], "strategies": scores, "by_task": pairs}
    baseline, evidence = (scores[mode] for mode in STRATEGIES)
    checks = {
        "at_least_two_native_successes": evidence["native_successes"] >= 2,
        "more_native_successes_than_control": evidence["native_successes"] > baseline["native_successes"],
        "fewer_wrong_submissions_than_control": evidence["wrong_submissions"] < baseline["wrong_submissions"],
    }
    return {"scope": SCOPE, "cohort_sha256": cohort_digest, "task_count": len(pairs),
            "max_model_calls": PROFILE["max_model_calls"], "strategies": scores, "by_task": pairs,
            "pairs_with_identical_reset_pixels": sum(pair["identical_reset_pixels"] for pair in pairs.values()),
            "acceptance_checks": checks, "target_met": all(checks.values())}


def run_pairs(run, cases, references, cohort_digest, model, listing_site, *, evidence_only=False):
    probe_instance()
    actor = QwenGroundingPolicy(model, device="cuda", max_new_tokens=128, max_pixels=784 * 784,
                                load_in_4bit=True, native_navigation=True)
    if not holdout_profile("evidence-cards", PROFILE["max_model_calls"], actor.config, "evidence-cards"):
        raise ValueError("This evaluation needs the registered Qwen 7B NF4 configuration")
    package = Path(__file__).resolve().parent.parent / "src/gui_recover"
    plan = {"cohort_sha256": cohort_digest, "listing_origin": listing_site, "model": actor.config,
            "source_files": {path.name: fingerprint(path.read_bytes()) for path in
                             [Path(__file__), Path(__file__).with_name("run_visualwebarena_live.py"),
                              *sorted(package.glob("*.py"))]}}
    strategies = ("evidence-cards",) if evidence_only else STRATEGIES
    if evidence_only:
        plan["strategies"] = list(strategies)
    run.mkdir(parents=True, exist_ok=True)
    plan_path = run / "run-plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != plan or (run / "audit.json").exists():
            raise ValueError("Select a fresh output path or resume the exact unfinished run")
    else:
        if any(run.iterdir()):
            raise ValueError("Select an empty output directory")
        write_json(plan_path, plan)
    reader = RapidScreenReader()
    for case in cases:
        for mode in strategies:
            branch = run / f"visualwebarena.{case['task_id']}-{mode}"
            if branch.is_dir():
                continue
            with_cards = mode == "evidence-cards"
            env = browser_case(case, references, cards=with_cards)
            try:
                temporary = Path(tempfile.mkdtemp(prefix=".episode-", dir=run)) / branch.name
                policy = EvidenceAnswerPolicy(
                    actor, confirm=True, use_navigation_context=True, screen_reader=reader, listing_site=listing_site,
                    listing_preview=env.preview_listing_links, ranked_capture=env.capture_cards,
                ) if with_cards else actor
                result = run_episode(
                    env, policy, env.task, PROFILE["seed"], PROFILE["max_predictions"], temporary,
                    recovery="scroll-feedback", invalid_action_retry="feedback", answer_check=not with_cards,
                    allow_observation_only=with_cards, submit_answer=True, success_check=exact_vwa_success,
                    max_model_calls=PROFILE["max_model_calls"],
                )
                if result["reason"] == "runtime_error":
                    raise RuntimeError(f"Browser episode failed: {temporary}")
                render_replay(temporary)
                temporary.rename(branch)
                temporary.parent.rmdir()
                print(f"{branch.name}: {result['reason']} ({result['model_calls']} model calls)", flush=True)
            finally:
                env.close()
    audit_path = run / "audit.json"
    temporary_audit = audit_path.with_suffix(".json.partial")
    write_json(temporary_audit, audit_run(run, cases, references, cohort_digest))
    temporary_audit.replace(audit_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "audit"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--evidence-only", action="store_true", help="record a clean evidence policy arm for a separate comparison")
    args = parser.parse_args()
    _, cases, references = load_cohort(args.cohort, args.sha256, os.getenv("VWA_HOMEPAGE", "__HOMEPAGE__"))
    if args.command == "preflight":
        preflight(cases, references)
    elif args.command == "run" and args.out is not None and args.model is not None and args.model.is_dir():
        run_pairs(args.out, cases, references, args.sha256, args.model, os.environ["VWA_CLASSIFIEDS"],
                  evidence_only=args.evidence_only)
    elif args.command == "audit" and args.out is not None:
        print(json.dumps(audit_run(args.out, cases, references, args.sha256), indent=2, ensure_ascii=False))
    else:
        parser.error("Run requires --out and a local --model. Audit requires --out.")


if __name__ == "__main__":
    main()
