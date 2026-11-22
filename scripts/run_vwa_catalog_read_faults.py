"""Run and audit controlled read faults on a sealed Classifieds cohort."""

import argparse
import json
import os
import re
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import gui_recover
from audit_visualwebarena_live import audit_episode
from gui_recover.answer_evidence import EvidenceAnswerPolicy, REFERENCE_TITLE_HEADING
from gui_recover.listing_cards import is_listing_breadcrumb
from gui_recover.listing_facts import LISTING_FACTS_REGION, ListingFacts, main_listing_price
from gui_recover.price_discovery import _displayed_cents
from gui_recover.qwen_grounding import QwenGroundingPolicy, read_reply
from gui_recover.runner import render_replay, run_episode, write_json
from gui_recover.screen_text import RapidScreenReader
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import read_json, require
from prepare_vwa_catalog_holdout import fingerprint
from run_visualwebarena_live import exact_vwa_success, holdout_profile, probe_instance
from run_vwa_catalog_holdout import PROFILE, browser_case, load_cohort


AMOUNT_ANCHORED_REREAD = "match-ocr-price-in-reread-amount-anchor"
PRICE_REPLY_FAULTS = {
    "match-ocr-price-in-confirmation": "initial",
    "match-ocr-price-in-reread": "isolated",
    AMOUNT_ANCHORED_REREAD: "isolated",
}
FAULT_FAMILIES = {
    "shorten-city": ("detail-price-city",),
    "omit-breadcrumb": ("gallery-reference", "search-reference"),
    "change-title-letter": ("search-highest-price",),
    "change-ocr-price-digit": ("detail-price-city",),
    **dict.fromkeys(PRICE_REPLY_FAULTS, ("detail-price-city",)),
}
OCR_FAULT_TARGETS = {
    "omit-breadcrumb": REFERENCE_TITLE_HEADING,
    "change-ocr-price-digit": LISTING_FACTS_REGION,
    **dict.fromkeys(PRICE_REPLY_FAULTS, LISTING_FACTS_REGION),
}
FORMAT = "gui-recover-catalog-read-faults-v1"


def changed_reply(fault, reply):
    require(fault in ("shorten-city", "change-title-letter"), "Expected a verification reply fault")
    field = "city" if fault == "shorten-city" else "title"
    data = read_reply(reply)
    require(isinstance(data, dict) and set(data) == {field, "dollars"} and isinstance(data[field], str),
            "Read fault requires a complete compound verification reply")
    value = data[field]
    if field == "city":
        words = value.split()
        require(len(words) >= 2, "City truncation needs at least two words")
        data[field] = words[0]
    else:
        letter = re.search(r"[A-Za-z]", value)
        require(letter is not None, "Title alteration needs a letter")
        replacement = "y" if letter[0].casefold() == "x" else "x"
        data[field] = value[:letter.start()] + replacement + value[letter.end():]
    return json.dumps(data, ensure_ascii=False)


def changed_ocr(fault, rows):
    if fault == "omit-breadcrumb":
        matches = [row for row in rows if is_listing_breadcrumb(row)]
        require(len(matches) == 1, "Clean OCR must show one listing breadcrumb")
        crumb, = matches
        return [row for row in rows if row is not crumb], {"fault": fault, "omitted_row": crumb}
    require(fault == "change-ocr-price-digit" or fault in PRICE_REPLY_FAULTS, "Expected an OCR fault")
    price = main_listing_price(rows)
    require(price is not None, "Clean OCR must show one listing price")
    (left, top, right, bottom), cents = price
    indices = [index for index, row in enumerate(rows)
               if left <= row["bounds"][0] < row["bounds"][2] <= right
               and top <= row["bounds"][1] < row["bounds"][3] <= bottom
               and re.search(r"[0-9]", row["text"])]
    require(len(indices) == 1, "Price digit fault needs a single numeric OCR row")
    index, = indices
    original = rows[index]
    text = re.sub(r"[0-9]", lambda match: "1" if match[0] == "9" else str(int(match[0]) + 1),
                  original["text"], count=1)
    mutated = {**original, "text": text}
    altered = [*rows[:index], mutated, *rows[index + 1:]]
    new_price = main_listing_price(altered)
    require(new_price is not None and new_price[0] == price[0] and new_price[1] != cents,
            "Price digit fault must change the amount within the same price box")
    return altered, {"fault": fault, "original_row": original, "injected_row": mutated}


def checked_price_reply(stage, reply, original_cents):
    key = "city" if stage == "initial" else "text"
    data = read_reply(reply, parse_float=Decimal)
    require(isinstance(data, dict) and set(data) == {key, "dollars"} and isinstance(data[key], str)
            and type(data["dollars"]) in (str, int, Decimal), "Expected an intact price and city reply")
    amount = str(data["dollars"]).strip()
    require(_displayed_cents(amount if "$" in amount else amount + " $") == original_cents,
            "The model must originally read the unmodified price")
    return key, data[key]


def matching_price_reply(stage, reply, original_cents, injected_cents):
    key, text = checked_price_reply(stage, reply, original_cents)
    return json.dumps({key: text, "dollars": format(Decimal(injected_cents) / 100, ".2f")}, ensure_ascii=False)


def inject_matching_price(kind, target, raw):
    stage = PRICE_REPLY_FAULTS[kind]
    rows = target["screen_text"]
    altered, _ = changed_ocr(kind, rows)
    return matching_price_reply(stage, raw, main_listing_price(rows)[1], main_listing_price(altered)[1])


def prediction_metrics(directory):
    for line in (directory / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "prediction":
            yield event["metrics"]


@dataclass(frozen=True)
class FaultCase:
    case: dict
    kind: str
    baseline: dict
    target: dict

    @property
    def episode(self):
        return f"visualwebarena.{self.case['task_id']}"


class FaultAnchorUnavailable(ValueError):
    """The clean episode did not reach the registered injection checkpoint."""


def clean_target(directory, fault, *, expected_cents=None):
    metrics = list(prediction_metrics(directory))
    if fault in OCR_FAULT_TARGETS:
        candidates = [step for step in metrics if step["provider_format"] == OCR_FAULT_TARGETS[fault]
                      and "reference_title_breadcrumb" not in step]
        if len(candidates) != 1:
            raise FaultAnchorUnavailable("Clean episode must have one OCR read at the registered evidence stage")
        step, = candidates
        changed_ocr(fault, step["screen_text"])
        if fault in PRICE_REPLY_FAULTS:
            confirmation = next((item for item in metrics if item.get("evidence_stage") == "confirm"), None)
            same_frame = confirmation is not None and confirmation["evidence_frame"] == step["evidence_frame"]
            if fault == AMOUNT_ANCHORED_REREAD:
                if not same_frame or main_listing_price(step["screen_text"])[1] != expected_cents:
                    raise FaultAnchorUnavailable("Expected the sealed amount in full-page OCR on the first read frame")
                try:
                    checked_price_reply("initial", confirmation["provider_output"], expected_cents)
                except (TypeError, ValueError, RecursionError) as exc:
                    raise FaultAnchorUnavailable("The first model read did not transcribe the sealed amount") from exc
            elif not same_frame or confirmation["provider_format"] != "evidence_verify_agreed":
                raise FaultAnchorUnavailable("Expected a successful first model read on the same screen")
            step = {**step, "clean_confirmation": confirmation["provider_output"]}
    else:
        step = next((step for step in metrics if step.get("evidence_stage") == "confirm"), None)
        require(step is not None and step["provider_format"] == "evidence_verify_agreed",
                "Clean episode must agree on the first compound confirmation")
        changed_reply(fault, step["provider_output"])
    return step


def audit_evidence(directory, plan):
    return audit_episode(directory, PROFILE["max_predictions"], "scroll-feedback", "feedback",
                         answer_study=True, evidence_check=True, confirm=True, use_navigation_context=True,
                         compact_navigation=plan["model"]["compact_navigation_replies"], use_screen_text=True,
                         guarded_ocr=True, direct_navigation=True, listing_site=plan["listing_origin"],
                         use_ranked_cards=True, use_browser_context=True, model_budget=PROFILE["max_model_calls"])


def load_study(cohort, digest, fault_plan, clean_run, *, allow_unexecuted=False):
    _, cases, references = load_cohort(cohort, digest, os.getenv("VWA_HOMEPAGE", "__HOMEPAGE__"))
    registration = read_json(fault_plan)
    require(isinstance(registration, dict), "Fault plan must be a JSON object")
    faults = registration.get("cases")
    require(set(registration) == {"cohort_sha256", "cases"} and registration["cohort_sha256"] == digest
            and isinstance(faults, dict) and faults, "Fault plan must name cases in the sealed cohort")
    evidence = {str(row["task_id"]): row for row in read_json(cohort / "source-evidence.json")}
    require(all(isinstance(fault, str) and evidence.get(task, {}).get("family") in FAULT_FAMILIES.get(fault, ())
                for task, fault in faults.items()), "Fault must apply to a registered task family")
    clean = read_json(clean_run / "run-plan.json")
    require(clean["cohort_sha256"] == digest
            and holdout_profile("evidence-cards", PROFILE["max_model_calls"], clean["model"], "evidence-cards"),
            "Clean run must use the registered cohort and model profile")
    plan = {"format": FORMAT, "cohort_sha256": digest, "faults": faults,
            "fault_plan_sha256": fingerprint(fault_plan.read_bytes()),
            "clean_plan_sha256": fingerprint((clean_run / "run-plan.json").read_bytes()),
            "clean_trajectory_sha256": {}, "model": clean["model"], "listing_origin": clean["listing_origin"]}
    selected = []
    unexecuted = {}
    for case in cases:
        task = str(case["task_id"])
        if task not in faults:
            continue
        directory = clean_run / f"visualwebarena.{task}-evidence-cards"
        audited = audit_evidence(directory, plan)
        with references(case["image"]) if case["image"] else nullcontext() as reference:
            images = [pixel_hash(reference.convert("RGB"))] if reference is not None else []
        require(audited["reset"] == (case["intent"], PROFILE["seed"]) and audited["reset_context"][1] == images,
                "Clean episode differs from the sealed task")
        plan["clean_trajectory_sha256"][task] = fingerprint((directory / "trajectory.jsonl").read_bytes())
        try:
            if not audited["row"]["success"]:
                raise FaultAnchorUnavailable(f"Clean episode did not succeed: {audited['row']['reason']}")
            expected_cents = None
            if faults[task] == AMOUNT_ANCHORED_REREAD:
                dollars = evidence[task]["source"]["visible"]["dollars"]
                require(type(dollars) is int and dollars > 0, "Expected a sealed whole dollar amount")
                expected_cents = dollars * 100
            target = clean_target(directory, faults[task], expected_cents=expected_cents)
        except FaultAnchorUnavailable as exc:
            if not allow_unexecuted:
                raise
            unexecuted[task] = str(exc)
            continue
        selected.append(FaultCase(case, faults[task], audited, target))
    if allow_unexecuted:
        plan["unexecuted"] = unexecuted
    return plan, selected, references


class FaultPolicy(EvidenceAnswerPolicy):
    """Change the registered reads, then use the regular recovery policy."""

    def __init__(self, actor, reader, case, **kwargs):
        self.fault_case = case
        self.injected = None
        self.reader = reader
        super().__init__(actor, screen_reader=self.read_screen, **kwargs)

    def reset(self, seed):
        self.injected = None
        super().reset(seed)

    def read_screen(self, screen, size):
        rows = self.reader(screen, size)
        fault, target = self.fault_case.kind, self.fault_case.target
        if fault not in OCR_FAULT_TARGETS or self.injected is not None or pixel_hash(screen) != target["evidence_frame"]:
            return rows
        require(rows == target["screen_text"], "Full-page OCR differs from the clean read")
        altered, self.injected = changed_ocr(fault, rows)
        return altered

    @contextmanager
    def intercept_reply(self, frame, candidate, stage):
        fault = self.fault_case.kind
        target = self.fault_case.target
        compound = fault in PRICE_REPLY_FAULTS
        paired = compound and self.injected is not None and "reply" not in self.injected and PRICE_REPLY_FAULTS[fault] == stage
        single = stage == "initial" and fault not in OCR_FAULT_TARGETS and self.injected is None
        if not ((paired or single) and isinstance(candidate, ListingFacts) and frame == target["evidence_frame"]):
            yield
            return
        ask = self.actor.ask

        def ask_with_fault(*args, **kwargs):
            raw = ask(*args, **kwargs)
            if paired:
                changed = inject_matching_price(fault, target, raw)
                self.injected["reply"] = {"stage": stage, "original_reply": raw, "injected_reply": changed}
            else:
                require(read_reply(raw) == read_reply(target["provider_output"]),
                        "Verification reply differs from the clean read")
                changed = changed_reply(fault, raw)
                self.injected = {"fault": fault, "original_reply": raw, "injected_reply": changed}
            self.actor.last_metrics["provider_output"] = changed
            return changed

        self.actor.ask = ask_with_fault
        try:
            yield
        finally:
            self.actor.ask = ask

    def verify(self, observation, frame, crop, bounds, candidate):
        with self.intercept_reply(frame, candidate, "initial"):
            return super().verify(observation, frame, crop, bounds, candidate)

    def reread_listing_facts(self, observation, frame, candidate):
        with self.intercept_reply(frame, candidate, "isolated"):
            return super().reread_listing_facts(observation, frame, candidate)


def audit_fault(directory, case, plan):
    checked = audit_evidence(directory, plan)
    require(all(checked[key] == case.baseline[key] for key in ("reset", "reset_context", "reset_pixels")),
            "Injected episode starts from a different observation")
    proof = read_json(directory / "injected-fault.json")
    kind, target = case.kind, case.target
    require(proof["fault"] == kind, "Recorded fault differs from the fault plan")
    metrics = list(prediction_metrics(directory))
    same_frame = [step for step in metrics if step.get("evidence_frame") == target["evidence_frame"]]
    row = checked["row"]
    delivery = "complete"
    if kind in OCR_FAULT_TARGETS:
        expected, expected_proof = changed_ocr(kind, target["screen_text"])
        if kind in PRICE_REPLY_FAULTS:
            stage = PRICE_REPLY_FAULTS[kind]
            reply = proof.get("reply")
            initial = [step for step in same_frame if step.get("evidence_stage") == "confirm"
                       and "listing_fact_reread_sha256" not in step]
            selected = initial if stage == "initial" else [
                step for step in same_frame if "listing_fact_reread_sha256" in step
            ]
            if (reply is None and stage == "isolated" and "unexecuted" in plan and len(initial) == 1 and not selected
                    and row["reason"] == "evidence_abstained" and row["answers_submitted"] == 0
                    and read_reply(initial[0]["provider_output"]) == read_reply(target["clean_confirmation"])):
                delivery = "second_read_not_reached"
            else:
                require(isinstance(reply, dict) and reply.get("stage") == stage, "Missing the registered model reply fault")
                raw = reply["original_reply"]
                require(len(selected) == 1 and len(initial) == 1, "Expected one confirmation at the registered stage")
                unmodified = raw if stage == "initial" else initial[0]["provider_output"]
                require(read_reply(unmodified) == read_reply(target["clean_confirmation"])
                        and selected[0]["provider_output"] == reply["injected_reply"],
                        "The matching price fault must change only the registered confirmation")
                expected_proof["reply"] = {"stage": stage, "original_reply": raw,
                                           "injected_reply": inject_matching_price(kind, target, raw)}
        require(proof == expected_proof
                and sum(step.get("evidence_stage") == "locate" and step.get("screen_text") == expected for step in same_frame) == 1,
                "Expected one registered OCR change and the rest of the original rows")
    else:
        raw = proof["original_reply"]
        altered = proof["injected_reply"]
        first_confirmation = next((step for step in metrics if step.get("evidence_stage") == "confirm"), {})
        require(read_reply(raw) == read_reply(target["provider_output"]) and altered == changed_reply(kind, raw)
                and first_confirmation in same_frame and first_confirmation.get("provider_output") == altered,
                "Expected one changed confirmation field from the clean read")
    result = {**{key: row[key] for key in ("success", "reason", "answers_submitted", "predictions", "model_calls")},
              "fault": kind, "evidence_stages": checked["evidence_stages"]}
    if "unexecuted" in plan:
        result["fault_delivery"] = delivery
    return result


def audit_fault_run(out, plan, cases):
    if (out / "run-plan.json").is_file():
        recorded = read_json(out / "run-plan.json")
        require(all(recorded.get(key) == value for key, value in plan.items()),
                "Run plan differs from the sealed faults or clean run")
    else:
        # The original cohort-004 experiment archived its runner next to the plan.
        recorded = read_json(out / "fault-plan.json")
        require(all(recorded.get(key) == plan[key] for key in ("cohort_sha256", "faults", "model"))
                and fingerprint((out / "runner.py").read_bytes()) == recorded["runner_sha256"],
                "Original experiment archive differs from its fault plan")
    rows = {str(case.case["task_id"]): audit_fault(out / case.episode, case, plan) for case in cases}
    report = {"cohort_sha256": plan["cohort_sha256"], "task_count": len(rows), "by_task": rows,
              "native_successes": sum(row["success"] for row in rows.values()),
              "wrong_submissions": sum(row["answers_submitted"] > 0 and not row["success"] for row in rows.values()),
              "model_calls": sum(row["model_calls"] for row in rows.values())}
    if "unexecuted" in plan:
        report.update(registered_task_count=len(plan["faults"]), unexecuted=plan["unexecuted"])
    if recorded.get("format") == FORMAT and (out / "audit.json").is_file():
        require(read_json(out / "audit.json") == report, "Saved audit differs from the episode traces")
    return report


def run_faults(out, plan, cases, references, model):
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A fault run is already writing to {out}") from exc
        return _run_faults(out, plan, cases, references, model)


def _run_faults(out, plan, cases, references, model):
    listing_site = os.environ["VWA_CLASSIFIEDS"].rstrip("/")
    require(listing_site == plan["listing_origin"], "Use the same Classifieds origin as the clean run")
    scripts = Path(__file__).resolve().parent
    package = Path(gui_recover.__file__).resolve().parent
    paths = [Path(__file__), *(scripts / filename for filename in ("audit_visualwebarena_live.py",
             "prepare_vwa_catalog_holdout.py", "run_visualwebarena_live.py", "run_vwa_catalog_holdout.py")),
             *sorted(package.glob("*.py"))]
    plan = {**plan, "source_files": {path.name: fingerprint(path.read_bytes()) for path in paths}}
    plan_path = out / "run-plan.json"
    if plan_path.exists():
        require(read_json(plan_path) == plan, "Use an empty output directory or resume the exact run")
    else:
        temporary_plan = plan_path.with_suffix(".json.partial")
        require(all(entry.name == ".run.lock" or
                    (entry == temporary_plan and entry.is_file() and not entry.is_symlink())
                    for entry in out.iterdir()),
                "New fault run requires an empty output directory")
        write_json(temporary_plan, plan)
        temporary_plan.replace(plan_path)
    needed = []
    for case in cases:
        if (out / case.episode).is_dir():
            audit_fault(out / case.episode, case, plan)
        else:
            needed.append(case)
    if needed:
        probe_instance()
        actor = QwenGroundingPolicy(model, device="cuda", max_new_tokens=128, max_pixels=784 * 784,
                                   load_in_4bit=True, native_navigation=True)
        require(actor.config == plan["model"], "Model settings differ from the clean run")
        reader = RapidScreenReader()
        for case in needed:
            directory = Path(tempfile.mkdtemp(prefix=".episode-", dir=out)) / case.episode
            env = browser_case(case.case, references, cards=True)
            try:
                policy = FaultPolicy(actor, reader, case, confirm=True, use_navigation_context=True,
                                     listing_site=listing_site, listing_preview=env.preview_listing_links,
                                     ranked_capture=env.capture_cards)
                row = run_episode(env, policy, env.task, PROFILE["seed"], PROFILE["max_predictions"], directory,
                                  recovery="scroll-feedback", invalid_action_retry="feedback", allow_observation_only=True,
                                  submit_answer=True, success_check=exact_vwa_success, max_model_calls=PROFILE["max_model_calls"])
                require(row["reason"] != "runtime_error" and policy.injected is not None,
                        f"Episode failed to apply the registered fault: {directory}")
                write_json(directory / "injected-fault.json", policy.injected)
                audit_fault(directory, case, plan)
                render_replay(directory)
                directory.rename(out / case.episode)
                directory.parent.rmdir()
                print(f"{case.episode}: {row['reason']} ({row['model_calls']} model calls)", flush=True)
            finally:
                env.close()
    summary = audit_fault_run(out, plan, cases)
    temporary = out / "audit.json.partial"
    write_json(temporary, summary)
    temporary.replace(out / "audit.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "audit", "compare"))
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--fault-plan", required=True, type=Path)
    parser.add_argument("--clean-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--allow-unexecuted", action="store_true",
                        help="report registered cases without a valid clean injection anchor")
    args = parser.parse_args()
    if args.command == "run" and (args.model is None or not args.model.is_dir()):
        parser.error("Run requires a local --model directory")
    if args.command == "compare" and (args.prior_run is None or args.prior_run.resolve() == args.out.resolve()):
        parser.error("Compare requires a separate --prior-run directory")
    plan, cases, references = load_study(args.cohort, args.sha256, args.fault_plan, args.clean_run,
                                         allow_unexecuted=args.allow_unexecuted)
    if args.command == "run":
        report = run_faults(args.out, plan, cases, references, args.model)
    elif args.command == "audit":
        report = audit_fault_run(args.out, plan, cases)
    else:
        report = {"prior": audit_fault_run(args.prior_run, plan, cases),
                  "current": audit_fault_run(args.out, plan, cases)}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
