"""Run and audit controlled read faults on a sealed Classifieds cohort."""

import argparse
import json
import os
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import gui_recover
from audit_visualwebarena_live import audit_episode
from gui_recover.answer_evidence import EvidenceAnswerPolicy, REFERENCE_TITLE_HEADING
from gui_recover.listing_cards import is_listing_breadcrumb
from gui_recover.listing_facts import ListingFacts
from gui_recover.qwen_grounding import QwenGroundingPolicy, read_reply
from gui_recover.runner import render_replay, run_episode, write_json
from gui_recover.screen_text import RapidScreenReader
from gui_recover.verification import pixel_hash
from gui_recover.visual_text_audit import read_json, require
from prepare_vwa_catalog_holdout import fingerprint
from run_visualwebarena_live import exact_vwa_success, holdout_profile, probe_instance
from run_vwa_catalog_holdout import PROFILE, browser_case, load_cohort


FAULT_FAMILIES = {
    "shorten-city": ("detail-price-city",),
    "omit-breadcrumb": ("gallery-reference", "search-reference"),
    "change-title-letter": ("search-highest-price",),
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


def clean_target(directory, fault):
    metrics = list(prediction_metrics(directory))
    if fault == "omit-breadcrumb":
        candidates = [step for step in metrics if step["provider_format"] == REFERENCE_TITLE_HEADING
                      and "reference_title_breadcrumb" not in step]
        require(len(candidates) == 1, "Clean episode must have one full-page title verification")
        step, = candidates
        require(sum(is_listing_breadcrumb(row) for row in step["screen_text"]) == 1,
                "Clean OCR must show one listing breadcrumb")
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


def load_study(cohort, digest, fault_plan, clean_run):
    _, cases, references = load_cohort(cohort, digest, os.getenv("VWA_HOMEPAGE", "__HOMEPAGE__"))
    registration = read_json(fault_plan)
    require(isinstance(registration, dict), "Fault plan must be a JSON object")
    faults = registration.get("cases")
    require(set(registration) == {"cohort_sha256", "cases"} and registration["cohort_sha256"] == digest
            and isinstance(faults, dict) and faults, "Fault plan must name cases in the sealed cohort")
    families = {str(row["task_id"]): row["family"] for row in read_json(cohort / "source-evidence.json")}
    require(all(isinstance(fault, str) and families.get(task) in FAULT_FAMILIES.get(fault, ())
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
    for case in cases:
        task = str(case["task_id"])
        if task not in faults:
            continue
        directory = clean_run / f"visualwebarena.{task}-evidence-cards"
        audited = audit_evidence(directory, plan)
        with references(case["image"]) if case["image"] else nullcontext() as reference:
            images = [pixel_hash(reference.convert("RGB"))] if reference is not None else []
        require(audited["row"]["success"] and audited["reset"] == (case["intent"], PROFILE["seed"])
                and audited["reset_context"][1] == images, "Expected a successful clean episode for the sealed task")
        plan["clean_trajectory_sha256"][task] = fingerprint((directory / "trajectory.jsonl").read_bytes())
        selected.append(FaultCase(case, faults[task], audited, clean_target(directory, faults[task])))
    return plan, selected, references


class FaultPolicy(EvidenceAnswerPolicy):
    """Change a single read at the recorded frame, then use the regular recovery policy."""

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
        target = self.fault_case.target
        if self.fault_case.kind != "omit-breadcrumb" or self.injected is not None or pixel_hash(screen) != target["evidence_frame"]:
            return rows
        require(rows == target["screen_text"], "Full-page OCR differs from the clean read")
        crumb, = (row for row in rows if is_listing_breadcrumb(row))
        self.injected = {"fault": self.fault_case.kind, "omitted_row": crumb}
        return [row for row in rows if row is not crumb]

    def verify(self, observation, frame, crop, bounds, candidate):
        fault = self.fault_case.kind
        if (fault == "omit-breadcrumb" or self.injected is not None
                or not isinstance(candidate, ListingFacts) or frame != self.fault_case.target["evidence_frame"]):
            return super().verify(observation, frame, crop, bounds, candidate)
        ask = self.actor.ask

        def ask_with_fault(*args, **kwargs):
            raw = ask(*args, **kwargs)
            require(read_reply(raw) == read_reply(self.fault_case.target["provider_output"]),
                    "Verification reply differs from the clean read")
            changed = changed_reply(fault, raw)
            self.injected = {"fault": fault, "original_reply": raw, "injected_reply": changed}
            self.actor.last_metrics["provider_output"] = changed
            return changed

        self.actor.ask = ask_with_fault
        try:
            return super().verify(observation, frame, crop, bounds, candidate)
        finally:
            self.actor.ask = ask


def audit_fault(directory, case, plan):
    checked = audit_evidence(directory, plan)
    require(all(checked[key] == case.baseline[key] for key in ("reset", "reset_context", "reset_pixels")),
            "Injected episode starts from a different observation")
    proof = read_json(directory / "injected-fault.json")
    kind, target = case.kind, case.target
    require(proof["fault"] == kind, "Recorded fault differs from the fault plan")
    metrics = list(prediction_metrics(directory))
    same_frame = [step for step in metrics if step.get("evidence_frame") == target["evidence_frame"]]
    if kind == "omit-breadcrumb":
        crumb, = (row for row in target["screen_text"] if is_listing_breadcrumb(row))
        expected = [row for row in target["screen_text"] if row != crumb]
        require(proof == {"fault": kind, "omitted_row": crumb}
                and sum(step.get("evidence_stage") == "locate" and step.get("screen_text") == expected for step in same_frame) == 1,
                "Expected one removed breadcrumb and the rest of the original OCR rows")
    else:
        raw = proof["original_reply"]
        altered = proof["injected_reply"]
        first_confirmation = next((step for step in metrics if step.get("evidence_stage") == "confirm"), {})
        require(read_reply(raw) == read_reply(target["provider_output"]) and altered == changed_reply(kind, raw)
                and first_confirmation in same_frame and first_confirmation.get("provider_output") == altered,
                "Expected one changed confirmation field from the clean read")
    row = checked["row"]
    return {**{key: row[key] for key in ("success", "reason", "answers_submitted", "predictions", "model_calls")},
            "fault": kind, "evidence_stages": checked["evidence_stages"]}


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
    args = parser.parse_args()
    if args.command == "run" and (args.model is None or not args.model.is_dir()):
        parser.error("Run requires a local --model directory")
    if args.command == "compare" and (args.prior_run is None or args.prior_run.resolve() == args.out.resolve()):
        parser.error("Compare requires a separate --prior-run directory")
    plan, cases, references = load_study(args.cohort, args.sha256, args.fault_plan, args.clean_run)
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
