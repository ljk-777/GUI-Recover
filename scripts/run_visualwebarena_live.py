"""Optional native BrowserGym adapter for a self-hosted VisualWebArena instance."""

import argparse
import hashlib
import importlib.metadata
from importlib.resources import files
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import urlopen

from gui_recover.actions import Action, compile_action, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.environment import MiniwobEnvironment
from gui_recover.listing_cards import capture_listing_cards
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.qwen_grounding import QWEN_7B_MIRROR_REVISION, QWEN_7B_MODEL, QwenGroundingPolicy
from gui_recover.runner import aggregate, render_replay, run_episode, write_json
from gui_recover.screen_text import RapidScreenReader
from gui_recover.verification import pixel_hash
from gui_recover.visible_links import visible_listing_links
from gui_recover.vwa_inputs import LocalReferenceArchive, contextual_observation, reference_name


BACKEND_VARIABLES = ("VWA_CLASSIFIEDS", "VWA_SHOPPING", "VWA_REDDIT", "VWA_WIKIPEDIA",
                     "VWA_HOMEPAGE", "VWA_CLASSIFIEDS_RESET_TOKEN")
READ_ONLY_TASKS = (118, 120, 149)
TRANSFER_TASKS = (11, 12, 22, 49, 119, 128, 148, 186, 221, 222)
HOLDOUT_TASK_SET = "evidence-links-holdout"
HOLDOUT_PROTOCOL_SHA256 = "df60ebc2e7cb257d63b22a1d6eac4322e5ea53eed75f695850efcc8d51407629"
INPUT_HOLDOUT_TASK_SET = "evidence-cards-input-holdout"
INPUT_PROTOCOL_SHA256 = "d97cc17118a061eb822879c109bdd6202431f60a41ad4dc0d1970e45ca2087fe"
INPUT_MEDIA_SHA256 = "88a7fcad2505d581b266427896265c71367cbc067ed1da461eba127a45b20f58"
INPUT_SHOPPING_TASKS = (233,)
TASK_SETS = {
    "diagnostic": READ_ONLY_TASKS,
    "transfer": TRANSFER_TASKS,
    "answer-diagnostic": (118, 128, 221),
    "answer-transfer": (10, 13, 41, 42, 150, 163, 192, 209),
    "evidence-diagnostic": (118, 221, 41, 42, 192, 209),
    "evidence-transfer": (23, 40, 43, 110, 161, 173, 174, 199),
    "evidence-confirm-transfer": (25, 27, 98, 100, 107, 109, 111, 178),
    "evidence-context-transfer": (14, 15, 16, 26, 50, 80, 83, 129),
    "evidence-navigation-diagnostic": (14, 15, 221),
    "evidence-ocr-diagnostic": (14, 15, 40, 83, 98, 221),
    "evidence-search-diagnostic": (14, 40, 98, 221),
    "evidence-direct-search-diagnostic": (40, 98, 221),
    "evidence-card-diagnostic": (34, 106),
    HOLDOUT_TASK_SET: (6, 7, 34, 35, 38, 39, 106, 108),
    INPUT_HOLDOUT_TASK_SET: (36, 102, 136, 137, 138, 140, 142, 233),
}
STUDY_MODES = {
    "visual": ("none", "feedback"),
    "format": ("none", "retry", "feedback"),
    "scroll": ("feedback", "scroll-feedback"),
    "answer": ("navigation", "answer-check"),
    "evidence": ("answer-check", "evidence-check"),
    "evidence-confirm": ("answer-check", "evidence-confirm"),
    "evidence-context": ("answer-check", "evidence-context"),
    "evidence-ocr": ("answer-check", "evidence-ocr"),
    "evidence-links": ("answer-check", "evidence-links"),
    "evidence-cards": ("answer-check", "evidence-cards"),
}
LINK_MODES = ("evidence-links", "evidence-cards")
OCR_MODES = ("evidence-ocr", *LINK_MODES)
NAVIGATION_CONTEXT_MODES = ("evidence-context", *OCR_MODES)
CONFIRM_MODES = ("evidence-confirm", *NAVIGATION_CONTEXT_MODES)
EVIDENCE_STUDIES = ("evidence", *CONFIRM_MODES)
ANSWER_STUDIES = ("answer", *EVIDENCE_STUDIES)


def live_preflight():
    missing = [name for name in BACKEND_VARIABLES if not os.getenv(name)]
    invalid = []
    for name in BACKEND_VARIABLES[:-1]:
        if name in missing:
            continue
        parsed = urlsplit(os.environ[name])
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            invalid.append(name)
    try:
        version = importlib.metadata.version("browsergym-visualwebarena")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"missing_environment": missing, "invalid_url_variables": invalid,
            "browsergym_visualwebarena_version": version,
            "ready_for_connection_check": not missing and not invalid and version == "0.14.3"}


def probe_instance(site="CLASSIFIEDS"):
    try:
        with urlopen(os.environ[f"VWA_{site}"], timeout=8) as response:
            if response.status != 200:
                raise RuntimeError(f"VisualWebArena {site} did not return HTTP 200")
    except OSError:
        raise RuntimeError(f"The configured VisualWebArena {site} backend is unreachable") from None


def exact_vwa_success(reward, terminated, truncated, _task_info):
    return bool(terminated and not truncated and reward == 1.0)


def holdout_profile(study, budget, model, expected_study="evidence-links"):
    return (study == expected_study and budget == 8 and model.get("model") == QWEN_7B_MODEL
            and model.get("revision") == QWEN_7B_MIRROR_REVISION and model.get("device") == "cuda"
            and model.get("quantization") == "bitsandbytes NF4 double quantization"
            and model.get("max_new_tokens") == 128 and model.get("max_pixels") == 784 * 784
            and model.get("do_sample") is False and model.get("native_navigation") is True)


def input_holdout_archive(directory):
    """Authenticate the registration and exact official task table before any rollout."""
    protocol_bytes = (directory / "cohort.json").read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != INPUT_PROTOCOL_SHA256:
        raise ValueError("Input holdout protocol differs from the registered cohort")
    protocol = json.loads(protocol_bytes)
    task_ids = TASK_SETS[INPUT_HOLDOUT_TASK_SET]
    native_table = json.loads(files("visualwebarena").joinpath("test_raw.json").read_text())
    entries = sorted((row for row in native_table if row["task_id"] in task_ids), key=lambda row: row["task_id"])
    for row in entries:
        image = row.get("image")
        if isinstance(image, str):
            row["image"] = reference_name(image)
        elif image:
            row["image"] = [reference_name(item) for item in image]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if (tuple(protocol["task_ids"]) != task_ids
            or hashlib.sha256(canonical).hexdigest() != protocol["entries_sha256"]
            or tuple(row["task_id"] for row in entries if "shopping" in row["sites"]) != INPUT_SHOPPING_TASKS):
        raise ValueError("Installed VisualWebArena tasks differ from the frozen native entries")
    archive = LocalReferenceArchive(directory, INPUT_MEDIA_SHA256, homepage=os.environ["VWA_HOMEPAGE"])
    input_manifest = {}
    references = set()
    for row in entries:
        images = row.get("image") or []
        images = [images] if isinstance(images, str) else images
        references.update(images)
        fingerprints = []
        for name in images:
            with archive(name) as image:
                fingerprints.append(pixel_hash(image.convert("RGB")))
        input_manifest[str(row["task_id"])] = {"reference_pixels": fingerprints,
                                               "start_pages": len(row["start_url"].split(" |AND| "))}
    if references != set(archive.images):
        raise ValueError("Task references differ from the registered file set")
    return archive, input_manifest


def probe_shopping():
    classifieds, shopping = (urlsplit(os.environ[f"VWA_{site}"]) for site in ("CLASSIFIEDS", "SHOPPING"))
    if (classifieds.scheme, classifieds.netloc) == (shopping.scheme, shopping.netloc):
        raise ValueError("The locked cross-site task requires a separately configured Shopping origin")
    probe_instance("SHOPPING")


def expand_listing_view():
    """Expand the browser viewport to inspect the rendered results grid.

    Examples:
        expand_listing_view()
    """
    page.set_viewport_size({"width": page.viewport_size["width"], "height": 1800})


class VisualWebArenaEnvironment(MiniwobEnvironment):
    """Use BrowserGym's screenshot and action contract with native VWA answers."""

    def __init__(self, task_id, *, enable_card_view=False, enable_browser_context=False, reference_archive=None):
        from browsergym.core.action.highlevel import HighLevelActionSet
        from browsergym.core.env import BrowserEnv
        from browsergym.visualwebarena.task import GenericVisualWebArenaTask

        if not any(task_id in tasks for tasks in TASK_SETS.values()):
            raise ValueError("Task is outside the registered Classifieds subset")
        # The upstream module constructs an OpenAI client even for its local
        # string evaluator. These tasks require no external judging service.
        os.environ["OPENAI_API_KEY"] = "local-string-evaluator"
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"
        class StringOnlyTask(GenericVisualWebArenaTask):
            def setup(self, page):
                permitted = {"classifieds", "shopping"} if enable_browser_context else {"classifieds"}
                sites = set()
                for config in self.task_configs:
                    evaluation = config["eval"]
                    config_sites = set(config["sites"])
                    if ("classifieds" not in config_sites or not config_sites <= permitted or config.get("require_reset")
                            or evaluation["eval_types"] != ["string_match"]
                            or set(evaluation["reference_answers"]) != {"must_include"}):
                        raise ValueError("Task requires more than the registered local string evaluator")
                    if config.get("image") and reference_archive is None:
                        raise ValueError("Task requires a registered local reference archive")
                    sites.update(config_sites)

                origins = {(url.scheme, url.netloc)
                           for url in (urlsplit(self.webarena_instance.urls[site]) for site in sites)}

                def restrict_origin(route):
                    requested = urlsplit(route.request.url)
                    if (requested.scheme, requested.netloc) in origins:
                        route.continue_()
                    else:
                        route.abort()

                page.context.route("**/*", restrict_origin)
                page.context.set_default_navigation_timeout(30_000)
                page.set_default_navigation_timeout(30_000)

                def unavailable_captioner(*args, **kwargs):
                    raise RuntimeError("Captioning must not be used by the local string evaluator")

                # This upstream release initializes the image model on every
                # task. Keep its native string judge and skip unused assets.
                with patch("visualwebarena.evaluation_harness.image_utils.get_captioning_fn",
                           return_value=unavailable_captioner), patch(
                               "browsergym.visualwebarena.task.image_url_to_pil_image",
                               side_effect=reference_archive or unavailable_captioner):
                    return super().setup(page)

        self.task = f"visualwebarena.{task_id}"
        self.visual_variant = "default"
        self.obs = None
        self.enable_card_view = enable_card_view
        self.enable_browser_context = enable_browser_context
        self.references = None
        subsets = ["coord", "chat", "custom"] if enable_card_view else ["coord", "chat"]
        if enable_browser_context:
            subsets.append("tab")
        actions = HighLevelActionSet(subsets=subsets, strict=True, multiaction=True,
                                     custom_actions=[expand_listing_view] if enable_card_view else None)
        self.env = BrowserEnv(StringOnlyTask, task_kwargs={"task_id": task_id}, headless=True,
                              action_mapping=actions.to_python_code)

    def reset(self, seed):
        self.references = None
        return super().reset(seed)

    def _visual(self, observation):
        base = super()._visual(observation)
        if not self.enable_browser_context:
            return base
        visual = contextual_observation(base, observation, references=self.references)
        self.references = visual.reference_images
        return visual

    def step(self, action: Action):
        checked = parse_action(json.dumps(action.to_dict()))
        if checked.action == "ANSWER":
            code = f"send_msg_to_user({checked.value!r})"
        elif checked.action == "TAB_FOCUS":
            index = int(checked.value) - 1
            if not self.enable_browser_context or self.obs is None or index >= len(self.obs.tabs):
                raise ValueError("Tab target must exist in the last browser observation")
            code = f"tab_focus({index})"
        elif checked.action == "LISTING_GRID":
            grid = self.capture_cards()["grid"] if self.enable_card_view else None
            if grid is None or tuple(grid) != checked.position:
                raise ValueError("Listing grid action must point to a visible grid control")
            code = compile_action(Action("CLICK", position=checked.position), *self.obs.screenshot.size)
            code += "\nexpand_listing_view()"
        else:
            code = compile_action(checked, *self.obs.screenshot.size)
        return self._step_code(code)

    def capture_cards(self):
        if not self.enable_card_view or self.obs is None:
            raise ValueError("Card capture requires a registered live viewport")
        return capture_listing_cards(self.env.unwrapped.page, os.environ["VWA_CLASSIFIEDS"], self.obs.screenshot)

    def preview_listing_links(self, positions):
        """Read selected visible destinations without opening the listings."""
        if self.obs is None:
            raise RuntimeError("Reset the browser before previewing links")
        page = self.env.unwrapped.page
        return {"source_pixels": pixel_hash(self.obs.screenshot),
                "links": visible_listing_links(page, os.environ["VWA_CLASSIFIEDS"], positions)}


def stage_input_holdout(out, phase, model, input_manifest):
    """Freeze the rollout code and shared environment before the first staged episode."""
    import gui_recover

    package_root = Path(gui_recover.__file__).parent
    versions = ("browsergym-core", "browsergym-visualwebarena", "torch", "transformers",
                "bitsandbytes", "playwright", "rapidocr", "onnxruntime")
    plan = {
        "protocol_sha256": INPUT_PROTOCOL_SHA256,
        "reference_files_sha256": INPUT_MEDIA_SHA256,
        "input_manifest": input_manifest,
        "model": model,
        "classifieds": os.environ["VWA_CLASSIFIEDS"],
        "homepage": os.environ["VWA_HOMEPAGE"],
        "versions": {name: importlib.metadata.version(name) for name in versions},
        "source_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [Path(__file__), *sorted(package_root.glob("*.py"))]},
    }
    state = out / "input-holdout-plan.json"
    full_summary = out / "summary.json"
    if full_summary.exists():
        raise ValueError("The registered input holdout already has a full summary")
    if out.exists():
        if not state.is_file() or json.loads(state.read_text()) != plan:
            raise ValueError("The staged holdout requires its original code, model, environment and inputs")
    else:
        if phase == "finish":
            raise ValueError("Run the Classifieds phase before the cross-site phase")
        out.mkdir(parents=True)
        write_json(state, plan)
    classifieds_tasks = tuple(task_id for task_id in TASK_SETS[INPUT_HOLDOUT_TASK_SET]
                              if task_id not in INPUT_SHOPPING_TASKS)
    if phase == "classifieds":
        return classifieds_tasks
    finished_pairs = all((out / f"visualwebarena.{task_id}-{mode}" / "episode.json").is_file()
                         for task_id in classifieds_tasks for mode in STUDY_MODES["evidence-cards"])
    if not finished_pairs:
        raise ValueError("Complete all seven Classifieds pairs before the cross-site phase")
    return TASK_SETS[INPUT_HOLDOUT_TASK_SET]


def run_study(actor, *, out, study, task_set, max_steps, reference_archive=None, input_manifest=None,
              input_phase="all"):
    if task_set == HOLDOUT_TASK_SET and not holdout_profile(study, max_steps, actor.config):
        raise ValueError("The locked cohort requires the registered 8-decision Qwen 7B NF4 comparison")
    use_inputs = task_set == INPUT_HOLDOUT_TASK_SET
    if use_inputs and (reference_archive is None or input_manifest is None
                       or not holdout_profile(study, max_steps, actor.config, "evidence-cards")):
        raise ValueError("The new holdout requires its media archive and registered 8-decision comparison")
    staged = input_phase != "all"
    if staged:
        if not use_inputs or input_phase not in ("classifieds", "finish"):
            raise ValueError("Staged execution is only available for the registered image-and-tab holdout")
        task_ids = stage_input_holdout(out, input_phase, actor.config, input_manifest)
    else:
        out.mkdir(parents=True, exist_ok=False)
        task_ids = TASK_SETS[task_set]
    screen_reader = RapidScreenReader() if study in OCR_MODES else None
    listing_site = os.environ["VWA_CLASSIFIEDS"] if study in LINK_MODES else None
    strategies = {mode: [] for mode in STUDY_MODES[study]}
    new_pairs = INPUT_SHOPPING_TASKS if input_phase == "finish" else task_ids
    for task_id in task_ids:
        for mode in strategies:
            task = f"visualwebarena.{task_id}"
            branch = out / f"{task}-{mode}"
            if staged and branch.is_dir():
                row = json.loads((branch / "episode.json").read_text())
                if row.get("task") != task or row.get("seed") != 0 or row.get("reason") == "runtime_error":
                    raise ValueError(f"Inconsistent staged episode {branch}")
                strategies[mode].append(row)
                continue
            if task_id not in new_pairs:
                raise ValueError(f"Missing required staged episode {branch}")
            if study in ANSWER_STUDIES:
                recovery, invalid_retry = "scroll-feedback", "feedback"
            elif study == "format":
                recovery, invalid_retry = "none", mode
            else:
                recovery = mode
                invalid_retry = "feedback" if study == "scroll" else "none"
            env = VisualWebArenaEnvironment(task_id, enable_card_view=mode == "evidence-cards",
                                            enable_browser_context=use_inputs, reference_archive=reference_archive)
            try:
                pending = Path(mkdtemp(prefix=".episode-", dir=out)) / branch.name if staged else branch
                using_evidence = study in EVIDENCE_STUDIES and mode != "answer-check"
                policy = actor
                if using_evidence:
                    policy = EvidenceAnswerPolicy(
                        actor, confirm=mode in CONFIRM_MODES, use_navigation_context=mode in NAVIGATION_CONTEXT_MODES,
                        screen_reader=screen_reader, listing_site=listing_site,
                        listing_preview=env.preview_listing_links if listing_site is not None else None,
                        ranked_capture=env.capture_cards if mode == "evidence-cards" else None,
                    )
                row = run_episode(env, policy, env.task, 0, max_steps, pending,
                                  recovery=recovery, invalid_action_retry=invalid_retry,
                                  answer_check=mode == "answer-check",
                                  allow_observation_only=using_evidence,
                                  submit_answer=True, success_check=exact_vwa_success)
                render_replay(pending)
                if staged:
                    if row["reason"] == "runtime_error":
                        raise RuntimeError(f"Infrastructure failure in {task} {mode}; trace retained in {pending}")
                    pending.rename(branch)
                    pending.parent.rmdir()
                strategies[mode].append(row)
                progress = "recorded" if staged else f"{row['reason']} ({row['predictions']} predictions)"
                print(f"{env.task} {mode}: {progress}", flush=True)
            finally:
                env.close()
    if input_phase == "classifieds":
        print("Seven Classifieds pairs recorded. Finish the registered run when Shopping is ready.", flush=True)
        return
    counts = ("predictions", "model_calls", "steps", "invalid_predictions",
              "invalid_action_retries", "repaired_predictions")
    if study in ("scroll", *ANSWER_STUDIES):
        counts += ("scroll_feedbacks", "scroll_detection_signals")
    if study in ANSWER_STUDIES:
        counts += ("answer_checks", "answers_submitted", "answers_from_checks")
    summary = {"benchmark": "visualwebarena", "task_ids": task_ids,
        "task_set": task_set, "study": study, "budget_unit": "predictions",
        "max_steps": max_steps, "model": actor.config, "strategies": {
            mode: {**aggregate(rows), **{key: sum(row[key] for row in rows) for key in counts}}
            for mode, rows in strategies.items()}}
    if screen_reader is not None:
        summary["screen_reader"] = {**screen_reader.config, "guarded_ocr": True, "direct_navigation": True}
    if listing_site is not None:
        site = urlsplit(listing_site)
        summary["listing_origin"] = f"{site.scheme}://{site.netloc}"
    if task_set == HOLDOUT_TASK_SET:
        summary["protocol_sha256"] = HOLDOUT_PROTOCOL_SHA256
    if use_inputs:
        summary.update(protocol_sha256=INPUT_PROTOCOL_SHA256, reference_files_sha256=INPUT_MEDIA_SHA256,
                       browser_context=True, input_manifest=input_manifest)
    final_summary = out / "summary.json"
    if staged:
        pending_summary = Path(mkdtemp(prefix=".summary-", dir=out)) / final_summary.name
        write_json(pending_summary, summary)
        pending_summary.rename(final_summary)
        pending_summary.parent.rmdir()
    else:
        write_json(final_summary, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--actor", choices=("showui", "qwen"), default="showui")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--qwen-7b-4bit", action="store_true")
    parser.add_argument("--task-set", choices=TASK_SETS, default="diagnostic")
    parser.add_argument("--study", choices=STUDY_MODES, default="visual")
    parser.add_argument("--cohort-dir", type=Path, help="Registered input holdout protocol and reference archive")
    parser.add_argument("--input-phase", choices=("all", "classifieds", "finish"), default="all",
                        help="Record the seven single-site pairs first or finish the same run with Shopping")
    args = parser.parse_args()
    ready = live_preflight()
    if args.preflight:
        print(json.dumps(ready, indent=2))
        return 0 if ready["ready_for_connection_check"] else 2
    if not ready["ready_for_connection_check"]:
        parser.error(f"Run the backend preflight first: {json.dumps(ready)}")
    if args.model is None or args.out is None or not 1 <= args.max_steps <= 12:
        parser.error("Provide local model, fresh output directory and a prediction budget of 1–12")
    if args.study != "visual" and args.actor != "qwen":
        parser.error("This study uses Qwen")
    if args.task_set == HOLDOUT_TASK_SET and (args.study != "evidence-links" or args.max_steps != 8
                                               or not args.qwen_7b_4bit):
        parser.error("The locked cohort needs --study evidence-links --max-steps 8 --qwen-7b-4bit")
    if args.task_set == INPUT_HOLDOUT_TASK_SET and (args.cohort_dir is None or args.study != "evidence-cards"
                                                    or args.max_steps != 8 or not args.qwen_7b_4bit):
        parser.error("The input holdout needs --cohort-dir, --study evidence-cards, --max-steps 8, and --qwen-7b-4bit")
    if args.input_phase != "all" and args.task_set != INPUT_HOLDOUT_TASK_SET:
        parser.error("--input-phase is only used by the registered image-and-tab holdout")
    if args.qwen_7b_4bit and (args.actor != "qwen" or args.device != "cuda"):
        parser.error("The Qwen 7B 4-bit profile requires --actor qwen --device cuda")
    probe_instance()
    reference_archive, input_manifest = (input_holdout_archive(args.cohort_dir)
                                         if args.task_set == INPUT_HOLDOUT_TASK_SET else (None, None))
    if args.task_set == INPUT_HOLDOUT_TASK_SET and args.input_phase != "classifieds":
        probe_shopping()
    if args.out.exists() and args.input_phase == "all":
        parser.error("Use a fresh output directory")
    if args.actor == "showui":
        if args.device != "cuda":
            parser.error("ShowUI requires the CUDA profile")
        actor = ShowUIPolicy(str(args.model), revision=MODEL_REVISION, max_new_tokens=96)
    else:
        actor = QwenGroundingPolicy(args.model, device=args.device,
                                    max_new_tokens=128 if args.qwen_7b_4bit else 96, max_pixels=784 * 784,
                                    load_in_4bit=args.qwen_7b_4bit,
                                    native_navigation=args.study in CONFIRM_MODES)
    run_study(actor, out=args.out, study=args.study, task_set=args.task_set, max_steps=args.max_steps,
              reference_archive=reference_archive, input_manifest=input_manifest, input_phase=args.input_phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
