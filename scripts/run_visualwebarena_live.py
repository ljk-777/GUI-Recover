"""Optional native BrowserGym adapter for a self-hosted VisualWebArena instance."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import urlopen

from gui_recover.actions import Action, compile_action, parse_action
from gui_recover.answer_evidence import EvidenceAnswerPolicy
from gui_recover.environment import MiniwobEnvironment
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.qwen_grounding import QwenGroundingPolicy
from gui_recover.runner import aggregate, render_replay, run_episode, write_json
from gui_recover.screen_text import RapidScreenReader


BACKEND_VARIABLES = ("VWA_CLASSIFIEDS", "VWA_SHOPPING", "VWA_REDDIT", "VWA_WIKIPEDIA",
                     "VWA_HOMEPAGE", "VWA_CLASSIFIEDS_RESET_TOKEN")
READ_ONLY_TASKS = (118, 120, 149)
TRANSFER_TASKS = (11, 12, 22, 49, 119, 128, 148, 186, 221, 222)
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
}
EVIDENCE_STUDIES = ("evidence", "evidence-confirm", "evidence-context", "evidence-ocr")
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


def probe_instance():
    try:
        with urlopen(os.environ["VWA_CLASSIFIEDS"], timeout=8) as response:
            if response.status != 200:
                raise RuntimeError("Classifieds did not return HTTP 200")
    except OSError:
        raise RuntimeError("The configured VisualWebArena Classifieds backend is unreachable") from None


def exact_vwa_success(reward, terminated, truncated, _task_info):
    return bool(terminated and not truncated and reward == 1.0)


class VisualWebArenaEnvironment(MiniwobEnvironment):
    """Use BrowserGym's screenshot and action contract with native VWA answers."""

    def __init__(self, task_id):
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
                for config in self.task_configs:
                    evaluation = config["eval"]
                    if (config["sites"] != ["classifieds"] or config.get("require_reset")
                            or evaluation["eval_types"] != ["string_match"]
                            or set(evaluation["reference_answers"]) != {"must_include"}):
                        raise ValueError("Task requires more than the registered local string evaluator")

                site = urlsplit(self.webarena_instance.urls["classifieds"])
                origin = site.scheme, site.netloc

                def restrict_origin(route):
                    requested = urlsplit(route.request.url)
                    if (requested.scheme, requested.netloc) == origin:
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
                           return_value=unavailable_captioner):
                    return super().setup(page)

        self.task = f"visualwebarena.{task_id}"
        self.visual_variant = "default"
        self.obs = None
        actions = HighLevelActionSet(subsets=["coord", "chat"], strict=True, multiaction=True)
        self.env = BrowserEnv(StringOnlyTask, task_kwargs={"task_id": task_id}, headless=True,
                              action_mapping=actions.to_python_code)

    def step(self, action: Action):
        checked = parse_action(json.dumps(action.to_dict()))
        if checked.action == "ANSWER":
            code = f"send_msg_to_user({checked.value!r})"
        else:
            code = compile_action(checked, *self.obs.screenshot.size)
        return self._step_code(code)


def run_study(actor, *, out, study, task_set, max_steps):
    out.mkdir(parents=True, exist_ok=False)
    screen_reader = RapidScreenReader() if study == "evidence-ocr" else None
    task_ids = TASK_SETS[task_set]
    strategies = {mode: [] for mode in STUDY_MODES[study]}
    for task_id in task_ids:
        for mode in strategies:
            if study in ANSWER_STUDIES:
                recovery, invalid_retry = "scroll-feedback", "feedback"
            elif study == "format":
                recovery, invalid_retry = "none", mode
            else:
                recovery = mode
                invalid_retry = "feedback" if study == "scroll" else "none"
            env = VisualWebArenaEnvironment(task_id)
            try:
                branch = out / f"{env.task}-{mode}"
                using_evidence = study in EVIDENCE_STUDIES and mode != "answer-check"
                policy = actor
                if using_evidence:
                    policy = EvidenceAnswerPolicy(actor, confirm=mode != "evidence-check",
                                                  use_navigation_context=mode in ("evidence-context", "evidence-ocr"),
                                                  screen_reader=screen_reader)
                row = run_episode(env, policy, env.task, 0, max_steps, branch,
                                  recovery=recovery, invalid_action_retry=invalid_retry,
                                  answer_check=mode == "answer-check",
                                  allow_observation_only=using_evidence,
                                  submit_answer=True, success_check=exact_vwa_success)
                render_replay(branch)
                strategies[mode].append(row)
                print(f"{env.task} {mode}: {row['reason']} ({row['predictions']} predictions)", flush=True)
            finally:
                env.close()
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
        summary["screen_reader"] = screen_reader.config
    write_json(out / "summary.json", summary)


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
    if args.qwen_7b_4bit and (args.actor != "qwen" or args.device != "cuda"):
        parser.error("The Qwen 7B 4-bit profile requires --actor qwen --device cuda")
    probe_instance()
    if args.out.exists():
        parser.error("Use a fresh output directory")
    if args.actor == "showui":
        if args.device != "cuda":
            parser.error("ShowUI requires the CUDA profile")
        actor = ShowUIPolicy(str(args.model), revision=MODEL_REVISION, max_new_tokens=96)
    else:
        actor = QwenGroundingPolicy(args.model, device=args.device,
                                    max_new_tokens=128 if args.qwen_7b_4bit else 96, max_pixels=784 * 784,
                                    load_in_4bit=args.qwen_7b_4bit,
                                    native_navigation=args.study in ("evidence-confirm", "evidence-context", "evidence-ocr"))
    run_study(actor, out=args.out, study=args.study, task_set=args.task_set, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
