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
from gui_recover.environment import MiniwobEnvironment
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.qwen_grounding import QwenGroundingPolicy
from gui_recover.runner import aggregate, render_replay, run_episode, write_json


BACKEND_VARIABLES = ("VWA_CLASSIFIEDS", "VWA_SHOPPING", "VWA_REDDIT", "VWA_WIKIPEDIA",
                     "VWA_HOMEPAGE", "VWA_CLASSIFIEDS_RESET_TOKEN")
READ_ONLY_TASKS = (118, 120, 149)


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

        if task_id not in READ_ONLY_TASKS:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--actor", choices=("showui", "qwen"), default="showui")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    ready = live_preflight()
    if args.preflight:
        print(json.dumps(ready, indent=2))
        return 0 if ready["ready_for_connection_check"] else 2
    if not ready["ready_for_connection_check"]:
        parser.error(f"Run the backend preflight first: {json.dumps(ready)}")
    if args.model is None or args.out is None or not 1 <= args.max_steps <= 12:
        parser.error("Provide local model, fresh output directory and a step budget of 1–12")
    probe_instance()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.actor == "showui":
        if args.device != "cuda":
            parser.error("ShowUI requires the CUDA profile")
        actor = ShowUIPolicy(str(args.model), revision=MODEL_REVISION, max_new_tokens=96)
    else:
        actor = QwenGroundingPolicy(args.model, device=args.device, max_new_tokens=96, max_pixels=784 * 784)
    rows = []
    for task_id in READ_ONLY_TASKS:
        for mode in ("none", "feedback"):
            env = VisualWebArenaEnvironment(task_id)
            try:
                branch = args.out / f"{env.task}-{mode}"
                row = run_episode(env, actor, env.task, 0, args.max_steps, branch, recovery=mode,
                                  submit_answer=True, success_check=exact_vwa_success)
                render_replay(branch)
                rows.append(row)
            finally:
                env.close()
    write_json(args.out / "summary.json", {"benchmark": "visualwebarena", "task_ids": READ_ONLY_TASKS,
        "max_steps": args.max_steps, "model": actor.config,
        "strategies": {mode: aggregate([r for r in rows if r["recovery"] == mode]) for mode in ("none", "feedback")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
