import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

from . import __version__

from .environment import MiniwobEnvironment
from .policies import MODEL_ID, MODEL_REVISION, OracleSmokePolicy, ShowUIPolicy
from .runner import aggregate, render_replay, run_episode, write_json
from .comparison import compare_runs, compare_precheck_runs
from .grounding import ScriptedGroundingPolicy


def fingerprint():
    packages = {}
    for name in ("gui-recover", "browsergym-core", "browsergym-miniwob", "gymnasium", "playwright", "torch", "transformers", "pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        repo = str(Path(__file__).resolve().parents[2])
        commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    miniwob_commit = None
    source = urlparse(os.getenv("MINIWOB_URL", ""))
    if source.scheme == "file":
        try:
            miniwob_commit = subprocess.check_output(["git", "-C", unquote(source.path), "rev-parse", "HEAD"],
                                                      text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "git_commit": commit, "git_dirty": dirty, "miniwob_url": os.getenv("MINIWOB_URL"),
            "miniwob_commit": miniwob_commit,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}


def main(argv=None, backend=None):
    parser = argparse.ArgumentParser(prog="gui-recover", description="GUI-Recover visual GUI evaluation and recovery")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--cuda", action="store_true", help="Probe CUDA only when explicitly requested")
    replay = sub.add_parser("replay")
    replay.add_argument("directory")
    compare = sub.add_parser("compare")
    compare.add_argument("--clean", required=True)
    compare.add_argument("--faulted", required=True)
    compare.add_argument("--recovered", required=True)
    compare.add_argument("--clean-recovered")
    compare.add_argument("--out", required=True)
    precheck = sub.add_parser("compare-precheck")
    precheck.add_argument("--baseline", required=True)
    precheck.add_argument("--checked", required=True)
    precheck.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--policy", choices=["oracle-smoke", "showui", "showui-scripted"], required=True)
    run.add_argument("--tasks", nargs="+", default=["click-test"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--out", required=True)
    run.add_argument("--model", default=MODEL_ID)
    run.add_argument("--revision", default=MODEL_REVISION)
    run.add_argument("--prompt-profile", choices=["baseline", "official-web", "official-miniwob"], default="baseline")
    run.add_argument("--parser", choices=["strict", "mixed"], default="strict")
    run.add_argument("--fault", choices=["none", "drop-click", "offset-click", "drop-input", "partial-input", "wrong-input"], default="none")
    run.add_argument("--fault-seed", type=int, default=0)
    run.add_argument("--recovery", choices=["none", "feedback", "input-guard", "input-retry"], default="none")
    run.add_argument("--input-guard-version", choices=["v1", "v2", "v3"], default="v2")
    run.add_argument("--input-precheck", choices=["off", "visual"], default="off")
    run.add_argument("--visual-variant", choices=["default", "shifted", "dark"], default="default")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        data = fingerprint()
        if args.cuda:
            import torch
            data["cuda_available"] = torch.cuda.is_available()
            if data["cuda_available"]:
                prop = torch.cuda.get_device_properties(0)
                data["gpu"] = {"name": prop.name, "total_bytes": prop.total_memory,
                               "cuda_runtime": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported()}
        print(json.dumps(data, indent=2))
        return
    if args.command == "replay":
        render_replay(args.directory)
        return
    if args.command == "compare":
        result = compare_runs(args.clean, args.faulted, args.recovered, args.clean_recovered)
        if Path(args.out).exists():
            parser.error("Comparison output already exists")
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "compare-precheck":
        if Path(args.out).exists():
            parser.error("Comparison output already exists")
        result = compare_precheck_runs(args.baseline, args.checked)
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return
    if args.max_steps < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("max-steps must be positive and seeds nonnegative")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate tasks or seeds are not allowed")
    if any(not task or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in task) for task in args.tasks):
        parser.error("Invalid MiniWoB task name")
    if args.policy == "oracle-smoke" and args.tasks != ["click-test"]:
        parser.error("oracle-smoke supports only click-test")
    if args.policy != "showui" and args.recovery == "feedback":
        parser.error("Feedback recovery requires a model policy")
    if args.recovery in ("input-guard", "input-retry") and args.policy != "showui-scripted":
        parser.error("Input guard requires the explicitly scripted planner")
    if args.policy == "showui-scripted" and any(t not in ("click-test", "click-button", "enter-text") for t in args.tasks):
        parser.error("Scripted planner supports click-test, click-button and enter-text only")
    if args.visual_variant != "default" and args.tasks != ["enter-text"]:
        parser.error("Visual fixtures require tasks=[enter-text]")
    if args.input_precheck != "off" and (args.policy != "showui-scripted" or args.tasks != ["enter-text"]):
        parser.error("Input precheck requires showui-scripted and tasks=[enter-text]")
    if args.input_precheck != "off" and args.recovery == "input-retry":
        parser.error("Input precheck cannot be combined with blind input retry")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "policy": args.policy, "diagnostic_only": args.policy == "oracle-smoke",
                "scripted_planner": args.policy == "showui-scripted",
                "config": vars(args), "runtime": fingerprint(),
                "success_rule": "terminated and not truncated and reward==1 and DONE_GLOBAL and RAW_REWARD_GLOBAL>0"}
    write_json(directory / "manifest.json", manifest)
    try:
        if backend is not None:
            if args.policy == "oracle-smoke":
                raise ValueError("Cannot reuse a VLM backend for the diagnostic oracle")
            for key, value in (("model", args.model), ("revision", args.revision), ("prompt_profile", args.prompt_profile)):
                if backend.config.get(key) != value:
                    raise ValueError(f"Reused backend has different {key}")
            model = backend
        else:
            model = ShowUIPolicy(model=args.model, revision=args.revision,
                                 prompt_profile=args.prompt_profile) if args.policy != "oracle-smoke" else None
        manifest["reused_model_backend"] = backend is not None
        if args.policy == "showui-scripted":
            model = ScriptedGroundingPolicy(model, input_guard=args.recovery == "input-guard",
                                            blind_input_retry=args.recovery == "input-retry",
                                            input_guard_version=args.input_guard_version,
                                            input_precheck=args.input_precheck == "visual")
    except Exception as exc:
        write_json(directory / "startup_error.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    if model is not None:
        manifest["model_config"] = model.config
        write_json(directory / "manifest.json", manifest)
    results = []
    for task in args.tasks:
        for seed in args.seeds:
            env = None
            try:
                env = MiniwobEnvironment(task, visual_variant=args.visual_variant)
                policy = model if model is not None else OracleSmokePolicy(env)
                result = run_episode(env, policy, task, seed, args.max_steps,
                                     directory / f"{task}-{seed}", parser_mode=args.parser,
                                     fault_kind=args.fault, fault_seed=args.fault_seed,
                                     recovery=args.recovery)
                render_replay(directory / f"{task}-{seed}")
            except Exception as exc:
                result = {"task": task, "seed": seed, "success": False, "reason": "setup_error",
                          "steps": 0, "error": str(exc)}
                write_json(directory / f"{task}-{seed}-error.json", result)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        result["close_error"] = str(exc)
            results.append(result)
            write_json(directory / "summary.json", {"policy": args.policy, **aggregate(results), "results": results})
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if any(r["reason"] in ("runtime_error", "setup_error") or r.get("close_error") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
