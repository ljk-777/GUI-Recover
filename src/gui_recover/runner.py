import hashlib
import html
import json
import time
from collections import Counter
from pathlib import Path

from .actions import ActionError, parse_action
from .recovery import ActionFault, FeedbackRecovery, detect


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_success(reward, terminated, truncated, task_info):
    # Pinned BrowserGym MiniWoB uses RAW_REWARD_GLOBAL > 0 as its binary reward.
    # Do not infer success from text, stop requests, partial progress or timeouts.
    return bool(terminated and not truncated and reward == 1.0
                and task_info.get("DONE_GLOBAL")
                and task_info.get("RAW_REWARD_GLOBAL", 0) > 0)


def run_episode(env, policy, task, seed, max_steps, out_dir, parser_mode="strict",
                fault_kind="none", fault_seed=0, recovery="none", *,
                submit_answer=False, success_check=is_success):
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=False)
    trajectory = directory / "trajectory.jsonl"
    start = time.perf_counter()
    steps = 0
    predictions = 0
    model_calls = 0
    model_calls_known = True
    execution_errors = 0
    inference_seconds = 0.0
    success = terminated = truncated = False
    reward = 0.0
    task_info = {}
    reason = "environment_error"
    history = []
    metadata = {}
    normalized_outputs = 0
    injector = ActionFault(fault_kind, fault_seed)
    controller = FeedbackRecovery(enabled=recovery == "feedback")
    detection_signals = 0

    def emit(event):
        with trajectory.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def screenshot(obs, index):
        path = directory / f"{index:04d}.png"
        obs.screenshot.save(path)
        return {"screenshot": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": obs.screenshot.size}

    try:
        if hasattr(policy, "reset"):
            policy.reset(seed)
        obs, info = env.reset(seed)
        task_info = info.get("task_info", {})
        metadata = env.metadata()
        emit({"event": "reset", "task": task, "seed": seed, "goal": obs.goal,
              "task_info": task_info, **screenshot(obs, 0)})
        reason = "step_limit"
        for index in range(max_steps):
            before = time.perf_counter()
            raw = policy.predict(obs, history)
            latency = time.perf_counter() - before
            predictions += 1
            invocation_count = getattr(policy, "last_metrics", {}).get("model_invocations")
            if invocation_count is None:
                model_calls_known = False
            else:
                model_calls += invocation_count
            inference_seconds += latency
            emit({"event": "prediction", "step": index, "raw": raw,
                  "inference_seconds": latency, "metrics": getattr(policy, "last_metrics", {})})
            try:
                action = parse_action(raw, mode=parser_mode)
                if parser_mode == "mixed":
                    try:
                        parse_action(raw)
                    except ActionError:
                        normalized_outputs += 1
                        emit({"event": "format_normalized", "step": index,
                              "action": action.to_dict(), "parser": "mixed"})
            except ActionError as exc:
                reason = "invalid_action"
                emit({"event": "error", "step": index, "error": str(exc)})
                break
            if action.action == "ANSWER" and not submit_answer:
                reason = "policy_stop"
                emit({"event": "stop", "step": index, "action": action.to_dict()})
                break
            executed, fault = injector.apply(action)
            before_obs = obs
            obs, reward, terminated, truncated, info, code = env.step(executed)
            steps += 1
            signal = detect(before_obs, obs, action)
            detection_signals += signal.unchanged_after_click or bool(signal.execution_error)
            task_info = info.get("task_info", {})
            execution_errors += bool(obs.last_action_error)
            emit({"event": "transition", "step": index, "action": executed.to_dict(),
                  "proposed_action": action.to_dict(),
                  "executed_code": code, "reward": reward, "terminated": terminated,
                  "truncated": truncated, "task_info": task_info,
                  "execution_error": obs.last_action_error, **screenshot(obs, steps)})
            emit({"event": "diagnostic", "step": index, "proposed_action": action.to_dict(),
                  "executed_action": executed.to_dict(), "fault": fault,
                  "unchanged_after_click": signal.unchanged_after_click,
                  "mean_pixel_difference": signal.mean_pixel_difference})
            if hasattr(policy, "observe_transition"):
                observation = policy.observe_transition(before_obs, obs, action)
                if observation:
                    emit({"event": "policy_observation", "step": index, **observation})
            # Proposed action is what the agent believes it executed. Actual injected
            # actions and labels are audit data only, not passed into policy history.
            history.append({"action": action.to_dict(), "execution_error": obs.last_action_error})
            success = success_check(reward, terminated, truncated, task_info)
            if terminated or truncated:
                reason = "success" if success else ("truncated" if truncated else "task_failure")
                break
            feedback = controller.feedback(signal)
            if hasattr(policy, "set_feedback"):
                policy.set_feedback(feedback)
            if feedback:
                emit({"event": "recovery_feedback", "step": index, "message": feedback})
    except Exception as exc:
        reason = "runtime_error"
        emit({"event": "error", "error_type": type(exc).__name__, "error": str(exc)})
    result = {"task": task, "seed": seed, "success": success, "reason": reason,
              "steps": steps, "predictions": predictions, "execution_errors": execution_errors,
              "reward": reward, "terminated": terminated, "truncated": truncated,
              "task_info": task_info, "inference_seconds": inference_seconds,
              "elapsed_seconds": time.perf_counter() - start, "environment": metadata,
              "metrics": getattr(policy, "last_metrics", {})}
    result["normalized_outputs"] = normalized_outputs
    result["model_calls"] = model_calls if model_calls_known else None
    result["input_guard"] = getattr(policy, "guard_metrics", None)
    result.update({"fault_kind": fault_kind, "fault_seed": fault_seed,
                   "injected": injector.injected, "recovery": recovery,
                   "recovery_feedbacks": controller.used, "detection_signals": detection_signals})
    write_json(directory / "episode.json", result)
    emit({"event": "episode_end", **result})
    return result


def aggregate(results):
    def summary(rows):
        n = len(rows)
        return {"episodes": n, "successes": sum(r["success"] for r in rows),
                "success_rate": sum(r["success"] for r in rows) / n if n else 0.0,
                "mean_steps": sum(r["steps"] for r in rows) / n if n else 0.0,
                "reasons": dict(Counter(r["reason"] for r in rows))}
    return {**summary(results), "by_task": {
        task: summary([r for r in results if r["task"] == task]) for task in sorted({r["task"] for r in results})}}


def render_replay(directory):
    """Offline visual inspection only; not deterministic browser state replay."""
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    cards = []
    for event in events:
        picture = ""
        if "screenshot" in event:
            name = event["screenshot"]
            if Path(name).name != name:
                raise ValueError("Screenshot must be an episode-local filename")
            picture = f'<img src="{html.escape(name, quote=True)}" alt="environment screenshot">'
        cards.append(f'<section>{picture}<pre>{html.escape(json.dumps(event, indent=2, ensure_ascii=False))}</pre></section>')
    (directory / "replay.html").write_text('<!doctype html><meta charset="utf-8"><title>GUI-Recover trajectory</title>'
        '<style>body{font:15px system-ui;max-width:1000px;margin:30px auto}section{border-top:1px solid #ccc;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style>'
        '<h1>GUI-Recover trajectory</h1>' + "".join(cards), encoding="utf-8")
