import hashlib
import html
import json
import time
from collections import Counter
from pathlib import Path

from .actions import Action, ActionError, parse_action
from .environment import frame_key
from .recovery import ActionFault, FeedbackRecovery, detect
from .verification import pixel_hash


INVALID_ACTION_FEEDBACK = (
    "The previous response was invalid and no browser action ran. "
    "Use one JSON object with action, value, and position. "
    "CLICK and INPUT require two coordinates, with a null value for CLICK. "
    "For SCROLL use a direction and null position. "
    "If the answer is already visible, use ANSWER with the answer as value and null position. "
    "Use INPUT only to enter text in a visible field."
)


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
                submit_answer=False, success_check=is_success, invalid_action_retry="none",
                answer_check=False, allow_observation_only=False, max_model_calls=None):
    """Run up to max_steps predictions, including answer checks and action retries."""
    if invalid_action_retry not in ("none", "retry", "feedback"):
        raise ValueError("Unknown invalid-action retry mode")
    if invalid_action_retry == "feedback" and not callable(getattr(policy, "set_feedback", None)):
        raise ValueError("Invalid-action feedback requires a policy with set_feedback")
    if answer_check and (not submit_answer or not callable(getattr(policy, "check_answer", None))):
        raise ValueError("Answer checks require a policy with check_answer and native answer submission")
    if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls < 1):
        raise ValueError("A model call budget must be a positive integer")
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
    invalid_predictions = invalid_action_retries = repaired_predictions = 0
    answer_checks = answers_submitted = answers_from_checks = 0
    checked_frames = set()
    retrying_invalid = False
    injector = ActionFault(fault_kind, fault_seed)
    controller = FeedbackRecovery(enabled=recovery in ("feedback", "scroll-feedback"),
                                  scroll=recovery == "scroll-feedback")
    detection_signals = 0
    scroll_detection_signals = 0

    def emit(event):
        with trajectory.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def screenshot(obs, index):
        path = directory / f"{index:04d}.png"
        obs.screenshot.save(path)
        record = {"screenshot": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "size": obs.screenshot.size}
        if obs.tabs:
            record["browser_tabs"] = {"tabs": obs.tabs, "active_tab": obs.active_tab}
        return record

    try:
        if hasattr(policy, "reset"):
            policy.reset(seed)
        obs, info = env.reset(seed)
        task_info = info.get("task_info", {})
        metadata = env.metadata()
        frame = screenshot(obs, 0)
        references = []
        for index, image in enumerate(obs.reference_images, start=1):
            path = directory / f"reference-{index:02d}.png"
            image.save(path)
            references.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "pixels": pixel_hash(image), "size": image.size})
        if references:
            frame["reference_images"] = references
        emit({"event": "reset", "task": task, "seed": seed, "goal": obs.goal,
              "task_info": task_info, **frame})
        reason = "step_limit"
        for index in range(max_steps):
            if max_model_calls is not None and model_calls >= max_model_calls:
                reason = "model_call_limit"
                break
            before = time.perf_counter()
            frame_digest = frame["sha256"]
            key = frame_key(frame_digest, obs.tabs, obs.active_tab)
            checking_answer = answer_check and key not in checked_frames
            if checking_answer:
                checked_frames.add(key)
                answer = policy.check_answer(obs)
                raw = None if answer is None else json.dumps(Action("ANSWER", answer).to_dict(), ensure_ascii=False)
            else:
                raw = policy.predict(obs, history)
            latency = time.perf_counter() - before
            predictions += 1
            invocation_count = getattr(policy, "last_metrics", {}).get("model_invocations")
            if max_model_calls is not None and (type(invocation_count) is not int or invocation_count not in (0, 1)):
                raise RuntimeError("A budgeted policy must report zero or one model invocation per prediction")
            if invocation_count is None:
                model_calls_known = False
            else:
                model_calls += invocation_count
            inference_seconds += latency
            prediction = {"event": "prediction", "step": index, "raw": raw,
                          "inference_seconds": latency, "metrics": getattr(policy, "last_metrics", {})}
            if checking_answer:
                prediction.update(decision="answer_check", observation_sha256=frame_digest)
            evidence_stage = prediction["metrics"].get("evidence_stage") if allow_observation_only else None
            answer_checks += checking_answer or evidence_stage == "locate"
            emit(prediction)
            if allow_observation_only and raw is None and prediction["metrics"].get("stop_after_evidence") is True:
                reason = "evidence_abstained"
                break
            if raw is None and (checking_answer or allow_observation_only):
                retrying_invalid = False
                continue
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
                invalid_predictions += 1
                emit({"event": "error", "step": index, "error": str(exc)})
                if invalid_action_retry != "none" and not invalid_action_retries and index + 1 < max_steps:
                    feedback = INVALID_ACTION_FEEDBACK if invalid_action_retry == "feedback" else None
                    if feedback:
                        policy.set_feedback(feedback)
                    invalid_action_retries += 1
                    retrying_invalid = True
                    emit({"event": "invalid_action_retry", "step": index, "mode": invalid_action_retry,
                          "next_prediction": index + 1, "message": feedback})
                    continue
                reason = "invalid_action"
                break
            if retrying_invalid:
                repaired_predictions += 1
                retrying_invalid = False
                emit({"event": "invalid_action_repaired", "step": index, "action": action.to_dict()})
            if action.action == "ANSWER" and not submit_answer:
                reason = "policy_stop"
                emit({"event": "stop", "step": index, "action": action.to_dict()})
                break
            executed, fault = injector.apply(action)
            before_obs = obs
            obs, reward, terminated, truncated, info, code = env.step(executed)
            steps += 1
            if executed.action == "ANSWER":
                answers_submitted += 1
                answers_from_checks += checking_answer or evidence_stage in ("read", "confirm", "link_confirm", "ranked_cards")
            signal = detect(before_obs, obs, action)
            detection_signals += signal.unchanged_after_click or bool(signal.execution_error)
            scroll_detection_signals += signal.unchanged_after_scroll
            task_info = info.get("task_info", {})
            execution_errors += bool(obs.last_action_error)
            frame = screenshot(obs, steps)
            emit({"event": "transition", "step": index, "action": executed.to_dict(),
                  "proposed_action": action.to_dict(),
                  "executed_code": code, "reward": reward, "terminated": terminated,
                  "truncated": truncated, "task_info": task_info,
                  "execution_error": obs.last_action_error, **frame})
            emit({"event": "diagnostic", "step": index, "proposed_action": action.to_dict(),
                  "executed_action": executed.to_dict(), "fault": fault,
                  "unchanged_after_click": signal.unchanged_after_click,
                  "unchanged_after_scroll": signal.unchanged_after_scroll,
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
            previous_scroll_feedbacks = controller.scroll_used
            feedback = controller.feedback(signal) if index + 1 < max_steps else None
            if hasattr(policy, "set_feedback"):
                policy.set_feedback(feedback)
            if feedback:
                emit({"event": "recovery_feedback", "step": index, "message": feedback,
                      "scroll": controller.scroll_used > previous_scroll_feedbacks})
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
    result.update({"invalid_action_retry": invalid_action_retry, "invalid_predictions": invalid_predictions,
                   "invalid_action_retries": invalid_action_retries, "repaired_predictions": repaired_predictions})
    result.update({"answer_check": answer_check, "answer_checks": answer_checks,
                   "answers_submitted": answers_submitted, "answers_from_checks": answers_from_checks})
    result["model_calls"] = model_calls if model_calls_known else None
    result["input_guard"] = getattr(policy, "guard_metrics", None)
    result.update({"fault_kind": fault_kind, "fault_seed": fault_seed,
                   "injected": injector.injected, "recovery": recovery,
                   "recovery_feedbacks": controller.used, "detection_signals": detection_signals,
                   "scroll_feedbacks": controller.scroll_used, "scroll_detection_signals": scroll_detection_signals})
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
        for reference in event.get("reference_images", []):
            name = reference["filename"]
            if Path(name).name != name:
                raise ValueError("Reference must be an episode-local filename")
            picture += f'<img src="{html.escape(name, quote=True)}" alt="task reference image">'
        cards.append(f'<section>{picture}<pre>{html.escape(json.dumps(event, indent=2, ensure_ascii=False))}</pre></section>')
    (directory / "replay.html").write_text('<!doctype html><meta charset="utf-8"><title>GUI-Recover trajectory</title>'
        '<style>body{font:15px system-ui;max-width:1000px;margin:30px auto}section{border-top:1px solid #ccc;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style>'
        '<h1>GUI-Recover trajectory</h1>' + "".join(cards), encoding="utf-8")
