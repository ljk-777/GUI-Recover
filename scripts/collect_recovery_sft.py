"""Mine actual frozen-base grounding failures before any new adapter training."""

import argparse
import json
from pathlib import Path
import shutil

from gui_recover.actions import Action, ActionError
from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.grounding import parse_point, point_hits_boxes
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.runner import is_success, write_json
from gui_recover.sft import audit_dataset, file_sha256
from gui_recover.sft_recovery import audit_corrections, center, select_online_cases, visible_target_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    runtime = fingerprint()
    if runtime["git_dirty"] is not False:
        raise ValueError("Freeze a clean mining protocol first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"images").mkdir()
    (out/"traces").mkdir()
    shutil.copyfile("vendor/miniwob-plusplus/LICENSE", out/"MINIWOB_LICENSE.txt")
    model = ShowUIPolicy(args.model, revision=MODEL_REVISION, max_new_tokens=32)
    manifest = {"config": config, "runtime": runtime, "model_config": model.config,
        "teacher": "Offline DOM labels and failure selection only; no reward/DOM labels in prompts",
        "failure_selection": "executed first proposal misses target and fails actual first-action postcondition",
        "filtered": [], "source_cohort": [], "corrections": {}}
    all_rows = {}

    def frame(obs, name):
        image = f"images/{name}.png"
        obs.screenshot.save(out/image)
        return {"image": image, "sha256": file_sha256(out/image)}

    def record(split, task, seed, kind, picture, query, boxes, previous=None, **extra):
        point = center(boxes[0])
        return {"id": f"{task}-{seed}-{kind}", "split": split, "task": task, "seed": seed,
            "kind": kind, "query": query, "image": picture["image"], "image_sha256": picture["sha256"],
            "target_boxes": boxes, "target_point": point, "answer": json.dumps(point,separators=(",", ":")),
            "previous_action": previous, **extra}

    for split, groups in config["splits"].items():
        rows = []
        for group in groups:
            task = group["task"]
            for seed in group["seeds"]:
                env = MiniwobEnvironment(task)
                name = f"{task}-{seed}"
                try:
                    obs, _ = env.reset(seed)
                    initial = frame(obs, name+"-initial")
                    query, boxes, kind, value = visible_target_boxes(env, obs)
                    if not boxes:
                        manifest["filtered"].append({"task":task,"seed":seed,"split":split,
                            "reason":"no_visible_initial_target_at_least_4px"})
                        continue
                    rows.append(record(split,task,seed,"normal_entry",initial,query,boxes))
                    model.reset(seed)
                    raw = model.ground(obs.screenshot, query)
                    trace = {"task":task,"seed":seed,"split":split,"goal":obs.goal,"query":query,
                        "initial":initial,"initial_target_boxes":boxes,"first_raw":raw,
                        "executed":False,"first_action_completed":False}
                    proposed = None
                    correction_row = None
                    try:
                        point = parse_point(raw)
                        proposed = Action(kind,value,point)
                    except ActionError:
                        trace["reason"] = "invalid_source_output"
                    if proposed is not None:
                        after,reward,terminated,truncated,info,code = env.step(proposed)
                        trace.update({"proposed_action":proposed.to_dict(),"executed_action":proposed.to_dict(),
                            "executed":True,"reward":reward,"terminated":terminated,"truncated":truncated,
                            "task_info":info.get("task_info",{}),"execution_error":after.last_action_error,
                            "executed_code":code,"point_hit":point_hits_boxes(point,boxes),
                            "after_action":frame(after,name+"-after")})
                        if task == "enter-text":
                            completed = env.env.unwrapped.page.locator("#tt").input_value() == value
                            trace["postcondition"] = "input value equals goal value; label only"
                        else:
                            completed = is_success(reward,terminated,truncated,info.get("task_info",{}))
                            trace["postcondition"] = "environment success"
                        trace["first_action_completed"] = completed
                        if not completed and not trace["point_hit"]:
                            if terminated or truncated:
                                current,_ = env.reset(seed)
                                current_frame = frame(current,name+"-restart")
                                if current_frame["sha256"] != initial["sha256"]:
                                    raise ValueError("Same-seed restart differs from source initial screenshot")
                                mode = "restart_after_terminal"
                            else:
                                current, current_frame = after, trace["after_action"]
                                mode = "retry_in_place"
                            trace["current"] = current_frame
                            trace["recovery_mode"] = mode
                            next_query,next_boxes,_,_ = visible_target_boxes(env,current)
                            if not next_boxes:
                                trace["reason"] = "no_visible_target_after_failure"
                            else:
                                trace_path = f"traces/{name}.json"
                                correction_row=record(split,task,seed,"correction",current_frame,next_query,next_boxes,
                                    proposed.to_dict(),recovery_mode=mode,source_trace=trace_path)
                                rows.append(correction_row)
                                trace["reason"] = "executed_grounding_miss"
                        else:
                            trace["reason"] = "completed" if completed else "on_target_execution_failure"
                    write_json(out/f"traces/{name}.json", trace)
                    if correction_row is not None:
                        correction_row["source_trace_sha256"]=file_sha256(out/f"traces/{name}.json")
                    manifest["source_cohort"].append({"task":task,"seed":seed,"split":split,
                        "executed":trace["executed"],"completed":trace["first_action_completed"],
                        "reason":trace["reason"]})
                    # Positive action history: Submit after a real correct input.
                    # A teacher input is clearly labelled when the base input failed.
                    if task == "enter-text":
                        teacher_prefix = not trace["first_action_completed"]
                        if teacher_prefix:
                            obs,_ = env.reset(seed)
                            _,input_boxes,_,value = visible_target_boxes(env,obs)
                            good = Action("INPUT",value,tuple(center(input_boxes[0])))
                            after,_,_,_,_,_ = env.step(good)
                            previous = good.to_dict()
                            picture = frame(after,name+"-teacher-input")
                        else:
                            previous = proposed.to_dict()
                            picture = trace["after_action"]
                        if env.env.unwrapped.page.locator("#tt").input_value() != value:
                            raise ValueError("Teacher normal prefix did not enter the goal value")
                        submit_query,submit_boxes,_,_ = visible_target_boxes(env,after,"submit")
                        if submit_boxes:
                            rows.append(record(split,task,seed,"normal_submit",picture,submit_query,submit_boxes,
                                               previous,teacher_prefix=teacher_prefix))
                finally:
                    env.close()
                print(split,name,trace["reason"],flush=True)
        (out/f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        all_rows[split] = rows
        manifest["corrections"][split] = audit_corrections(out,rows)
        write_json(out/"progress.json", {s:{"rows":len(r),"corrections":sum(x["kind"]=="correction" for x in r)} for s,r in all_rows.items()})
    manifest["audit"] = audit_dataset(out)
    if sum(r["kind"]=="correction" for r in all_rows["train"]) < config["minimum_train_corrections"]:
        raise ValueError("Not enough genuine train corrections for the frozen pilot")
    manifest["online_case_ids"] = select_online_cases([r for r in all_rows["test"] if r["kind"]=="correction"],
                                                       config["online_correction_limit"])
    manifest["split_sha256"] = {s:file_sha256(out/f"{s}.jsonl") for s in all_rows}
    write_json(out/"manifest.json",manifest)
    print(json.dumps({"audit":manifest["audit"],"corrections":manifest["corrections"],
                      "online_cases":manifest["online_case_ids"]},indent=2))


if __name__ == "__main__":
    main()
