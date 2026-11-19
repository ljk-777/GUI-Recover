"""Bounded GUI acceptance for cached grounding predictions, with full task evidence."""

import json
import re

from .actions import Action, parse_action
from .environment import MiniwobEnvironment
from .runner import is_success, render_replay, write_json
from .sft import file_sha256


def task_action(task,goal,point):
    if task == "enter-text":
        value = re.fullmatch(r'Enter "(.*)" into the text field and press Submit\.',goal).group(1)
        return Action("INPUT",value,tuple(point))
    return Action("CLICK",position=tuple(point))


def execute_case(model,head,row,prediction,data,out,conditional):
    directory = out/"online"/f"{row['id']}-{head}"
    directory.mkdir(parents=True,exist_ok=False)
    trajectory = directory/"trajectory.jsonl"
    frames = 0
    result = {"id":row["id"],"task":row["task"],"seed":row["seed"],"head":head,
        "conditional":conditional,"success":False,"reason":"runtime_error","source_steps":0,
        "repair_steps":0,"resets_after_failure":0,"source_model_calls_reused":int(conditional),
        "candidate_model_calls_reused":1,"additional_submit_calls":0,"prefix_matched":False,
        "total_logical_model_calls":int(conditional)+1}
    env = MiniwobEnvironment(row["task"])
    def emit(event):
        nonlocal frames
        if "obs" in event:
            obs = event.pop("obs")
            name=f"{frames:04d}.png"
            obs.screenshot.save(directory/name)
            event.update({"screenshot":name,"sha256":file_sha256(directory/name)})
            frames += 1
        with trajectory.open("a") as f: f.write(json.dumps(event)+"\n")
        return event.get("sha256")
    try:
        obs,_ = env.reset(row["seed"])
        initial_sha = emit({"event":"reset","goal":obs.goal,"obs":obs})
        if conditional:
            trace=json.loads((data/row["source_trace"]).read_text())
            if initial_sha != trace["initial"]["sha256"]:
                result["reason"]="mismatched_initial_state"
                return result
            source=parse_action(json.dumps(trace["proposed_action"]))
            obs,reward,done,truncated,info,code=env.step(source)
            result["source_steps"]=1
            after_sha=emit({"event":"source_action","action":source.to_dict(),"reward":reward,
                "terminated":done,"truncated":truncated,"task_info":info.get("task_info",{}),
                "executed_code":code,"execution_error":obs.last_action_error,"obs":obs})
            if (after_sha != trace["after_action"]["sha256"] or done != trace["terminated"]
                    or truncated != trace["truncated"] or reward != trace["reward"]):
                result["reason"]="mismatched_source_transition"
                return result
            if done or truncated:
                obs,_=env.reset(row["seed"])
                current_sha=emit({"event":"restart_after_terminal","obs":obs})
                result["resets_after_failure"]=1
            else:
                current_sha=after_sha
            if current_sha != row["image_sha256"]:
                result["reason"]="mismatched_correction_state"
                return result
        elif initial_sha != row["image_sha256"]:
            result["reason"]="mismatched_initial_state"
            return result
        result["prefix_matched"]=True
        result["recovery_mode"]=row.get("recovery_mode")
        emit({"event":"candidate_prediction", "prediction":prediction})
        if prediction["point"] is None:
            result["reason"]="invalid_prediction"
            return result
        action=task_action(row["task"],obs.goal,prediction["point"])
        obs,reward,done,truncated,info,code=env.step(action)
        result["repair_steps"]+=1
        emit({"event":"candidate_action","action":action.to_dict(),"reward":reward,
              "terminated":done,"truncated":truncated,"task_info":info.get("task_info",{}),
                "executed_code":code,"execution_error":obs.last_action_error,"obs":obs})
        if action.action == "INPUT" and not done and not truncated:
            previous=action.to_dict()
            result["additional_submit_calls"]=1
            result["total_logical_model_calls"]+=1
            submit=model.predict(obs.screenshot,"Submit button",previous,head)
            emit({"event":"submit_prediction","prediction":submit})
            if submit["point"] is None:
                result["reason"]="invalid_submit_prediction"
                return result
            click=Action("CLICK",position=tuple(submit["point"]))
            obs,reward,done,truncated,info,code=env.step(click)
            result["repair_steps"]+=1
            emit({"event":"submit_action","action":click.to_dict(),"reward":reward,
                  "terminated":done,"truncated":truncated,"task_info":info.get("task_info",{}),
                "executed_code":code,"execution_error":obs.last_action_error,"obs":obs})
        result["success"]=is_success(reward,done,truncated,info.get("task_info",{}))
        result["reason"]="success" if result["success"] else "task_not_completed"
        result["total_logical_model_calls"]=int(conditional)+1+result["additional_submit_calls"]
        assert result["total_logical_model_calls"]<=4
        return result
    except Exception as exc:
        result["error"]=str(exc)
        return result
    finally:
        env.close()
        emit({"event":"episode_end",**result})
        write_json(directory/"episode.json",result)
        render_replay(directory)
