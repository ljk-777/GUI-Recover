"""Six-way prompt/training ablation plus bounded real GUI correction replay."""

import argparse
import json
from pathlib import Path
import re

from PIL import Image

from gui_recover.actions import Action, parse_action
from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.grounding import point_hits_boxes
from gui_recover.runner import is_success, render_replay, write_json
from gui_recover.sft import audit_dataset, file_sha256, grounding_query, load_records
from gui_recover.sft_ablation import AblationGrounder, DEFAULT_HEADS, HEADS
from gui_recover.sft_recovery import audit_corrections


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
                "terminated":done,"truncated":truncated,"obs":obs})
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
        if prediction["point"] is None:
            result["reason"]="invalid_prediction"
            return result
        action=task_action(row["task"],obs.goal,prediction["point"])
        emit({"event":"candidate_prediction","raw":prediction["raw"],"action":action.to_dict(),
              "input_query":prediction["input_query"]})
        obs,reward,done,truncated,info,code=env.step(action)
        result["repair_steps"]+=1
        emit({"event":"candidate_action","action":action.to_dict(),"reward":reward,
              "terminated":done,"truncated":truncated,"obs":obs})
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
                  "terminated":done,"truncated":truncated,"obs":obs})
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--visual-adapter",required=True)
    parser.add_argument("--history-adapter",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    data,out=Path(args.data),Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    audit=audit_dataset(data)
    rows=load_records(data,"test")
    corrections=audit_corrections(data,rows)
    source=json.loads((data/"manifest.json").read_text())
    runtime=fingerprint()
    if runtime["git_dirty"] is not False: raise ValueError("Freeze clean evaluation code")
    model=AblationGrounder(args.model,data,args.visual_adapter,args.history_adapter)
    write_json(out/"manifest.json",{"runtime":runtime,"data_audit":audit,"corrections":corrections,
        "dataset_manifest_sha256":file_sha256(data/"manifest.json"),"training_code":model.training[0]["runtime"]["git_commit"],
        "design":"same BF16 base and budgets; six weight/context heads; reused identical normal prompts",
        "heads":HEADS,"online_defaults":DEFAULT_HEADS,"source_model":source["model_config"],
        "limitations":"oracle-selected real failures; conditional repair, at most one same-seed restart, not autonomous detection"})
    predictions=[]
    index={}
    for row in rows:
        picture=Image.open(data/row["image"]).convert("RGB")
        result={k:row[k] for k in ("id","task","seed","kind","image_sha256")}
        result["recovery_mode"]=row.get("recovery_mode")
        result["previous_action"]=row.get("previous_action")
        result["target_boxes"]=row["target_boxes"]
        result["copy_previous_point_hit"]=point_hits_boxes((row.get("previous_action") or {}).get("position"),row["target_boxes"])
        result["heads"]={}
        cache={}
        for head,(weights,history) in HEADS.items():
            previous=row.get("previous_action") if history else None
            key=(weights,grounding_query(row["query"],previous))
            if key not in cache:
                cache[key]=model.predict(picture,row["query"],row.get("previous_action"),head)
            result["heads"][head]={**cache[key],"hit":point_hits_boxes(cache[key]["point"],row["target_boxes"])}
        predictions.append(result)
        index[row["id"]]=result
        with (out/"predictions.jsonl").open("a") as f: f.write(json.dumps(result)+"\n")
        print("grounded",row["id"],flush=True)
    def summary(items):
        return {"queries":len(items),"heads":{head:{"hits":sum(r["heads"][head]["hit"] for r in items),
            "invalid":sum(r["heads"][head]["point"] is None for r in items),
            "newly_hit_vs_base_plain":sum(not r["heads"]["base_plain"]["hit"] and r["heads"][head]["hit"] for r in items),
            "degraded_vs_base_plain":sum(r["heads"]["base_plain"]["hit"] and not r["heads"][head]["hit"] for r in items)} for head in HEADS},
            "paired_effects":[{"from":a,"to":b,
                "newly_hit":sum(not r["heads"][a]["hit"] and r["heads"][b]["hit"] for r in items),
                "degraded":sum(r["heads"][a]["hit"] and not r["heads"][b]["hit"] for r in items)}
                for a,b in (("base_plain","base_history"),("base_plain","visual_plain"),
                            ("visual_history","history_history"),("history_plain","history_history"))],
            "copy_previous_point_hits":sum(r["copy_previous_point_hit"] for r in items)}
    offline={"all":summary(predictions),"normal":summary([r for r in predictions if r["kind"]!="correction"]),
        "correction":summary([r for r in predictions if r["kind"]=="correction"]),
        "by_task":{task:summary([r for r in predictions if r["task"]==task]) for task in sorted({r["task"] for r in predictions})}}
    write_json(out/"offline-summary.json",offline)
    selected=set(source["online_case_ids"])
    normal_seeds=set(source["config"]["online_normal_seeds"])
    online=[]
    for row in rows:
        conditional=row["id"] in selected
        normal=row["kind"]=="normal_entry" and row["seed"] in normal_seeds
        if not conditional and not normal: continue
        for head in DEFAULT_HEADS:
            result=execute_case(model,head,row,index[row["id"]]["heads"][head],data,out,conditional)
            online.append(result)
            write_json(out/"online-results.json",online)
            print("executed",row["id"],head,result["reason"],flush=True)
    def online_summary(items):
        return {head:{"episodes":sum(r["head"]==head for r in items),
            "successes":sum(r["head"]==head and r["success"] for r in items),
            "prefix_mismatches":sum(r["head"]==head and not r["prefix_matched"] for r in items),
            "resets":sum(r["resets_after_failure"] for r in items if r["head"]==head)} for head in DEFAULT_HEADS}
    totals={"offline":offline,"online_normal":online_summary([r for r in online if not r["conditional"]]),
        "online_conditional":online_summary([r for r in online if r["conditional"]]),
        "actual_generation_calls_including_submit":model.calls,
        "peak_cuda_allocated_bytes":model.torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes":model.torch.cuda.max_memory_reserved()}
    write_json(out/"summary.json",totals)
    print(json.dumps(totals,indent=2))
    if any(r["reason"]=="runtime_error" for r in online): raise SystemExit(1)


if __name__=="__main__": main()
