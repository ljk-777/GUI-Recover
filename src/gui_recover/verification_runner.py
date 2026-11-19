"""Bounded recovery routing. Only the explicit oracle may call the teacher."""

import json
from pathlib import Path
import time

from .actions import Action,ActionError
from .grounding import parse_point
from .runner import is_success,render_replay,write_json
from .sft import file_sha256
from .verification import TransitionView,appearance_verdict,pixel_hash
from .verification_env import TransitionExecution,task_request,teacher_effect


class GroundingCache:
    def __init__(self,actor):
        self.actor=actor
        self.cache={}
        self.requests=self.calls=0

    def ground(self,picture,target):
        self.requests+=1
        key=(pixel_hash(picture),target)
        if key not in self.cache:
            raw=self.actor.ground(picture,target)
            try:
                point=parse_point(raw)
            except ActionError:
                point=None
            self.cache[key]={"raw":raw,"point":point}
            self.calls+=1
        return self.cache[key]


def run_verification_case(row,strategy,grounder,verifier,out):
    from .actions import parse_action
    directory=Path(out)/"online"/f"{row['id']}-{strategy}"
    directory.mkdir(parents=True,exist_ok=False)
    trajectory=directory/"trajectory.jsonl"
    result={"id":row["id"],"scene_id":row["scene_id"],"task":row["task"],"seed":row["seed"],
        "regime":row["regime"],"strategy":strategy,"success":False,"reason":"invalid_source",
        "prefix_matched":False,"waits":0,"repairs":0,"logical_actor_calls":1,
        "invalid_predictions":0,"source_prediction_reused":True,"verifier_calls":0,"verifier_seconds":0.0}
    execution=None
    emitted=0
    frames=0
    started=time.perf_counter()
    calls_before=grounder.calls
    def emit(event,obs=None):
        nonlocal frames
        if obs is not None:
            name=f"{frames:04d}.png"
            obs.screenshot.save(directory/name)
            event.update({"screenshot":name,"sha256":file_sha256(directory/name)})
            frames+=1
        with trajectory.open("a") as f:
            f.write(json.dumps(event)+"\n")
        return event.get("sha256")
    def events():
        nonlocal emitted
        while emitted<len(execution.events):
            emit(execution.events[emitted],execution.obs if emitted==len(execution.events)-1 else None)
            emitted+=1
    def decide(phase):
        view=TransitionView(execution.before.screenshot,execution.obs.screenshot,execution.proposed,
                            execution.obs.goal,target,phase)
        before=time.perf_counter()
        if strategy=="oracle":
            # Deliberately privileged diagnostic reference, isolated from all other paths.
            status="unknown" if execution.pending is not None else (
                "achieved" if teacher_effect(execution) else "not_achieved")
            value={"status":status,"privileged_oracle":True}
        elif strategy=="appearance":
            value=appearance_verdict(view)
        else:
            value=verifier.verify(view)
        result["verifier_calls"]+=1
        result["verifier_seconds"]+=time.perf_counter()-before
        emit({"event":"verification","phase":phase,"verdict":value})
        return value["status"]
    try:
        if row["status"]!="ok":
            result["reason"]=row["status"]
            return result
        proposed=parse_action(json.dumps(row["proposed_action"]))
        execution=TransitionExecution(row["task"],row["seed"],proposed,row["regime"])
        target,_,_=task_request(row["task"],execution.obs.goal)
        initial=emit({"event":"reset","goal":execution.obs.goal,"target":target},execution.obs)
        if initial!=row["before"]["sha256"]:
            result["reason"]="prefix_mismatch"
            return result
        execution.first()
        events()
        if file_sha256(directory/f"{frames-1:04d}.png")!=row["immediate"]["sha256"]:
            result["reason"]="prefix_mismatch"
            return result
        result["prefix_matched"]=True
        if not (execution.done or execution.truncated):
            if strategy=="none":
                trigger=False
            elif strategy in ("retry","wait_retry"):
                if strategy=="wait_retry":
                    execution.wait_once();result["waits"]+=1;events()
                trigger=True
            else:
                status=decide("immediate")
                if status=="unknown":
                    execution.wait_once();result["waits"]+=1;events()
                    status=decide("after_wait")
                trigger=status=="not_achieved"
            result["triggered_recovery"]=bool(trigger)
            if trigger and not (execution.done or execution.truncated):
                result["repairs"]+=1
                result["logical_actor_calls"]+=1
                prediction=grounder.ground(execution.obs.screenshot,target)
                emit({"event":"repair_prediction","prediction":prediction})
                if prediction["point"] is not None:
                    execution.repair(prediction["point"]);events()
                else:
                    result["invalid_predictions"]+=1
            # Fixture delivery is independent of verifier output and uses no teacher labels.
            execution.flush_pending();events()
            if row["task"]=="enter-text" and not (execution.done or execution.truncated):
                result["logical_actor_calls"]+=1
                prediction=grounder.ground(execution.obs.screenshot,"Submit button")
                emit({"event":"submit_prediction","prediction":prediction})
                if prediction["point"] is not None:
                    execution._step(Action("CLICK",position=tuple(prediction["point"])),"submit");events()
                else:
                    result["invalid_predictions"]+=1
        result["success"]=is_success(execution.reward,execution.done,execution.truncated,execution.info.get("task_info",{}))
        result["reason"]="success" if result["success"] else "task_not_completed"
        result.update({"reward":execution.reward,"terminated":execution.done,"truncated":execution.truncated,
                       "task_info":execution.info.get("task_info",{}),"environment_steps":len(execution.events)})
        assert result["logical_actor_calls"]<=3 and result["waits"]<=1 and result["repairs"]<=1
        return result
    except Exception as exc:
        result.update({"reason":"runtime_error","error":str(exc),"success":False})
        return result
    finally:
        if execution is not None:
            execution.close()
        result["new_actor_generations"]=grounder.calls-calls_before
        result["elapsed_seconds"]=time.perf_counter()-started
        emit({"event":"episode_end",**result})
        write_json(directory/"episode.json",result)
        render_replay(directory)
