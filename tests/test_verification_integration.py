import hashlib
import os

import pytest

from gui_recover.actions import Action
from gui_recover.sft_recovery import center,visible_target_boxes
from gui_recover.verification_env import TransitionExecution,task_request,teacher_effect


pytestmark=[pytest.mark.integration,pytest.mark.skipif(not os.getenv("MINIWOB_URL"),reason="Set local MiniWoB URL")]


def prepared(regime,seed=4000):
    execution=TransitionExecution("enter-text",seed,None,regime)
    _,boxes,_,_=visible_target_boxes(execution.env,execution.obs)
    _,_,value=task_request("enter-text",execution.obs.goal)
    execution.proposed=Action("INPUT",value,tuple(center(boxes[0])))
    return execution


def test_replacement_repairs_wrong_content_without_appending():
    execution=prepared("wrong-input")
    try:
        execution.first()
        assert not teacher_effect(execution)
        execution.repair(execution.proposed.position)
        assert teacher_effect(execution)
    finally:
        execution.close()


def test_queued_delivery_distinguishes_wait_from_premature_retry():
    execution=prepared("deferred-input")
    try:
        execution.first()
        assert not teacher_effect(execution)
        execution.repair(execution.proposed.position)
        execution.flush_pending()
        assert not teacher_effect(execution)
    finally:
        execution.close()
    execution=prepared("deferred-input")
    try:
        execution.first()
        execution.wait_once()
        assert teacher_effect(execution)
        execution.flush_pending()
        assert teacher_effect(execution)
    finally:
        execution.close()


def test_visual_runtime_never_calls_teacher_and_completes_real_recovery(tmp_path,monkeypatch):
    import gui_recover.verification_runner as runner
    execution=prepared("drop-input")
    try:
        initial=tmp_path/"before.png";execution.obs.screenshot.save(initial)
        execution.first()
        after=tmp_path/"after.png";execution.obs.screenshot.save(after)
        row={"id":"diagnostic","scene_id":"engineering-only","task":"enter-text","seed":4000,
             "regime":"drop-input","status":"ok","proposed_action":execution.proposed.to_dict(),
             "before":{"sha256":hashlib.sha256(initial.read_bytes()).hexdigest()},
             "immediate":{"sha256":hashlib.sha256(after.read_bytes()).hexdigest()}}
    finally:
        execution.close()
    active=[]
    original=runner.TransitionExecution
    def factory(*args):
        obj=original(*args);active.append(obj);return obj
    monkeypatch.setattr(runner,"TransitionExecution",factory)
    def forbidden(*args):
        raise AssertionError("Teacher crossed visual runtime boundary")
    monkeypatch.setattr(runner,"teacher_effect",forbidden)
    class DiagnosticGrounder:
        # Privileged test double for execution plumbing only, never a model score.
        calls=0
        def ground(self,picture,target):
            self.calls+=1
            execution=active[-1]
            _,boxes,_,_=visible_target_boxes(execution.env,execution.obs,"submit" if target=="Submit button" else "entry")
            return {"raw":"diagnostic", "point":center(boxes[0])}
    class VisualOnlyVerifier:
        def verify(self,view):
            assert not hasattr(view,"task_info") and not hasattr(view,"regime")
            return {"status":"unknown" if view.phase=="immediate" else "not_achieved"}
    result=runner.run_verification_case(row,"verifier",DiagnosticGrounder(),VisualOnlyVerifier(),tmp_path/"out")
    assert result["success"] and result["prefix_matched"],result
    assert result["waits"]==1 and result["repairs"]==1 and result["logical_actor_calls"]==3
