import json
import os
import subprocess
import sys

import pytest

from gui_recover.actions import Action
from gui_recover.sft_recovery import center, visible_target_boxes
from gui_recover.verification import TransitionView
from gui_recover.verification_env import TransitionExecution, task_request, teacher_evidence
from gui_recover.visual_text import VisualTextVerifier


pytestmark = [pytest.mark.integration, pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")]


def test_rendered_verifier_detects_case_change_but_not_invisible_trailing_space(tmp_path):
    seed=4001
    sample=TransitionExecution("enter-text",seed,None)
    try:
        target,kind,value=task_request("enter-text",sample.obs.goal)
        plan=tmp_path/"plan.json"
        plan.write_text(json.dumps({"selected":[{"task":"enter-text","goal":sample.obs.goal}]}))
    finally:
        sample.close()
    bank=tmp_path/"bank.json"
    subprocess.run([sys.executable,"-m","gui_recover.visual_text","--plan",str(plan),"--out",str(bank)],check=True)
    verifier=VisualTextVerifier(json.loads(bank.read_text()))
    results={}
    for regime in ("none","case-input","trailing-space-input"):
        actual=TransitionExecution("enter-text",seed,None,regime)
        try:
            point=center(visible_target_boxes(actual.env,actual.obs)[1][0])
            actual.proposed=Action(kind,value,tuple(point))
            actual.first()
            view=TransitionView(actual.before.screenshot,actual.obs.screenshot,actual.proposed,actual.obs.goal,target)
            results[regime]=verifier.verify(view)["status"], teacher_evidence(actual)["observed"], view.after.tobytes()
        finally:
            actual.close()
    assert results["none"][0:2] == ("matched",value)
    assert results["case-input"][0:2] == ("different",value[:1].swapcase()+value[1:])
    assert results["trailing-space-input"][0:2] == ("matched",value+" ")
    assert results["none"][2] == results["trailing-space-input"][2]
