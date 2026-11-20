import json
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_recover.sft import file_sha256, grounding_messages, grounding_query
from gui_recover.sft_recovery import audit_corrections, select_online_cases, visible_target_boxes


def test_action_context_accepts_only_the_proposed_action_and_preserves_plain_prompt():
    action={"action":"CLICK","value":None,"position":[0.1,0.2]}
    assert grounding_query("target") == "target"
    query=grounding_query("target",action)
    assert 'Previous proposed action:' in query and '[0.1,0.2]' in query
    with pytest.raises(ValueError):
        grounding_query("target",{**action,"reward":-1})
    messages=grounding_messages(Image.new("RGB",(10,10)),"target","[0.5,0.5]",action)
    assert messages[0]["content"][-1]["text"] == query
    assert '[0.5,0.5]' not in str(messages[0])


def test_online_selection_balances_task_and_mode_without_reading_outcomes():
    rows=[{"id":"b2","task":"b","recovery_mode":"restart_after_terminal"},
          {"id":"a2","task":"a","recovery_mode":"retry_in_place"},
          {"id":"a1","task":"a","recovery_mode":"retry_in_place"}]
    assert select_online_cases(rows,2)==["a1","b2"]
    assert select_online_cases(rows,3)==["a1","b2","a2"]


def test_teacher_boxes_respect_scaling_and_visible_area():
    class Element:
        def bounding_box(self): return {"x":100,"y":10,"width":50,"height":10}
    class Page:
        _bgym_scale_factor=1.5
        def locator(self,selector): return SimpleNamespace(all=lambda:[Element()])
    env=SimpleNamespace(task="focus-text",env=SimpleNamespace(unwrapped=SimpleNamespace(page=Page())))
    obs=SimpleNamespace(goal="Focus into the textbox.",screenshot=Image.new("RGB",(200,100)))
    _,boxes,kind,_=visible_target_boxes(env,obs)
    assert boxes==[[0.75,0.15,1,0.3]] and kind=="CLICK"


def trace_case(root):
    for name,color in (("before",0),("after",255)):
        Image.new("RGB",(10,10),(color,0,0)).save(root/f"{name}.png")
    def frame(name): return {"image":f"{name}.png","sha256":file_sha256(root/f"{name}.png")}
    action={"action":"CLICK","value":None,"position":[0.1,0.1]}
    trace={"proposed_action":action,"executed_action":action,"executed":True,
        "first_action_completed":False,"initial_target_boxes":[[0.4,0.4,0.6,0.6]],
        "initial":frame("before"),"after_action":frame("after"),"current":frame("after"),
        "terminated":False,"truncated":False}
    (root/"trace.json").write_text(json.dumps(trace))
    row={"kind":"correction","source_trace":"trace.json","previous_action":action,
         "image_sha256":trace["current"]["sha256"],"recovery_mode":"retry_in_place"}
    return trace,row


def test_real_trace_audit_requires_execution_failure_and_matching_frame(tmp_path):
    trace,row=trace_case(tmp_path)
    assert audit_corrections(tmp_path,[row])=={"retry_in_place":1}
    trace["first_action_completed"]=True
    (tmp_path/"trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError,match="genuinely executed failed"):
        audit_corrections(tmp_path,[row])


def test_restart_audit_requires_terminal_source_and_identical_reset(tmp_path):
    trace,row=trace_case(tmp_path)
    row["recovery_mode"]="restart_after_terminal"
    trace["terminated"]=True
    (tmp_path/"trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError,match="Restart"):
        audit_corrections(tmp_path,[row])
