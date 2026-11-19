import ast
import json

import pytest

from gui_recover.actions import Action, ActionError, compile_action, parse_action


@pytest.mark.parametrize("raw", [
    "__import__('os').system('echo unwanted')",
    '{"action":"CLICK","value":null,"position":[1.1,0.5]}',
    '{"action":"CLICK","value":null,"position":[true,0.5]}',
    '{"action":"CLICK","value":null,"position":[NaN,0.5]}',
    "[{'action':'CLICK','value':None,'position':[0.5,0.5]}]",
    "{'action':'SELECT','value':None,'position':[0.5,0.5]}",
    "{'action':'ENTER','value':None,'position':[0,0]}",
    "{'action':'CLICK','value':None,'position':[0,0],'code':'anything'}",
])
def test_reject_unsafe_or_ambiguous_output(raw):
    with pytest.raises(ActionError):
        parse_action(raw)


def test_text_stays_inert_and_coordinates_stay_in_screenshot_space():
    value = "'); mouse_click(0, 0)\n#"
    action = parse_action(json.dumps({"action": "INPUT", "value": value, "position": [0.5, 0.5]}))
    tree = ast.parse(compile_action(action, 498, 321))
    assert len(tree.body) == 2
    assert [n.value for n in tree.body[0].value.args] == [249, 160.5]
    assert tree.body[1].value.args[0].value == value


def test_boundary_and_stop():
    assert compile_action(Action("CLICK", position=(1, 1)), 498, 321) == "mouse_click(497, 320)"
    with pytest.raises(ActionError):
        compile_action(Action("ANSWER", "done"), 498, 321)


def test_code_fence():
    assert parse_action('```json\n{"action":"ENTER","value":null,"position":null}\n```').action == "ENTER"


def test_mixed_parser_preserves_string_values():
    raw = "{'action':'INPUT','value':'null true false None','position':[0.1,0.2]}"
    assert parse_action(raw, mode="mixed").value == "null true false None"
    raw = "{'action':'CLICK','value':null,'position':[0.1,0.2]}"
    with pytest.raises(ActionError):
        parse_action(raw)
    assert parse_action(raw, mode="mixed").action == "CLICK"


@pytest.mark.parametrize("raw", [
    "{'action':'CLICK','value':null,'position':[true,0.2]}",
    "{'action':'CLICK','value':null,'position':[float('nan'),0.2]}",
    "{'action':'CLICK','value':null,'position':__import__('os').getcwd()}",
    '{"action":"CLICK","action":"ENTER","value":null,"position":null}',
    "{'action':'CLICK','action':'ENTER','value':null,'position':null}",
    "{'action':'CLICK','value':null,'position':[0,0]}, {'action':'ENTER','value':null,'position':null}",
])
def test_mixed_parser_keeps_action_boundary(raw):
    with pytest.raises(ActionError):
        parse_action(raw, mode="mixed")


@pytest.mark.parametrize("action", [
    Action("CLICK", position=(0.1, 0.2)), Action("HOVER", position=(0.1, 0.2)),
    Action("INPUT", "quoted ' text\nline two", (0.1, 0.2)),
    Action("ENTER"), Action("SCROLL", "down"),
])
def test_compiled_actions_accepted_by_pinned_browsergym(action):
    from browsergym.core.action.highlevel import HighLevelActionSet
    mapper = HighLevelActionSet(subsets=["coord"], strict=True, multiaction=True)
    executable = mapper.to_python_code(compile_action(action, 498, 321))
    compile(executable, "<browsergym-action>", "exec")
