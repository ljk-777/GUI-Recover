import hashlib
import json
import os

import pytest

from gui_recover.actions import Action
from gui_recover.contextual_repair import ContextualRepairPolicy, RepairGrounder, prepare_repair
from gui_recover.sft_recovery import center, visible_target_boxes
from gui_recover.verification import TransitionView
from gui_recover.verification_env import TransitionExecution, task_request


pytestmark = [pytest.mark.integration, pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")]


def test_zoom_coordinates_really_repair_input_without_teacher_in_visual_runtime(tmp_path, monkeypatch):
    import gui_recover.verification_runner as runner
    source = TransitionExecution("enter-text", 4001, None, "drop-input")
    try:
        entry = center(visible_target_boxes(source.env, source.obs)[1][0])
        submit = center(visible_target_boxes(source.env, source.obs, "submit")[1][0])
        _, _, value = task_request("enter-text", source.obs.goal)
        source.proposed = Action("INPUT", value, tuple(entry))
        before = source.obs.screenshot.copy()
        before_path = tmp_path/"before.png"; before.save(before_path)
        source.first()
        after = source.obs.screenshot.copy()
        after_path = tmp_path/"after.png"; after.save(after_path)
        v = TransitionView(before, after, source.proposed, source.obs.goal, "text field")
        prepared = prepare_repair(v, "feedback_zoom")
        left, top, right, bottom = prepared.crop
        local = [(entry[0]*before.width-left)/(right-left), (entry[1]*before.height-top)/(bottom-top)]
        row = {"id":"engineering", "scene_id":"engineering", "task":"enter-text", "seed":4001,
            "regime":"drop-input", "status":"ok", "proposed_action":source.proposed.to_dict(),
            "before":{"sha256":hashlib.sha256(before_path.read_bytes()).hexdigest()},
            "immediate":{"sha256":hashlib.sha256(after_path.read_bytes()).hexdigest()}}
    finally:
        source.close()
    class Actor:
        # Privileged geometry is confined to this execution-only test double.
        last_metrics = {"input_tokens":10,"generated_tokens":5}
        def ground(self, picture, query):
            return json.dumps(submit if query == 'Submit button' else local)
    class Verifier:
        def verify(self, view):
            return {'status':'unknown' if view.phase=='immediate' else 'not_achieved'}
    def forbidden(*args):
        raise AssertionError('Teacher entered non-oracle runtime')
    monkeypatch.setattr(runner, 'teacher_effect', forbidden)
    out = tmp_path/'run'
    grounder = RepairGrounder(Actor(), out/'inference')
    image = out/'online/engineering-verifier_zoom/repair-input.png'
    policy = ContextualRepairPolicy(grounder, 'feedback_zoom', input_path=image)
    result = runner.run_verification_case(row, 'verifier', grounder, Verifier(), out,
                                          repairer=policy, run_name='verifier_zoom')
    assert result['success'] and result['prefix_matched'], result
    assert result['logical_actor_calls'] == 3 and result['repairs'] == result['waits'] == 1
    assert image.is_file() and result['new_actor_generations'] == 2
