import os

import pytest

from gui_recover.multistep_study import TASKS, first_click, register_multistep


pytestmark = [pytest.mark.integration, pytest.mark.skipif(not os.getenv("MINIWOB_URL"), reason="Set MiniWoB URL")]


@pytest.mark.parametrize("task", TASKS)
def test_normal_first_click_is_live_and_completable_and_blank_does_not_progress(task, tmp_path):
    plan = register_multistep({"multistep_tasks": [
        {"task": task, "candidate_seeds": list(range(52000, 52006)), "count": 1}
    ]})
    scene = plan["accepted"][0]

    def detector(view):
        return {"status": "unknown" if view.phase == "immediate" else "not_achieved"}

    healthy = first_click(scene, "correct", None, {"test": detector}, tmp_path/"healthy")
    blank = first_click(scene, "blank", None, {"test": detector}, tmp_path/"blank")
    assert healthy["achieved"] and healthy["live"] and healthy["completed_with_oracle_continuation"]
    assert healthy["decisions"]["test"]["wait"] and healthy["decisions"]["test"]["trigger"]
    assert blank["live"] and not blank["achieved"] and blank["immediate_state"] == []
