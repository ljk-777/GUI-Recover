from copy import deepcopy
import json

import pytest

from gui_recover.sft import file_sha256
from gui_recover.verification_revision import revise_focus_record, assemble_revised_data


def test_focus_revision_preserves_evidence_and_uses_original_success_rule():
    state = {"reward": 1.0, "terminated": True, "truncated": False,
        "task_info": {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 1},
        "effect_achieved": False, "teacher_evidence": {"kind": "input_focus", "effect_achieved": False}}
    row = {"task": "focus-text", "immediate": state, "after_wait": None}
    before = deepcopy(row)
    revised = revise_focus_record(row)
    assert row == before and revised['immediate']['effect_achieved'] is True
    assert revised['immediate']['teacher_evidence']['original_teacher_evidence'] == state['teacher_evidence']
    assert revise_focus_record(revised) == revised
    for key, value in (("terminated", False), ("truncated", True), ("reward", 0)):
        failed = deepcopy(row)
        failed['immediate'][key] = value
        assert revise_focus_record(failed)['immediate']['effect_achieved'] is False


def test_revision_rejects_reusing_an_exposed_test_seed_before_writing_output(tmp_path):
    development, fresh = tmp_path/'development', tmp_path/'fresh'
    development.mkdir(); fresh.mkdir()
    config={'splits':{'train':[], 'validation':[], 'test':[{'task':'click-test','seeds':[11000]}]}}
    (development/'manifest.json').write_text(json.dumps({'config':config}))
    (fresh/'manifest.json').write_text(json.dumps({'config':config,'partial_split':'test','runtime':{'git_commit':'new'}}))
    (development/'trace.json').write_text('{}')
    row={'id':'old','scene_id':'click-test-11000','split':'test','seed':11000,'status':'invalid_source',
         'trace':'trace.json','trace_sha256':file_sha256(development/'trace.json')}
    (development/'test.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Previously exposed test seeds'):
        assemble_revised_data(development,fresh,tmp_path/'out',config,{'git_commit':'new'})
    assert not (tmp_path/'out').exists()
